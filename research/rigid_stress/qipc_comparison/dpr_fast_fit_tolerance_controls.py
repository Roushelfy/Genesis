"""Bounded research-only pressure-stop controls on frozen mechanical inputs.

Only the instrumented Newton corrector uses the selected stopping tolerance.
The native initial solve, source-wrench acceptance and elastic checks retain
their original thresholds. Execute on Slurm rtx-mid/image1_26, one stop/rule
per process; no rigid step or production-module mutation occurs.
"""

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path

from .artifacts import runtime_identity
from .dpr_fast_fit_diagnostic import _algorithm_proof, _api, _arrays, _identity, _make_trace, _trace_summary, _write

PHYSICAL_FIELDS = (
    "valid",
    "position",
    "force",
    "normal",
    "friction",
    "radius",
    "radius_status",
    "center",
    "anchor_face",
    "frame_position",
    "frame_quaternion",
)
CONTROL_FIELDS = ("status", "is_refined", "candidate_count", "evaluations", "weight_sum", "coefficient")


def _equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _selected_problem(api, contacts, surface, slot):
    count = surface.info.face_origin.shape[0] * 3 * surface.info.shape.shape[0]
    coordinates = api.V_VEC(3, dtype=api.gs.qd_float, shape=(count,))
    weights = api.V(dtype=api.gs.qd_float, shape=(count,))
    positions = api.V_VEC(3, dtype=api.gs.qd_float, shape=(count,))
    shapes = api.V_VEC(6, dtype=api.gs.qd_float, shape=(count,))
    faces = api.V(dtype=api.gs.qd_int, shape=(count,))
    api._samples(contacts, surface.info, 0, slot, coordinates, weights, positions, shapes, faces)
    return {
        name: api.qd_to_numpy(tensor, transpose=True, copy=True)
        for name, tensor in (
            ("coordinates", coordinates),
            ("weights", weights),
            ("positions", positions),
            ("shapes", shapes),
            ("faces", faces),
        )
    }


def _sample_summary(api, samples, physical, fit, slot):
    np = api.np
    weights = samples["weights"]
    if not np.isfinite(weights).all() or (weights < 0).any():
        return {"finite_nonnegative_weights": False, "raw_samples_preserved": True}
    positive = weights > 0
    x, w, p = (samples[name][positive] for name in ("coordinates", "weights", "positions"))
    if not len(w) or not all(np.isfinite(value).all() for value in (x, w, p)):
        return {"finite_nonnegative_weights": True, "nonempty_finite_problem": False}
    coefficient = fit["coefficient"][0, slot]
    force, point, radius = (physical[name][0, slot] for name in ("force", "position", "radius"))
    weight_sum = w.sum()
    if not np.isfinite(weight_sum) or not weight_sum > 0 or not np.isfinite(coefficient).all():
        return {
            "finite_nonnegative_weights": True,
            "nonempty_finite_problem": False,
            "coefficient_and_weight_sum_finite": False,
        }
    a = coefficient * weight_sum
    profiles = x @ a
    normalized = w / weight_sum
    mass = w * np.maximum(x @ coefficient, 0)
    active = profiles > 0
    gradient = x.T @ (normalized * np.maximum(profiles, 0)) - np.array([1.0, 0.0, 0.0])
    hessian = x[active].T @ (normalized[active, None] * x[active])
    gram = x.T @ (normalized[:, None] * x)
    moment = np.cross(p - point, mass[:, None] * force).sum(axis=0)
    force_error = abs(mass.sum() - 1) * np.linalg.norm(force)
    feasible = api.scipy.optimize.linprog(
        np.zeros(len(w)), A_eq=x.T, b_eq=[1.0, 0.0, 0.0], bounds=(0, None), method="highs"
    )
    return {
        "finite_nonnegative_weights": True,
        "nonempty_finite_problem": True,
        "coefficient_and_weight_sum_finite": True,
        "positive_weight_samples": len(w),
        "positive_active_samples": int(active.sum()),
        "weight_effective_samples": float(weight_sum**2 / (w @ w)),
        "pressure_effective_samples": float(mass.sum() ** 2 / (mass @ mass)) if mass @ mass > 0 else None,
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
        "active_hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        "sampled_support_LP_success": bool(feasible.success),
        "sampled_support_LP_status": int(feasible.status),
        "force_error_N": float(force_error),
        "force_error_relative": float(force_error / np.linalg.norm(force)),
        "original_point_moment_Nm": moment.tolist(),
        "original_point_moment_error_relative": float(np.linalg.norm(moment) / (np.linalg.norm(force) * radius)),
        "meaning": "Diagnostic reintegration, including rejected coefficients. Analytically contained affine disks use exact native moments; this quadrature diagnostic does not replace the native scatter wrench gate.",
    }


def _run(api, rules, model, options, surface, frame, args, name, native):
    directory = args.output / name
    directory.mkdir()
    contacts = rules.prepare_contacts(frame)
    before = _arrays(api, contacts)
    state = model.create_state(1, "full")
    workspace = api.create_scatter_workspace(1, args.tasks_per_env, reuse_moments=True)
    rules.kernel_prepare_frozen(float(api.np.finfo(api.gs.np_float).eps), contacts, surface.info)
    trace = None if native else _make_trace(api)
    if native:
        api.kernel_pressure_moments(contacts, state, model.load_info, surface.info, workspace, True)
    else:
        api._trace_pipeline(contacts, state, model.load_info, surface.info, workspace, trace, 0, args.slot)
    fit = _arrays(api, contacts)
    api.np.savez_compressed(directory / "fit-contacts.npz", **fit)
    samples = _selected_problem(api, contacts, surface, args.slot)
    api.np.savez_compressed(directory / "sampled-problem.npz", **samples)
    summary = {"selected_sampled_problem": _sample_summary(api, samples, before, fit, args.slot)}
    if trace is not None:
        trace_arrays = {
            field.name: api.qd_to_numpy(vars(trace)[field.name], transpose=True, copy=True)
            for field in dataclasses.fields(trace)
        }
        api.np.savez_compressed(directory / "trace.npz", **trace_arrays)
        summary["trace_passes"] = [_trace_summary(api, trace_arrays, phase) for phase in range(2)]
        summary["selected_corrector_entered"] = bool(trace_arrays["count"].sum())
    rules.kernel_scatter_faces(
        contacts, state, model.load_info, surface.info, workspace, cached_bounds=True, reuse_moments=True
    )
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(1,))
    omega.from_numpy(api.np.ascontiguousarray(frame.omega))
    model.recover(omega, state, options, surface_load=True)
    api.qd.sync()
    final = _arrays(api, contacts)
    fields = {
        field.name: api.qd_to_numpy(vars(state)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(state)
    }
    api.np.savez_compressed(directory / "final-contacts.npz", **final)
    api.np.savez_compressed(directory / "state.npz", **fields, omega=frame.omega)
    physical_unchanged = {
        name: _equal(before[name], fit[name]) and _equal(before[name], final[name]) for name in PHYSICAL_FIELDS
    }
    loaded = frame.valid & (api.np.linalg.norm(frame.force, axis=-1) > 1e-30)
    force_scale = api.np.maximum(api.np.linalg.norm(frame.force, axis=-1), 1e-30)
    force_error = final["force_error"] / force_scale
    moment_error = final["moment_error"] / api.np.maximum(force_scale * frame.radius, 1e-30)
    source_passed = bool(
        api.np.all((final["status"] == 0) | ~frame.valid)
        and api.np.all((force_error <= 1e-8) | ~loaded)
        and api.np.all((moment_error <= 1e-8) | ~loaded)
        and api.np.all(api.np.isfinite(final["coefficient"]).all(axis=-1) | ~loaded)
    )
    residual = api.np.sqrt(fields["residual_norm_squared"])
    residual_budget = api.np.maximum(
        options.absolute_tolerance, options.tolerance * api.np.sqrt(fields["rhs_norm_squared"])
    )
    finite_fields = all(
        api.np.isfinite(fields[name]).all()
        for name in (
            "force",
            "rhs",
            "displacement",
            "stress_tensor",
            "von_mises",
            "peak",
            "residual_norm_squared",
            "rhs_norm_squared",
        )
    )
    elastic_passed = bool(fields["valid"].all() and (residual <= residual_budget).all() and finite_fields)
    summary.update(
        complete=True,
        physical_inputs_unchanged=physical_unchanged,
        all_original_source_wrench_gates_passed=source_passed,
        original_elastic_gates_passed=elastic_passed,
        eligible_for_field_comparison=source_passed and elastic_passed and all(physical_unchanged.values()),
        source_status_before_scatter=fit["status"].tolist(),
        source_status_after_scatter=final["status"].tolist(),
        source_force_error_relative=force_error.tolist(),
        source_original_point_moment_error_relative=moment_error.tolist(),
        native_error_fields_computed_for_successful_fit=(fit["status"] == 0).tolist(),
        native_error_field_scope="Status3 errors can be uncomputed zeros; only status0 fits enter the native wrench check. Independent selected-source reintegration is recorded separately even after rejection.",
        evaluations=fit["evaluations"].tolist(),
        evaluations_total=int(fit["evaluations"].sum()),
        elastic_residual=residual.tolist(),
        original_elastic_residual_budget=residual_budget.tolist(),
        selected_status_before_scatter=int(fit["status"][0, args.slot]),
        selected_status_after_scatter=int(final["status"][0, args.slot]),
        workspace_count=int(api.qd_to_numpy(workspace.count, transpose=True)),
        workspace_capacity=workspace.tasks.shape[0],
        scope="A rejected pressure environment's saved field is incomplete diagnostic data, never qualified stress.",
    )
    _write(directory / "result.json", summary)
    return fit, fields, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--capture-result", type=Path, required=True)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), required=True)
    parser.add_argument("--fit-tolerance", type=float, choices=(1e-9, 1e-7, 1e-5), required=True)
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid/image1_26.")
    if min(args.env, args.slot) < 0 or args.tasks_per_env < 1:
        parser.error("Use existing nonnegative indices and positive workspace capacity.")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    capture = json.loads(args.capture_result.read_text())
    archive_identity = _identity(args.archive)
    if (
        capture["schema"] != "genesis.dpr-fast-strict-nearby-capture.v1"
        or capture["complete"] is not True
        or capture["passed"] is not True
        or archive_identity not in [row["snapshot"] for row in capture["captures"]]
    ):
        parser.error("Use an exact snapshot identified by its completed nearby mechanical capture.")
    args.output.mkdir(parents=True, exist_ok=False)
    _write(
        args.output / "result.json",
        {
            "schema": "genesis.dpr-fast-fit-tolerance-controls.v1",
            "complete": False,
            "passed": False,
            "goal_complete": False,
            "application_qualified": False,
            "primary_timing": False,
            "phase": "validating_native_source_and_captured_model",
            "archive": archive_identity,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    from . import dpr_fast_rule_qualification as rules

    api = _api(args.fit_tolerance)
    api.gs.init(backend=api.gs.cuda, precision="64", seed=510000, logging_level="warning")
    if Path(api.gs.__file__).resolve().parent.parent != args.git_source.resolve():
        raise RuntimeError("Imported Genesis differs from the declared frozen source.")
    from genesis.engine.solvers.rigid.stress import contact as native_contact

    native_constant_before = native_contact._PRESSURE_FIT_TOLERANCE
    if native_constant_before != 1e-9 or api.native_fit_tolerance != 1e-9:
        raise RuntimeError("This bounded control requires the original strict native stop.")
    native_path = Path(native_contact.__file__)
    proof = _algorithm_proof(native_path)
    frame = rules.load_frame(args.archive, "selected-original-environment", False, [args.env])
    if (
        args.slot >= frame.valid.shape[1]
        or not frame.valid[0, args.slot]
        or api.np.linalg.norm(frame.force[0, args.slot]) <= 1e-30
        or frame.frame_position is None
        or frame.frame_quaternion is None
    ):
        raise ValueError("Require a loaded selected source and actual complete material-frame context.")
    if not api.np.isfinite(frame.frame_position).all() or not api.np.isfinite(frame.frame_quaternion).all():
        raise ValueError("The archived material pose contains nonfinite values.")
    mesh = Path(capture["mesh"]["path"])
    if _identity(mesh) != capture["mesh"]:
        raise RuntimeError("The physical mesh differs from its capture identity.")
    options = api.gs.options.RigidStressOptions(
        **{
            **capture["stress_options"],
            "mesh": mesh,
            "output_mode": "full",
            "quadrature": 16,
            "pressure_quadrature": args.pressure_quadrature,
        }
    )
    model = api.StressModel(options)
    if model.load_info.load_nodes.shape[0]:
        raise ValueError("This bounded study is volume-only.")
    source_model = args.capture_result.parent / "native-model.npz"
    current_model = {
        field.name: api.qd_to_numpy(vars(model.info)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(model.info)
    }
    with api.np.load(source_model, allow_pickle=False) as saved:
        model_equal = {key: _equal(value, saved[key]) for key, value in current_model.items()}
    api.np.savez_compressed(args.output / "replay-model.npz", **current_model)
    if not all(model_equal.values()):
        _write(
            args.output / "model-mismatch.json",
            {
                "complete": False,
                "passed": False,
                "model_constant_arrays_bitwise_equal": model_equal,
                "capture": _identity(source_model),
                "replay": _identity(args.output / "replay-model.npz"),
            },
        )
        raise RuntimeError("The replay elastic model must match every captured native constant array exactly.")
    identities = {
        "archive": archive_identity,
        "capture_result": _identity(args.capture_result),
        "mesh": _identity(mesh),
        "captured_native_model": _identity(source_model),
        "worker": _identity(Path(__file__)),
        "diagnostic": _identity(Path(__file__).with_name("dpr_fast_fit_diagnostic.py")),
        "frozen_input_helpers": _identity(Path(rules.__file__)),
        "native_contact": _identity(native_path),
    }
    result = {
        "schema": "genesis.dpr-fast-fit-tolerance-controls.v1",
        "complete": False,
        "passed": False,
        "goal_complete": False,
        "application_qualified": False,
        "primary_timing": False,
        "identity": runtime_identity(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "inputs": identities,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "trace_algorithm_control": proof,
        "model_constant_arrays_bitwise_equal": model_equal,
        "native_initial_fit_and_stop_tolerance": native_constant_before,
        "research_corrector_stop_tolerance": args.fit_tolerance,
        "unchanged_controls": {
            "Newton_evaluations": 80,
            "max_Armijo_trials": 40,
            "Armijo_slope": 1e-4,
            "Armijo_absolute_slack": 1e-15,
            "source_force_relative": 1e-8,
            "source_original_point_moment_relative": 1e-8,
            "elastic_relative": options.tolerance,
            "elastic_absolute": options.absolute_tolerance,
        },
        "options": options.model_dump(mode="json"),
        "scope": "Only this process's research corrector closure changes its stop. Native module, initial fit, physical sources, model, nonnegative clipping, final wrench and K gates are unchanged. No rigid advancement, primary speed claim or integration-reference/application qualification.",
    }
    _write(args.output / "result.json", result)
    api.np.savez_compressed(
        args.output / "physical-inputs.npz",
        **{
            field.name: vars(frame)[field.name]
            for field in dataclasses.fields(frame)
            if field.name not in ("path", "name", "is_complete_mechanical_input")
            and vars(frame)[field.name] is not None
        },
    )
    surface = api.StressSurface(16, model.load_info, angular=args.pressure_quadrature == "angle")
    baseline_fit, baseline_fields, baseline = _run(
        api, rules, model, options, surface, frame, args, "native-strict", True
    )
    candidate_fit, candidate_fields, candidate = _run(
        api, rules, model, options, surface, frame, args, "research-stop", False
    )
    controls = {
        key: {
            "whole_array_bitwise_equal": _equal(baseline_fit[key], candidate_fit[key]),
            "selected_bitwise_equal": _equal(baseline_fit[key][0, args.slot], candidate_fit[key][0, args.slot]),
        }
        for key in CONTROL_FIELDS
    }
    comparable = baseline["eligible_for_field_comparison"] and candidate["eligible_for_field_comparison"]
    field_comparison = {
        "eligible": comparable,
        "reference": "Same-rule unmodified native strict stop; not an integration reference.",
    }
    if comparable:
        field_comparison["arrays"] = {}
        for name in ("force", "rhs", "displacement", "stress_tensor", "von_mises", "peak"):
            expected, actual = baseline_fields[name], candidate_fields[name]
            delta_norm, reference_norm = api.np.linalg.norm(actual - expected), api.np.linalg.norm(expected)
            field_comparison["arrays"][name] = {
                "absolute_L2": float(delta_norm),
                "relative_L2": float(delta_norm / reference_norm) if reference_norm > 0 else None,
                "maximum_absolute": float(api.np.max(api.np.abs(actual - expected))),
                "bitwise_equal": _equal(expected, actual),
            }
    native_unchanged = (
        native_contact._PRESSURE_FIT_TOLERANCE == native_constant_before
        and _identity(native_path) == identities["native_contact"]
    )
    input_unchanged = all(_identity(Path(value["path"])) == value for value in identities.values())
    physical_unchanged = all(baseline["physical_inputs_unchanged"].values()) and all(
        candidate["physical_inputs_unchanged"].values()
    )
    selected_control = all(value["selected_bitwise_equal"] for value in controls.values())
    strict_control = args.fit_tolerance != 1e-9 or selected_control
    samples_valid = all(
        row["selected_sampled_problem"].get("nonempty_finite_problem", False)
        and row["selected_sampled_problem"].get("finite_nonnegative_weights", False)
        for row in (baseline, candidate)
    )
    result.update(
        complete=True,
        passed=input_unchanged and native_unchanged and physical_unchanged and strict_control and samples_valid,
        sampled_problem_diagnostics_valid=samples_valid,
        native_module_and_constant_unchanged=native_unchanged,
        input_files_unchanged=input_unchanged,
        native_vs_research_fit_controls=controls,
        original_strict_selected_control=selected_control if args.fit_tolerance == 1e-9 else None,
        original_strict_whole_array_control=all(value["whole_array_bitwise_equal"] for value in controls.values())
        if args.fit_tolerance == 1e-9
        else None,
        baseline=baseline,
        candidate=candidate,
        same_rule_strict_field_comparison=field_comparison,
        corrector_tolerance_branch_exercised_selected_source=candidate["selected_corrector_entered"],
        corrector_tolerance_branch_exercised_any_source=candidate["evaluations_total"] > 0,
        whole_environment_evaluation_change=candidate["evaluations_total"] - baseline["evaluations_total"],
        pass_semantics="Completed trustworthy bounded control, including retained rejected candidates. A candidate is numerically eligible only under its separate original source/elastic gates. The strict selected control never hides whole-array drift; no iteration reduction is claimed when the corrector was not entered.",
    )
    _write(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("Bounded replay control failed; retain all raw results without qualification.")


if __name__ == "__main__":
    main()
