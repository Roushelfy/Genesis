"""Audit one predeclared authored support/force path through native DPR stages.

This is a pressure-integration mechanism control, not a Hertz geometry-radius
history. All scientific work runs on Slurm rtx-mid/image1_26.
"""

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json

IS_SLURM = (
    bool(os.environ.get("SLURM_JOB_ID"))
    and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    and os.environ.get("GENESIS_IMAGE_VER") == "1_26"
)
if IS_SLURM:
    import numpy as np
    import quadrants as qd

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.contact import StressContactState, func_contact_section
    from genesis.engine.solvers.rigid.stress.model import StressModel
    from genesis.engine.solvers.rigid.stress.surface import (
        StressSurface,
        StressSurfaceInfo,
        func_surface_face_admitted,
    )
    from genesis.utils.array_class import V_VEC
    from genesis.utils.misc import qd_to_numpy

    from .dpr_fast_adaptive_replay import (
        add_application,
        checked_identity,
        evaluate,
        exact_array,
        finite_json,
        legacy_error,
        load_frame,
        read_npz,
        region_errors,
        runtime,
        validate_budgets,
    )

    @qd.kernel
    def kernel_geometry_routes(branches: qd.Tensor, contacts: StressContactState, surface: StressSurfaceInfo):
        for i_f, i_c, i_b in qd.ndrange(branches.shape[0], branches.shape[1], branches.shape[2]):
            supported, contained, analytical = False, False, False
            if contacts.valid[i_c, i_b] and contacts.radius[i_c, i_b] > 0.0:
                _closest, remaining, contained = func_contact_section(
                    i_f, contacts.center[i_c, i_b], contacts.radius[i_c, i_b], surface
                )
                supported = remaining > 0.0 and func_surface_face_admitted(i_f, i_c, i_b, surface)
                beta = contacts.coefficient[i_c, i_b]
                analytical = contained and contacts.hertz_profile[None]
                analytical = analytical and beta[0] - qd.sqrt(beta[1] ** 2 + beta[2] ** 2) > 1e-12 * beta.norm()
            branches[i_f, i_c, i_b] = qd.Vector(
                [gs.qd_int(supported), gs.qd_int(supported and contained), gs.qd_int(supported and analytical)]
            )


def valid_report(report):
    return (
        all(report["source_and_elastic_eligible_per_environment"])
        and all(report["public_valid_per_environment"])
        and not any(report["errno_per_environment"])
    )


def masks_for(frame, positions):
    masks = []
    for env in range(len(frame.valid)):
        loaded = frame.valid[env] & (np.linalg.norm(frame.force[env], axis=-1) > 1e-30)
        distances = positions[:, :, None, :] - frame.center[env, loaded]
        region = (np.linalg.norm(distances, axis=-1) <= 2 * frame.radius[env, loaded]).any(axis=-1)
        masks.append({"global": np.ones_like(region), "loaded": region, "remote": ~region})
    return masks


def continuity_pair(candidate, reference, previous_candidate, previous_reference, weights):
    """Separate the increment of approximation error from the physical reference increment."""
    total_weight = weights.sum()
    current_error = candidate["tensor_full9"] - reference["tensor_full9"]
    previous_error = previous_candidate["tensor_full9"] - previous_reference["tensor_full9"]
    increments = (
        current_error - previous_error,
        reference["tensor_full9"] - previous_reference["tensor_full9"],
        candidate["tensor_full9"] - previous_candidate["tensor_full9"],
        reference["tensor_full9"],
        previous_reference["tensor_full9"],
    )
    rms = [
        np.sqrt(np.sum(weights * np.sum(value**2, axis=(-2, -1)), axis=(1, 2)) / total_weight) for value in increments
    ]
    denominator = np.maximum(np.maximum(rms[3], rms[4]), 1.0)
    candidate_peak = candidate["von_mises"].max(axis=(1, 2))
    previous_candidate_peak = previous_candidate["von_mises"].max(axis=(1, 2))
    reference_peak = reference["von_mises"].max(axis=(1, 2))
    previous_reference_peak = previous_reference["von_mises"].max(axis=(1, 2))
    peak_denominator = np.maximum(np.maximum(reference_peak, previous_reference_peak), 1.0)
    signed_peak_error = candidate_peak - reference_peak
    previous_signed_peak_error = previous_candidate_peak - previous_reference_peak
    peak_error_increment = np.abs(signed_peak_error - previous_signed_peak_error)
    underestimate_growth = np.maximum(np.maximum(-signed_peak_error, 0) - np.maximum(-previous_signed_peak_error, 0), 0)
    passed = rms[0] / denominator <= 0.05
    passed &= peak_error_increment / peak_denominator <= 0.05
    passed &= underestimate_growth / peak_denominator <= 0.02
    return {
        "delta_error_tensor_RMS_Pa": rms[0].tolist(),
        "physical_reference_tensor_increment_RMS_Pa": rms[1].tolist(),
        "physical_candidate_tensor_increment_RMS_Pa": rms[2].tolist(),
        "reference_RMS_pair_denominator_Pa": denominator.tolist(),
        "delta_error_tensor_relative": (rms[0] / denominator).tolist(),
        "peak_signed_error_increment_Pa": peak_error_increment.tolist(),
        "physical_reference_peak_increment_Pa": (reference_peak - previous_reference_peak).tolist(),
        "reference_peak_pair_denominator_Pa": peak_denominator.tolist(),
        "peak_error_increment_relative": (peak_error_increment / peak_denominator).tolist(),
        "peak_underestimate_growth_Pa": underestimate_growth.tolist(),
        "peak_underestimate_growth_relative": (underestimate_growth / peak_denominator).tolist(),
        "passed_per_environment": passed.tolist(),
        "passed": bool(passed.all()),
    }


def repeat_comparison(actual, previous):
    errors = {
        key: legacy_error(actual[key], previous[key]).tolist()
        for key in ("nodal_force", "rhs", "displacement", "stress_tensor", "tensor_full9", "peak")
    }
    return {"relative_errors": errors, "passed": all(value <= 1e-8 for values in errors.values() for value in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific continuation requires Slurm rtx-mid/image1_26.")
    if not args.output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output directory.")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol["schema"] != "genesis.dpr-fast-authored-continuation-protocol.v1"
        or protocol["physical_path"]["integer_exponent_indices"] != [*range(21), *range(19, -1, -1)]
        or protocol["execution"]["reference256_subset_path_indices"] != [0, 10, 20, 30, 40]
        or protocol["execution"]["tasks_per_env"] != 128
    ):
        raise ValueError("Use the exact predeclared bounded41-point path and reference subset.")
    gates = protocol["continuity_experiment"]
    if [
        gates[key]
        for key in (
            "tensor_delta_error_relative_max",
            "peak_delta_error_relative_max",
            "peak_underestimate_growth_relative_max",
        )
    ] != [0.05, 0.05, 0.02]:
        raise ValueError("The separately predeclared continuity gates must stay5/5/2.")
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        source_revision != protocol["engine_revision"]
        or Path(gs.__file__).resolve().parent.parent != args.git_source.resolve()
    ):
        raise ValueError("Imported Genesis differs from the explicit frozen engine.")
    bound = [
        checked_identity(protocol[key])
        for key in (
            "original_manual_result",
            "original_model",
            "original_physical_input",
            "original_mesh",
            "application_budgets",
        )
    ]
    validate_budgets(Path(protocol["application_budgets"]["path"]))
    saved = json.loads(Path(protocol["original_manual_result"]["path"]).read_text())
    if (
        saved["complete"] is not True
        or saved["passed"] is not True
        or saved["authored_manual_control"] != "convex-edge"
    ):
        raise ValueError("The original authored convex-edge reference must remain qualified and unchanged.")
    gs.init(backend=gs.cuda, precision="64", seed=510000, logging_level="warning")
    base = load_frame(Path(protocol["original_physical_input"]["path"]), "manual-convex-edge", False, None)
    if base.valid.shape != (1, 1) or not base.valid.all() or np.count_nonzero(base.omega):
        raise ValueError("The continuation requires the exact one-source manual fixture with known zero omega.")
    options = gs.options.RigidStressOptions(**saved["options"])
    model = StressModel(options)
    positions = qd_to_numpy(model.info.vertices, transpose=True)[
        qd_to_numpy(model.info.elements, transpose=False)[:, :4]
    ]
    weights = np.broadcast_to(qd_to_numpy(model.info.volumes, transpose=True)[:, None] / 4, positions.shape[:-1])
    model_arrays = {
        field.name: qd_to_numpy(vars(model.info)[field.name], transpose=False)
        for field in dataclasses.fields(model.info)
    }
    model_arrays.update(sample_positions=positions, sample_weights=weights)
    old_model = read_npz(protocol["original_model"])
    if set(model_arrays) != set(old_model) or any(
        not exact_array(value, old_model[key]) for key, value in model_arrays.items()
    ):
        raise ValueError("The original K/material/mass/gauge/topology/output mapping changed.")
    args.output.mkdir(parents=True, exist_ok=False)
    indices = protocol["physical_path"]["integer_exponent_indices"]
    ratios = 64.0 ** ((np.arange(21) - 20) / 20)
    unique_frames = []
    physical = {}
    for ratio in ratios:
        radius = base.radius * ratio
        force = base.force * (radius / base.radius)[..., None] ** 3
        unique_frames.append(dataclasses.replace(base, radius=radius, force=force))
    frames = [unique_frames[index] for index in indices]
    for field in dataclasses.fields(base):
        if field.name not in ("path", "name", "is_complete_mechanical_input") and vars(base)[field.name] is not None:
            physical[field.name] = np.stack([vars(frame)[field.name] for frame in frames])
    np.savez_compressed(args.output / "frozen-physical-path.npz", exponent_indices=np.array(indices), **physical)
    np.savez_compressed(args.output / "model.npz", **model_arrays)
    candidate = runtime(model, options, frames[0], "adaptive", "cached", 128)
    reference = runtime(model, options, frames[0], "angle128", "cached", 128)
    higher = runtime(model, options, frames[0], "angle128", "cached", 128)
    higher = dataclasses.replace(higher, surface=StressSurface(256, model.load_info, angular=True))
    branches = V_VEC(3, dtype=gs.qd_int, shape=(candidate.surface.info.face_origin.shape[0], 1, 1))
    result = {
        "schema": "genesis.dpr-fast-authored-continuation.v1",
        "complete": False,
        "passed": False,
        "continuation_qualified": False,
        "goal_complete": False,
        "primary_timing": False,
        "source_revision": source_revision,
        "worker": file_id(Path(__file__)),
        "runtime_helper": file_id(Path(__file__).with_name("dpr_fast_adaptive_replay.py")),
        "identity": runtime_identity(),
        "cuda_graphs_process_environment": os.environ.get("QD_GRAPH"),
        "graph_provenance": "Process environment set before imports by the job helper; no direct imported Quadrants constant readback is claimed.",
        "protocol": file_id(args.protocol),
        "frozen_physical_path": file_id(args.output / "frozen-physical-path.npz"),
        "model": file_id(args.output / "model.npz"),
        "original_model_arrays_byte_exact": True,
        "scope": "One authored pressure-integration mechanism path, not Hertz radius sizing or a rigid trajectory. Same original model/F/point/anchor/R per comparison; no reference injection, radius repair or threshold change.",
        "rows": [],
    }
    write_json(args.output / "result.json", result)
    fields_by_index, reports_by_index, high_by_index = {}, {}, {}
    previous = None
    for step, (exponent, frame) in enumerate(zip(indices, frames)):
        directory = args.output / f"point-{step:02d}"
        directory.mkdir()
        report, fields, contact = evaluate(model, frame, "adaptive", "cached", candidate, directory / "adaptive")
        ref_report, ref_fields, _ = evaluate(model, frame, "angle128", "cached", reference, directory / "angle128")
        row = {
            "step": step,
            "exponent_index": exponent,
            "radius_m": frame.radius.tolist(),
            "force_N": frame.force.tolist(),
            "adaptive": report,
            "reference128": ref_report,
        }
        numeric = valid_report(report) and valid_report(ref_report)
        stable = True
        if step in (0, 10, 20):
            high_report, high_fields, _ = evaluate(model, frame, "angle256", "cached", higher, directory / "angle256")
            high_report["actual_private_surface_quadrature"] = 256
            high_report["public_options_quadrature_unchanged"] = higher.options.quadrature
            high_by_index[exponent] = (high_report, high_fields)
        if step in protocol["execution"]["reference256_subset_path_indices"]:
            high_report, high_fields = high_by_index[exponent]
            row["reference256"] = {
                "report": high_report,
                "reused_mirror": step in (30, 40),
                "original_path_step": exponent,
            }
            if valid_report(ref_report) and valid_report(high_report):
                stability = region_errors(ref_fields, high_fields, weights, masks_for(frame, positions))
                stable = all(
                    v["tensor_relative_L2"] <= 0.01 and v["peak_bidirectional_relative"] <= 0.005
                    for env in stability
                    for v in env.values()
                    if v["samples"]
                )
                row["reference256_stability"] = stability
            else:
                stable = False
                row["reference256_stability"] = None
        row["reference_stability_passed"] = stable
        row["original_numerical_passed"] = numeric
        if numeric and stable:
            add_application(report, fields, ref_fields, frame, positions, weights, [True])
            row["application_passed"] = all(value is True for value in report["application_passed_per_environment"])
        else:
            row["application_passed"] = False
            row["application_fields_evaluated"] = False
        kernel_geometry_routes(branches, candidate.contacts, candidate.surface.info)
        geometry = np.swapaxes(qd_to_numpy(branches, transpose=True), 1, 2)
        np.savez_compressed(directory / "geometry-routing.npz", geometry=geometry)
        row["geometry_routing"] = {
            "arrays": file_id(directory / "geometry-routing.npz"),
            "columns": [
                "supported_original_face",
                "contained_original_face",
                "accepted_beta_full_disk_analytic_criterion",
            ],
            "counts_per_source": geometry.sum(axis=2).tolist(),
            "post_fit_affine_cache_flags": qd_to_numpy(candidate.workspace.affine_pressure, transpose=True)[
                :1
            ].tolist(),
            "scope": "Actual original section geometry and post-fit evaluator criterion; not a count of every Newton/backtracking analytic call. Native selected rule/checker/attempts and counter delta are recorded separately.",
        }
        selected = contact["admission.rule"][..., 0].tolist()
        row["selected_rule_indices"] = selected
        row["selected_rule_degrees"] = (np.array([8, 16, 64, 128])[contact["admission.rule"][..., 0]]).tolist()
        row["continuity"] = None
        if previous is not None and numeric and stable and previous["numeric"]:
            row["continuity"] = continuity_pair(fields, ref_fields, previous["fields"], previous["reference"], weights)
            row["selected_rule_changed"] = selected != previous["selected"]
        else:
            row["selected_rule_changed"] = False
        if step > 20:
            first = fields_by_index[exponent]
            row["round_trip"] = repeat_comparison(fields, first)
            row["round_trip"]["selected_rule_exact"] = selected == reports_by_index[exponent]["selected_rule_indices"]
            row["round_trip"]["passed"] &= row["round_trip"]["selected_rule_exact"]
            row["round_trip"]["physical_inputs_byte_exact"] = all(
                exact_array(vars(frame)[key], vars(frames[exponent])[key]) for key in physical
            )
            row["round_trip"]["passed"] &= row["round_trip"]["physical_inputs_byte_exact"]
        else:
            fields_by_index[exponent], reports_by_index[exponent] = fields, row
        result["rows"].append(row)
        previous = {"fields": fields, "reference": ref_fields, "selected": selected, "numeric": numeric and stable}
        write_json(args.output / "result.json", finite_json(result))
    released = dataclasses.replace(frames[0], valid=np.zeros_like(base.valid), force=np.zeros_like(base.force))
    release_report, release_fields, _ = evaluate(
        model, released, "adaptive", "cached", candidate, args.output / "release"
    )
    reload_report, reload_fields, reload_contact = evaluate(
        model, frames[0], "adaptive", "cached", candidate, args.output / "reload"
    )
    release_zero = all(
        np.count_nonzero(release_fields[key]) == 0
        for key in ("nodal_force", "rhs", "displacement", "stress_tensor", "peak")
    )
    reload_parity = repeat_comparison(reload_fields, fields_by_index[0])
    reload_rule_exact = (
        reload_contact["admission.rule"][..., 0].tolist() == reports_by_index[0]["selected_rule_indices"]
    )
    result["lifecycle_tail"] = {
        "release": release_report,
        "reload": reload_report,
        "release_exact_zero": release_zero,
        "reload_parity": reload_parity,
        "reload_rule_exact": reload_rule_exact,
        "passed": bool(
            valid_report(release_report)
            and valid_report(reload_report)
            and release_zero
            and reload_parity["passed"]
            and reload_rule_exact
        ),
    }
    rows = result["rows"]
    outward_switches = [row["step"] for row in rows[1:21] if row["selected_rule_changed"]]
    return_switches = [row["step"] for row in rows[21:] if row["selected_rule_changed"]]
    result["mechanism"] = {
        "outward_switch_steps": outward_switches,
        "return_switch_steps": return_switches,
        "higher_rule_points": sum(
            any(value > 1 for source in row["selected_rule_indices"] for value in source) for row in rows
        ),
        "switching_exercised": bool(outward_switches and return_switches),
        "repair_exercised": any(
            any(value > 1 for source in row["selected_rule_indices"] for value in source) for row in rows
        ),
    }
    result["all_original_numerical_passed"] = all(row["original_numerical_passed"] for row in rows)
    result["all_point_application_passed"] = all(row["application_passed"] for row in rows)
    result["all_declared_reference_stability_passed"] = all(row["reference_stability_passed"] for row in rows)
    result["continuity_experiment_passed"] = all(
        row["continuity"] is not None and row["continuity"]["passed"] for row in rows[1:]
    )
    result["round_trip_passed"] = all(row["round_trip"]["passed"] for row in rows[21:])
    errors6 = [value for row in rows for value in row["adaptive"].get("legacy_errors", {}).get("stress_tensor", [])]
    peaks = [value for row in rows for value in row["adaptive"].get("legacy_errors", {}).get("peak", [])]
    result["legacy_spatial_5mean_10worst_5peak"] = {
        "loaded_environment_frames": len(errors6),
        "mean_error6": sum(errors6) / len(errors6) if errors6 else None,
        "worst_error6": max(errors6, default=None),
        "peak_worst": max(peaks, default=None),
        "passed": len(errors6) == 41 and sum(errors6) / 41 <= 0.05 and max(errors6) <= 0.1 and max(peaks) <= 0.05,
    }
    result["legacy_volume_parity_passed"] = all(
        all(row["adaptive"].get("legacy_volume_parity_passed_per_environment", [False])) for row in rows
    )
    result["input_and_reference_artifacts_unchanged"] = bound == [file_id(row["path"]) for row in bound]
    result["frozen_physical_path_unchanged"] = (
        file_id(args.output / "frozen-physical-path.npz") == result["frozen_physical_path"]
    )
    result["complete"] = len(rows) == 41
    result["all_declared_nonmechanism_gates_passed"] = bool(
        result["complete"]
        and result["all_original_numerical_passed"]
        and result["all_point_application_passed"]
        and result["all_declared_reference_stability_passed"]
        and result["continuity_experiment_passed"]
        and result["round_trip_passed"]
        and result["lifecycle_tail"]["passed"]
        and result["input_and_reference_artifacts_unchanged"]
        and result["frozen_physical_path_unchanged"]
    )
    result["passed"] = bool(
        result["all_declared_nonmechanism_gates_passed"]
        and result["mechanism"]["switching_exercised"]
        and result["mechanism"]["repair_exercised"]
    )
    result["continuation_qualified"] = result["passed"]
    result["outcome"] = (
        "qualified_bounded_authored_path"
        if result["passed"]
        else (
            "inconclusive_no_switch"
            if result["all_declared_nonmechanism_gates_passed"] and not result["mechanism"]["switching_exercised"]
            else "retained_negative"
        )
    )
    write_json(args.output / "result.json", finite_json(result))
    if not result["passed"]:
        raise RuntimeError(
            "The predeclared authored continuation was negative or did not exercise switching; preserve all evidence."
        )


if __name__ == "__main__":
    main()
