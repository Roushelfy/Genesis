"""Replay production adaptive DPR against saved frozen-input angular references.

Scientific execution is restricted to Slurm rtx-mid/image1_26. This worker
reuses saved q128 fields; it never solves a new reference or advances rigid
state. The original raw detector may reject inputs: those are retained results.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json

IS_SLURM = (
    bool(os.environ.get("SLURM_JOB_ID"))
    and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    and os.environ.get("GENESIS_IMAGE_VER") == "1_26"
)
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
)
CONTEXT_FIELDS = ("frame_position", "frame_quaternion")
COUNTER_COLUMNS = (
    "attempted_contacts",
    "checker_evaluations",
    "angle16_attempts",
    "angle64_attempts",
    "angle128_attempts",
    "repaired_contacts",
    "rejected_contacts",
    "overflow_replays",
    "initial_affine_fits",
)

if IS_SLURM:
    import numpy as np
    import quadrants as qd

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.adaptive_pressure import kernel_adaptive_pressure
    from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts, kernel_scatter
    from genesis.engine.solvers.rigid.stress.data import StressState
    from genesis.engine.solvers.rigid.stress.model import StressModel
    from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
    from genesis.engine.solvers.rigid.stress.scatter import (
        StressScatterWorkspace,
        create_scatter_workspace,
        kernel_pressure_moments,
        kernel_scatter_faces,
    )
    from genesis.engine.solvers.rigid.stress.surface import StressSurface
    from genesis.utils.array_class import V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    from .dpr_fast_rule_qualification import (
        finite_json,
        kernel_prepare_frozen,
        legacy_error,
        load_frame,
        region_errors,
        tensor_matrix,
    )

    @dataclasses.dataclass(frozen=True)
    class ReplayRuntime:
        options: gs.options.RigidStressOptions
        surface: StressSurface
        contacts: StressContactState
        workspace: StressScatterWorkspace
        state: StressState
        omega: qd.Tensor
        errno: qd.Tensor
        reuse_moments: bool


def checked_identity(record):
    """Fail closed when a referenced immutable artifact changed."""
    actual = file_id(Path(record["path"]))
    if actual != record:
        raise ValueError(f"Changed frozen artifact: {record['path']}")
    return actual


def array_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def exact_array(actual, expected):
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C")
    )


def validate_cpu_model_control(control, reference_result, reference_model, reference_options):
    """Validate the explicit CPU arithmetic receipt without relaxing array identity."""
    flags = (
        "complete",
        "diagnostic_execution_completed",
        "original_artifacts_unchanged",
        "physical_topology_gauge_samples_exact",
        "options_exact",
    )
    if control.get("schema") != "genesis.dpr-fast-CPU-model-roundoff-diagnostic.v1" or any(
        control.get(key) is not True for key in flags
    ):
        raise ValueError("CPU model control requires a completed, exactly bound model diagnostic.")
    if (
        control["original_result"] != reference_result
        or control["original_model"] != reference_model
        or control["options"] != reference_options
    ):
        raise ValueError("CPU model diagnostic belongs to a different original reference/model/options.")
    exact_fields = (
        "vertices",
        "elements",
        "surface_nodes",
        "row_start",
        "columns",
        "pins",
        "is_free",
        "sample_positions",
    )
    if any(control["field_comparisons"][key]["bytes_exact"] is not True for key in exact_fields):
        raise ValueError("CPU model diagnostic changed physical topology, gauge or sample positions.")
    for values in control["field_comparisons"].values():
        if values["dtype_and_shape_exact"] is not True:
            raise ValueError("CPU model diagnostic changed an array type or shape.")
        if values["CPU"]["dtype"].endswith(("f4", "f8")):
            if values["CPU_nonfinite_entries"] != 0 or values["reference_nonfinite_entries"] != 0:
                raise ValueError("CPU model diagnostic contains nonfinite model values.")
        elif values["bytes_exact"] is not True:
            raise ValueError("CPU model diagnostic changed integer or Boolean topology.")
    checks = control["independent_CSR_mass_relief_controls"]
    for key in ("CSR_values", "recomputed_relief", "own_K_solved_displacement", "own_mass_relieved_rhs"):
        values = checks[key]
        if (
            values["dtype_and_shape_exact"] is not True
            or not math.isfinite(values["relative_L2_difference"])
            or not 0.0 <= values["relative_L2_difference"] <= 1e-8
        ):
            raise ValueError("CPU model diagnostic failed the original numerical operator control scale.")
    for key in ("CPU_original_K_checks", "reference_original_K_checks"):
        if not checks[key]:
            raise ValueError("Missing original own-K residual controls.")
        for row in checks[key]:
            residual, budget = row["original_free_K_residual"], row["original_K_residual_budget"]
            wrench = row["relieved_rigid_wrench_relative"]
            if (
                row["original_K_gate_passed_for_independent_solve"] is not True
                or row["pinned_displacement_exact_zero"] is not True
                or not math.isfinite(residual)
                or not math.isfinite(budget)
                or not math.isfinite(wrench)
                or not 0.0 <= residual <= budget
                or budget < reference_options["absolute_tolerance"]
                or not 0.0 <= wrench <= 1e-8
            ):
                raise ValueError("CPU model diagnostic failed original K, gauge or relief checks.")


def read_npz(identity):
    checked_identity(identity)
    with np.load(identity["path"], allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def tensor_arrays(value, prefix=""):
    """Read nested public typed state without assuming every field is a tensor."""
    result = {}
    for field in dataclasses.fields(value):
        item = vars(value)[field.name]
        key = prefix + field.name
        if dataclasses.is_dataclass(item):
            result.update(tensor_arrays(item, key + "."))
        else:
            result[key] = qd_to_numpy(item, transpose=True, copy=True)
    return result


def clear_tensors(value):
    for field in dataclasses.fields(value):
        item = vars(value)[field.name]
        if dataclasses.is_dataclass(item):
            clear_tensors(item)
        else:
            item.fill(0)


def set_physical(contacts, frame):
    for key in PHYSICAL_FIELDS:
        vars(contacts)[key].from_numpy(np.ascontiguousarray(np.swapaxes(vars(frame)[key], 0, 1)))
    for key in CONTEXT_FIELDS:
        value = vars(frame)[key]
        if value is not None:
            vars(contacts)[key].from_numpy(np.ascontiguousarray(value))


def physical_identity(contacts, omega, frame):
    result = {key: array_identity(qd_to_numpy(vars(contacts)[key], transpose=True)) for key in PHYSICAL_FIELDS}
    result["omega"] = array_identity(qd_to_numpy(omega, transpose=True))
    for key in CONTEXT_FIELDS:
        if vars(frame)[key] is not None:
            result[key] = array_identity(qd_to_numpy(vars(contacts)[key], transpose=True))
    return result


def state_arrays(state):
    arrays = {
        name: qd_to_numpy(tensor, transpose=True, copy=True)
        for name, tensor in (
            ("nodal_force", state.force),
            ("rhs", state.rhs),
            ("displacement", state.displacement),
            ("stress_tensor", state.stress_tensor),
            ("von_mises", state.von_mises),
            ("peak", state.peak),
            ("elastic_valid", state.valid),
            ("residual_squared", state.residual_norm_squared),
            ("rhs_squared", state.rhs_norm_squared),
        )
    }
    arrays["von_mises"] = arrays["von_mises"][..., 0]
    arrays["tensor_full9"] = tensor_matrix(arrays["stress_tensor"])
    return arrays


def runtime(model, baseline_options, frame, rule, mode, tasks_per_env):
    adaptive = rule == "adaptive"
    quadrature = 16 if adaptive else int(rule[5:] if rule.startswith("angle") else rule[4:])
    option_values = baseline_options.model_dump(mode="json")
    option_values.update(
        pressure_quadrature="adaptive" if adaptive else ("angle" if rule.startswith("angle") else "edge"),
        quadrature=quadrature,
    )
    options = gs.options.RigidStressOptions(**option_values)
    surface = StressSurface(quadrature, model.load_info, angular=rule.startswith("angle"), adaptive=adaptive)
    contacts = create_contacts(frame.valid.shape[1], frame.valid.shape[0], 0.001, hertz_profile=True, adaptive=adaptive)
    clear_tensors(contacts)
    contacts.hertz_profile.fill(True)
    reuse = mode in ("cached", "overflow") and gs.backend == gs.cuda
    capacity = 1 if mode == "overflow" else tasks_per_env
    workspace = create_scatter_workspace(frame.valid.shape[0], capacity, reuse_moments=reuse, adaptive=adaptive)
    state = model.create_state(frame.valid.shape[0], "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(frame.valid.shape[0],))
    omega.from_numpy(np.ascontiguousarray(frame.omega))
    errno = V(dtype=gs.qd_int, shape=(frame.valid.shape[0],))
    errno.fill(0)
    return ReplayRuntime(options, surface, contacts, workspace, state, omega, errno, reuse)


def evaluate(model, frame, rule, mode, native, directory):
    directory.mkdir()
    options, surface, contacts = native.options, native.surface, native.contacts
    workspace, state, omega = native.workspace, native.state, native.omega
    errno, reuse = native.errno, native.reuse_moments
    counter_before = qd_to_numpy(contacts.admission.counters, transpose=True, copy=True) if rule == "adaptive" else None
    set_physical(contacts, frame)
    before = physical_identity(contacts, omega, frame)
    expected = {key: array_identity(vars(frame)[key]) for key in PHYSICAL_FIELDS}
    expected["omega"] = array_identity(frame.omega)
    expected.update({key: array_identity(vars(frame)[key]) for key in CONTEXT_FIELDS if vars(frame)[key] is not None})
    if before != expected:
        raise ValueError("Native source/anchor/omega/pose storage differs from the frozen typed input.")
    errno.fill(0)
    kernel_begin_step(state)
    kernel_prepare_frozen(float(np.finfo(gs.np_float).eps), contacts, surface.info)
    qd.sync()
    start = time.perf_counter()
    if rule == "adaptive":
        kernel_adaptive_pressure(
            contacts, state, model.load_info, surface.info, workspace, mode != "serial", mode != "uncached", reuse
        )
        if mode == "serial" or gs.backend != gs.cuda:
            kernel_scatter(contacts, state, model.load_info, surface.info, cooperative=False)
        else:
            kernel_scatter_faces(contacts, state, model.load_info, surface.info, workspace, mode != "uncached", reuse)
    else:
        kernel_pressure_moments(contacts, state, model.load_info, surface.info, workspace, cached_bounds=True)
        kernel_scatter_faces(contacts, state, model.load_info, surface.info, workspace, True, True)
    model.recover(omega, state, options, surface_load=True)
    qd.sync()
    elapsed = time.perf_counter() - start
    raw = state_arrays(state)
    contact = tensor_arrays(contacts)
    after = physical_identity(contacts, omega, frame)
    unchanged = {key: before[key] == after[key] for key in before}
    if not all(unchanged.values()):
        raise ValueError("A physical source, original anchor, body omega or pose changed during replay.")
    loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
    force_scale = np.linalg.norm(frame.force, axis=-1)
    force_error = contact["force_error"] / np.maximum(force_scale, 1e-30)
    moment_error = contact["moment_error"] / np.maximum(force_scale * frame.radius, 1e-30)
    source = np.all((contact["status"] == 0) | ~frame.valid, axis=1)
    source &= np.all((force_error <= 1e-8) | ~loaded, axis=1)
    source &= np.all((moment_error <= 1e-8) | ~loaded, axis=1)
    source &= np.all(np.isfinite(contact["coefficient"]).all(axis=-1) | ~loaded, axis=1)
    residual = np.sqrt(raw["residual_squared"])
    residual_budget = np.maximum(options.absolute_tolerance, options.tolerance * np.sqrt(raw["rhs_squared"]))
    finite = np.ones(len(frame.valid), dtype=bool)
    for value in raw.values():
        finite &= np.isfinite(value).reshape((len(finite), -1)).all(axis=1)
    eligible = source & raw["elastic_valid"] & (residual <= residual_budget) & finite
    np.savez_compressed(directory / "raw-fields.npz", **raw, omega=frame.omega)
    np.savez_compressed(directory / "contacts.npz", **contact)
    kernel_accept(state, contacts, errno)
    qd.sync()
    accepted = state_arrays(state)
    accepted["step_valid"] = qd_to_numpy(state.step_valid, transpose=True, copy=True)
    accepted["step_peak"] = qd_to_numpy(state.step_peak, transpose=True, copy=True)
    error_codes = qd_to_numpy(errno, transpose=True, copy=True)
    public = accepted["step_valid"]
    np.savez_compressed(directory / "accepted-fields.npz", **accepted, errno=error_codes, omega=frame.omega)
    admission = {
        key.removeprefix("admission."): value.tolist() for key, value in contact.items() if key.startswith("admission.")
    }
    report = {
        "rule": rule,
        "mode": mode,
        "options": options.model_dump(mode="json"),
        "raw_fields": file_id(directory / "raw-fields.npz"),
        "accepted_fields": file_id(directory / "accepted-fields.npz"),
        "contacts": file_id(directory / "contacts.npz"),
        "intrusive_seconds_including_runtime_JIT": elapsed,
        "timing_scope": "Diagnostic one-pass replay, including first-use JIT; not a throughput measurement.",
        "source_and_elastic_eligible_per_environment": eligible.tolist(),
        "public_valid_per_environment": public.tolist(),
        "errno_per_environment": error_codes.tolist(),
        "physical_identity_before": before,
        "physical_identity_after": after,
        "physical_inputs_bitwise_unchanged": unchanged,
        "unavailable_context_fields": [key for key in CONTEXT_FIELDS if vars(frame)[key] is None],
        "status": contact["status"].tolist(),
        "evaluations": contact["evaluations"].tolist(),
        "force_error_relative": force_error.tolist(),
        "original_point_moment_error_relative": moment_error.tolist(),
        "K_residual": residual.tolist(),
        "K_residual_budget": residual_budget.tolist(),
        "workspace_actual_count": int(qd_to_numpy(workspace.count)),
        "workspace_capacity": workspace.tasks.shape[0],
        "workspace_overflow_calls": int(qd_to_numpy(workspace.overflow_calls)),
        "admission": admission,
        "admission_counter_delta": (contact["admission.counters"] - counter_before).tolist()
        if counter_before is not None
        else None,
        "admission_counter_scope": "Cumulative typed counters retained across these phases; delta is this evaluate call only.",
        "admission_counter_columns": list(COUNTER_COLUMNS),
        "stage_rules": [16, 64, 128],
        "stage_checker_rules": [8, 16, 64],
        "admission_version": 4,
        "admission_error_columns": [
            "independently_fitted_checker_gradient_norm",
            "face_P2_L1",
            "candidate_gradient_norm",
            "source_assembled_P2_nodal_L1",
        ],
        "stage_source_error_columns": [
            "candidate_relative_force",
            "candidate_relative_point_moment",
            "checker_relative_force",
            "checker_relative_point_moment",
        ],
        "complete": True,
    }
    return report, raw, contact


def compare_fields(actual, expected):
    return {
        key: legacy_error(actual[key], expected[key]).tolist()
        for key in ("nodal_force", "displacement", "stress_tensor", "peak")
    }


def parity_passed(errors):
    count = len(errors["peak"])
    return [
        all(values[env] < (1e-6 if key == "nodal_force" else 1e-4) for key, values in errors.items())
        for env in range(count)
    ]


def independent_scores(frame, references, face_nodes, node_count):
    """Assemble saved independently fitted face differences with NumPy add.at."""
    loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
    scores = np.zeros((*frame.valid.shape, 3))
    identities = []
    saved_moments = {}
    for rule in ("angle8", "angle16", "angle64", "angle128"):
        identity = references[rule]["moment_witness"]
        saved_moments[rule] = read_npz(identity)["face_load_fractions"]
        identities.append(identity)
    for stage, (candidate, checker) in enumerate(
        (("angle16", "angle8"), ("angle64", "angle16"), ("angle128", "angle64"))
    ):
        difference = saved_moments[candidate] - saved_moments[checker]
        if difference.shape != (*frame.valid.shape, len(face_nodes), 6):
            raise ValueError("Saved P2 witness has a different source/face/node direction.")
        for env, contact in np.argwhere(loaded):
            nodal = np.zeros(node_count)
            for local in range(6):
                np.add.at(nodal, face_nodes[:, local], difference[env, contact, :, local])
            scores[env, contact, stage] = np.abs(nodal).sum()
    return scores, identities


def compare_stage_scores(report, contact, frame, scores):
    attempts = contact["admission.attempts"][..., 0]
    loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
    applicable = loaded[..., None] & (np.arange(3) < attempts[..., None])
    applicable &= contact["admission.stage_status"] == 0
    applicable &= contact["admission.stage_checker_status"] == 0
    native_scores = contact["admission.stage_errors"][..., 3]
    difference = np.abs(native_scores - scores)
    score_passed = (difference <= 1e-8) | ~applicable
    reason = contact["admission.stage_reason"]
    admitted = applicable & (reason == 0)
    false_admission = admitted & (scores > 1e-3)
    report["independent_stage_operator_check"] = {
        "scope": "Saved independently fitted source-node operator assembled using original topology. No new reference fit; dimensionless source fraction RHS scale is one.",
        "absolute_score_parity_tolerance": 1e-8,
        "admission_threshold_unchanged": 1e-3,
        "saved_source_node_L1": scores.tolist(),
        "native_source_node_L1": native_scores.tolist(),
        "comparable_stage_mask": applicable.tolist(),
        "absolute_score_difference": difference.tolist(),
        "maximum_comparable_absolute_difference": float(difference[applicable].max(initial=0.0)),
        "native_accepted_but_saved_score_failed": false_admission.tolist(),
        "passed": bool(score_passed.all() and not false_admission.any()),
    }


def add_application(report, fields, reference, frame, positions, weights, reference_valid):
    loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
    masks = []
    for env in range(len(frame.valid)):
        delta = positions[:, :, None, :] - frame.center[env, loaded[env]]
        region = (np.linalg.norm(delta, axis=-1) <= 2 * frame.radius[env, loaded[env]]).any(axis=-1)
        masks.append({"global": np.ones_like(region), "loaded": region, "remote": ~region})
    regions = region_errors(fields, reference, weights, masks)
    application = []
    for env, values in enumerate(regions):
        passed = report["source_and_elastic_eligible_per_environment"][env]
        passed &= report["public_valid_per_environment"][env] and report["errno_per_environment"][env] == 0
        passed &= all(row["passed"] is not False for row in values.values())
        if not loaded[env].any() and not np.any(frame.omega[env]):
            maximum = float(np.linalg.norm(fields["tensor_full9"][env], axis=(-2, -1)).max())
            values["zero_load"] = {"maximum_tensor_Frobenius_Pa": maximum, "passed": maximum <= 1.0}
            passed &= maximum <= 1.0
        application.append(bool(passed) if reference_valid[env] else None)
    errors = compare_fields(fields, reference)
    spatial = [
        bool(
            reference_valid[env]
            and report["source_and_elastic_eligible_per_environment"][env]
            and errors["stress_tensor"][env] <= 0.1
            and errors["peak"][env] <= 0.05
        )
        for env in range(len(frame.valid))
    ]
    mean = float(np.array(errors["stress_tensor"])[loaded.any(axis=1)].mean()) if loaded.any() else None
    report.update(
        application_regions=regions,
        application_passed_per_environment=application,
        truth_evaluable_per_environment=reference_valid,
        legacy_errors=errors,
        legacy_volume_parity_passed_per_environment=parity_passed(errors),
        legacy_volume_parity_definition="Numerical parity only: load<1e-6; displacement/stored6 tensor/peak<1e-4. This is separate from application accuracy.",
        original_spatial_worst10_peak5_passed_per_environment=spatial,
        original_spatial_loaded_mean_error6=mean,
        original_spatial_mean5_passed=mean is not None and mean <= 0.05,
    )


def validate_budgets(path):
    budgets = json.loads(path.read_text())
    gates = {
        "tensor_relative_L2_max": 0.05,
        "sampled_peak_bidirectional_relative_max": 0.05,
        "sampled_peak_underestimate_relative_max": 0.02,
    }
    if any(budgets["application_gates"][key] != gates for key in ("global", "loaded", "remote")):
        raise ValueError("Replay must retain the frozen v1 regional application thresholds.")
    if budgets["metrics"]["floor_values"] != {
        "absolute_stress_floor_Pa": 1.0,
        "regional_floor_fraction_of_reference_global_peak": 0.01,
    }:
        raise ValueError("Replay must retain the original stress normalization floors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-run", type=Path, required=True, help="Existing rule worker directory or result.json."
    )
    parser.add_argument("--case", help="Exact original rows[].name; required for multi-row references.")
    parser.add_argument("--modes", nargs="+", choices=("cached", "uncached", "serial", "overflow"), default=["cached"])
    parser.add_argument("--fixed-controls", nargs="*", choices=("edge16", "angle128"), default=[])
    parser.add_argument(
        "--lifecycle", action="store_true", help="Reuse each runtime for release then source-slot permutation/reload."
    )
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--cpu-model-control",
        type=Path,
        help="Explicit separately diagnosed CPU arithmetic model; CUDA keeps the original exact guard.",
    )
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific execution requires Slurm rtx-mid/image1_26.")
    if (
        args.tasks_per_env < 1
        or len(set(args.modes)) != len(args.modes)
        or len(set(args.fixed_controls)) != len(args.fixed_controls)
    ):
        parser.error("Require positive workspace capacity and unique modes/fixed controls.")
    if args.backend == "cpu" and args.fixed_controls:
        parser.error(
            "Fixed-route parity reuses the original CUDA face-parallel control; omit --fixed-controls for CPU."
        )
    if args.cpu_model_control is not None and args.backend != "cpu":
        parser.error("--cpu-model-control is restricted to the explicit portable CPU control.")
    if Path(gs.__file__).resolve().parent.parent != args.git_source.resolve():
        raise ValueError("Imported Genesis differs from the explicit immutable --git-source.")
    validate_budgets(args.budgets)
    reference_path = args.reference_run / "result.json" if args.reference_run.is_dir() else args.reference_run
    saved = json.loads(reference_path.read_text())
    if (
        saved["schema"] != "genesis.dpr-fast-frozen-rule-qualification.v1"
        or saved["complete"] is not True
        or saved["passed"] is not True
    ):
        raise ValueError("Use a terminal trustworthy frozen-rule reference.")
    rows = [row for row in saved["rows"] if args.case is None or row["name"] == args.case]
    if len(rows) != 1:
        raise ValueError("Select exactly one original row by its exact name.")
    row = rows[0]
    references = {record["rule"]: record for record in row["rules"]}
    reference_record = references["angle128"]
    if not all(row["reference_evaluable_per_environment"]) or not all(
        reference_record["source_and_elastic_eligible_per_environment"]
    ):
        raise ValueError("Saved q128 reference did not pass its original source/K/reference gates.")
    bound = [
        file_id(reference_path),
        checked_identity(saved["model"]),
        checked_identity(row["physical_inputs"]),
        checked_identity(reference_record["fields"]),
        file_id(args.budgets),
    ]
    bound.extend(
        checked_identity(references[rule][key]) for rule in args.fixed_controls for key in ("fields", "contacts")
    )
    bound.extend(
        checked_identity(identity)
        for identity in (
            row["archived_material_binding"],
            row["archived_topology_binding"],
            saved.get("authored_manual_plan"),
        )
        if identity is not None
    )
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", seed=510000, logging_level="warning")
    frame = load_frame(Path(row["physical_inputs"]["path"]), row["name"], False, None)
    frame = dataclasses.replace(
        frame,
        environment_indices=np.array(row["original_environment_indices"]),
        is_complete_mechanical_input=row["whole_frame_qualification"],
    )
    options = gs.options.RigidStressOptions(**saved["options"])
    if options.backend != "volume" or options.radius_model != "hertz" or options.pressure_profile != "hertz":
        raise ValueError("This bounded replay implements the frozen volume/Hertz pressure model only.")
    model = StressModel(options)
    original_model = read_npz(saved["model"])
    positions = qd_to_numpy(model.info.vertices, transpose=True)[
        qd_to_numpy(model.info.elements, transpose=False)[:, :4]
    ]
    weights = np.broadcast_to(qd_to_numpy(model.info.volumes, transpose=True)[:, None] / 4, positions.shape[:-1])
    current_model = {
        field.name: qd_to_numpy(vars(model.info)[field.name], transpose=False)
        for field in dataclasses.fields(model.info)
    }
    current_model.update(sample_positions=positions, sample_weights=weights)
    model_exact = set(original_model) == set(current_model) and all(
        exact_array(current_model[key], original_model[key]) for key in current_model
    )
    cpu_model_control = None
    if args.cpu_model_control is not None:
        control_identity = file_id(args.cpu_model_control)
        control = json.loads(args.cpu_model_control.read_text())
        validate_cpu_model_control(control, file_id(reference_path), saved["model"], saved["options"])
        control_model = read_npz(control["CPU_model"])
        mesh_binding = next(
            (
                identity
                for identity in saved["inputs"]
                if Path(identity["path"]).resolve() == Path(options.mesh).resolve()
            ),
            None,
        )
        if mesh_binding is None:
            raise ValueError("Original mesh identity is missing from the bound reference inputs.")
        checked_identity(mesh_binding)
        physical_exact = all(
            exact_array(current_model[key], original_model[key])
            for key in (
                "vertices",
                "elements",
                "surface_nodes",
                "row_start",
                "columns",
                "pins",
                "is_free",
                "sample_positions",
            )
        )
        cpu_exact = set(current_model) == set(control_model) and all(
            exact_array(current_model[key], control_model[key]) for key in current_model
        )
        if not physical_exact or not cpu_exact:
            np.savez_compressed(args.output / "failed-fresh-CPU-model.npz", **current_model)
            write_json(
                args.output / "CPU-model-control-failure.json",
                {
                    "schema": "genesis.dpr-fast-CPU-model-exact-control-failure.v1",
                    "passed": False,
                    "diagnostic": control_identity,
                    "expected_CPU_model": control["CPU_model"],
                    "actual_CPU_model": file_id(args.output / "failed-fresh-CPU-model.npz"),
                    "physical_fields_exact": physical_exact,
                    "all_CPU_fields_exact": cpu_exact,
                    "field_identities": {
                        key: {
                            "actual": array_identity(value),
                            "diagnosed_CPU": array_identity(control_model[key]) if key in control_model else None,
                            "exact": key in control_model and exact_array(value, control_model[key]),
                        }
                        for key, value in current_model.items()
                    },
                    "scope": "Fresh CPU byte guard failed; no allclose or reference upload was substituted.",
                },
            )
            raise ValueError("Fresh CPU model differs from its separately diagnosed exact CPU arithmetic control.")
        cpu_model_control = {
            "schema": "genesis.dpr-fast-portable-CPU-model-control.v1",
            "diagnostic": control_identity,
            "CPU_model": control["CPU_model"],
            "original_mesh": mesh_binding,
            "fresh_CPU_all_arrays_bitwise_equal": cpu_exact,
            "original_physical_topology_gauge_samples_exact": physical_exact,
            "diagnostic_qualification_passed": control["qualification_passed"],
            "scope": "Exact separately constructed CPU arithmetic control; original GPU arrays need not be byte-identical. No reference model data is uploaded to production.",
        }
        bound.extend(
            [control_identity, control["CPU_model"], checked_identity(control["independent_controls"]), mesh_binding]
        )
    if not model_exact and cpu_model_control is None:
        raise ValueError("Original model/K/mass/gauge/topology/sample arrays differ from the saved reference.")
    reference = read_npz(reference_record["fields"])
    if not exact_array(reference["omega"], frame.omega):
        raise ValueError("Saved q128 field omega differs from the replay mechanical input.")
    scores, moment_identities = independent_scores(
        frame, references, qd_to_numpy(model.info.surface_nodes, transpose=False), model.info.vertices.shape[0]
    )
    bound.extend(moment_identities)
    result = {
        "schema": "genesis.dpr-fast-adaptive-replay.v1",
        "complete": False,
        "passed": False,
        "goal_complete": False,
        "default_qualified": False,
        "primary_timing": False,
        "source_revision": source_revision,
        "source": str(args.git_source),
        "worker": file_id(Path(__file__)),
        "worker_reference_helper": file_id(Path(__file__).with_name("dpr_fast_rule_qualification.py")),
        "identity": runtime_identity(require_gpu_inventory=args.backend == "cuda"),
        "reference_result": file_id(reference_path),
        "reference_source_revision": saved["source_revision"],
        "reference_native_producers": saved["native_producers"],
        "native_producers": [
            file_id(args.git_source / "genesis/engine/solvers/rigid/stress" / name)
            for name in (
                "adaptive_pressure.py",
                "contact.py",
                "scatter.py",
                "surface.py",
                "model.py",
                "solve.py",
                "balance.py",
                "pipeline.py",
                "recovery.py",
            )
        ],
        "reference_model": saved["model"],
        "reference_fields": reference_record["fields"],
        "physical_inputs": row["physical_inputs"],
        "archived_material_binding": row["archived_material_binding"],
        "archived_topology_binding": row["archived_topology_binding"],
        "application_budgets": file_id(args.budgets),
        "case": row["name"],
        "original_environment_indices": row["original_environment_indices"],
        "whole_frame_qualification": row["whole_frame_qualification"],
        "mechanical_scope": row["mechanical_scope"],
        "authored_manual_control": saved.get("authored_manual_control"),
        "authored_manual_plan": saved.get("authored_manual_plan"),
        "reference_stability_status": row["reference_stability_status"],
        "reference256_stability": row["reference256_stability"],
        "reference_solves_performed_here": False,
        "independent_stage_moment_references": moment_identities,
        "fixed_controls_are_candidate_regressions": True,
        "original_model_all_arrays_bitwise_equal": model_exact,
        "model_identity_mode": "diagnosed_CPU_arithmetic_exact.v1"
        if cpu_model_control is not None
        else "original_GPU_arrays_exact.v1",
        "portable_CPU_model_control": cpu_model_control,
        "modes": [],
        "fixed_controls": [],
        "scope": "Frozen native replay and finite-model regional application comparison. Saved R/anchor/F/friction/omega/pose and all original elastic constants are retained. Conditional historical and authored manual support scopes are not promoted to automatic-radius whole-frame qualification.",
    }
    write_json(args.output / "result.json", finite_json(result))
    baseline_fields = None
    baseline_status = None
    baseline_rules = None
    for mode in args.modes:
        native = runtime(model, options, frame, "adaptive", mode, args.tasks_per_env)
        directory = args.output / mode
        directory.mkdir()
        report, fields, contact = evaluate(model, frame, "adaptive", mode, native, directory / "load")
        add_application(
            report, fields, reference, frame, positions, weights, row["reference_evaluable_per_environment"]
        )
        compare_stage_scores(report, contact, frame, scores)
        if baseline_fields is None:
            baseline_fields, baseline_status = fields, contact["status"]
            baseline_rules = contact["admission.rule"]
            report["scheduling_control"] = {"baseline": True, "passed": True}
        else:
            errors = compare_fields(fields, baseline_fields)
            same_status = np.array_equal(contact["status"], baseline_status)
            same_rules = np.array_equal(contact["admission.rule"], baseline_rules)
            report["scheduling_control"] = {
                "baseline": False,
                "errors": errors,
                "status_equal": same_status,
                "selected_rules_equal": same_rules,
                "passed": same_status and same_rules and all(parity_passed(errors)),
            }
        if args.lifecycle:
            released = dataclasses.replace(frame, valid=np.zeros_like(frame.valid), force=np.zeros_like(frame.force))
            release, release_fields, _ = evaluate(model, released, "adaptive", mode, native, directory / "release")
            release["raw_contact_force_exact_zero"] = bool(np.all(release_fields["nodal_force"] == 0.0))
            release["passed"] = (
                release["raw_contact_force_exact_zero"]
                and all(release["public_valid_per_environment"])
                and not any(release["errno_per_environment"])
            )
            permutation = np.roll(np.arange(frame.valid.shape[1]), 1)
            reloaded = dataclasses.replace(
                frame, **{key: vars(frame)[key][:, permutation].copy() for key in PHYSICAL_FIELDS}
            )
            reload, reload_fields, _ = evaluate(
                model, reloaded, "adaptive", mode, native, directory / "reload-permuted"
            )
            errors = compare_fields(reload_fields, fields)
            reload["slot_permutation"] = permutation.tolist()
            reload["unpermuted_raw_field_errors"] = errors
            reload["passed"] = (
                all(parity_passed(errors))
                and reload["source_and_elastic_eligible_per_environment"]
                == report["source_and_elastic_eligible_per_environment"]
            )
            report["lifecycle"] = {
                "release": release,
                "reload": reload,
                "passed": release["passed"] and reload["passed"],
            }
        report["passed"] = (
            all(value is True for value in report["application_passed_per_environment"])
            and report["scheduling_control"]["passed"]
            and report["independent_stage_operator_check"]["passed"]
            and report.get("lifecycle", {"passed": True})["passed"]
        )
        write_json(directory / "result.json", finite_json(report))
        result["modes"].append(report)
        write_json(args.output / "result.json", finite_json(result))
    for rule in args.fixed_controls:
        native = runtime(model, options, frame, rule, "cached", args.tasks_per_env)
        report, fields, contact = evaluate(model, frame, rule, "cached", native, args.output / rule)
        expected_fields = read_npz(references[rule]["fields"])
        expected_contact = read_npz(references[rule]["contacts"])
        errors = compare_fields(fields, expected_fields)
        same_status = np.array_equal(contact["status"], expected_contact["status"])
        same_eligibility = (
            report["source_and_elastic_eligible_per_environment"]
            == references[rule]["source_and_elastic_eligible_per_environment"]
        )
        report["fixed_reference"] = {"fields": references[rule]["fields"], "contacts": references[rule]["contacts"]}
        report["raw_before_accept_errors"] = errors
        report["original_status_equal"] = same_status
        report["original_source_and_elastic_eligibility_equal"] = same_eligibility
        report["passed"] = same_status and same_eligibility and all(parity_passed(errors))
        report["scope"] = (
            "Common-sampler fixed-route regression only. Original failed fits stay failed; raw pre-accept fields are compared, and public acceptance is reported independently. No new reference solve."
        )
        write_json(args.output / rule / "result.json", finite_json(report))
        result["fixed_controls"].append(report)
    result["bound_inputs_unchanged"] = [checked_identity(record) for record in bound] == bound
    result["complete"] = True
    result["passed"] = result["bound_inputs_unchanged"] and all(
        record["passed"] for record in [*result["modes"], *result["fixed_controls"]]
    )
    write_json(args.output / "result.json", finite_json(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
