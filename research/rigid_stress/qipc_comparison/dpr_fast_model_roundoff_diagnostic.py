"""Diagnose a CPU model's exact-array mismatch against a frozen native model.

Run only on Slurm rtx-mid/image1_26. This tool rebuilds the model, never fits
pressure, changes an acceptance gate or injects reference operators.
"""

import argparse
import dataclasses
import hashlib
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
    import scipy.linalg

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.model import StressModel
    from genesis.utils.misc import qd_to_numpy


def array_identity(value):
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "logical_C_bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def compare_array(actual, reference):
    result = {"CPU": array_identity(actual), "reference": array_identity(reference)}
    matching_layout = actual.shape == reference.shape and actual.dtype == reference.dtype
    result["dtype_and_shape_exact"] = matching_layout
    result["bytes_exact"] = matching_layout and actual.tobytes() == reference.tobytes()
    if not matching_layout:
        return result
    unequal = actual != reference
    result["numerically_unequal_entries"] = int(np.count_nonzero(unequal))
    result["element_count"] = actual.size
    result.update(max_ULP_distance_finite=None, max_absolute_difference=None, relative_L2_difference=None)
    result["numeric_difference_metrics_applicability"] = "not_applicable_to_exact_integer_or_boolean_topology"
    indices = np.argwhere(unequal)
    result["first_unequal_indices"] = indices[:8].tolist()
    if np.issubdtype(actual.dtype, np.floating):
        result["numeric_difference_metrics_applicability"] = "finite_floating_pairs"
        finite = np.isfinite(actual) & np.isfinite(reference)
        result["CPU_nonfinite_entries"] = int(np.count_nonzero(~np.isfinite(actual)))
        result["reference_nonfinite_entries"] = int(np.count_nonzero(~np.isfinite(reference)))
        delta = np.abs(actual[finite] - reference[finite])
        scale = np.abs(reference[finite])
        result["max_absolute_difference"] = float(delta.max(initial=0))
        result["relative_L2_difference"] = float(np.linalg.norm(delta) / max(np.linalg.norm(scale), 1e-300))
        result["max_reference_absolute"] = float(scale.max(initial=0))
        result["max_pointwise_relative_on_nonzero_reference"] = (
            float((delta[scale > 0] / scale[scale > 0]).max(initial=0)) if (scale > 0).any() else None
        )
        result["changed_reference_zero_entries"] = int(np.count_nonzero((reference == 0) & unequal))
        unsigned = np.uint64 if actual.dtype.itemsize == 8 else np.uint32
        high_bit = unsigned(1 << (8 * actual.dtype.itemsize - 1))
        a_bits, r_bits = np.ascontiguousarray(actual).view(unsigned), np.ascontiguousarray(reference).view(unsigned)
        a_order = np.where((a_bits & high_bit) != 0, ~a_bits, a_bits | high_bit)
        r_order = np.where((r_bits & high_bit) != 0, ~r_bits, r_bits | high_bit)
        ulps = np.maximum(a_order, r_order) - np.minimum(a_order, r_order)
        result["max_ULP_distance_finite"] = int(ulps[finite].max(initial=0))
        result["ULP_definition"] = (
            "Distance between ordered IEEE binary bit patterns, including distinct signed zeros; finite pairs only."
        )
        for label, values in (
            ("absolute", np.where(finite, np.abs(actual - reference), -1)),
            ("ULP", np.where(finite, ulps, 0)),
        ):
            if actual.size:
                where = np.unravel_index(np.argmax(values), actual.shape)
                result[label + "_max_location"] = {
                    "index": [int(index) for index in where],
                    "CPU": float(actual[where]).hex(),
                    "reference": float(reference[where]).hex(),
                }
    return result


def independent_operator(arrays, nodal_force, omega, options):
    """Reconstruct CSR K and mass relief independently from one exported model."""
    starts, columns, blocks = arrays["row_start"], arrays["columns"], arrays["stiffness"]
    n = len(starts) - 1
    rows = np.repeat(np.arange(n), np.diff(starts))
    matrix = np.zeros((n, 3, n, 3))
    matrix[rows, :, columns, :] = blocks
    matrix = matrix.reshape((3 * n, 3 * n))
    free = arrays["is_free"].reshape(-1).astype(bool)
    modes, mass_modes = arrays["modes"].reshape((3 * n, 6)), arrays["mass_modes"].reshape((3 * n, 6))
    gram = modes.T @ mass_modes
    relief = scipy.linalg.solve(gram, mass_modes.T, assume_a="sym").T
    centrifugal = arrays["centrifugal"].reshape((3 * n, 6))
    factor = scipy.linalg.cho_factor(matrix[np.ix_(free, free)])
    displacements, right_sides, control_rows = [], [], []
    for env, spin in enumerate(omega):
        terms = np.r_[spin**2, spin[0] * spin[1], spin[0] * spin[2], spin[1] * spin[2]]
        force = nodal_force[env].reshape(-1) + centrifugal @ terms
        rhs = force - relief @ (modes.T @ force)
        q = np.zeros_like(rhs)
        q[free] = scipy.linalg.cho_solve(factor, rhs[free]) / options["young"]
        residual = options["young"] * (matrix @ q) - rhs
        residual_norm = np.linalg.norm(residual[free])
        budget = max(options["absolute_tolerance"], options["tolerance"] * np.linalg.norm(rhs[free]))
        displacements.append(q)
        right_sides.append(rhs)
        control_rows.append(
            {
                "environment": env,
                "original_free_K_residual": float(residual_norm),
                "original_K_residual_budget": float(budget),
                "original_K_gate_passed_for_independent_solve": bool(residual_norm <= budget),
                "relieved_rigid_wrench_relative": float(
                    np.linalg.norm(modes.T @ rhs) / max(np.linalg.norm(force), 1e-300)
                ),
                "pinned_displacement_exact_zero": bool(np.count_nonzero(q[~free]) == 0),
            }
        )
    return matrix, relief, np.stack(displacements), np.stack(right_sides), control_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific diagnosis requires Slurm rtx-mid/image1_26.")
    if not args.output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output directory.")
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        source_revision != args.source_revision
        or Path(gs.__file__).resolve().parent.parent != args.git_source.resolve()
    ):
        raise ValueError("Imported native source differs from its explicit engine pin.")
    reference_path = args.reference_run / "result.json"
    saved = json.loads(reference_path.read_text())
    if saved["complete"] is not True or saved["passed"] is not True or len(saved["rows"]) != 1:
        raise ValueError("Use the one completed original frozen-rule reference.")
    case = saved["rows"][0]
    if case["whole_frame_qualification"] is not True:
        raise ValueError("This diagnostic requires complete archived mechanical inputs including actual omega.")
    ref128 = next(row for row in case["rules"] if row["rule"] == "angle128")
    mesh = Path(saved["options"]["mesh"])
    mesh_id = file_id(mesh)
    if mesh_id not in saved["inputs"]:
        raise ValueError("The original mesh no longer matches its frozen identity.")
    bound = [file_id(reference_path), mesh_id, saved["model"], case["physical_inputs"], ref128["fields"]]
    if bound != [file_id(row["path"]) for row in bound]:
        raise ValueError("An original reference/input artifact changed.")
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cpu, precision="64", seed=510000, logging_level="warning")
    options = gs.options.RigidStressOptions(**saved["options"])
    model = StressModel(options)
    current = {
        field.name: qd_to_numpy(vars(model.info)[field.name], transpose=False)
        for field in dataclasses.fields(model.info)
    }
    vertices = qd_to_numpy(model.info.vertices, transpose=True)
    elements = qd_to_numpy(model.info.elements, transpose=False)
    positions = vertices[elements[:, :4]]
    current.update(
        sample_positions=positions,
        sample_weights=np.broadcast_to(
            qd_to_numpy(model.info.volumes, transpose=True)[:, None] / 4, positions.shape[:-1]
        ),
    )
    with np.load(saved["model"]["path"], allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    np.savez_compressed(args.output / "CPU-model.npz", **current)
    keys = sorted(set(current) | set(reference))
    comparisons = {
        key: compare_array(current[key], reference[key]) if key in current and key in reference else {"missing": True}
        for key in keys
    }
    exact_required = ("vertices", "elements", "surface_nodes", "row_start", "columns", "is_free", "sample_positions")
    exact_inputs = all(comparisons[key]["bytes_exact"] for key in exact_required)
    result = {
        "schema": "genesis.dpr-fast-CPU-model-roundoff-diagnostic.v1",
        "complete": False,
        "diagnostic_execution_completed": False,
        "qualification_passed": False,
        "goal_complete": False,
        "source_revision": source_revision,
        "worker": file_id(Path(__file__)),
        "identity": runtime_identity(require_gpu_inventory=False),
        "original_result": bound[0],
        "original_model": saved["model"],
        "CPU_model": file_id(args.output / "CPU-model.npz"),
        "original_model_guard_passed": all(row.get("bytes_exact", False) for row in comparisons.values()),
        "physical_topology_gauge_samples_exact": exact_inputs,
        "options_exact": options.model_dump(mode="json") == saved["options"],
        "options": options.model_dump(mode="json"),
        "field_comparisons": comparisons,
        "scope": "Diagnostic only. Original CPU replay byte guard remains unchanged. No pressure/source solve, operator injection, allclose substitution or new numerical acceptance rule. Independent linear algebra binds each model's own K, mass relief and exact same saved q128 nodal force/omega.",
    }
    write_json(args.output / "result.json", result)
    if exact_inputs and result["options_exact"]:
        with np.load(ref128["fields"]["path"], allow_pickle=False) as archive:
            nodal_force, omega = archive["nodal_force"], archive["omega"]
        with np.load(case["physical_inputs"]["path"], allow_pickle=False) as archive:
            if (
                omega.dtype != archive["omega"].dtype
                or omega.shape != archive["omega"].shape
                or omega.tobytes() != archive["omega"].tobytes()
            ):
                raise ValueError("The frozen q128 load/control uses different omega from the physical input.")
        native_K, native_relief, native_q, native_rhs, native_rows = independent_operator(
            current, nodal_force, omega, saved["options"]
        )
        ref_K, ref_relief, ref_q, ref_rhs, ref_rows = independent_operator(
            reference, nodal_force, omega, saved["options"]
        )
        result["independent_CSR_mass_relief_controls"] = {
            "CSR_values": compare_array(native_K, ref_K),
            "recomputed_relief": compare_array(native_relief, ref_relief),
            "own_K_solved_displacement": compare_array(native_q, ref_q),
            "own_mass_relieved_rhs": compare_array(native_rhs, ref_rhs),
            "CPU_original_K_checks": native_rows,
            "reference_original_K_checks": ref_rows,
            "CPU_K_symmetry_relative": float(np.linalg.norm(native_K - native_K.T) / np.linalg.norm(native_K)),
            "reference_K_symmetry_relative": float(np.linalg.norm(ref_K - ref_K.T) / np.linalg.norm(ref_K)),
        }
        np.savez_compressed(
            args.output / "independent-controls.npz",
            CPU_displacement=native_q,
            reference_displacement=ref_q,
            CPU_rhs=native_rhs,
            reference_rhs=ref_rhs,
            CPU_relief=native_relief,
            reference_relief=ref_relief,
            frozen_nodal_force=nodal_force,
            frozen_omega=omega,
        )
        result["independent_controls"] = file_id(args.output / "independent-controls.npz")
    else:
        result["independent_controls_skipped"] = (
            "Exact physical/topology/gauge/sample/material binding failed; do not reinterpret this as harmless model roundoff."
        )
    result["original_artifacts_unchanged"] = bound == [file_id(row["path"]) for row in bound]
    result["complete"] = True
    result["diagnostic_execution_completed"] = result["original_artifacts_unchanged"]
    write_json(args.output / "result.json", result)
    if not result["original_artifacts_unchanged"]:
        raise RuntimeError("An immutable original input changed during diagnosis.")


if __name__ == "__main__":
    main()
