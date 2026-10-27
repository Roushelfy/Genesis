"""Compare fused and separate volume recovery on one native frozen rigid state.

Run in Slurm rtx-mid/image 1_26. The two scenes own their original operators and
consume device-copied physical inputs. Scientific arrays are saved for audit.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from .check_planar_native_trajectory import _api as trajectory_api


def _api():
    from examples.speed_benchmark.rigid_stress import (
        GeometryFrame,
        anchor_execution_modes,
        prepare_recovery_snapshot,
    )
    from genesis.utils.array_class import iter_data

    return SimpleNamespace(
        **vars(trajectory_api()),
        GeometryFrame=GeometryFrame,
        anchor_execution_modes=anchor_execution_modes,
        prepare_recovery_snapshot=prepare_recovery_snapshot,
        iter_data=iter_data,
    )


def copy_native(api, source, target, name):
    """Copy every declared native tensor, preserving scalar/vector element types."""
    kernels = (api.kernel_copy_scalar, api.kernel_copy_vector, api.kernel_copy_matrix, api.kernel_copy_cube)
    rows = []
    originals, copies = list(api.iter_data(source, name)), list(api.iter_data(target, name))
    for original, copied in zip(originals, copies, strict=True):
        if original.name != copied.name or original.value.shape != copied.value.shape:
            raise ValueError(f"Native clone layout differs at {original.name}.")
        if len(original.value.shape) >= len(kernels):
            raise ValueError(f"Unsupported native clone rank at {original.name}.")
        if all(original.value.shape):
            kernels[len(original.value.shape)](original.value, copied.value)
            rows.append(original.name)
    return rows


def native_digest(api, structs):
    """Hash bulk readbacks of physical solver arrays without supplying CPU values."""
    digest = hashlib.sha256()
    for name, struct in structs:
        for item in api.iter_data(struct, name):
            if all(item.value.shape):
                value = api.qd_to_numpy(item.value, transpose=True)
                digest.update(item.name.encode())
                digest.update(value.dtype.str.encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read_outputs(api, entry):
    return {
        name: api.qd_to_numpy(tensor, transpose=True)
        for name, tensor in (
            ("force", entry.state.force),
            ("rhs", entry.state.rhs),
            ("displacement", entry.state.displacement),
            ("stress_tensor", entry.state.stress_tensor),
            ("von_mises", entry.state.von_mises),
            ("peak", entry.state.peak),
            ("step_valid", entry.state.step_valid),
            ("invalid_steps", entry.state.invalid_steps),
            ("residual_squared", entry.state.residual_norm_squared),
            ("rhs_squared", entry.state.rhs_norm_squared),
            ("contact_valid", entry.contacts.valid),
            ("point", entry.contacts.position),
            ("source_force", entry.contacts.force),
            ("normal", entry.contacts.normal),
            ("radius", entry.contacts.radius),
            ("friction", entry.contacts.friction),
            ("omega", entry.omega),
            ("anchor_face", entry.contacts.anchor_face),
            ("anchor_center", entry.contacts.center),
            ("status", entry.contacts.status),
            ("force_error", entry.contacts.force_error),
            ("moment_error", entry.contacts.moment_error),
        )
    }


def conservation(api, output, options):
    np = api.np
    valid = output["contact_valid"]
    magnitude = np.linalg.norm(output["source_force"], axis=-1)
    loaded = valid & (magnitude > 0)
    physical_finite = all(np.isfinite(output[name][valid]).all() for name in ("point", "source_force", "normal"))
    physical_finite = physical_finite and all(
        np.isfinite(output[name][loaded]).all() for name in ("radius", "friction")
    )
    force_error = output["force_error"][loaded] / magnitude[loaded]
    moment_error = output["moment_error"][loaded] / (output["radius"][loaded] * magnitude[loaded])
    residual = np.sqrt(output["residual_squared"])
    rhs = np.sqrt(output["rhs_squared"])
    finite = all(
        np.isfinite(output[name]).all()
        for name in ("force", "rhs", "displacement", "stress_tensor", "von_mises", "peak", "omega")
    )
    report = {
        "physical_input_finite": bool(physical_finite),
        "loaded_contacts": int(loaded.sum()),
        "all_output_finite": bool(finite),
        "all_step_valid": bool(output["step_valid"].all()),
        "invalid_steps": output["invalid_steps"].tolist(),
        "loaded_radius_positive": bool((output["radius"][loaded] > 0).all()),
        "loaded_status_zero": bool((output["status"][loaded] == 0).all()),
        "force_error_relative_max": float(force_error.max(initial=0)),
        "moment_error_relative_max": float(moment_error.max(initial=0)),
        "complete_K_residual": residual.tolist(),
        "complete_K_budget": np.maximum(options.absolute_tolerance, options.tolerance * rhs).tolist(),
        "source_force_gate": bool((force_error <= 1e-8).all()),
        "source_point_moment_gate": bool((moment_error <= 1e-8).all()),
        "complete_K_gate": bool((residual <= np.maximum(options.absolute_tolerance, options.tolerance * rhs)).all()),
    }
    report["passed"] = bool(
        physical_finite
        and finite
        and loaded.any()
        and output["step_valid"].all()
        and (output["invalid_steps"] == 0).all()
        and report["loaded_radius_positive"]
        and report["loaded_status_zero"]
        and report["source_force_gate"]
        and report["source_point_moment_gate"]
        and report["complete_K_gate"]
    )
    return report


def compare(api, first, second):
    np = api.np
    exact_fields = (
        "contact_valid",
        "point",
        "source_force",
        "normal",
        "radius",
        "friction",
        "omega",
        "anchor_face",
        "anchor_center",
    )
    exact = {name: first[name].tobytes(order="C") == second[name].tobytes(order="C") for name in exact_fields}
    comparisons = {}
    for name in ("force", "rhs", "displacement", "stress_tensor", "von_mises", "peak"):
        axes = tuple(range(1, first[name].ndim))
        reference = np.sqrt(np.sum(first[name] ** 2, axis=axes))
        difference = np.sqrt(np.sum((second[name] - first[name]) ** 2, axis=axes))
        gate = difference <= 1e-11 + 1e-8 * reference
        comparisons[name] = {
            "difference_L2_per_environment": difference.tolist(),
            "reference_L2_per_environment": reference.tolist(),
            "max_absolute_component_difference": float(np.abs(second[name] - first[name]).max(initial=0)),
            "relative_1e_8_absolute_1e_11_gate": bool(gate.all()),
        }
    return {
        "exact_physical_and_anchor_face_bytes": exact,
        "same_input_numerical_equivalence": comparisons,
        "passed": all(exact.values()) and all(row["relative_1e_8_absolute_1e_11_gate"] for row in comparisons.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=730)
    parser.add_argument("--quadrature", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if args.warmup <= 0:
        parser.error("A loaded frozen module snapshot requires positive warmup.")
    args.output.mkdir(parents=True, exist_ok=False)
    api = _api()
    np = api.np
    actual_source = Path(api.gs.__file__).resolve().parent.parent
    if actual_source != args.git_source.resolve():
        raise ValueError(f"Imported source {actual_source} differs from requested {args.git_source}.")
    revision = subprocess.check_output(["git", "-C", str(actual_source), "rev-parse", "HEAD"], text=True).strip()
    api.gs.init(backend=api.gs.gpu, precision="64", seed=510000, logging_level="warning")
    started = time.perf_counter()
    fused = api.FrankaEgg(
        args.envs,
        level=1,
        seed=510000,
        conditions=args.conditions,
        egg_geometry="mesh64",
        stress_backend="volume",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        pressure_quadrature="angle",
        quadrature=args.quadrature,
        fused_pipeline=True,
        output_mode="full",
    )
    fused_solver = fused.scene.rigid_solver
    fused_entry = fused_solver.stress_recovery.links[0]
    fused_entry.cooperative_anchor = False
    geometry = None
    for tick in range(args.warmup):
        if tick == args.warmup - 1:
            fused_solver.update_forward_pos()
            geometry = api.GeometryFrame(
                np.ascontiguousarray(api.qd_to_numpy(fused_solver.dyn_state.geoms.pos, transpose=True, copy=True)),
                np.ascontiguousarray(api.qd_to_numpy(fused_solver.dyn_state.geoms.quat, transpose=True, copy=True)),
            )
        fused.step()
    fused_solver.check_errno()
    original_snapshot = api.prepare_recovery_snapshot(fused, geometry)
    separate = api.FrankaEgg(
        args.envs,
        level=1,
        seed=510000,
        conditions=args.conditions,
        egg_geometry="mesh64",
        stress_backend="volume",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        pressure_quadrature="angle",
        quadrature=args.quadrature,
        fused_pipeline=False,
        output_mode="full",
    )
    separate_solver = separate.scene.rigid_solver
    separate_entry = separate_solver.stress_recovery.links[0]
    separate_entry.cooperative_anchor = True
    if not fused_entry.fused_pipeline or separate_entry.fused_pipeline:
        raise ValueError("Requested fused/separate recovery paths were not built.")
    cloned = []
    for name, original, target in (
        ("dyn", fused_solver.dyn_state, separate_solver.dyn_state),
        ("collider", fused_solver.collider.collider_state, separate_solver.collider.collider_state),
        ("stress", fused_entry.state, separate_entry.state),
        ("contacts", fused_entry.contacts, separate_entry.contacts),
        ("scatter", fused_entry.scatter, separate_entry.scatter),
    ):
        cloned.extend(copy_native(api, original, target, name))
    api.kernel_copy_vector(fused_entry.omega, separate_entry.omega)
    api.qd.sync()
    physical_structs = (
        ("dyn", fused_solver.dyn_state),
        ("collider", fused_solver.collider.collider_state),
    )
    other_structs = (
        ("dyn", separate_solver.dyn_state),
        ("collider", separate_solver.collider.collider_state),
    )
    before_fused = native_digest(api, physical_structs)
    before_separate = native_digest(api, other_structs)
    fused_solver.stress_recovery.recover(0)
    separate_solver.stress_recovery.recover(0)
    api.qd.sync()
    fused_solver.check_errno()
    separate_solver.check_errno()
    after_fused = native_digest(api, physical_structs)
    after_separate = native_digest(api, other_structs)
    fused_output, separate_output = read_outputs(api, fused_entry), read_outputs(api, separate_entry)
    np.savez_compressed(args.output / "fused-full.npz", **fused_output)
    np.savez_compressed(args.output / "separate-full.npz", **separate_output)
    numerical = compare(api, fused_output, separate_output)
    fused_gates = conservation(api, fused_output, fused_entry.options)
    separate_gates = conservation(api, separate_output, separate_entry.options)
    physical_unchanged = before_fused == before_separate == after_fused == after_separate
    payload = {
        "complete": True,
        "passed": physical_unchanged and numerical["passed"] and fused_gates["passed"] and separate_gates["passed"],
        "engine_revision": revision,
        "engine_source": str(actual_source),
        "harness": api.file_id(__file__),
        "runtime": api.runtime_identity(),
        "cuda_graphs_enabled": os.environ.get("QD_GRAPH", "1") == "1",
        "conditions": api.file_id(args.conditions),
        "envs": args.envs,
        "warmup": args.warmup,
        "quadrature": args.quadrature,
        "pressure_quadrature": "angle",
        "output_mode": "full",
        "elapsed_seconds": time.perf_counter() - started,
        "original_frozen_snapshot_check": original_snapshot,
        "native_copied_fields": cloned,
        "physical_input_sha256": {
            "fused_before": before_fused,
            "separate_before": before_separate,
            "fused_after": after_fused,
            "separate_after": after_separate,
        },
        "same_physical_input_and_no_rigid_advance": physical_unchanged,
        "actual_routes": {
            "fused": {"fused_pipeline": fused_entry.fused_pipeline, "anchor": api.anchor_execution_modes(fused)},
            "separate": {
                "fused_pipeline": separate_entry.fused_pipeline,
                "anchor": api.anchor_execution_modes(separate),
            },
        },
        "fused_original_gates": fused_gates,
        "separate_original_gates": separate_gates,
        "cross_path": numerical,
        "scope": "Intrusive frozen-module numerical audit. Two independently built original volume operators; native device-to-device state clone; recovery only, without rigid integration. Timing campaign remains separate.",
    }
    api.write_json(args.output / "result.json", payload)
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
