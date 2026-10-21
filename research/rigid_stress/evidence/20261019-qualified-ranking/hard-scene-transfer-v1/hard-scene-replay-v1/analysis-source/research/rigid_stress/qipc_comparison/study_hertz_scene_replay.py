"""Attribute fixed-input stress variation to load assembly, runtime reduction and operator preparation."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .live_hertz_inputs import array
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-snapshot", type=Path, required=True)
    parser.add_argument("--full-snapshot", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--env", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "snapshots": [file_id(args.max_snapshot), file_id(args.full_snapshot)],
            "existing_peak_comparison": {"relative_tolerance": 1e-11, "absolute_tolerance_pa": 1e-9},
        },
    )
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    with np.load(args.max_snapshot) as source:
        maximum = {key: source[key] for key in source.files}
    with np.load(args.full_snapshot) as source:
        full = {key: source[key] for key in source.files}
    assert maximum["scene_status"][args.env] == full["scene_status"][args.env] == 0
    with np.load(args.mesh) as source:
        cpu = FreeElasticP2(source["vertices"], source["tetrahedra"], 1e10, 0.3, 2000)
    models = [
        StressModel(RigidStressOptions(mesh=args.mesh)),
        StressModel(RigidStressOptions(mesh=args.mesh)),
        StressModel(RigidStressOptions(mesh=args.mesh, method="direct")),
    ]
    assert models[0].surface_inverse is not None and models[1].surface_inverse is not None
    preparation = []
    for name, left, right in [
        ("force_blocks", models[0].surface_inverse.info.force_blocks, models[1].surface_inverse.info.force_blocks),
        (
            "centrifugal_response",
            models[0].surface_inverse.info.centrifugal,
            models[1].surface_inverse.info.centrifugal,
        ),
        ("full_inverse", models[0].inverse.info.blocks, models[1].inverse.info.blocks),
    ]:
        first, second = array(left), array(right)
        preparation.append(
            {
                "field": name,
                "maximum_absolute_difference": float(np.max(np.abs(first - second))),
                "relative_norm": float(np.linalg.norm(first - second) / np.linalg.norm(first)),
            }
        )
    batch = len(maximum["scene_status"])
    omega = V_VEC(3, dtype=gs.qd_float, shape=(batch,))
    coordinates = array(models[0].info.vertices)
    permutation = cpu.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
    source_differences = []
    for key in ["source_force", "force", "moment", "nodal_force", "omega", "balanced_rhs", "displacement"]:
        first, second = maximum[key], full[key]
        source_differences.append(
            {
                "field": key,
                "maximum_absolute_difference": float(np.max(np.abs(first - second))),
                "relative_norm": float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-300)),
            }
        )
    rows, tensors = [], []
    for source_name, source in [("max_capture", maximum), ("full_capture", full)]:
        load = np.empty_like(source["balanced_rhs"][args.env])
        load[permutation] = source["balanced_rhs"][args.env]
        reference, _, _, residual = cpu.solve(load)
        reference_peak = stress_measures(reference)[1].max()
        assert np.linalg.norm(residual) < 1e-7 * np.linalg.norm(load)
        omega.from_numpy(source["omega"])
        for variant, model, packed in [
            ("inverse_a_packed", models[0], True),
            ("inverse_a_ordered", models[0], False),
            ("inverse_b_packed", models[1], True),
            ("direct", models[2], True),
        ]:
            options = model.options.model_copy(update={"packed_surface_loads": packed})
            for mode in ["max", "full"]:
                state = model.create_state(batch, mode)
                state.force.from_numpy(source["nodal_force"].transpose(1, 0, 2).copy())
                peaks, rhs_changes, tensor_errors = [], [], []
                baseline_rhs = None
                for repeat in range(12):
                    kernel_begin_step(state)
                    model.recover(omega, state, options, surface_load=True)
                    assert array(state.valid).all()
                    peaks.append(float(array(state.step_peak)[args.env]))
                    rhs = array(state.rhs)[args.env]
                    if baseline_rhs is None:
                        baseline_rhs = rhs.copy()
                    rhs_changes.append(float(np.linalg.norm(rhs - baseline_rhs) / np.linalg.norm(baseline_rhs)))
                    if mode == "full":
                        field = array(state.stress_tensor)[args.env]
                        tensor = field[..., [[0, 3, 4], [3, 1, 5], [4, 5, 2]]]
                        tensors.append(tensor)
                        tensor_errors.append(float(np.linalg.norm(tensor - reference) / np.linalg.norm(reference)))
                values = np.array(peaks)
                rows.append(
                    {
                        "source": source_name,
                        "variant": variant,
                        "output": mode,
                        "peaks_pa": peaks,
                        "spread_pa": float(np.ptp(values)),
                        "cpu_same_balanced_rhs_peak_pa": float(reference_peak),
                        "peak_relative_errors_to_cpu": ((values - reference_peak) / reference_peak).tolist(),
                        "maximum_relative_rhs_change": max(rhs_changes),
                        "tensor_relative_errors_to_cpu": tensor_errors,
                        "existing_cross_output_limit_pa": float(1e-9 + 1e-11 * abs(reference_peak)),
                    }
                )
    np.savez_compressed(args.output / "replay-tensors.npz", tensors=tensors)
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "environment": args.env,
            "batch": batch,
            "source_differences": source_differences,
            "independent_preparation_differences": preparation,
            "replays": rows,
            "scope": "Fixed captured nodal loads and angular velocity, 12 repeated max/full recoveries, shared versus independently prepared inverse operators, packed versus ordered application, native direct control and independent same-balanced-RHS P2 elasticity. This attributes numerical variation; it does not relax the original cross-run peak assertion or qualify physical stress.",
        },
    )


if __name__ == "__main__":
    main()
