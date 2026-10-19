"""Heterogeneous live-source batching, permutation and empty-environment control."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_batch_recovery import sample_batch
from .native_pad_recovery import NativePadRecovery
from .plates import plate_xml


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--method", choices=("dpr", "scr"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    group = 0 if args.method == "dpr" else 1
    root = args.data_root / "native-pad-full-source-stability-v1"
    runs = [(root / f"group{group}/native", level) for level in (0, 1, 2)]
    runs.append((root / f"group{group + 2}/native", 1))
    configs = [json.loads((run / "start.json").read_text())["args"] for run, _ in runs]
    config = SimpleNamespace(**configs[0])
    config.assets = Path(config.assets)
    config.envs, config.primary_cost = 4, True
    with np.load(config.mesh) as saved:
        outer, faces = saved["outer_vertices"], saved["mid_faces"]
    points = outer[faces].mean(axis=1)
    fields, forces, rotations, angular, loads, receipts = [], [], [], [], [], []
    for run, level in runs:
        field_path = run / f"frames/stress-level{level}.npz"
        motion_path = run / f"frames/level{level}.npz"
        receipts.extend((file_id(field_path), file_id(motion_path)))
        with np.load(motion_path) as saved:
            rotation, omega = saved["rotation"], saved["angular_velocity_rad_per_s"]
        with np.load(field_path) as saved:
            np.testing.assert_array_equal(saved["original_contact_points_m"], points)
            fields.append(saved["stress_pa"])
            forces.append(saved["original_contact_forces_n"] @ rotation.T)
            loads.append(float(saved["load_n"]))
        rotations.append(rotation)
        angular.append(omega)
    forces, rotations, angular, loads = map(np.array, (forces, rotations, angular, loads))
    friction = np.array([c["friction"] for c in configs])
    xml = plate_xml(float(outer[:, 2].min() - config.clearance), float(outer[:, 2].max() + config.clearance))
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    controls, base_peaks = [], None
    for output_mode in ("full", "max"):
        config.stress_output = output_mode
        observer = NativePadRecovery.allocate(config, outer, faces, points, xml)
        for name, permutation, empty in (
            ("original", np.arange(4), False),
            ("permuted", np.array([3, 1, 0, 2]), False),
            ("one-empty", np.arange(4), True),
        ):
            selected_force = forces[permutation].copy()
            selected_angular = angular[permutation].copy()
            if empty:
                selected_force[2] = 0
                selected_angular[2] = 0
            paths = [args.output / f"{output_mode}-{name}-{env}.npz" for env in range(4)]
            rows, timing = sample_batch(
                observer,
                selected_force,
                rotations[permutation],
                selected_angular,
                friction[permutation],
                loads[permutation],
                paths,
            )
            errors = []
            for env, source in enumerate(permutation):
                assert rows[env]["native_valid"]
                if empty and env == 2:
                    assert rows[env]["source_count"] == 0 and abs(rows[env]["raw_vm_peak_pa"]) <= 1e-9
                elif output_mode == "full":
                    with np.load(paths[env]) as saved:
                        tensor = saved["stress_pa"]
                    error = float(np.linalg.norm(tensor - fields[source]) / np.linalg.norm(fields[source]))
                    assert error <= 1e-6, (name, env, error)
                    errors.append(error)
            peaks = np.array([r["raw_vm_peak_pa"] for r in rows])
            if output_mode == "full" and name == "original":
                base_peaks = peaks
            target = base_peaks[permutation].copy()
            if empty:
                target[2] = 0
            peak_error = float(np.max(np.abs(peaks - target)) / max(np.max(np.abs(target)), 1e-30))
            assert peak_error <= 1e-6, (output_mode, name, peak_error)
            controls.append(
                {
                    "mode": output_mode,
                    "case": name,
                    "rows": rows,
                    "relative_field_errors": errors,
                    "relative_peak_error": peak_error,
                    "timing": timing,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "method": args.method,
            "passed": True,
            "controls": controls,
            "receipts": receipts,
            "scope": "Four independent original complete source packets: three normal loads and sustained friction. One native batched recovery call, with per-environment source counts, rotations, angular velocities and friction. Permutation and one-empty controls check adapter isolation, then max/full agreement. This is a recovery-input control, not a batched moving-trajectory or throughput claim.",
        },
    )


if __name__ == "__main__":
    main()
