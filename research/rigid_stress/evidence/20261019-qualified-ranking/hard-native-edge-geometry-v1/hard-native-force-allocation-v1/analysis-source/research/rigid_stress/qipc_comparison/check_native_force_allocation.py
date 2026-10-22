"""Compare native cone-constrained region forces with independent convex allocations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.force_allocation import StressForceAllocation

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--references", nargs="+", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(p) for p in args.references],
            "targets": {"independent_force_relative": 1e-8, "scaled_wrench_residual": 1e-10},
            "scope": "Six-coordinate native dual Newton force allocation with exact Coulomb-cone projections. Compare certified independent SLSQP minima under original total force/moment, physical centers/normals and free couples. Exercise load scales, rotations, translations, invalid/singular isolation and unload/restoration. Numerical allocation only; nearby-load stress acceptance, pressure-support updates and native scene binding remain separate.",
        },
    )
    original, centers, normals, friction, targets, lengths, expected, names = [], [], [], [], [], [], [], []
    for path in args.references:
        record = json.loads(path.read_text())
        assert record["constrained"]["qualified_constrained_minimum"]
        for scale in [1.0, 1e-3, 1e3]:
            transform = Rotation.identity() if scale == 1 else Rotation.from_rotvec([0.3, -0.6, 0.8])
            translation = np.array([0.04, -0.03, 0.02]) if scale != 1 else np.zeros(3)
            rows = record["contacts"]
            force = transform.apply([row["force_n"] for row in rows]) * scale
            center = transform.apply([row["pressure_position_m"] for row in rows]) + translation
            application = transform.apply([row["original_position_m"] for row in rows]) + translation
            couple = transform.apply([row["original_free_moment_nm"] for row in rows]) * scale
            length = record["length_scale_m"]
            moment = (np.cross(application - center.mean(axis=0), force) + couple).sum(axis=0)
            original.append(force)
            centers.append(center)
            normals.append(transform.apply([row["normal"] for row in rows]))
            friction.append([record["friction"]] * len(rows))
            targets.append(np.r_[force.sum(axis=0), moment / length])
            lengths.append(length)
            expected.append(transform.apply(record["constrained"]["forces_n"]) * scale)
            names.append({"reference": str(path), "scale": scale, "is_rotated": scale != 1})
    original, centers, normals, friction = np.array(original), np.array(centers), np.array(normals), np.array(friction)
    targets, lengths, expected = np.array(targets), np.array(lengths), np.array(expected)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    allocation = StressForceAllocation(original.shape[1], original.shape[0])
    state = allocation.state
    state.original.from_numpy(original.transpose(1, 0, 2))
    state.center.from_numpy(centers.transpose(1, 0, 2))
    state.normal.from_numpy(normals.transpose(1, 0, 2))
    state.friction.from_numpy(friction.T)
    state.target.from_numpy(targets)
    state.length.from_numpy(lengths)
    allocation.update()
    force, status, residual = array(state.force), array(state.status), array(state.residual)
    iterations, changes = array(state.iterations), array(state.maximum_change)
    relative = np.linalg.norm(force - expected, axis=2) / np.linalg.norm(expected, axis=2)
    axial = np.sum(force * normals, axis=2)
    margins = friction * axial - np.linalg.norm(force - axial[..., None] * normals, axis=2)
    margins /= np.linalg.norm(original, axis=2)
    np.savez_compressed(
        args.output / "allocations.npz",
        original=original,
        centers=centers,
        normals=normals,
        friction=friction,
        target=targets,
        length=lengths,
        expected=expected,
        native=force,
        status=status,
        residual=residual,
        iterations=iterations,
        maximum_change=changes,
    )
    write_json(
        args.output / "numerical.json",
        {
            "status": status.tolist(),
            "residual": residual.tolist(),
            "iterations": iterations.tolist(),
            "maximum_force_relative": float(relative.max()),
            "minimum_cone_margin_relative": float(margins.min()),
            "maximum_changes": changes.tolist(),
        },
    )
    np.testing.assert_array_equal(status, 0)
    assert relative.max() < 1e-8, relative
    assert residual.max() < 1e-10 and margins.min() > -1e-12
    # Numerical feasibility leaves the original >5% force adjustment visible.
    assert changes[0] > 0.05 and np.all(changes[3:] < 0.05)
    corrupted = original.copy()
    corrupted[0, 0, 0] = np.nan
    state.original.from_numpy(corrupted.transpose(1, 0, 2))
    allocation.update()
    assert array(state.status)[0] == 2
    np.testing.assert_array_equal(array(state.force)[1:], force[1:])
    corrupted[0] = 0
    zero_target = targets.copy()
    zero_target[0] = 0
    state.original.from_numpy(corrupted.transpose(1, 0, 2))
    state.target.from_numpy(zero_target)
    allocation.update()
    assert array(state.status)[0] == 1
    np.testing.assert_array_equal(array(state.force)[0], 0)
    state.original.from_numpy(original.transpose(1, 0, 2))
    state.target.from_numpy(targets)
    collapsed = centers.copy()
    collapsed[0] = centers[0].mean(axis=0)
    state.center.from_numpy(collapsed.transpose(1, 0, 2))
    allocation.update()
    assert array(state.status)[0] == 3
    np.testing.assert_array_equal(array(state.force)[1:], force[1:])
    state.center.from_numpy(centers.transpose(1, 0, 2))
    allocation.update()
    np.testing.assert_array_equal(array(state.status), 0)
    np.testing.assert_array_equal(array(state.force), force)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "cases": names,
            "maximum_force_relative": float(relative.max()),
            "maximum_scaled_wrench_residual": float(residual.max()),
            "minimum_cone_margin_relative": float(margins.min()),
            "maximum_iterations": int(iterations.max()),
            "maximum_relative_changes": changes.tolist(),
            "invalid_singular_unload_isolation_restoration": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
