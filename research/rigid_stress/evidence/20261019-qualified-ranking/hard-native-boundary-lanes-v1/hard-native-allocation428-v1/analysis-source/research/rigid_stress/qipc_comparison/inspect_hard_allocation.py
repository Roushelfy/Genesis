"""Replay captured region-force allocation and test linear wrench representability."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.force_allocation import StressForceAllocation

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"), "input": file_id(args.hard)},
    )
    with np.load(args.hard) as saved:
        data = {
            key: saved["allocation_" + key] for key in ["original", "center", "normal", "friction", "target", "length"]
        }
        captured = {key: saved["allocation_" + key] for key in ["status", "force", "residual", "iterations"]}
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    batch, slots = data["original"].shape[:2]
    allocation = StressForceAllocation(slots, batch)
    for key in ["original", "center", "normal", "friction"]:
        vars(allocation.state)[key].from_numpy(np.swapaxes(data[key], 0, 1))
    allocation.state.target.from_numpy(data["target"])
    allocation.state.length.from_numpy(data["length"])
    allocation.update()
    qd.sync()
    actual = {key: array(vars(allocation.state)[key]) for key in captured}
    np.testing.assert_array_equal(actual["status"], captured["status"])
    np.testing.assert_array_equal(actual["iterations"], captured["iterations"])
    np.testing.assert_allclose(actual["force"], captured["force"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(actual["residual"], captured["residual"], rtol=1e-9, atol=1e-12)
    rows, certificates = [], {}
    for env in range(batch):
        selected = np.flatnonzero(np.linalg.norm(data["original"][env], axis=1) > 0)
        original = data["original"][env, selected]
        centers = data["center"][env, selected]
        force_scale = np.linalg.norm(original, axis=1).sum()
        position = (centers - centers.mean(axis=0)) / data["length"][env]
        cross = np.cross(position[:, None], np.eye(3)[None]).transpose(0, 2, 1)
        matrix = np.concatenate([np.tile(np.eye(3), (len(selected), 1, 1)), cross], axis=1)
        matrix = matrix.transpose(1, 0, 2).reshape(6, -1)
        target = data["target"][env] / force_scale
        vectors, singular, _ = np.linalg.svd(matrix, full_matrices=True)
        threshold = 128 * np.finfo(float).eps * max(matrix.shape) * singular[0]
        rank = int((singular > threshold).sum())
        missing = vectors[:, rank:] @ (vectors[:, rank:].T @ target)
        minimum = np.linalg.norm(missing)
        certificate = missing / minimum if minimum > 0 else np.zeros(6)
        null_error = np.linalg.norm(matrix.T @ certificate)
        if rank < 6 and minimum > 1e-12:
            assert null_error < 1e-12
            np.testing.assert_allclose(certificate @ target, minimum, rtol=1e-12, atol=1e-15)
        certificates[f"env{env}_matrix"] = matrix
        certificates[f"env{env}_target"] = target
        certificates[f"env{env}_certificate"] = certificate
        row = {
            "environment": env,
            "regions": selected.tolist(),
            "linear_rank": rank,
            "singular_values": singular.tolist(),
            "unrestricted_normalized_wrench_residual_lower_bound": float(minimum),
            "certificate_null_error": float(null_error),
            "linear_infeasibility_proved_at_native_1e_12_gate": bool(rank < 6 and minimum > 1e-12),
            "native_status": int(actual["status"][env]),
            "native_residual": float(actual["residual"][env]),
        }
        if len(selected) == 1:
            force = data["target"][env, :3]
            moment = data["target"][env, 3:] * data["length"][env]
            offset = np.cross(force, moment) / np.dot(force, force)
            axial = force * np.dot(force, moment) / np.dot(force, force)
            row.update(
                required_free_moment_nm=moment.tolist(),
                force_line_offset_m=offset.tolist(),
                force_line_offset_norm_m=float(np.linalg.norm(offset)),
                axial_free_moment_nm=axial.tolist(),
                axial_free_moment_norm_nm=float(np.linalg.norm(axial)),
            )
        rows.append(row)
    np.savez_compressed(args.output / "allocation.npz", **actual, **certificates)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Exact captured-input replay and independent singular-value decomposition of the point-region wrench map. The lower bound allows arbitrary signed region forces, so a positive certified defect also excludes cone-feasible forces at the original centers. It does not prove infeasibility of a distributed physical pressure/traction field with additional first moments.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
