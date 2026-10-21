"""Compare the complete moving-input pilot against its matched rigid baseline."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--rigid", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((args.inputs / "result.json").read_text())
    rigid = json.loads((args.rigid / "result.json").read_text())
    assert inputs["execution_completed"] and rigid["execution_completed"]
    assert (inputs["envs"], inputs["steps"]) == (rigid["envs"], rigid["steps"])
    trajectory = {}
    with np.load(args.inputs / "trajectory.npz") as current, np.load(args.rigid / "trajectory.npz") as reference:
        for key, tolerance in (("position", 1e-9), ("quaternion", 1e-9), ("joints", 1e-8)):
            difference = np.abs(current[key] - reference[key])
            assert np.isfinite(difference).all()
            trajectory[key] = {
                "maximum_absolute_difference": float(difference.max()),
                "tolerance": tolerance,
                "passed": bool((difference <= tolerance).all()),
            }
    with np.load(args.inputs / "conditions.npz") as current, np.load(args.rigid / "conditions.npz") as reference:
        assert current.files == reference.files
        for key in current.files:
            np.testing.assert_array_equal(current[key], reference[key])
    with gzip.open(args.inputs / "history.jsonl.gz", "rt") as stream:
        history = [json.loads(line) for line in stream]
    assert len(history) == inputs["steps"]
    status_counts = np.bincount([status for row in history for status in row["source_status"]], minlength=6)
    valid = np.array([row["is_valid"] for row in history])
    loaded = np.zeros_like(valid)
    for step, row in enumerate(history):
        for env, _ in row["source_indices"]:
            loaded[step, env] = True
    pressure_axes, eccentricity = [], []
    snapshots = []
    for path in sorted(args.inputs.glob("step*.npz")):
        with np.load(path) as data:
            selected = data["source_count"] > 0
            count = selected.shape[1]
            axes = data["hertz_axes"].reshape(count, inputs["envs"], 2).transpose(1, 0, 2)
            status = data["hertz_status"].reshape(count, inputs["envs"]).T
            selected &= status == 0
            pressure_axes.extend(axes[selected].tolist())
            moment = np.linalg.norm(data["moment"][selected], axis=1)
            scale = np.linalg.norm(data["force"][selected], axis=1) * axes[selected, 0]
            eccentricity.extend((moment / scale).tolist())
            active = data["source_valid"] & (np.linalg.norm(data["source_force"], axis=2) > 0)
            snapshots.append(
                {
                    "file": path.name,
                    "source_force_magnitude": np.linalg.norm(data["source_force"][active], axis=1).tolist(),
                    "source_status": data["status"][active].tolist(),
                    "geometry_kind": data["geometry_kind"][active].tolist(),
                    "normal_residual": data["normal_residual"][active].tolist(),
                    "source_witness": data["source_witness"][active].tolist(),
                    "root_normal": data["normal"][active].tolist(),
                    "source_normal": data["source_normal"][active].tolist(),
                }
            )
            if "probe_normals" in data:
                normals = data["probe_normals"][active]
                snapshots[-1]["probe_geoms"] = data["probe_geoms"][active].tolist()
                snapshots[-1]["probe_types"] = data["probe_types"][active].tolist()
                snapshots[-1]["probe_normal_dot"] = np.sum(normals[:, 0] * normals[:, 1], axis=1).tolist()
                snapshots[-1]["probe_normals"] = normals.tolist()
                snapshots[-1]["probe_statuses"] = data["probe_statuses"][active].tolist()
                snapshots[-1]["probe_curvature_statuses"] = data["probe_curvature_statuses"][active].tolist()
    timings = {}
    for name, result in (("inputs", inputs), ("rigid", rigid)):
        samples = np.array([row["seconds"] for row in result["rollouts"] if not row["audited"]]) / result["steps"]
        timings[name] = {"scene_step_seconds": samples.tolist(), "median_scene_step_seconds": float(np.median(samples))}
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs_result": file_id(args.inputs / "result.json"),
            "rigid_result": file_id(args.rigid / "result.json"),
            "trajectory": trajectory,
            "motion_matches": all(row["passed"] for row in trajectory.values()),
            "source_status_counts": status_counts.tolist(),
            "loaded_environment_steps": loaded.sum(axis=0).tolist(),
            "invalid_environment_steps": (~valid).sum(axis=0).tolist(),
            "valid_loaded_environment_steps": (valid & loaded).sum(axis=0).tolist(),
            "timing": timings,
            "incremental_scene_step_seconds": timings["inputs"]["median_scene_step_seconds"]
            - timings["rigid"]["median_scene_step_seconds"],
            "sampled_axes_m": pressure_axes,
            "snapshots": snapshots,
            "sampled_moment_over_force_major_axis": eccentricity,
            "scope": "Full audit history, matched rigid trajectories and complete-run input-stage cost. Region validity is not physical pressure/stress admission. Nonzero region moment is retained; a centered Hertz pressure alone cannot represent an arbitrary wrench. Failed environment steps exclude a valid-throughput claim.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
