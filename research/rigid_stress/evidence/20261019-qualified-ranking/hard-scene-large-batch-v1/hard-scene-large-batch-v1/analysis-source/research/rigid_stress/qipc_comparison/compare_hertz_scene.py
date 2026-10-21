"""Compare moving-scene max/full admission, tensors, trajectories and paired rigid cost."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", dest="maximum", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {"max": args.maximum, "full": args.full, "rigid": args.rigid, "inputs": args.inputs}
    results = {name: json.loads((path / "result.json").read_text()) for name, path in paths.items()}
    assert all(result["execution_completed"] for result in results.values())
    batch, steps = results["max"]["envs"], results["max"]["steps"]
    assert all((result["envs"], result["steps"]) == (batch, steps) for result in results.values())
    trajectory_differences = {}
    with np.load(args.rigid / "conditions.npz") as reference, np.load(args.rigid / "trajectory.npz") as motion:
        for name, path in paths.items():
            with np.load(path / "conditions.npz") as conditions, np.load(path / "trajectory.npz") as trajectory:
                for key in reference.files:
                    np.testing.assert_array_equal(conditions[key], reference[key])
                differences = {}
                for key, tolerance in (("position", 1e-9), ("quaternion", 1e-9), ("joints", 1e-8)):
                    delta = np.abs(trajectory[key] - motion[key])
                    assert np.isfinite(delta).all() and (delta <= tolerance).all(), (name, key)
                    differences[key] = float(delta.max())
                trajectory_differences[name] = differences
    histories = {}
    for name in ("max", "full", "inputs"):
        with gzip.open(paths[name] / "history.jsonl.gz", "rt") as stream:
            histories[name] = [json.loads(line) for line in stream]
        assert len(histories[name]) == steps
    status = np.array([row["scene_status"] for row in histories["max"]])
    loaded = np.zeros((steps, batch), dtype=bool)
    peak_failures = []
    for step, (maximum, full, inputs) in enumerate(
        zip(histories["max"], histories["full"], histories["inputs"], strict=True)
    ):
        for key in ("step", "reset_count", "source_indices", "source_status", "region_count", "is_valid"):
            assert maximum[key] == full[key] == inputs[key], (step, key)
        for key in ("scene_status", "patch_indices", "patch_status", "source_projection_status"):
            assert maximum[key] == full[key], (step, key)
        for key in ("curvature", "normal_residual"):
            np.testing.assert_allclose(
                maximum[key], inputs[key], rtol=1e-11, atol=1e-9 if key == "curvature" else 1e-12
            )
        for env, _ in maximum["source_indices"]:
            loaded[step, env] = True
        for env, code in enumerate(maximum["scene_status"]):
            if code:
                assert maximum["scene_peak_pa"][env] is None and full["scene_peak_pa"][env] is None
            else:
                try:
                    np.testing.assert_allclose(
                        maximum["scene_peak_pa"][env], full["scene_peak_pa"][env], rtol=1e-11, atol=1e-9
                    )
                except AssertionError as error:
                    peak_failures.append(
                        {
                            "step": step,
                            "env": env,
                            "max": maximum["scene_peak_pa"][env],
                            "full": full["scene_peak_pa"][env],
                            "error": str(error),
                        }
                    )
    snapshots = []
    admitted_nonzero_snapshots = 0
    for path in sorted(args.maximum.glob("step*.npz")):
        with np.load(path) as maximum, np.load(args.full / path.name) as full:
            np.testing.assert_array_equal(maximum["scene_status"], full["scene_status"])
            valid = full["scene_status"] == 0
            np.testing.assert_allclose(maximum["nodal_force"], full["nodal_force"], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(
                maximum["displacement"][valid], full["displacement"][valid], rtol=1e-10, atol=1e-14
            )
            np.testing.assert_array_equal(full["nodal_force"][~valid], 0)
            assert np.isnan(full["stress_tensor"][~valid]).all()
            assert np.isnan(full["von_mises"][~valid]).all()
            components = full["stress_tensor"][valid]
            assert np.isfinite(components).all()
            tensor = components[..., [[0, 3, 4], [3, 1, 5], [4, 5, 2]]]
            trace = np.trace(tensor, axis1=-2, axis2=-1)
            deviator = tensor - trace[..., None, None] * np.eye(3) / 3
            vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-1, -2)))
            np.testing.assert_allclose(vm, full["von_mises"][valid, ..., 0], rtol=1e-11, atol=1e-8)
            np.testing.assert_allclose(vm.max(axis=(1, 2)), maximum["scene_peak_pa"][valid], rtol=1e-11, atol=1e-8)
            nonzero = valid & (np.linalg.norm(full["nodal_force"], axis=(1, 2)) > 0)
            admitted_nonzero_snapshots += int(nonzero.sum())
            snapshots.append(
                {
                    "file": path.name,
                    "admitted": int(valid.sum()),
                    "admitted_with_nonzero_contact_load": int(nonzero.sum()),
                }
            )
    timing = {}
    for name, result in results.items():
        samples = np.array([row["seconds"] / steps for row in result["rollouts"] if not row["audited"]])
        assert len(samples) >= 3
        timing[name] = {"seconds_per_step": samples.tolist(), "median_seconds_per_step": float(np.median(samples))}
    counts = {
        "total_environment_steps": int(status.size),
        "loaded_environment_steps": int(loaded.sum()),
        "input_valid_loaded_environment_steps": int(
            (loaded & np.array([row["is_valid"] for row in histories["max"]])).sum()
        ),
        "admitted_environment_steps": int((status == 0).sum()),
        "admitted_loaded_environment_steps": int(((status == 0) & loaded).sum()),
        "rejection_bits": {
            name: int(((status & bit) != 0).sum())
            for name, bit in [("geometry", 1), ("source", 2), ("transfer", 4), ("facet_cone", 8), ("elastic_solve", 16)]
        },
        "status_combinations": {str(code): int((status == code).sum()) for code in np.unique(status)},
    }
    for mode in ("max", "full"):
        timing[mode]["added_seconds_per_step"] = (
            timing[mode]["median_seconds_per_step"] - timing["rigid"]["median_seconds_per_step"]
        )
        timing[mode]["raw_environment_steps_per_second"] = batch / timing[mode]["median_seconds_per_step"]
        timing[mode]["audited_admitted_loaded_fraction"] = (
            counts["admitted_loaded_environment_steps"] / counts["loaded_environment_steps"]
        )
    write_json(
        args.output / "result.json",
        {
            "passed": not peak_failures,
            "peak_failures": peak_failures,
            "identity": runtime_identity(require_gpu_inventory=False),
            "results": {name: file_id(path / "result.json") for name, path in paths.items()},
            "counts": counts,
            "timing": timing,
            "trajectory_differences": trajectory_differences,
            "snapshots": snapshots,
            "admitted_nonzero_contact_snapshots": admitted_nonzero_snapshots,
            "scope": "Every-step max/full status, source and peak comparison; input-stage conformance; saved complete-field von Mises identity and rejection masking; paired rigid trajectories and warmed inclusive cost. Admission is a strict mechanical gate, not independent pressure, finite-shell or friction-history accuracy. Zero admitted loaded steps provide no useful stress-throughput evidence. The input-only timing is retained from its earlier allocation; only max/full/rigid share the present allocation.",
            "whole_goal_complete": False,
        },
    )
    assert not peak_failures, peak_failures


if __name__ == "__main__":
    main()
