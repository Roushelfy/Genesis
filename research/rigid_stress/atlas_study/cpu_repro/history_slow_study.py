"""Numerically qualify the declared B64 speed-one-quarter timing control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .history import BatchHistory, RecoverySystem
from .history_scaled_trajectory import SPEED_DEFINITION, prepare_scaled_trajectory
from .history_study import compare_trajectory
from .model import Mother


def source_hashes() -> dict[str, str]:
    folder = Path(__file__).parent
    names = ("history_slow_study.py", "history_scaled_trajectory.py", "history_speed_study.py",
             "history_study.py", "history.py", "model.py", "geometry.py", "loads.py")
    paths = [folder / name for name in names] + [folder.parents[1] / "shell_cpu" / "core.py"]
    return {str(path.relative_to(folder.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("history_slow_results.json"))
    parser.add_argument("--speed-results", type=Path, default=Path(__file__).with_name("history_speed_results.json"))
    args = parser.parse_args()
    initial_hashes = source_hashes()
    speed_results_sha = hashlib.sha256(args.speed_results.read_bytes()).hexdigest()
    batch, frames, order, speed, tolerance = 64, 80, 8, 0.25, 0.01
    with threadpool_limits(limits=1):
        mother = Mother()
        system = RecoverySystem.from_mother(mother)
        trajectory = prepare_scaled_trajectory(mother, speed, batch, frames=frames, order=order)
        sensitivity = json.loads(args.speed_results.read_text())
        small_case = next(item for item in sensitivity["rows"] if item["speed"] == speed)
        reference_small = prepare_scaled_trajectory(mother, speed, 16, frames=frames, order=order)
        if reference_small.force_sha256 != small_case["force_sha256"]:
            raise ArithmeticError("The B16 force trace differs from the frozen sensitivity.")
        prefix_sha = hashlib.sha256(np.ascontiguousarray(trajectory.forces[:, :, :16]).tobytes()).hexdigest()
        prefix_difference = trajectory.forces[:, :, :16] - reference_small.forces
        prefix_relative_difference = float(np.linalg.norm(prefix_difference) / np.linalg.norm(reference_small.forces))
        prefix_maximum_difference = float(np.max(np.abs(prefix_difference)))
        if prefix_relative_difference > 1e-12:
            raise ArithmeticError("The B64 first sixteen force traces differ beyond matrix-rounding tolerance.")
        row = compare_trajectory(system, trajectory, tolerance)
        engines = {
            "rotate": BatchHistory(system, batch, tolerance=tolerance),
            "cached_rotate": BatchHistory(system, batch, tolerance=tolerance, cached_product=True),
        }
        # Retain every actual gate value in a second independent numerical replay.
        # This replay is outside throughput measurement and is not used to repair history.
        gate_frames = []
        for frame in range(frames):
            gate_row = {"frame": frame, "active": trajectory.active[frame].tolist(),
                        "reset_envs": trajectory.resets[frame].tolist()}
            for name, engine in engines.items():
                output = engine.step(trajectory.forces[frame], trajectory.active[frame],
                                     reset=trajectory.resets[frame], audit_cache=True)
                gate_row[name] = {
                    "scaled_residual_ratio": output.residual_ratio.tolist(),
                    "accepted_prediction": output.accepted_prediction.tolist(),
                    "depth": output.depth.tolist(),
                    "cache_shadow_decision_difference": output.cache_decision_disagreement,
                    "cache_direct_rechecks": output.cache_direct_rechecks,
                }
            if not np.array_equal(gate_row["rotate"]["accepted_prediction"],
                                  gate_row["cached_rotate"]["accepted_prediction"]):
                raise ArithmeticError("Direct and cached gate decisions differ.")
            gate_frames.append(gate_row)
    if source_hashes() != initial_hashes:
        raise RuntimeError("A source changed during the slow-control numerical qualification.")
    if hashlib.sha256(args.speed_results.read_bytes()).hexdigest() != speed_results_sha:
        raise RuntimeError("The frozen sensitivity input changed during qualification.")
    result = {
        "reconstruction": True, "study": "Predeclared B64 slow numerical qualification, no timing",
        "speed": speed, "speed_definition": SPEED_DEFINITION,
        "batch": batch, "frames": frames, "order": order, "tolerance": tolerance,
        "source_hashes": initial_hashes, "force_sha256": trajectory.force_sha256,
        "speed_results_input": str(args.speed_results), "speed_results_sha256": speed_results_sha,
        "reference_b16_force_sha256": reference_small.force_sha256,
        "reference_b16_matches_frozen_sensitivity_sha": True,
        "first_sixteen_force_sha256": prefix_sha,
        "first_sixteen_relative_difference": prefix_relative_difference,
        "first_sixteen_maximum_absolute_difference": prefix_maximum_difference,
        "cross_batch_rounding_note": "Mass-relief dense matrix products can use different reduction order for B16 and B64; SHA is checked within the same batch.",
        "row": row, "gate_frames": gate_frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "force_sha256": trajectory.force_sha256,
                      "acceptance": row["rows"]["rotate"]["acceptance"],
                      "field_max": row["rows"]["rotate"]["field_max"],
                      "peak_max": row["rows"]["rotate"]["peak_max"]}), flush=True)


if __name__ == "__main__":
    main()
