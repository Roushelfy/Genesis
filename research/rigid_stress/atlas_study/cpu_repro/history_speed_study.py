"""Measure history sensitivity to smooth per-frame load evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .history import RecoverySystem
from .history_study import Trajectory, compare_trajectory
from .loads import Source, assemble_sources
from .model import Mother


SPEEDS = (0.25, 0.5, 1.0, 2.0)


def scaled_sources(frame: int, batch: int, speed: float) -> tuple[list[Source], np.ndarray, np.ndarray]:
    """Scale smooth time while keeping release, reset and shock frames fixed."""
    time = speed * frame
    sources = []
    active = np.ones(batch, dtype=bool)
    if 24 <= frame < 28:
        active[np.arange(batch) % 4 == 0] = False
    resets = np.empty(0, dtype=np.int64)
    if frame == 35:
        resets = np.flatnonzero(np.arange(batch) % 5 == 0)
    if frame == 61:
        resets = np.flatnonzero(np.arange(batch) % 7 == 2)
    # Repeat the frozen main formula here so the main experiment source stays immutable.
    # The speed-one complete balanced-force SHA256 is checked against its saved result.
    for i_env in np.flatnonzero(active):
        phase = 2.0 * np.pi * ((i_env * 0.6180339887498949) % 1.0)
        for i_contact in range(2):
            azimuth = phase + i_contact * (np.pi + 0.16)
            azimuth += time * (0.009 + 0.003 * np.sin(phase + i_contact))
            azimuth += 0.08 * np.sin(0.037 * time + phase + 1.1 * i_contact)
            latitude = 0.38 * np.sin(0.73 * phase + 1.5 * i_contact)
            latitude += 0.13 * np.sin(0.043 * time + phase + 0.6 * i_contact)
            force = (0.8 - 0.15 * i_contact) * (1.0 + 0.22 * np.sin(0.052 * time + phase + i_contact))
            if frame == 42 and i_env % 4 == 1 and i_contact == 0:
                azimuth += 0.18
                force *= 1.6
            direction = np.array([
                np.cos(latitude) * np.cos(azimuth),
                np.cos(latitude) * np.sin(azimuth),
                np.sin(latitude),
            ])
            tangent = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
            radius = (0.0016 + 0.0002 * i_contact) * (force / (0.8 - 0.15 * i_contact)) ** (1.0 / 3.0)
            friction = 0.3 + 0.1 * np.sin(0.031 * time + phase + i_contact)
            sources.append(Source(
                direction=direction,
                radius=radius,
                force=force * (-direction + friction * tangent),
                normal_force=force,
                friction=friction,
                tangent=tangent,
                side=1,
                name=f"env{i_env}-contact{i_contact}",
            ))
    return sources, active, resets


def prepare_scaled(
    mother: Mother, speed: float, batch: int, frames: int, order: int, cases_file,
) -> Trajectory:
    start = perf_counter()
    forces = np.empty((frames, len(mother.free), batch))
    active = np.empty((frames, batch), dtype=bool)
    resets = []
    for frame in range(frames):
        sources, active[frame], reset = scaled_sources(frame, batch, speed)
        raw = assemble_sources(mother, sources, order=order, audit=False)
        full = np.zeros((mother.n_dof, batch))
        full[:, active[frame]] = raw[:, 0::2] + raw[:, 1::2]
        forces[frame] = mother.balance(full)[mother.free]
        resets.append(reset)
        cases_file.write(json.dumps({
            "speed": speed, "frame": frame, "smooth_time": speed * frame,
            "active_envs": np.flatnonzero(active[frame]).tolist(), "reset_envs": reset.tolist(),
            "sources": [{"name": source.name, "direction": source.direction.tolist(),
                         "radius": source.radius, "force": source.force.tolist(),
                         "radial_force_scale": source.normal_force, "friction": source.friction,
                         "tangent": source.tangent.tolist(), "side": source.side} for source in sources],
        }, separators=(",", ":")) + "\n")
    return Trajectory(forces, active, tuple(resets), perf_counter() - start,
                      hashlib.sha256(forces.tobytes()).hexdigest(), order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("history_speed_results.json"))
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("history_speed_cases.jsonl"))
    parser.add_argument("--main-results", type=Path, default=Path(__file__).with_name("history_results.json"))
    args = parser.parse_args()
    baseline = json.loads(args.main_results.read_text())
    main_case = next(row for row in baseline["rows"] if row["batch"] == 16 and row["tolerance"] < 0.011)
    batch, frames, order, tolerance = 16, 80, 8, 0.01
    if main_case["frames"] != frames or main_case["order"] != order:
        raise ValueError("The saved main numerical trajectory has another definition.")
    root = Path(__file__).parents[3]
    source_paths = [Path(__file__), Path(__file__).with_name("history_study.py"),
                    Path(__file__).with_name("history.py"), Path(__file__).with_name("model.py"),
                    Path(__file__).with_name("geometry.py"), Path(__file__).with_name("loads.py"),
                    Path(__file__).parents[2] / "shell_cpu" / "core.py"]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1), args.cases.open("w") as cases_file:
        mother = Mother()
        system = RecoverySystem.from_mother(mother)
        for speed in SPEEDS:
            trajectory = prepare_scaled(mother, speed, batch, frames, order, cases_file)
            if abs(speed - 1.0) < 1e-15 and trajectory.force_sha256 != main_case["force_sha256"]:
                raise ArithmeticError("Speed-one force trace differs from the frozen main trajectory.")
            row = compare_trajectory(system, trajectory, tolerance)
            row["speed"] = speed
            rows.append(row)
            print(json.dumps({"speed": speed, "batch": batch, "frames": frames,
                              "acceptance": row["rows"]["rotate"]["acceptance"],
                              "field_max": row["rows"]["rotate"]["field_max"],
                              "peak_max": row["rows"]["rotate"]["peak_max"]}), flush=True)
    final_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_paths}
    if final_hashes != source_hashes:
        raise RuntimeError("A sensitivity source changed during the numerical run.")
    result = {
        "reconstruction": True, "study": "Predeclared smooth-time sensitivity, separate from the frozen main trace",
        "speeds": SPEEDS, "batch": batch, "frames": frames, "order": order, "tolerance": tolerance,
        "speed_definition": "Every smooth azimuth/latitude/force/friction/radius evolution uses time=speed*frame. Release/reset/shock event frames stay fixed.",
        "comparison_limit": "A slower speed covers a different total path over 80 frames. This is per-frame evolution sensitivity, not equal-total-motion throughput.",
        "main_force_sha256": main_case["force_sha256"], "speed_one_force_sha_equal": True,
        "source_hashes": source_hashes, "cases_file": args.cases.name,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": str(args.cases)}), flush=True)


if __name__ == "__main__":
    main()
