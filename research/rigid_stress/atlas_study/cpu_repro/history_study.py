"""Numerically compare sustained batch prediction against current-load recovery."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .history import BatchHistory, RecoverySystem, field_error, sampled_maximum
from .loads import Source, assemble_sources
from .model import Mother


@dataclass(frozen=True)
class Trajectory:
    forces: np.ndarray
    active: np.ndarray
    resets: tuple[np.ndarray, ...]
    construction_seconds: float
    force_sha256: str
    order: int


def sources_at(frame: int, batch: int) -> tuple[list[Source], np.ndarray, np.ndarray]:
    """Construct independent asymmetric source pairs, releases, resets and shocks."""
    sources = []
    active = np.ones(batch, dtype=bool)
    if 24 <= frame < 28:
        active[np.arange(batch) % 4 == 0] = False
    resets = np.empty(0, dtype=np.int64)
    if frame == 35:
        resets = np.flatnonzero(np.arange(batch) % 5 == 0)
    if frame == 61:
        resets = np.flatnonzero(np.arange(batch) % 7 == 2)
    for i_env in np.flatnonzero(active):
        phase = 2.0 * np.pi * ((i_env * 0.6180339887498949) % 1.0)
        for i_contact in range(2):
            azimuth = phase + i_contact * (np.pi + 0.16)
            azimuth += frame * (0.009 + 0.003 * np.sin(phase + i_contact))
            azimuth += 0.08 * np.sin(0.037 * frame + phase + 1.1 * i_contact)
            latitude = 0.38 * np.sin(0.73 * phase + 1.5 * i_contact)
            latitude += 0.13 * np.sin(0.043 * frame + phase + 0.6 * i_contact)
            force = (0.8 - 0.15 * i_contact) * (1.0 + 0.22 * np.sin(0.052 * frame + phase + i_contact))
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
            friction = 0.3 + 0.1 * np.sin(0.031 * frame + phase + i_contact)
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


def current_force(mother: Mother, frame: int, batch: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources, active, resets = sources_at(frame, batch)
    raw = assemble_sources(mother, sources, order=order, audit=False)
    forces = np.zeros((mother.n_dof, batch))
    forces[:, active] = raw[:, 0::2] + raw[:, 1::2]
    return mother.balance(forces)[mother.free], active, resets


def prepare_trajectory(mother: Mother, batch: int, frames: int = 80, order: int = 8) -> Trajectory:
    start = perf_counter()
    forces = np.empty((frames, len(mother.free), batch))
    active = np.empty((frames, batch), dtype=bool)
    resets = []
    for frame in range(frames):
        forces[frame], active[frame], reset = current_force(mother, frame, batch, order)
        resets.append(reset)
    return Trajectory(
        forces,
        active,
        tuple(resets),
        perf_counter() - start,
        hashlib.sha256(forces.tobytes()).hexdigest(),
        order,
    )


def compare_trajectory(system: RecoverySystem, trajectory: Trajectory, tolerance: float) -> dict:
    batch = trajectory.active.shape[1]
    engines = {
        "copy": BatchHistory(system, batch, tolerance=tolerance, rotation=False),
        "rotate": BatchHistory(system, batch, tolerance=tolerance),
        "cached_rotate": BatchHistory(system, batch, tolerance=tolerance, cached_product=True),
    }
    exact = BatchHistory(system, batch, prediction=False)
    metrics = {name: {
        "field_errors": [], "peak_errors": [], "accepted": 0, "loaded": 0,
        "failed_columns": [], "release_absolute": 0.0, "stress_displacement_error": 0.0,
        "vs_copy_displacement": 0.0, "vs_copy_stress": 0.0,
        "gate_decision_difference": 0, "cache_shadow_decision_difference": 0,
        "cache_direct_rechecks": 0,
        "cache_gate_drift": 0.0, "cache_product_drift": 0.0, "closest_threshold_gap": float("inf"),
        "logical_history_absolute_difference": 0.0, "frames": [],
    } for name in engines}
    reset_isolation = 0.0
    reset_selected = 0.0
    for frame, forces in enumerate(trajectory.forces):
        active, resets = trajectory.active[frame], trajectory.resets[frame]
        truth = exact.step(forces, active, reset=resets)
        baseline = engines["copy"].step(forces, active, reset=resets)
        outputs = {"copy": baseline}
        for name in ("rotate", "cached_rotate"):
            outputs[name] = engines[name].step(forces, active, reset=resets, audit_cache=True)
        peak_reference = sampled_maximum(truth.stress)
        for name, output in outputs.items():
            row = metrics[name]
            errors = field_error(system, output.stress, truth.stress)
            peaks = sampled_maximum(output.stress)
            peak_error = np.abs(peaks - peak_reference) / np.maximum(peak_reference, 1e-30)
            row["field_errors"].extend(errors[active].tolist())
            row["peak_errors"].extend(peak_error[active].tolist())
            row["accepted"] += int(np.count_nonzero(output.accepted_prediction))
            row["loaded"] += int(np.count_nonzero(active))
            row["failed_columns"].append(len(output.failed_environments))
            if np.any(~active):
                row["release_absolute"] = max(row["release_absolute"], float(np.max(np.abs(output.stress[:, ~active]))))
                for array in engines[name].q_buffers + engines[name].s_buffers + engines[name].g_buffers:
                    row["release_absolute"] = max(row["release_absolute"], float(np.max(np.abs(array[:, ~active]))))
            sigma_from_q = system.stress @ output.displacement
            row["stress_displacement_error"] = max(
                row["stress_displacement_error"],
                float(np.linalg.norm(sigma_from_q - output.stress) / max(np.linalg.norm(sigma_from_q), 1e-30)),
            )
            row["vs_copy_displacement"] = max(row["vs_copy_displacement"], float(np.max(np.abs(output.displacement - baseline.displacement))))
            row["vs_copy_stress"] = max(row["vs_copy_stress"], float(np.max(np.abs(output.stress - baseline.stress))))
            row["gate_decision_difference"] += int(np.count_nonzero(output.accepted_prediction != baseline.accepted_prediction))
            row["cache_shadow_decision_difference"] += output.cache_decision_disagreement
            row["cache_direct_rechecks"] += output.cache_direct_rechecks
            row["cache_gate_drift"] = max(row["cache_gate_drift"], output.cache_gate_drift)
            row["cache_product_drift"] = max(row["cache_product_drift"], output.cache_product_drift)
            row["closest_threshold_gap"] = min(row["closest_threshold_gap"], output.threshold_gap)
            for reference, actual in zip(engines["copy"].q_buffers[:2], engines[name].q_buffers[:2], strict=True):
                row["logical_history_absolute_difference"] = max(
                    row["logical_history_absolute_difference"], float(np.max(np.abs(reference - actual)))
                )
            for reference, actual in zip(engines["copy"].s_buffers[:2], engines[name].s_buffers[:2], strict=True):
                row["logical_history_absolute_difference"] = max(
                    row["logical_history_absolute_difference"], float(np.max(np.abs(reference - actual)))
                )
            row["frames"].append({
                "frame": frame, "loaded": int(np.count_nonzero(active)),
                "accepted": int(np.count_nonzero(output.accepted_prediction)),
                "failed_ids": output.failed_environments.tolist(),
                "field_max": float(np.max(errors[active], initial=0.0)),
                "peak_max": float(np.max(peak_error[active], initial=0.0)),
            })
        if frame in (35, 61):
            # Exercise reset on private state clones, leaving actual predictor histories intact.
            for source_engine in engines.values():
                engine = BatchHistory(
                    system, batch, tolerance=tolerance, rotation=source_engine.rotation,
                    cached_product=source_engine.cached_product,
                )
                engine.depth[:] = source_engine.depth
                for destination, source in zip(
                    engine.q_buffers + engine.s_buffers + engine.g_buffers,
                    source_engine.q_buffers + source_engine.s_buffers + source_engine.g_buffers,
                    strict=True,
                ):
                    destination[:] = source
                untouched = np.setdiff1d(np.arange(batch), resets)
                saved = [array[:, untouched].copy() for array in engine.q_buffers + engine.s_buffers + engine.g_buffers]
                engine.reset(resets)
                for before, array in zip(saved, engine.q_buffers + engine.s_buffers + engine.g_buffers, strict=True):
                    reset_isolation = max(reset_isolation, float(np.max(np.abs(before - array[:, untouched]))))
                    reset_selected = max(reset_selected, float(np.max(np.abs(array[:, resets]))))
    rows = {}
    for name, row in metrics.items():
        field_values = row.pop("field_errors")
        peak_values = row.pop("peak_errors")
        failed = row.pop("failed_columns")
        row.update(
            field_mean=float(np.mean(field_values)), field_max=float(np.max(field_values)),
            peak_mean=float(np.mean(peak_values)), peak_max=float(np.max(peak_values)),
            acceptance=row["accepted"] / max(row["loaded"], 1), failed_mean=float(np.mean(failed)),
        )
        rows[name] = row
    return {
        "batch": batch,
        "frames": len(trajectory.forces),
        "tolerance": tolerance,
        "order": trajectory.order,
        "force_sha256": trajectory.force_sha256,
        "load_preparation_seconds": trajectory.construction_seconds,
        "reset_untouched_absolute_difference": reset_isolation,
        "reset_selected_history_absolute": reset_selected,
        "outputs": "all declared mother top/bottom local in-plane stress components",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("history_results.json"))
    args = parser.parse_args()
    source_paths = [Path(__file__), Path(__file__).with_name("history.py"),
                    Path(__file__).with_name("model.py"), Path(__file__).with_name("geometry.py"),
                    Path(__file__).with_name("loads.py"), Path(__file__).parents[2] / "shell_cpu" / "core.py"]
    hashes = {str(path.relative_to(Path(__file__).parents[3])): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    with threadpool_limits(limits=1):
        mother = Mother()
        system = RecoverySystem.from_mother(mother)
        rows = []
        for batch in args.batches:
            trajectory = prepare_trajectory(mother, batch, frames=args.frames, order=args.order)
            for tolerance in args.tolerances:
                rows.append(compare_trajectory(system, trajectory, tolerance))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(Path(__file__).parents[3]))]
               for path in source_paths):
            raise RuntimeError("A history numerical source changed during the captured run.")
        args.output.write_text(json.dumps({"reconstruction": True, "source_hashes": hashes, "rows": rows}, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "cases": len(rows)}))


if __name__ == "__main__":
    main()
