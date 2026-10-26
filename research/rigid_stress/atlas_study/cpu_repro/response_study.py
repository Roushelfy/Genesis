"""Run fixed-source response, sampled-maximum and global energy-basis controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .loads import Source, assemble_sources
from .model import Mother
from .response import EnergyPOD, ExactTopResponse, MaximumBounds, relative_tensor_error, sampled_maximum


@dataclass(frozen=True)
class ResponseCase:
    name: str
    sources: tuple[Source, ...]


def fibonacci_directions(count: int, phase: float = 0.0) -> np.ndarray:
    index = np.arange(count)
    z = 1.0 - 2.0 * (index + 0.5) / count
    angle = index * np.pi * (3.0 - np.sqrt(5.0)) + phase
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def response_cases() -> tuple[ResponseCase, ...]:
    directions = fibonacci_directions(36, phase=0.371)
    cases = []
    for index, direction in enumerate(directions):
        source = Source(
            direction,
            radius=(1 + index % 3) * 0.001,
            normal_force=0.4 + 0.08 * (index % 5),
            friction=(0.0, 0.3, 0.6)[index % 3],
            name=f"heldout-{index:02d}",
        )
        cases.append(ResponseCase(source.name, (source,)))
    for index, direction in enumerate(np.vstack((np.eye(3), -np.eye(3)))):
        source = Source(direction, radius=0.002, normal_force=0.7, friction=0.3, name=f"axis-{index}")
        cases.append(ResponseCase(source.name, (source,)))
    cases.append(ResponseCase("two-far", (cases[1].sources[0], cases[27].sources[0])))
    cases.append(ResponseCase("three-asymmetric", (cases[4].sources[0], cases[16].sources[0], cases[31].sources[0])))
    cases.append(ResponseCase("release", ()))
    return tuple(cases)


def raw_case_loads(mother: Mother, cases: tuple[ResponseCase, ...], order: int) -> np.ndarray:
    force = np.zeros((mother.model.ndof, len(cases)))
    for index, case in enumerate(cases):
        if case.sources:
            force[:, index] = assemble_sources(mother, list(case.sources), order=order).sum(axis=1)
    return force


def point_weights(mother: Mother) -> np.ndarray:
    return np.tile(mother.model.gauss_weights.reshape(-1), 2)


def peak_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    actual = sampled_maximum(candidate)
    target = sampled_maximum(reference)
    return np.abs(actual - target) / np.maximum(target, 1e-30)


def run_bound_trajectory(mother: Mother, response: ExactTopResponse, quantized: ExactTopResponse, order: int) -> dict:
    """Check independent bound histories under moving loads and partial lifecycle events."""
    bounds = MaximumBounds.build(response, block_elements=6)
    force = np.zeros((mother.n_dof, 2))
    weights = point_weights(mother)
    rows = []
    identities = np.array([11, 19])
    for frame in range(80):
        force.fill(0.0)
        if frame in (35, 61):
            bounds.reset(identities[[1 if frame == 35 else 0]])
        for environment in range(2):
            if environment == 0 and 24 <= frame < 28:
                continue
            angle = 0.008 * frame + 0.19 * environment
            if frame == 42:
                angle += 0.7
            first = np.array([np.cos(angle), np.sin(angle), 0.21 + 0.05 * np.sin(0.4 * angle)])
            second = np.array([-0.43, np.cos(0.7 * angle + 1.0), np.sin(0.7 * angle + 1.0)])
            normal_force = 0.7 + 0.1 * np.sin(0.07 * frame + environment)
            sources = [
                Source(first, radius=0.002 * normal_force ** (1.0 / 3.0), normal_force=normal_force, friction=0.3),
                Source(second, radius=0.0018 * normal_force ** (1.0 / 3.0), normal_force=0.8 * normal_force, friction=0.6),
            ]
            force[:, environment] = assemble_sources(mother, sources, order=order).sum(axis=1)
        target = mother.S @ mother.solve(mother.balance(force))
        exact = response.apply(force)
        reduced = quantized.apply(force)
        permutation = np.array([1, 0]) if frame % 2 else np.array([0, 1])
        screened = bounds.apply(force[:, permutation], env_ids=identities[permutation], use_history=True)
        maxima = screened.maxima[np.argsort(permutation)]
        peak = sampled_maximum(exact)
        difference = np.max(np.abs(maxima - peak) / np.maximum(peak, 1e-30))
        rows.append({
            "frame": frame,
            "fp64_field_worst": float(relative_tensor_error(target, exact, weights).max()),
            "fp32_field_worst": float(relative_tensor_error(target, reduced, weights).max()),
            "screen_peak_worst": float(difference),
            "evaluated_blocks": screened.evaluated_blocks.tolist(),
            "release_nonzeros": int(np.count_nonzero(exact[:, 0])) if 24 <= frame < 28 else 0,
        })
    return {
        "frames": len(rows),
        "environments": 2,
        "different_environment_ids_and_order": True,
        "events": {"partial_release": [24, 27], "partial_reset": [35, 61], "position_shock": 42},
        "fp64_field_worst": max(row["fp64_field_worst"] for row in rows),
        "fp32_field_worst": max(row["fp32_field_worst"] for row in rows),
        "screen_peak_worst": max(row["screen_peak_worst"] for row in rows),
        "release_nonzeros": sum(row["release_nonzeros"] for row in rows),
        "rows": rows,
    }


def run_response(mother: Mother, order: int, include_bounds: bool = True) -> tuple[dict, ExactTopResponse]:
    cases = response_cases()
    raw = raw_case_loads(mother, cases, order)
    reference = mother.S @ mother.solve(mother.balance(raw))
    response = ExactTopResponse.build(mother)
    quantized = response.quantized()
    exact = response.apply(raw)
    reduced = quantized.apply(raw)
    weights = point_weights(mother)
    exact_error = relative_tensor_error(reference, exact, weights)
    quantized_error = relative_tensor_error(reference, reduced, weights)
    exact_peak = peak_error(reference, exact)
    quantized_peak = peak_error(reference, reduced)
    load_map_error = np.linalg.norm(
        raw.reshape((-1, 6, raw.shape[1]))[:, 3:]
        - 0.5 * mother.model.h * np.cross(
            mother.model.directors[:, :, None], raw.reshape((-1, 6, raw.shape[1]))[:, :3],
            axisa=1, axisb=1, axisc=1,
        )
    )
    dense_difference = relative_tensor_error(exact, response.apply(raw, sparse_input=False), weights)
    changed = raw[:, :4].copy()
    untouched = response.apply(changed)[:, 1:].copy()
    changed[:, 0] = 0.0
    released = response.apply(changed)
    release_nonzeros = np.count_nonzero(released[:, 0])
    isolation_difference = np.max(np.abs(released[:, 1:] - untouched))
    rejected_side = False
    rejected_couple = False
    try:
        response.apply(raw[:, :1], side=-1)
    except ValueError:
        rejected_side = True
    altered = raw[:, :1].copy()
    altered[3, 0] += 0.001
    try:
        response.apply(altered)
    except ValueError:
        rejected_couple = True
    rows = [
        {
            "case": case.name,
            "sources": len(case.sources),
            "fp64_field_error": float(exact_error[index]),
            "fp32_field_error": float(quantized_error[index]),
            "fp64_peak_error": float(exact_peak[index]),
            "fp32_peak_error": float(quantized_peak[index]),
        }
        for index, case in enumerate(cases)
    ]
    result = {
        "scope": "newly reconstructed fixed-mother top-force response, same-order discrete FEM",
        "case_count": len(cases),
        "loaded_case_count": len(cases) - 1,
        "quadrature_order": order,
        "H_shape": list(response.values.shape),
        "H64_bytes": response.nbytes,
        "H32_bytes": quantized.nbytes,
        "response_build_seconds": response.build_seconds,
        "fp64_field_worst": float(exact_error.max()),
        "fp32_field_worst": float(quantized_error.max()),
        "fp64_peak_worst": float(exact_peak.max()),
        "fp32_peak_worst": float(quantized_peak.max()),
        "dense_sparse_field_worst": float(dense_difference.max()),
        "raw_top_couple_absolute_difference": float(load_map_error),
        "release_nonzeros": int(release_nonzeros),
        "untouched_environment_absolute_difference": float(isolation_difference),
        "bottom_side_rejected": rejected_side,
        "independent_couple_rejected": rejected_couple,
        "rows": rows,
    }
    if include_bounds:
        bound_rows = []
        for elements in (6, 24, 72):
            bounds = MaximumBounds.build(response, block_elements=elements)
            direct = bounds.apply(raw)
            first = bounds.apply(raw, env_ids=np.arange(len(cases)), use_history=True)
            second = bounds.apply(raw, env_ids=np.arange(len(cases)), use_history=True)
            target = sampled_maximum(exact)
            denominator = np.maximum(target, 1e-30)
            differences = [
                float(np.max(np.abs(value.maxima - target) / denominator))
                for value in (direct, first, second)
            ]
            bounds.reset(np.array([1, 3]))
            reset_result = bounds.apply(raw[:, [1, 3]], env_ids=np.array([1, 3]), use_history=True)
            reset_difference = float(np.max(np.abs(reset_result.maxima - target[[1, 3]]) / denominator[[1, 3]]))
            bound_rows.append({
                "block_elements": elements,
                "block_count": direct.total_blocks,
                "bounds_auxiliary_bytes": bounds.nbytes,
                "maximum_relative_difference": max(differences),
                "reset_relative_difference": reset_difference,
                "loaded_skipped_fraction": float(1.0 - direct.evaluated_blocks[:-1].mean() / direct.total_blocks),
                "history_repeat_skipped_fraction": float(1.0 - second.evaluated_blocks[:-1].mean() / second.total_blocks),
                "evaluated_blocks": direct.evaluated_blocks.tolist(),
            })
        result["maximum_bounds"] = bound_rows
        result["temporal_response"] = run_bound_trajectory(mother, response, quantized, order)
    temporal = result.get("temporal_response", {})
    result["passed"] = bool(
        exact_error.max() < 1e-9 and quantized_error.max() < 1e-5
        and exact_peak.max() < 1e-9 and quantized_peak.max() < 1e-5
        and dense_difference.max() < 1e-9 and not release_nonzeros
        and isolation_difference < 1e-20 and rejected_side and rejected_couple
        and all(row["maximum_relative_difference"] < 1e-9 for row in result.get("maximum_bounds", []))
        and temporal.get("fp64_field_worst", 0.0) < 1e-9
        and temporal.get("fp32_field_worst", 0.0) < 1e-5
        and temporal.get("screen_peak_worst", 0.0) < 1e-9
        and not temporal.get("release_nonzeros", 0)
    )
    return result, response


def run_pod(mother: Mother, order: int) -> dict:
    directions = fibonacci_directions(24, phase=0.0)
    sources = [
        Source(direction, radius=radius, force=force, name=f"train-{i_direction}-{radius}-{i_force}")
        for i_direction, direction in enumerate(directions)
        for radius in (0.001, 0.002, 0.003)
        for i_force, force in enumerate(np.eye(3))
    ]
    start = perf_counter()
    raw = assemble_sources(mother, sources, order=order)
    snapshots = mother.solve(mother.balance(raw))
    training_seconds = perf_counter() - start
    cases = response_cases()
    force = mother.balance(raw_case_loads(mother, cases, order))
    target = mother.S @ mother.solve(force)
    weights = point_weights(mother)
    rows = []
    for rank in (32, 64, 128, 192):
        basis = EnergyPOD.build(mother, snapshots, rank)
        candidate = basis.apply(force)
        error = relative_tensor_error(target, candidate, weights)
        peak = peak_error(target, candidate)
        rows.append({
            "requested_rank": rank,
            "actual_rank": basis.rank,
            "loaded_field_mean": float(error[:-1].mean()),
            "field_worst": float(error.max()),
            "sampled_peak_worst": float(peak.max()),
            "inventory_bytes": basis.nbytes,
        })
    return {
        "scope": "global energy POD with current-force Galerkin coefficients",
        "training_snapshots": len(sources),
        "training_seconds": training_seconds,
        "training_phase": 0.0,
        "heldout_phase": 0.371,
        "rows": rows,
        "note": "Low-rank errors are measured outcomes; no accuracy target was fitted to the historical table.",
    }


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--no-pod", action="store_true")
    parser.add_argument("--no-bounds", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    mother = Mother()
    result, response = run_response(mother, arguments.order, include_bounds=not arguments.no_bounds)
    with (arguments.output / "response_results.json").open("w") as output:
        json.dump(result, output, indent=2, default=serialize)
        output.write("\n")
    del response
    if not arguments.no_pod:
        pod = run_pod(mother, arguments.order)
        with (arguments.output / "pod_results.json").open("w") as output:
            json.dump(pod, output, indent=2, default=serialize)
            output.write("\n")
    with (arguments.output / "response_cases.json").open("w") as output:
        json.dump([asdict(case) for case in response_cases()], output, indent=2, default=serialize)
        output.write("\n")
    if not result["passed"]:
        raise SystemExit("Response numerical acceptance failed; inspect retained JSON.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
