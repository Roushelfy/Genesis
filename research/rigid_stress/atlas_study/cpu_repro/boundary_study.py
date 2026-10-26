"""Resolve selector jumps separately from smooth changes in the current load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits

from .fixtures import unit
from .loads import Source, assemble_sources
from .model import Mother
from .selection import build_bank, field_weights, relative_field_error, sampled_maximum


def boundary_locations(centers: np.ndarray) -> tuple[dict, ...]:
    hull = ConvexHull(centers)
    simplex = sorted(tuple(sorted(int(value) for value in face)) for face in hull.simplices)[0]
    i, j, k = simplex
    vertex = unit(np.cross(centers[j] - centers[i], centers[k] - centers[i]))
    if np.dot(vertex, centers[i]) < 0.0:
        vertex = -vertex
    adjacent_faces = [face for face in hull.simplices if i in face and j in face]
    dual_vertices = []
    for face in adjacent_faces:
        a, b, c = centers[face]
        normal = unit(np.cross(b - a, c - a))
        if np.dot(normal, a) < 0.0:
            normal = -normal
        dual_vertices.append(normal)
    assert len(dual_vertices) == 2
    pair = unit(dual_vertices[0] + dual_vertices[1])
    triple_tangent = unit(centers[i] - centers[j])
    pair_tangent = unit(centers[i] - centers[j])
    exchange_normal = unit(centers[j] - centers[k])
    second_third = vertex + 0.15 * centers[i]
    second_third -= np.dot(second_third, exchange_normal) * exchange_normal
    second_third = unit(second_third)
    exchange_tangent = centers[j] - centers[k]
    exchange_tangent -= np.dot(exchange_tangent, second_third) * second_third
    exchange_tangent = unit(exchange_tangent)
    return (
        {"name": "nearest_pair_boundary", "center": pair, "tangent": pair_tangent,
         "expected_ties": (i, j)},
        {"name": "second_third_exchange", "center": second_third, "tangent": exchange_tangent,
         "expected_ties": (j, k)},
        {"name": "triple_junction", "center": vertex, "tangent": triple_tangent,
         "expected_ties": (i, j, k)},
    )


def study(output: Path, count: int = 12, order: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    mother = Mother()
    bank = build_bank(mother, count=count)
    weights = field_weights(mother)
    rows = []
    locations = []
    for boundary in boundary_locations(bank.centers):
        center, tangent = boundary["center"], boundary["tangent"]
        fixed_force = -0.1 * center + 0.025 * np.cross(center, tangent)
        center_angles = np.arccos(np.clip(bank.centers @ center, -1.0, 1.0))
        locations.append({"name": boundary["name"], "direction": center.tolist(),
                          "crossing_tangent": tangent.tolist(), "force": fixed_force.tolist(),
                          "expected_ties": boundary["expected_ties"],
                          "sorted_center_angles": np.sort(center_angles).tolist()})
        for epsilon in (1e-4, 1e-5, 1e-6):
            directions = (np.cos(epsilon) * center - np.sin(epsilon) * tangent,
                          np.cos(epsilon) * center + np.sin(epsilon) * tangent)
            sources = tuple(Source(direction, 0.002, force=fixed_force, cap_tangent=tangent,
                                   name=f"{boundary['name']}-{side}")
                            for side, direction in enumerate(directions))
            raw = assemble_sources(mother, sources, order=order, audit=True)
            balanced = mother.balance(raw)
            q_reference = mother.solve(balanced)
            reference = mother.S @ q_reference
            cached_q = tuple(space.lift(space.solve(balanced)) for space in bank.spaces)
            reference_jump = float(relative_field_error(reference[:, 1], reference[:, 0], weights))
            reference_peaks = sampled_maximum(reference)
            for strategy in ("continuous", "nearest", "top2", "narrow"):
                q = np.zeros_like(q_reference)
                assignments = []
                for side, source in enumerate(sources):
                    indices, coefficients = bank.select(source, strategy)
                    for index, coefficient in zip(indices, coefficients):
                        q[:, side] += coefficient * cached_q[index][:, side]
                    assignments.append({"indices": indices.tolist(), "weights": coefficients.tolist()})
                sigma = mother.S @ q
                peaks = sampled_maximum(sigma)
                peak_scale = max(float(np.mean(reference_peaks)), 1e-30)
                rows.append({"boundary": boundary["name"], "epsilon_rad": epsilon,
                             "strategy": strategy, "reference_relative_field_change": reference_jump,
                             "relative_field_jump": float(relative_field_error(sigma[:, 1], sigma[:, 0], weights)),
                             "relative_sampled_peak_jump": float(abs(peaks[1] - peaks[0]) / peak_scale),
                             "reference_relative_peak_change": float(abs(reference_peaks[1] - reference_peaks[0]) / peak_scale),
                             "field_errors": relative_field_error(sigma, reference, weights).tolist(),
                             "sampled_peaks_pa": peaks.tolist(), "assignments": assignments})
    result = {"kind": "reconstructed_boundary_numerics", "library_size": count,
              "quadrature_order": order, "radius_m": 0.002,
              "force_policy": "One fixed global force and safe cap tangent on both sides of each crossing",
              "field_jump_denominator": "Area/Frobenius norm of the left field",
              "peak_jump_denominator": "Mean mother sampled peak across the crossing",
              "locations": locations, "rows": rows}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-size", type=int, default=12)
    parser.add_argument("--order", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = study(args.output, args.library_size, args.order)
    for row in result["rows"]:
        if row["epsilon_rad"] <= 1.1e-6:
            print(json.dumps({key: row[key] for key in ("boundary", "strategy", "relative_field_jump",
                                                      "relative_sampled_peak_jump")}))


if __name__ == "__main__":
    main()
