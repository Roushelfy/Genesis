"""Validate source-local feature selection against the fixed fine shell mother."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .fixtures import cases_manifest, make_cases
from .loads import assemble_sources
from .model import Mother
from .selection import build_bank, field_weights, relative_field_error, sampled_maximum


def study(output: Path, count: int = 12, order: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    manifest = cases_manifest(cases)
    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    (output / "fixtures.json").write_text(serialized + "\n")
    mother = Mother()
    bank = build_bank(mother, count=count)
    sources = tuple(source for case in cases for source in case.sources)
    raw = assemble_sources(mother, sources, order=order, audit=True)
    balanced = mother.balance(raw)
    offsets = np.cumsum([0] + [len(case.sources) for case in cases])
    reference_q = mother.solve(balanced)
    reference_sigma = mother.S @ reference_q
    cached_z = tuple(space.solve(balanced) for space in bank.spaces)
    coarse_nodes = set(mother.geometry.coarse_nodes.tolist())
    n_coarse_free = len(mother.coarse_free)
    bubble_nodes = tuple(set(np.flatnonzero(space.T[:, n_coarse_free:].getnnz(axis=1)) // 6)
                         for space in bank.spaces)
    weights = field_weights(mother)
    model_rows = []
    for i, space in enumerate(bank.spaces):
        model_rows.append({"index": i, "center": bank.centers[i].tolist(),
                           "independent_coordinates": int(space.T.shape[1]),
                           "stiffness_nonzeros": int(space.K.nnz),
                           "bubble_nodes": len(bubble_nodes[i])})
    rows = []
    worst_residual = 0.0
    worst_merged_difference = 0.0
    coverage_failures = []
    negative_coverage_failures = []
    strategies = ("continuous", "nearest", "narrow", "top2", "wrong_all_loads")
    for case_index, case in enumerate(cases):
        start, end = offsets[case_index:case_index + 2]
        columns = np.arange(start, end)
        total_reference = np.sum(reference_sigma[:, columns], axis=1)
        total_force = np.sum(balanced[:, columns], axis=1)
        reference_peak = sampled_maximum(total_reference)
        for strategy in strategies:
            groups = {}
            contribution_count = 0
            source_assignment = []
            superposed_q = np.zeros(mother.n_dof)
            if strategy == "wrong_all_loads":
                selected = sorted({int(bank.select(source, "nearest")[0][0]) for source in case.sources})
                for i in selected:
                    groups[i] = total_force / len(selected)
                contribution_count = len(selected) * len(case.sources)
                for column in columns:
                    nodal = raw[:, column].reshape((-1, 6))
                    threshold = np.max(np.abs(nodal)) * 1e-14
                    loaded_nodes = set(np.flatnonzero(np.max(np.abs(nodal), axis=1) > threshold))
                    for index in selected:
                        uncovered = loaded_nodes - coarse_nodes - bubble_nodes[index]
                        if uncovered:
                            negative_coverage_failures.append({"case": case.name, "feature": index,
                                                              "source_column": int(column),
                                                              "uncovered_nodes": [int(node) for node in sorted(uncovered)]})
            else:
                for local, source in enumerate(case.sources):
                    column = start + local
                    indices, mixture = bank.select(source, strategy)
                    contribution_count += len(indices)
                    source_assignment.append({"source": source.name, "indices": indices.tolist(),
                                              "weights": mixture.tolist()})
                    nodal = raw[:, column].reshape((-1, 6))
                    threshold = np.max(np.abs(nodal)) * 1e-14
                    loaded_nodes = set(np.flatnonzero(np.max(np.abs(nodal), axis=1) > threshold))
                    for index, coefficient in zip(indices, mixture):
                        i = int(index)
                        if i not in groups:
                            groups[i] = np.zeros(mother.n_dof)
                        groups[i] += coefficient * balanced[:, column]
                        superposed_q += coefficient * bank.spaces[i].lift(cached_z[i][:, column])
                        uncovered = loaded_nodes - coarse_nodes - bubble_nodes[i]
                        if uncovered:
                            coverage_failures.append({"case": case.name, "strategy": strategy,
                                                      "source": source.name, "feature": i,
                                                      "uncovered_nodes": [int(node) for node in sorted(uncovered)]})
            q = np.zeros(mother.n_dof)
            for index, force in groups.items():
                space = bank.spaces[index]
                z = space.solve(force)
                projected = space.T.T @ force
                residual = np.linalg.norm(space.K @ z - projected) / max(np.linalg.norm(projected), 1e-30)
                worst_residual = max(worst_residual, float(residual))
                q += space.lift(z)
            if strategy != "wrong_all_loads":
                merged_error = np.linalg.norm(q - superposed_q) / max(np.linalg.norm(q), 1e-30)
                worst_merged_difference = max(worst_merged_difference, float(merged_error))
            sigma = mother.S @ q
            error = float(relative_field_error(sigma, total_reference, weights))
            peak = float(sampled_maximum(sigma))
            peak_error = abs(peak - reference_peak) / max(reference_peak, 1e-30)
            rows.append({"case": case.name, "family": case.family, "loaded": bool(case.sources),
                         "source_count": len(case.sources), "strategy": strategy,
                         "groups": len(groups), "contributions": contribution_count,
                         "relative_field_error": error, "relative_peak_error": float(peak_error),
                         "sampled_peak_pa": peak, "reference_peak_pa": float(reference_peak),
                         "assignments": source_assignment})
    summary = []
    for strategy in strategies:
        loaded = [row for row in rows if row["strategy"] == strategy and row["loaded"]]
        summary.append({"strategy": strategy, "loaded_cases": len(loaded),
                        "mean_field_error": float(np.mean([row["relative_field_error"] for row in loaded])),
                        "worst_field_error": max(row["relative_field_error"] for row in loaded),
                        "worst_peak_error": max(row["relative_peak_error"] for row in loaded),
                        "mean_groups": float(np.mean([row["groups"] for row in loaded])),
                        "maximum_groups": max(row["groups"] for row in loaded),
                        "mean_contributions_per_source": float(np.mean([
                            row["contributions"] / row["source_count"] for row in loaded]))})
    result = {"kind": "reconstructed_selection_numerics", "historical_sources_recovered": False,
              "position_convention": "Authored egg at coarse nodes is the midsurface, then frozen bilinear refinement",
              "fixture_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
              "mother_dof": mother.n_dof, "mother_elements": len(mother.model.quads),
              "coarse_coordinates": mother.P.shape[1], "library_size": count,
              "fill_angle": bank.fill_angle, "support_angle": bank.support_angle,
              "core_angle": bank.core_angle, "narrow_fraction": bank.narrow_fraction,
              "quadrature_order": order, "maximum_prepared_radius_m": bank.maximum_radius,
              "coverage_load_relative_cutoff": 1e-14, "coverage_failures": coverage_failures,
              "wrong_all_loads_coverage_failures": negative_coverage_failures,
              "worst_reduced_numerical_residual": worst_residual,
              "worst_merged_rhs_vs_source_superposition": worst_merged_difference,
              "summary": summary, "models": model_rows, "cases": rows}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if coverage_failures:
        raise AssertionError("A selected feature missed a finite-source nodal support. See results.json.")
    if worst_residual > 1e-9 or worst_merged_difference > 1e-9:
        raise ArithmeticError("Reduced solve or merged-RHS superposition failed. See results.json.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-size", type=int, default=12)
    parser.add_argument("--order", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = study(args.output, args.library_size, args.order)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
