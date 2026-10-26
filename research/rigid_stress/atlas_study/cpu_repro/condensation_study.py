"""Compare exact block-condensation directions on actual enriched shell spaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter, process_time

import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_info, threadpool_limits

from .condensation import CondensedSystem, DirectSystem, solve_without_cross_patch_coupling
from .loads import Source, assemble_sources
from .model import Mother


@dataclass(frozen=True)
class Case:
    name: str
    centers: tuple[np.ndarray, ...]
    sources: tuple[Source, ...]


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def patch_radius() -> float:
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array([(0.0, a, b) for a in (-1.0, 1.0) for b in (-phi, phi)]
                        + [(a, b, 0.0) for a in (-1.0, 1.0) for b in (-phi, phi)]
                        + [(b, 0.0, a) for a in (-1.0, 1.0) for b in (-phi, phi)])
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = vertices[ConvexHull(vertices).simplices].mean(axis=1)
    faces /= np.linalg.norm(faces, axis=1)[:, None]
    fill = np.max(np.arccos(np.clip((faces @ vertices.T).max(axis=1), -1.0, 1.0)))
    return float(1.2 * fill + 0.003 / 0.022)


def make_cases() -> tuple[Case, ...]:
    first = normalize(np.array([0.0, 1.0, 0.5 * (1.0 + np.sqrt(5.0))]))
    far = -first
    nearby = normalize(first + np.array([0.27, -0.04, 0.02]))
    third = normalize(np.array([1.0, -0.23, 0.31]))
    directions = ((first,), (first, far), (first, nearby), (first, far, third))
    names = ("single_patch", "two_far", "two_near", "three_asymmetric")
    cases = []
    for name, centers in zip(names, directions, strict=True):
        sources = tuple(Source(direction=center, radius=0.0015 + 0.0003 * i,
                               normal_force=0.2 + 0.07 * i, friction=0.3 + 0.1 * (i % 2),
                               name=f"{name}_{i}") for i, center in enumerate(centers))
        cases.append(Case(name, centers, sources))
    return tuple(cases)


def relative_error(value: np.ndarray, reference: np.ndarray, weights: np.ndarray | None = None) -> float:
    delta = value - reference
    if weights is not None:
        shape = weights if value.ndim == 1 else weights[:, None]
        numerator = np.sum(shape * delta * delta)
        denominator = np.sum(shape * reference * reference)
        return float(np.sqrt(numerator / max(denominator, 1e-60)))
    return float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30))


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    names = ("condensation.py", "condensation_study.py", "geometry.py", "model.py", "loads.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def build_case(mother: Mother, case: Case, order: int):
    features = tuple(mother.feature(center, patch_radius(), halo=1, name=f"{case.name}_{i}")
                     for i, center in enumerate(case.centers))
    active = np.logical_or.reduce([feature.active_macros for feature in features])
    space = mother.space(active, name=case.name)
    physical = assemble_sources(mother, list(case.sources), order=order)
    total = np.sum(physical, axis=1)
    balanced = mother.balance(total)
    rng = np.random.default_rng(613027)
    distributed = rng.normal(size=(mother.n_dof, 3))
    distributed.reshape((-1, 6, 3))[:, 3:] *= 0.0005
    distributed = mother.balance(distributed)
    force = np.column_stack((balanced, distributed))
    rhs = np.asfortranarray(space.T.T @ force)
    return space, rhs


def numerical_study(mother: Mother, order: int) -> dict:
    rows = []
    areas = np.tile(mother.model.gauss_weights.reshape(-1), 2)
    stress_weights = np.repeat(areas, 3) * np.tile(np.array([1.0, 1.0, 2.0]), len(areas))
    for case in make_cases():
        space, rhs = build_case(mother, case, order)
        direct = DirectSystem(space.K)
        reference = direct.solve(rhs)
        reference_field = space.S @ reference
        methods = []
        for direction in ("coarse_first", "local_first"):
            system = CondensedSystem(space.K, space.n_coarse, direction)
            solution = system.solve(rhs)
            field = space.S @ solution
            residual = space.K @ solution - rhs
            column_residuals = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
            methods.append({
                "direction": direction,
                "coordinate_relative": relative_error(solution, reference),
                "complete_weighted_tensor_relative": relative_error(field, reference_field, stress_weights),
                "complete_weighted_tensor_relative_by_rhs": [
                    relative_error(field[:, j], reference_field[:, j], stress_weights) for j in range(rhs.shape[1])
                ],
                "projected_residual_relative_max": float(column_residuals.max()),
                "projected_residual_by_rhs": column_residuals.tolist(),
                "payload_bytes": system.payload_bytes,
                "construction_seconds_diagnostic": system.build_seconds,
            })
            del system
            gc.collect()
        drop = None
        if len(case.centers) > 1:
            local_directions = mother.geometry.directions[space.local_nodes]
            node_groups = np.argmax(local_directions @ np.stack(case.centers).T, axis=1)
            labels = np.repeat(node_groups, 6)
            if len(labels) != space.n_dof - space.n_coarse:
                raise ValueError("Local coordinate ordering differs from the node-major bubble convention.")
            result = solve_without_cross_patch_coupling(space.K, space.n_coarse, rhs, labels)
            dropped_field = space.S @ result.solution
            drop = {
                "groups": result.n_groups,
                "removed_schur_norm_relative": result.eliminated_cross_norm_relative,
                "coordinate_relative": relative_error(result.solution, reference),
                "complete_weighted_tensor_relative": relative_error(dropped_field, reference_field, stress_weights),
                "complete_weighted_tensor_relative_by_rhs": [
                    relative_error(dropped_field[:, j], reference_field[:, j], stress_weights)
                    for j in range(rhs.shape[1])
                ],
                "physical_cap_tensor_relative": relative_error(
                    dropped_field[:, 0], reference_field[:, 0], stress_weights
                ),
                "projected_residual_relative": relative_error(space.K @ result.solution, rhs),
                "is_exact_control": False,
            }
        rows.append({
            "case": case.name,
            "source_directions": [center.tolist() for center in case.centers],
            "coarse_dof": space.n_coarse,
            "local_dof": space.n_dof - space.n_coarse,
            "total_dof": space.n_dof,
            "active_macros": int(np.count_nonzero(space.active_macros)),
            "rhs_count": rhs.shape[1],
            "rhs_columns": ["physical_finite_caps", "distributed_0", "distributed_1", "distributed_2"],
            "direct_payload_bytes": direct.payload_bytes,
            "direct_construction_seconds_diagnostic": direct.build_seconds,
            "methods": methods,
            "drop_cross_patch_negative": drop,
        })
        del direct, space
        gc.collect()
    passed = all(all(error < 1e-8 for error in method["complete_weighted_tensor_relative_by_rhs"])
                 and method["projected_residual_relative_max"] < 1e-8
                 for row in rows for method in row["methods"])
    return {"passed": passed, "rows": rows}


def timing_study(mother: Mother, order: int, repeats: int) -> dict:
    rows = []
    for case in make_cases():
        space, rhs = build_case(mother, case, order)
        systems = {
            "direct": DirectSystem(space.K),
            "coarse_first": CondensedSystem(space.K, space.n_coarse, "coarse_first"),
            "local_first": CondensedSystem(space.K, space.n_coarse, "local_first"),
        }
        for batch in (1, 16, 64):
            loads = np.asfortranarray(rhs[:, np.arange(batch) % rhs.shape[1]])
            raw = {name: [] for name in systems}
            checksums = {name: [] for name in systems}
            for system in systems.values():
                field = space.S @ system.solve(loads)
            for repeat in range(repeats):
                names = list(systems) if repeat % 2 == 0 else list(reversed(systems))
                for name in names:
                    start_wall, start_cpu = perf_counter(), process_time()
                    field = space.S @ systems[name].solve(loads)
                    wall, cpu = perf_counter() - start_wall, process_time() - start_cpu
                    checksum = [float(np.linalg.norm(field)), float(np.sum(field))]
                    raw[name].append({"repeat": repeat, "wall_ms": wall * 1000.0, "cpu_ms": cpu * 1000.0})
                    checksums[name].append(checksum)
            methods = {}
            for name, samples in raw.items():
                wall = np.array([sample["wall_ms"] for sample in samples])
                cpu = np.array([sample["cpu_ms"] for sample in samples])
                methods[name] = {
                    "median_wall_ms": float(np.median(wall)),
                    "min_wall_ms": float(wall.min()),
                    "max_wall_ms": float(wall.max()),
                    "median_cpu_ms": float(np.median(cpu)),
                    "raw_repeats": samples,
                    "checksums_outside_timing": checksums[name],
                    "payload_bytes": systems[name].payload_bytes,
                }
            rows.append({"case": case.name, "batch": batch, "dof": space.n_dof, "methods": methods})
        del systems, space
        gc.collect()
    return {
        "scope": ("reduced RHS -> reduced solve -> all common local stress components; source assembly, "
                  "projection, world tensors and max reduction excluded"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--order", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    with threadpool_limits(limits=1):
        mother = Mother()
        result = {
            "model": {
                "ndof": mother.n_dof,
                "elements": len(mother.geometry.quads),
                "coarse_free": len(mother.coarse_free),
            },
            "scope": ("exact algebra within identical common-mother Galerkin spaces, "
                      "not independent spatial accuracy or a GPU benchmark"),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "threads": threadpool_info(),
            },
            "source_sha256": before,
            "finite_cap_order": args.order,
        }
        if args.timing:
            result["timing"] = timing_study(mother, args.order, args.repeats)
            name = "condensation_timing.json"
        else:
            result["numerics"] = numerical_study(mother, args.order)
            name = "condensation_results.json"
        result["source_unchanged"] = source_hashes() == before
        if not result["source_unchanged"]:
            raise RuntimeError("A measured source changed during the run.")
        path = args.output / name
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps({
            "output": str(path), "passed": result.get("numerics", {}).get("passed"), "source_unchanged": True
        }), flush=True)
        if not args.timing and not result["numerics"]["passed"]:
            raise RuntimeError("Condensation numerical comparison failed.")


if __name__ == "__main__":
    main()
