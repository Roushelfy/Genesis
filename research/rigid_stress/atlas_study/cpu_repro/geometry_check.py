"""Compare closed-form ray mapping with an exhaustive physical Newton oracle."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import FRAMES, authored_egg, make_geometry, normalize
from .loads import Source, assemble_sources, force_for_source, source_directions
from .model import Mother


def newton_oracle(geometry, directions: np.ndarray,
                  preferred_macro: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Solve all macro charts in physical coordinates with three Newton unknowns."""
    directions = normalize(directions)
    rays = authored_egg(directions)
    nq, nm = len(rays), len(geometry.macro_quads)
    coefficients = geometry.macro_coefficients[None]
    ray = np.broadcast_to(rays[:, None], (nq, nm, 3))
    xi = np.zeros((nq, nm))
    eta = np.zeros_like(xi)
    lam = np.einsum("mni,mni->mn", np.broadcast_to(coefficients[:, :, 0], ray.shape), ray) / np.sum(ray * ray, axis=2)
    for _ in range(14):
        point = (coefficients[:, :, 0] + coefficients[:, :, 1] * xi[:, :, None]
                 + coefficients[:, :, 2] * eta[:, :, None]
                 + coefficients[:, :, 3] * (xi * eta)[:, :, None])
        residual = point - lam[:, :, None] * ray
        first = coefficients[:, :, 1] + coefficients[:, :, 3] * eta[:, :, None]
        second = coefficients[:, :, 2] + coefficients[:, :, 3] * xi[:, :, None]
        jacobian = np.stack([first, second, -ray], axis=-1)
        determinant = np.linalg.det(jacobian)
        stable = ((np.abs(determinant) > 1e-25) & np.isfinite(determinant)
                  & (np.abs(xi) < 100) & (np.abs(eta) < 100))
        delta = np.zeros_like(residual)
        delta[stable] = np.linalg.solve(jacobian[stable], residual[stable][..., None])[..., 0]
        xi -= delta[:, :, 0]
        eta -= delta[:, :, 1]
        lam -= delta[:, :, 2]
    point = (coefficients[:, :, 0] + coefficients[:, :, 1] * xi[:, :, None]
             + coefficients[:, :, 2] * eta[:, :, None]
             + coefficients[:, :, 3] * (xi * eta)[:, :, None])
    residual = np.linalg.norm(point - lam[:, :, None] * ray, axis=2)
    first = coefficients[:, :, 1] + coefficients[:, :, 3] * eta[:, :, None]
    second = coefficients[:, :, 2] + coefficients[:, :, 3] * xi[:, :, None]
    outward = np.einsum("nmi,nmi->nm", np.cross(first, second), ray) > 0
    valid = ((np.abs(xi) <= 1 + 2e-10) & (np.abs(eta) <= 1 + 2e-10)
             & (lam > 0) & outward & (residual < 2e-13))
    if np.any(~np.any(valid, axis=1)):
        raise ArithmeticError("Exhaustive Newton oracle found no admissible chart.")
    macro = np.argmax(valid, axis=1)
    if preferred_macro is not None:
        if not np.all(valid[np.arange(nq), preferred_macro]):
            raise ArithmeticError("Closed-form chart is not independently admissible to Newton.")
        macro = preferred_macro
    row = np.arange(nq)
    return macro, np.clip(np.column_stack([xi[row, macro], eta[row, macro]]), -1, 1)


def oracle_load(mother, sources: list[Source], order: int = 8) -> np.ndarray:
    """Assemble independently located loads using the original physical force."""
    result = np.zeros((mother.n_dof, len(sources)))
    g = mother.geometry
    n = g.ncoarse * g.subdiv
    for i_source, source in enumerate(sources):
        force = force_for_source(mother, source)
        directions, weights = source_directions(source, order)
        macro, local = newton_oracle(g, directions)
        fine = np.clip(np.floor((local + 1) * g.subdiv / 2).astype(int), 0, g.subdiv - 1)
        face = macro // g.ncoarse**2
        macro_cell = macro % g.ncoarse**2
        ix = (macro_cell % g.ncoarse) * g.subdiv + fine[:, 0]
        iy = (macro_cell // g.ncoarse) * g.subdiv + fine[:, 1]
        element = face * n * n + iy * n + ix
        nodes = g.quads[element]
        sublocal = (local + 1) * g.subdiv - 2 * fine - 1
        x, y = sublocal.T
        shape = 0.25 * np.column_stack([(1 - x) * (1 - y), (1 + x) * (1 - y),
                                        (1 + x) * (1 + y), (1 - x) * (1 + y)])
        nodal = shape[:, :, None] * weights[:, None, None] * force
        couples = source.side * mother.model.h / 2 * np.cross(g.directors[nodes], nodal)
        for component in range(3):
            np.add.at(result[:, i_source], (6 * nodes + component).reshape(-1), nodal[:, :, component].reshape(-1))
            np.add.at(result[:, i_source], (6 * nodes + 3 + component).reshape(-1), couples[:, :, component].reshape(-1))
    return result


def material_queries(ncoarse: int, random_count: int) -> np.ndarray:
    rng = np.random.default_rng(7141 + ncoarse)
    queries = [normalize(rng.normal(size=(random_count, 3)))]
    grid = np.linspace(-1, 1, ncoarse + 1)
    for normal, first, second in FRAMES:
        for u in grid:
            for v in np.linspace(-1, 1, 9):
                base = normal + u * first + v * second
                for eps in (0.0, -1e-10, 1e-10, -1e-6, 1e-6):
                    queries.append(normalize((base + eps * first)[None]))
    return np.concatenate(queries)


def run(output: Path, random_count: int = 4000) -> dict:
    rows = []
    for ncoarse, subdiv in ((2, 3), (3, 4), (4, 2)):
        geometry = make_geometry(ncoarse, subdiv)
        directions = material_queries(ncoarse, random_count)
        canonical_macro_difference = 0
        max_local = 0.0
        max_position = 0.0
        max_shape = 0.0
        for start in range(0, len(directions), 64):
            current = directions[start:start + 64]
            fast = geometry.locate(current)
            macro, local = newton_oracle(geometry, current)
            canonical_macro_difference += np.count_nonzero(macro != fast.macro_ids)
            macro, local = newton_oracle(geometry, current, preferred_macro=fast.macro_ids)
            max_local = max(max_local, np.max(np.abs(local - fast.local)))
            x = geometry.macro_coefficients[macro]
            position = x[:, 0] + x[:, 1] * local[:, :1] + x[:, 2] * local[:, 1:] + x[:, 3] * np.prod(local, axis=1)[:, None]
            max_position = max(max_position, np.max(np.abs(position - fast.positions)))
            max_shape = max(max_shape, np.max(np.abs(np.sum(fast.weights, axis=1) - 1)))
        row = {"ncoarse": ncoarse, "subdiv": subdiv, "queries": len(directions),
               "canonical_macro_id_differences": int(canonical_macro_difference),
               "all_fast_charts_independently_admissible": True,
               "maximum_local_difference": float(max_local),
               "maximum_position_difference_m": float(max_position), "shape_partition_error": float(max_shape)}
        row["passed"] = bool(max_local < 2e-10 and max_position < 1e-12 and max_shape < 1e-13)
        rows.append(row)
    mother = Mother()
    sources = [Source(normalize(np.array([1.0, 0.13, -0.24])), 0.002, friction=0.3),
               Source(normalize(np.array([-0.3, 1.0, 0.42])), 0.003, friction=0.6),
               Source(normalize(np.array([1.0, 1.0, 0.3])), 0.001, force=np.array([-0.09, 0.02, 0.01]))]
    raw = assemble_sources(mother, sources, audit=True)
    reference = oracle_load(mother, sources)
    q = mother.solve(mother.balance(raw))
    qref = mother.solve(mother.balance(reference))
    sigma, reference_sigma = mother.S @ q, mother.S @ qref
    raw_error = np.linalg.norm(raw - reference, axis=0) / np.linalg.norm(reference, axis=0)
    field_error = np.linalg.norm(sigma - reference_sigma, axis=0) / np.linalg.norm(reference_sigma, axis=0)
    explicit = [replace(source, force=force_for_source(mother, source)) for source in sources]
    shortcut = assemble_sources(mother, explicit, audit=False)
    shortcut_error = np.max(np.abs(raw - shortcut))
    result = {"geometry_rows": rows, "source_load_relative_errors": raw_error.tolist(),
              "source_field_relative_errors": field_error.tolist(),
              "explicit_force_shortcut_max_absolute_difference": float(shortcut_error),
              "position_convention": mother.geometry.position_convention,
              "geometry_source_sha256": hashlib.sha256(Path(__file__).with_name("geometry.py").read_bytes()).hexdigest()}
    result["boundary_contract"] = (
        "All CF charts must be independently Newton-admissible. Same-chart local values use the same domain clip. "
        "Canonical chart IDs may differ when tolerance admits adjacent macros. Initial strict-ID failures are preserved."
    )
    result["passed"] = bool(all(row["passed"] for row in rows) and np.max(field_error) < 1e-10 and shortcut_error < 1e-15)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=4000)
    args = parser.parse_args()
    result = run(args.output, args.random_count)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
