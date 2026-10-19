"""Integrate affine P1/P2 contact-field differences exactly through shared material columns."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .contact_fem_convergence import material_faces


@dataclass
class ColumnField:
    stress: np.ndarray
    selected: np.ndarray
    weights: np.ndarray
    layers: int
    n_faces: int
    tetrahedra: np.ndarray
    n_surface_vertices: int
    sorted_faces: np.ndarray
    wall_depths: np.ndarray | None = None

    def __post_init__(self):
        if self.wall_depths is None:
            self.wall_depths = np.linspace(0, 1, self.layers + 1)
        assert self.wall_depths.shape == (self.layers + 1,)
        assert self.wall_depths[0] == 0 and self.wall_depths[-1] == 1 and np.all(np.diff(self.wall_depths) > 0)

    def breaks(self):
        boundaries = [np.zeros(len(self.selected))]
        for layer in range(self.layers):
            boundaries.extend(
                (
                    self.wall_depths[layer] + np.diff(self.wall_depths)[layer] * self.weights[:, 2],
                    self.wall_depths[layer] + np.diff(self.wall_depths)[layer] * self.weights[:, 1:].sum(axis=1),
                    np.full(len(self.selected), self.wall_depths[layer + 1]),
                )
            )
        result = np.array(boundaries)
        assert np.diff(result, axis=0).min() >= -1e-12
        return result

    def sample(self, depth, basis):
        layer = np.clip(np.searchsorted(self.wall_depths, depth, side="right") - 1, 0, self.layers - 1)
        local = (depth - self.wall_depths[layer]) / np.diff(self.wall_depths)[layer]
        kind = (local >= self.weights[:, 2]).astype(int) + (local >= self.weights[:, 1:].sum(axis=1))
        elements = (3 * layer + kind) * self.n_faces + self.selected
        tensor = self.stress[elements]
        if self.stress.ndim == 4:
            # Barycentric weights in the sorted-column prism, in its actual tetrahedron ordering.
            node = self.tetrahedra[elements]
            column = node % self.n_surface_vertices
            lower_rank = (column[:, :, None] == self.sorted_faces[self.selected, None, :]).argmax(axis=2)
            local_vertex = node // self.n_surface_vertices - layer[:, None]
            w0, w1, w2 = self.weights.T
            bary = np.zeros_like(node, dtype=float)
            for corner in range(4):
                rank, upper = lower_rank[:, corner], local_vertex[:, corner]
                bary[:, corner] = np.where(
                    kind == 0,
                    np.where(upper == 1, local, np.where(rank == 0, w0, np.where(rank == 1, w1, w2 - local))),
                    np.where(
                        kind == 1,
                        np.where(
                            rank == 0, w0, np.where(rank == 2, w2, np.where(upper == 1, local - w2, w1 + w2 - local))
                        ),
                        np.where(
                            upper == 0, 1 - local, np.where(rank == 0, local - w1 - w2, np.where(rank == 1, w1, w2))
                        ),
                    ),
                )
            np.testing.assert_allclose(bary.sum(axis=1), 1, atol=1e-10)
            assert bary.min() >= -1e-9
            tensor = np.einsum("qc,qcij->qij", bary, tensor)
        return basis.transpose(0, 2, 1) @ tensor @ basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--meshes", type=Path, nargs="+", required=True)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(args.meshes) == len(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.meshes[-1]) as saved:
        outer, inner, faces = (saved[key] for key in ("outer_vertices", "inner_vertices", "mid_faces"))
        material_unit = saved["material_unit"] if "material_unit" in saved else None
    unit = material_unit
    if unit is None:
        unit = outer.copy()
        unit[:, 2] /= 0.03
        unit[:, :2] /= (0.022 * (1 - 0.18 * unit[:, 2]))[:, None]
    polar_radius = np.linalg.norm(outer[faces].mean(axis=1)[:, :2], axis=1)
    direction = unit[faces].mean(axis=1)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    normal = (outer - inner)[faces].mean(axis=1)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    first = np.cross(normal, np.eye(3)[np.abs(normal).argmin(axis=1)])
    first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(normal, first)), axis=-1)
    triangle = ((outer + inner) / 2)[faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
    rows = []
    for load in args.loads:
        fields, raw = [], []
        for mesh, run in zip(args.meshes, args.runs):
            with np.load(mesh) as saved:
                vertices, tets, outer, faces = (
                    saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces")
                )
                material_unit = saved["material_unit"] if "material_unit" in saved else None
                wall_depths = saved["wall_depths"] if "wall_depths" in saved else None
            path = run / f"load-{load:g}.npz"
            with np.load(path) as saved:
                stress = saved["stress_pa"]
            selected, weights = material_faces(outer, faces, direction, material_unit)
            weights = np.take_along_axis(weights, np.argsort(faces[selected], axis=1), axis=1)
            fields.append(
                ColumnField(
                    stress,
                    selected,
                    weights,
                    len(vertices) // len(outer) - 1,
                    len(faces),
                    tets,
                    len(outer),
                    np.sort(faces, axis=1),
                    wall_depths,
                )
            )
            dev = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
            if stress.ndim == 3:
                peak_position = vertices[tets[mises.argmax()]].mean(axis=0)
            else:
                element, corner = np.unravel_index(mises.argmax(), mises.shape)
                peak_position = vertices[tets[element, corner]]
            raw.append(
                {
                    "mesh": file_id(mesh),
                    "field": file_id(path),
                    "raw_maximum_mises_pa": float(mises.max()),
                    "raw_maximum_tension_pa": float(np.linalg.eigvalsh(stress)[..., -1].max()),
                    "raw_peak_position_m": peak_position.tolist(),
                }
            )
        comparisons = []
        for index, (actual, reference) in enumerate(zip(fields[:-1], fields[1:])):
            breaks = np.sort(np.concatenate((actual.breaks(), reference.breaks())), axis=0)
            np.testing.assert_allclose(np.diff(breaks, axis=0).sum(axis=0), 1, rtol=0, atol=1e-13)
            membrane = np.zeros((2, len(direction), 2, 2))
            bending = np.zeros_like(membrane)
            error_by_column = np.zeros(len(direction))
            reference_by_column = np.zeros(len(direction))
            for lower, upper in zip(breaks[:-1], breaks[1:]):
                midpoint = (lower + upper) / 2
                width = upper - lower
                for offset in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                    depth = midpoint + offset * width / 2
                    a, r = actual.sample(depth, basis), reference.sample(depth, basis)
                    weight = width * area / 2
                    error_by_column += np.einsum("qij,qij,q->q", a - r, a - r, weight)
                    reference_by_column += np.einsum("qij,qij,q->q", r, r, weight)
                    values = np.stack((a, r))
                    membrane += values * width[None, :, None, None] / 2 * 0.0005
                    bending += values * (width / 2 * (0.5 - depth))[None, :, None, None] * 0.0005**2
            differences = []
            for values in (membrane, bending):
                difference = values[0] - values[1]
                differences.append(
                    float(
                        np.sqrt(
                            np.einsum("qij,qij,q->", difference, difference, area)
                            / np.einsum("qij,qij,q->", values[1], values[1], area)
                        )
                    )
                )
            error_square, reference_square = error_by_column.sum(), reference_by_column.sum()
            regions = []
            for lower, upper in ((0.0, 0.0005), (0.0005, 0.002), (0.002, float("inf"))):
                selected = (polar_radius >= lower) & (polar_radius < upper)
                regions.append(
                    {
                        "polar_radius_min_m": lower,
                        "polar_radius_max_m": upper if np.isfinite(upper) else None,
                        "area_fraction": float(area[selected].sum() / area.sum()),
                        "squared_error_fraction": float(error_by_column[selected].sum() / error_square),
                        "squared_reference_fraction": float(reference_by_column[selected].sum() / reference_square),
                    }
                )
            comparisons.append(
                {
                    "regions_diagnostic_only": regions,
                    "coarse_index": index,
                    "fine_index": index + 1,
                    "tangent_relative_l2": float(np.sqrt(error_square / reference_square)),
                    "membrane_relative_l2": differences[0],
                    "bending_relative_l2": differences[1],
                    "raw_mises_relative_change": abs(
                        raw[index]["raw_maximum_mises_pa"] / raw[index + 1]["raw_maximum_mises_pa"] - 1
                    ),
                    "raw_tension_relative_change": abs(
                        raw[index]["raw_maximum_tension_pa"] / raw[index + 1]["raw_maximum_tension_pa"] - 1
                    ),
                }
            )
            np.savez_compressed(
                args.output / f"columns-{load:g}-{index}.npz",
                direction=direction,
                error_square_pa2_m2=error_by_column,
                reference_square_pa2_m2=reference_by_column,
                area_m2=area,
                basis=basis,
                membrane_n_per_m=membrane,
                bending_n=bending,
            )
        rows.append({"load_n": load, "raw": raw, "adjacent_comparisons": comparisons})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "cases": rows,
            "physical_reference_qualified": False,
            "scope": "Exact integration of piecewise-constant P1 or piecewise-linear P2 stress along material columns, including unioned "
            "tetrahedron partitions for squared field differences and analytic membrane/bending moments. "
            "Angular material rays, geometry mapping and finest-face centroid area quadrature remain "
            "unchanged from the midpoint control. No polar trim, peak smoothing or selected depth samples.",
        },
    )


if __name__ == "__main__":
    main()
