"""Integrate affine P1/P2 contact-field differences exactly through shared material columns."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .contact_fem_convergence import material_faces
from .common_metric import corotation


def surface_quadrature(count):
    """Return barycentric points and normalized physical-area weights on a triangle."""
    if count == 1:
        return np.full((1, 3), 1 / 3), np.ones(1)
    if count == 3:
        return np.eye(3) / 2 + 1 / 6, np.full(3, 1 / 3)
    order = int(np.sqrt(count))
    assert order**2 == count
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    area = u * weights[:, None] * weights[None, :] / 2
    return np.stack((1 - u, u * (1 - v), u * v), axis=-1).reshape(-1, 3), area.reshape(-1)


@dataclass
class ShellColumnField:
    stress: np.ndarray

    def breaks(self):
        return np.stack((np.zeros(len(self.stress)), np.ones(len(self.stress))))

    def sample_tensor(self, depth):
        return (1 - depth[:, None, None]) * self.stress[:, 2] + depth[:, None, None] * self.stress[:, 0]

    def sample(self, depth, basis):
        return basis.transpose(0, 2, 1) @ self.sample_tensor(depth) @ basis


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
    volumes: np.ndarray | None = None
    stvk_gradient: np.ndarray | None = None
    material_rotation: np.ndarray | None = None
    young: float = 1e10
    poisson: float = 0.3

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

    def locate(self, depth):
        layer = np.clip(np.searchsorted(self.wall_depths, depth, side="right") - 1, 0, self.layers - 1)
        local = (depth - self.wall_depths[layer]) / np.diff(self.wall_depths)[layer]
        kind = (local >= self.weights[:, 2]).astype(int) + (local >= self.weights[:, 1:].sum(axis=1))
        elements = (3 * layer + kind) * self.n_faces + self.selected
        return layer, local, kind, elements

    def volume_density(self, depth):
        """Physical volume per normalized triangle-area weight and wall-depth increment."""
        assert self.volumes is not None
        layer, _, _, elements = self.locate(depth)
        return 3 * self.volumes[elements] / np.diff(self.wall_depths)[layer]

    def sample_tensor(self, depth):
        layer, local, kind, elements = self.locate(depth)
        tensor = self.stress[elements] if self.stvk_gradient is None else self.stvk_gradient[elements]
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
        if self.stvk_gradient is not None:
            from .p2_stvk import cauchy_from_gradient

            tensor = cauchy_from_gradient(tensor, self.young, self.poisson)
            assert self.material_rotation is not None
            tensor = self.material_rotation.T @ tensor @ self.material_rotation
        return tensor

    def sample(self, depth, basis):
        return basis.transpose(0, 2, 1) @ self.sample_tensor(depth) @ basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--meshes", type=Path, nargs="+", required=True)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1])
    cli.add_argument("--shell-indices", type=int, nargs="*", default=[])
    cli.add_argument("--surface-quadrature", type=int, choices=(1, 3, 9, 25), default=1)
    cli.add_argument("--physical-volume", action="store_true")
    cli.add_argument(
        "--export-surface-fields", action="store_true", help="Retain unsmoothed common outer/inner tensors for maps"
    )
    cli.add_argument("--depth-quadrature", type=int, choices=(2, 4, 8), default=2)
    cli.add_argument(
        "--common-peaks", action="store_true", help="Also evaluate shared tangent peaks and 3D-only components"
    )
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    depth_nodes, depth_weights = np.polynomial.legendre.leggauss(args.depth_quadrature)
    assert len(args.meshes) == len(args.runs)
    assert all(0 <= index < len(args.runs) for index in args.shell_indices)
    assert all(not run.is_file() or len(args.loads) == 1 for run in args.runs)
    assert not args.physical_volume or len(args.runs) - 1 not in args.shell_indices
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.meshes[-1]) as saved:
        outer, inner, faces = (saved[key] for key in ("outer_vertices", "inner_vertices", "mid_faces"))
        material_unit = saved["material_unit"] if "material_unit" in saved else None
    unit = material_unit
    if unit is None:
        unit = outer.copy()
        unit[:, 2] /= 0.03
        unit[:, :2] /= (0.022 * (1 - 0.18 * unit[:, 2]))[:, None]
    bary, angular_weights = surface_quadrature(args.surface_quadrature)
    common_outer = np.einsum("qc,fci->fqi", bary, outer[faces]).reshape(-1, 3)
    common_inner = np.einsum("qc,fci->fqi", bary, inner[faces]).reshape(-1, 3)
    polar_radius = np.linalg.norm(np.einsum("qc,fci->fqi", bary, outer[faces])[..., :2], axis=-1).reshape(-1)
    direction = np.einsum("qc,fci->fqi", bary, unit[faces]).reshape(-1, 3)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    normal = np.einsum("qc,fci->fqi", bary, (outer - inner)[faces]).reshape(-1, 3)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    first = np.cross(normal, np.eye(3)[np.abs(normal).argmin(axis=1)])
    first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(normal, first)), axis=-1)
    triangle = ((outer + inner) / 2)[faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
    area = (area[:, None] * angular_weights).reshape(-1)
    column_angular_weights = np.tile(angular_weights, len(faces))
    rows = []
    for load in args.loads:
        fields, raw = [], []
        for index, (mesh, run) in enumerate(zip(args.meshes, args.runs)):
            with np.load(mesh) as saved:
                vertices, tets, outer, faces = (
                    saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces")
                )
                material_unit = saved["material_unit"] if "material_unit" in saved else None
                wall_depths = saved["wall_depths"] if "wall_depths" in saved else None
                midpoint = saved["mid_vertices"] if index in args.shell_indices else None
            path = run if run.is_file() else run / f"load-{load:g}.npz"
            selected, weights = material_faces(outer, faces, direction, material_unit)
            weights = np.take_along_axis(weights, np.argsort(faces[selected], axis=1), axis=1)
            if index in args.shell_indices:
                with np.load(path.parent.parent / "rest.npz") as saved:
                    rest, rest_faces, masses = saved["vertices"], saved["faces"], saved["masses_kg"]
                np.testing.assert_allclose(rest, midpoint, atol=1e-15, rtol=0)
                np.testing.assert_array_equal(rest_faces, faces)
                with np.load(path) as saved:
                    rotation = corotation(rest, saved["vertices_world_m"], masses)
                    stress = rotation.T @ saved["cauchy_world_pa"] @ rotation
                np.testing.assert_allclose(stress[:, 1], (stress[:, 0] + stress[:, 2]) / 2, rtol=1e-10, atol=1e-8)
                fields.append(ShellColumnField(stress[selected]))
            else:
                with np.load(path) as saved:
                    stress = saved["stress_pa"]
                    constitutive = {}
                    if "stvk_gradient" in saved:
                        constitutive = {
                            "stvk_gradient": saved["stvk_gradient"],
                            "material_rotation": saved["rotation_material_to_world"],
                            "young": float(saved["young_pa"]),
                            "poisson": float(saved["poisson"]),
                        }
                volumes = None
                if args.physical_volume and index == len(args.runs) - 1:
                    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
                    assert volumes.min() > 0
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
                        volumes,
                        **constitutive,
                    )
                )
            dev = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
            if index in args.shell_indices:
                face, fiber = np.unravel_index(mises.argmax(), mises.shape)
                tri = midpoint[faces[face]]
                normal_raw = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                normal_raw /= np.linalg.norm(normal_raw)
                peak_position = tri.mean(axis=0) + (fiber - 1) * 0.00025 * normal_raw
            elif stress.ndim == 3:
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
            if args.common_peaks:
                from .common_column_peaks import CommonColumnMeasures

                common = CommonColumnMeasures(
                    basis,
                    normal,
                    common_outer,
                    common_inner,
                    (isinstance(actual, ShellColumnField), isinstance(reference, ShellColumnField)),
                )
            partitions = [actual.breaks(), reference.breaks()]
            if args.physical_volume:
                partitions.append(fields[-1].breaks())
            breaks = np.sort(np.concatenate(partitions), axis=0)
            np.testing.assert_allclose(np.diff(breaks, axis=0).sum(axis=0), 1, rtol=0, atol=1e-13)
            membrane = np.zeros((2, len(direction), 2, 2))
            bending = np.zeros_like(membrane)
            error_by_column = np.zeros(len(direction))
            reference_by_column = np.zeros(len(direction))
            integration_measure = 0.0
            for lower, upper in zip(breaks[:-1], breaks[1:]):
                midpoint = (lower + upper) / 2
                width = upper - lower
                if args.common_peaks:
                    # Both one-sided traces of every unioned tetrahedron boundary.
                    for depth in (np.nextafter(lower, upper), np.nextafter(upper, lower)):
                        common.update_peaks([field.sample(depth, basis) for field in (actual, reference)], depth)
                for offset, depth_weight in zip(depth_nodes, depth_weights):
                    depth = midpoint + offset * width / 2
                    if args.common_peaks:
                        tensors = [field.sample_tensor(depth) for field in (actual, reference)]
                        a, r = [basis.transpose(0, 2, 1) @ tensor @ basis for tensor in tensors]
                    else:
                        a, r = actual.sample(depth, basis), reference.sample(depth, basis)
                    weight = width * area / 2 * depth_weight
                    if args.physical_volume:
                        weight = width / 2 * column_angular_weights * fields[-1].volume_density(depth) * depth_weight
                    integration_measure += weight.sum()
                    if args.common_peaks:
                        common.update_peaks((a, r), depth)
                        common.integrate_transverse(tensors, weight)
                    error_by_column += np.einsum("qij,qij,q->q", a - r, a - r, weight)
                    reference_by_column += np.einsum("qij,qij,q->q", r, r, weight)
                    values = np.stack((a, r))
                    membrane += values * width[None, :, None, None] / 2 * 0.0005 * depth_weight
                    bending += values * (width / 2 * (0.5 - depth))[None, :, None, None] * 0.0005**2 * depth_weight
            if args.physical_volume:
                np.testing.assert_allclose(integration_measure, fields[-1].volumes.sum(), rtol=1e-9, atol=1e-18)
            differences, resultant_rms_errors = [], []
            for values in (membrane, bending):
                difference = values[0] - values[1]
                resultant_rms_errors.append(
                    float(np.sqrt(np.einsum("qij,qij,q->", difference, difference, area) / area.sum()))
                )
                differences.append(
                    float(
                        np.sqrt(
                            np.einsum("qij,qij,q->", difference, difference, area)
                            / np.einsum("qij,qij,q->", values[1], values[1], area)
                        )
                    )
                )
            error_square, reference_square = error_by_column.sum(), reference_by_column.sum()
            surface_errors = {}
            surface_fields = {}
            for name, depth in (("outer", 0.0), ("inner", 1.0)):
                a, r = (field.sample(np.full(len(direction), depth), basis) for field in (actual, reference))
                if args.export_surface_fields:
                    surface_fields[name + "_tangent_pa"] = np.stack((a, r))
                if args.common_peaks:
                    common.update_peaks((a, r), np.full(len(direction), depth))
                numerator = np.einsum("qij,qij,q->", a - r, a - r, area)
                denominator = np.einsum("qij,qij,q->", r, r, area)
                surface_errors[name + "_tangent_relative_l2"] = float(np.sqrt(numerator / denominator))
                surface_errors[name + "_tangent_rms_error_pa"] = float(np.sqrt(numerator / area.sum()))
            if args.export_surface_fields:
                np.savez_compressed(
                    args.output / f"surface-fields-{load:g}-{index}.npz",
                    outer_rest_position_m=common_outer,
                    inner_rest_position_m=common_inner,
                    direction=direction,
                    basis=basis,
                    **surface_fields,
                )
            regions = []
            for lower, upper in ((0.0, 0.0005), (0.0005, 0.002), (0.002, float("inf"))):
                selected = (polar_radius >= lower) & (polar_radius < upper)
                regions.append(
                    {
                        "polar_radius_min_m": lower,
                        "polar_radius_max_m": upper if np.isfinite(upper) else None,
                        "area_fraction": float(area[selected].sum() / area.sum()),
                        "squared_error_fraction": float(error_by_column[selected].sum() / error_square)
                        if error_square
                        else None,
                        "squared_reference_fraction": float(reference_by_column[selected].sum() / reference_square),
                    }
                )
            comparisons.append(
                {
                    "regions_diagnostic_only": regions,
                    "coarse_index": index,
                    "fine_index": index + 1,
                    "tangent_relative_l2": float(np.sqrt(error_square / reference_square)),
                    "tangent_rms_error_pa": float(np.sqrt(error_square / integration_measure)),
                    "integration_measure": float(integration_measure),
                    "integration_measure_unit": "m3" if args.physical_volume else "m2",
                    **surface_errors,
                    **({"common_spatial_measures": common.result()} if args.common_peaks else {}),
                    "membrane_relative_l2": differences[0],
                    "bending_relative_l2": differences[1],
                    "membrane_rms_error_n_per_m": resultant_rms_errors[0],
                    "bending_rms_error_n": resultant_rms_errors[1],
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
                **{
                    f"error_square_pa2_m{3 if args.physical_volume else 2}": error_by_column,
                    f"reference_square_pa2_m{3 if args.physical_volume else 2}": reference_by_column,
                },
                area_m2=area,
                basis=basis,
                membrane_n_per_m=membrane,
                bending_n=bending,
            )
        rows.append({"load_n": load, "raw": raw, "adjacent_comparisons": comparisons})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "cases": rows,
            "shell_indices": args.shell_indices,
            "surface_quadrature_points_per_face": args.surface_quadrature,
            "physical_volume_weighting": args.physical_volume,
            "depth_quadrature_points": args.depth_quadrature,
            "common_peaks_evaluated": args.common_peaks,
            "physical_reference_qualified": False,
            "scope": "Exact integration of piecewise-constant P1 or piecewise-linear P2 stress along material columns. StVK uses the exact P2 displacement gradient at each sample and constitutive Cauchy evaluation, with explicit depth-quadrature sensitivity required. Integration includes unioned "
            "tetrahedron partitions for squared field differences and analytic membrane/bending moments. "
            "Shell uses its complete linear fiber field after rigid corotation in the same material basis. "
            "Optional physical volume weights use exact reference-tetrahedron Jacobians, including every reference partition. "
            "Membrane and bending retain the declared 500um normal-material thickness and physical mid-surface area weights. "
            "Angular material rays and the declared finest-face area quadrature retain surface sampling uncertainty. "
            "All faces, raw peaks and both boundary fibers are retained. This field comparison does not establish mechanical admission.",
        },
    )


if __name__ == "__main__":
    main()
