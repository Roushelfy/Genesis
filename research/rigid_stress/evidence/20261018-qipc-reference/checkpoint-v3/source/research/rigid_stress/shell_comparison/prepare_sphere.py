"""A compact curved analytic control using the existing 3D Lame reference."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from skfem import FacetBasis, asm

from ..skfem_reference import SkfemElasticReference
from ..spherical_shell_validation import analytic_shell, pressure_load
from .common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    radius, thickness, pressure = 0.022, 0.0005, 100.0
    inner, outer = radius - thickness / 2, radius + thickness / 2
    rows = []
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    for level in (1, 2):
        sphere = trimesh.creation.icosphere(subdivisions=level)
        count = len(sphere.vertices)
        vertices = np.concatenate([sphere.vertices * r for r in (outer, radius, inner)])
        tets = []
        for layer in range(2):
            for i, j, k in np.sort(sphere.faces, axis=1):
                a, b, c = np.array([i, j, k]) + layer * count
                d, e, f = np.array([i, j, k]) + (layer + 1) * count
                tets.extend(((a, b, c, f), (a, b, e, f), (a, d, e, f)))
        tets = np.array(tets)
        boundary, counts = np.unique(
            np.sort(tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1),
            axis=0,
            return_counts=True,
        )
        midpoint = radius * sphere.vertices
        np.savez_compressed(
            args.output / f"sphere-level{level}.npz",
            vertices=vertices,
            tetrahedra=tets,
            surface_triangles=boundary[counts == 1],
            mid_vertices=midpoint,
            mid_faces=sphere.faces,
        )
        with (args.output / f"sphere-mid-level{level}.obj").open("w") as stream:
            np.savetxt(stream, midpoint, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, sphere.faces + 1, fmt="f %d %d %d")
        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000.0)
        facets = reference.mesh.boundary_facets()
        inner_facets = facets[(reference.mesh.facets[:, facets] >= 2 * count).all(axis=0)]
        basis = FacetBasis(reference.mesh, reference.element, facets=inner_facets, intorder=4)
        force = asm(pressure_load, basis, pressure=pressure)
        solution = reference.solve(force)
        assert solution.residual_budget_fraction <= 1
        triangles = vertices[2 * count :][sphere.faces]
        area_normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
        shell_force = np.zeros_like(midpoint)
        for corner in range(3):
            np.add.at(shell_force, sphere.faces[:, corner], pressure * area_normal / 3)
        edge_vertices = np.unique(np.sort(tets[:, edges].reshape(-1, 2), axis=1), axis=0)
        coordinates = np.concatenate((vertices, vertices[edge_vertices].mean(axis=1)))
        permutation = reference.native_dof_permutation(coordinates)
        material_directions = midpoint[sphere.faces].mean(axis=1)
        material_directions /= np.linalg.norm(material_directions, axis=1)[:, None]
        mapped_material_fibers = material_directions[:, None] * np.array([inner, radius, outer])[None, :, None]
        _, target, _ = analytic_shell(mapped_material_fibers.reshape(-1, 3), inner, outer, pressure, 1e10, 0.3)
        _, corner_target, _ = analytic_shell(vertices[tets].reshape(-1, 3), inner, outer, pressure, 1e10, 0.3)
        corner_target = corner_target.reshape(solution.tensor_pa.shape)
        np.savez_compressed(
            args.output / f"level{level}.npz",
            shell_force_n=shell_force,
            native_coordinates_m=coordinates,
            native_force_n=force[permutation].reshape(-1, 3),
            reference_tensor_pa=target.reshape(len(sphere.faces), 3, 3, 3),
            independent_corner_tensor_pa=solution.tensor_pa,
            analytic_corner_tensor_pa=corner_target,
            thin_tangent_sigma_pa=pressure * inner**2 / (2 * radius * thickness),
        )
        rows.append(
            {
                "level": level,
                "radius_m": radius,
                "thickness_m": thickness,
                "inner_pressure_pa": pressure,
                "thin_tangent_sigma_pa": pressure * inner**2 / (2 * radius * thickness),
                "independent_corner_lame_relative_error": float(
                    np.linalg.norm(solution.tensor_pa - corner_target) / np.linalg.norm(corner_target)
                ),
                "independent_residual_budget_fraction": float(solution.residual_budget_fraction),
                "point_convention": "Volume analytic tensors at physical tet corners; shell fiber target uses radial material-coordinate mapping, exposing faceting rather than pretending flat facets coincide with a sphere.",
            }
        )
    write_json(args.output / "result.json", {"kind": "analytic-sphere-inputs", "cases": rows})


if __name__ == "__main__":
    main()
