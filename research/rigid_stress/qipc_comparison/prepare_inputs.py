"""Refine the existing hollow egg's fixed piecewise-planar volume independently in surface and wall directions."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from ..mechanics import shell_mesh
from .artifacts import file_id, write_json


def subdivide(outer, inner, faces):
    count = len(outer)
    edges, inverse = np.unique(
        np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    outer = np.concatenate((outer, outer[edges].mean(axis=1)))
    inner = np.concatenate((inner, inner[edges].mean(axis=1)))
    ab, bc, ca = (count + inverse.reshape(-1, 3)).T
    a, b, c = faces.T
    faces = np.stack(
        (
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3)
    return outer, inner, faces


def make_mesh(outer, inner, faces, layers):
    vertices = np.concatenate([outer * (1 - depth) + inner * depth for depth in np.linspace(0, 1, layers + 1)])
    count = len(outer)
    sorted_faces = np.sort(faces, axis=1)
    tets = []
    for layer in range(layers):
        a, b, c = (sorted_faces + layer * count).T
        d, e, f = a + count, b + count, c + count
        tets.extend((np.stack((a, b, c, f), axis=1), np.stack((a, b, e, f), axis=1), np.stack((a, d, e, f), axis=1)))
    tets = np.concatenate(tets)
    signed = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])
    negative = signed < 0
    tets[negative, :2] = tets[negative, 1::-1]
    assert np.abs(signed).min() > 0
    oriented = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, indices, multiplicity = np.unique(np.sort(oriented, axis=1), axis=0, return_index=True, return_counts=True)
    assert (multiplicity <= 2).all()
    boundary = oriented[indices[multiplicity == 1]]
    return vertices, tets, boundary, np.abs(signed) / 6


def mass_properties(vertices, tets, volume):
    density = 2000.0
    mass = volume * density
    corners = vertices[tets]
    centers = corners.mean(axis=1)
    total = mass.sum()
    com = mass @ centers / total
    summed = corners.sum(axis=1)
    second = np.einsum("t,tvi,tvj->ij", mass / 20, corners, corners) + np.einsum(
        "t,ti,tj->ij", mass / 20, summed, summed
    )
    second -= total * np.outer(com, com)
    inertia = np.eye(3) * np.trace(second) - second
    lumped = np.bincount(tets.reshape(-1), weights=np.repeat(mass / 4, 4), minlength=len(vertices))
    relative = vertices - com
    lump_second = np.einsum("v,vi,vj->ij", lumped, relative, relative)
    lump_inertia = np.eye(3) * np.trace(lump_second) - lump_second
    return {
        "mass_kg": float(total),
        "com_m": com.tolist(),
        "exact_volume_inertia_kg_m2": inertia.tolist(),
        "lumped_inertia_kg_m2": lump_inertia.tolist(),
        "lumped_inertia_relative_difference": float(np.linalg.norm(lump_inertia - inertia) / np.linalg.norm(inertia)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["1:0:2", "1:1:2", "1:1:4", "1:2:4", "1:2:8", "2:0:2"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.conditions, args.output / "conditions.npz")
    records = []
    base_mass = {}
    for specification in args.cases:
        base, refinement, layers = (int(value) for value in specification.split(":"))
        original, _, original_faces, geometry = shell_mesh(base, 2, 0.0005)
        count = geometry.outer_surface_vertices
        outer, inner, faces = original[:count], original[2 * count :], original_faces
        for _ in range(refinement):
            outer, inner, faces = subdivide(outer, inner, faces)
        vertices, tets, boundary, volume = make_mesh(outer, inner, faces, layers)
        properties = mass_properties(vertices, tets, volume)
        if base in base_mass:
            np.testing.assert_allclose(properties["mass_kg"], base_mass[base], rtol=2e-13, atol=0)
        else:
            base_mass[base] = properties["mass_kg"]
        name = f"base{base}-surface{refinement}-wall{layers}"
        path = args.output / f"{name}.npz"
        np.savez_compressed(
            path,
            vertices=vertices,
            tetrahedra=tets,
            surface_triangles=boundary,
            mid_vertices=(outer + inner) / 2,
            mid_faces=faces,
            outer_vertices=outer,
            inner_vertices=inner,
        )
        records.append(
            {
                "name": name,
                "base_geometry_level": base,
                "surface_refinements": refinement,
                "wall_layers": layers,
                "vertices": len(vertices),
                "tetrahedra": len(tets),
                "minimum_tet_volume_m3": float(volume.min()),
                **properties,
                "input": file_id(path),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "fixed-domain-hollow-egg-refinement",
            "cases": records,
            "condition_bank": file_id(args.conditions),
            "geometry_contract": "Within a base geometry level, refinement subdivides the same planar outer/inner facets without rounding or changing wall thickness. Base-level changes are separate geometry cohorts, not numerical refinement.",
        },
    )


if __name__ == "__main__":
    main()
