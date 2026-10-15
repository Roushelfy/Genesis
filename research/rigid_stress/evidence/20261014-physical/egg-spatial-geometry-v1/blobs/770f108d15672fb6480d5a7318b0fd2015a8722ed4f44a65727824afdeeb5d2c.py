"""Re-layer an existing full egg shell without changing either polyhedral material boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .refine_fixed_geometry import geometric_integrals


def canonical_rows(values):
    values = np.sort(values, axis=1)
    return values[np.lexsort(values.T[::-1])]


def prism_tetrahedra(surface, count, layers):
    bottom = surface[None] + np.arange(layers)[:, None, None] * count
    top = bottom + count
    return np.stack(
        (
            np.stack((bottom[..., 0], bottom[..., 1], bottom[..., 2], top[..., 2]), axis=-1),
            np.stack((bottom[..., 0], bottom[..., 1], top[..., 1], top[..., 2]), axis=-1),
            np.stack((bottom[..., 0], top[..., 0], top[..., 1], top[..., 2]), axis=-1),
        ),
        axis=-2,
    ).reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=[3, 4], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".json").exists()
    metadata_path = args.seed.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["geometry"]["name"] == "egg" and metadata["shell_layers"] == 2
    assert hashlib.sha256(args.seed.read_bytes()).hexdigest() == metadata["mesh_sha256"]
    with np.load(args.seed) as saved:
        original_vertices, original_tets, original_boundary = [
            saved[key].copy() for key in ["vertices", "tetrahedra", "surface_triangles"]
        ]
    assert len(original_vertices) % 3 == 0
    count = len(original_vertices) // 3
    outer = canonical_rows(original_boundary[np.all(original_boundary < count, axis=1)])
    inner = canonical_rows(original_boundary[np.all(original_boundary >= 2 * count, axis=1)] - 2 * count)
    np.testing.assert_array_equal(outer, inner)
    assert 2 * len(outer) == len(original_boundary)
    np.testing.assert_array_equal(canonical_rows(prism_tetrahedra(outer, count, 2)), canonical_rows(original_tets))
    np.testing.assert_allclose(
        original_vertices[count : 2 * count],
        (original_vertices[:count] + original_vertices[2 * count :]) / 2,
        rtol=0,
        atol=1e-15,
    )
    fractions = np.linspace(0, 1, args.layers + 1)[:, None, None]
    vertices = (
        (1 - fractions) * original_vertices[None, :count] + fractions * original_vertices[None, 2 * count :]
    ).reshape(-1, 3)
    vertices[:count], vertices[-count:] = original_vertices[:count], original_vertices[-count:]
    tetrahedra = prism_tetrahedra(outer, count, args.layers)
    determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
    assert np.all(determinant != 0)
    flipped = determinant < 0
    tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
    local_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces, multiplicity = np.unique(np.sort(tetrahedra[:, local_faces].reshape(-1, 3)), axis=0, return_counts=True)
    assert multiplicity.max() == 2
    boundary = faces[multiplicity == 1]
    expected_boundary = canonical_rows(np.concatenate((outer, outer + args.layers * count)))
    np.testing.assert_array_equal(canonical_rows(boundary), expected_boundary)
    np.testing.assert_array_equal(vertices[:count], original_vertices[:count])
    np.testing.assert_array_equal(vertices[-count:], original_vertices[-count:])
    original_volume, original_area, original_com = geometric_integrals(
        original_vertices, original_tets, original_boundary
    )
    volume, area, com = geometric_integrals(vertices, tetrahedra, boundary)
    np.testing.assert_allclose(volume.sum(), original_volume.sum(), rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(area.sum(), original_area.sum(), rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(com, original_com, rtol=0, atol=1e-15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary)
    result = {
        "scope": __doc__,
        "geometry": metadata["geometry"],
        "seed": str(args.seed),
        "seed_sha256": metadata["mesh_sha256"],
        "seed_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "seed_shell_layers": 2,
        "shell_layers": args.layers,
        "surface_mesh_size": metadata["mesh_size"],
        "nodes": len(vertices),
        "tetrahedra": len(tetrahedra),
        "boundary_triangles": len(boundary),
        "outer_and_inner_boundary_vertices_byte_identical": True,
        "outer_and_inner_boundary_triangles_identical": True,
        "seed_prism_connectivity_independently_reconstructed": True,
        "volume_m3": float(volume.sum()),
        "seed_volume_m3": float(original_volume.sum()),
        "surface_area_m2": float(area.sum()),
        "seed_surface_area_m2": float(original_area.sum()),
        "com_m": com.tolist(),
        "seed_com_m": original_com.tolist(),
        "mass_kg_at_unchanged_density2000": float(volume.sum() * 2000),
        "minimum_tetrahedron_volume_m3": float(volume.min()),
        "goal_completion": False,
        "limitations": "Only the through-thickness volume discretization changes. The existing affine outer/inner "
        "surface geometry and its CAD approximation are fixed, not corrected. Physical traction "
        "profiles, material, density and reference quadrature must remain unchanged in subsequent "
        "solves. Mesh identity alone does not qualify stress or coupled contact convergence.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
