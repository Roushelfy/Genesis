"""Build matched-connectivity CAD and fixed-polyhedron egg meshes without changing material or density."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .audit_egg_continuum_geometry import EggGeometry, continuum_integrals, meridian, mesh_integrals
from .audit_franka_reference import rotation
from .refine_egg_thickness import canonical_rows, prism_tetrahedra
from .summarize_fine_references import closed_source


def refine_surface(unit, faces):
    edges, inverse = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)), axis=0, return_inverse=True)
    midpoints = unit[edges].sum(axis=1)
    midpoints /= np.linalg.norm(midpoints, axis=1)[:, None]
    ij, jk, ki = (inverse.reshape(-1, 3) + len(unit)).T
    i, j, k = faces.T
    triangles = np.stack(
        (
            np.stack((i, ij, ki), axis=1),
            np.stack((j, jk, ij), axis=1),
            np.stack((k, ki, jk), axis=1),
            np.stack((ij, jk, ki), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3)
    return np.concatenate((unit, midpoints)), triangles, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--maximum-level", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.reference_run, args.capture]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    geometry = EggGeometry()
    continuum = continuum_integrals(geometry, 256, 4)
    reference = args.reference_run / "results/reference_geometry.npz"
    capture = args.capture / "results/summary.json"
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.seed, reference, capture]}
    with np.load(args.seed) as saved:
        seed, seed_tets, faces = [saved[key].copy() for key in ["vertices", "tetrahedra", "surface_triangles"]]
    count = len(seed) // 3
    assert seed.shape == (3 * count, 3) and faces.max() < count
    np.testing.assert_array_equal(
        canonical_rows(prism_tetrahedra(np.sort(faces, axis=1), count, 2)), canonical_rows(seed_tets)
    )
    parameter = seed[:count, 2] / geometry.polar_radius_m
    scale = 1 - geometry.taper * parameter
    unit = np.column_stack((seed[:count, :2] / (geometry.equatorial_radius_m * scale[:, None]), parameter))
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=0, atol=1e-14)
    radius, height, radial_normal, vertical_normal, _, _ = meridian(geometry, parameter, 0)
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    expected_outer = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height))
    normal = np.column_stack((radial_normal * np.cos(phi), radial_normal * np.sin(phi), vertical_normal))
    for layer in range(3):
        np.testing.assert_allclose(
            seed[layer * count : (layer + 1) * count],
            expected_outer - layer * geometry.thickness_m / 2 * normal,
            rtol=0,
            atol=1e-14,
        )
    frame = [row for row in json.loads(capture.read_text())["captures"] if row["mode"] == "stationary_lift_hold"][-1]
    transform = rotation(frame["egg_quaternion_wxyz"])
    shift = np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
    with np.load(reference) as saved:
        fixed = (saved["fixed_initializer_points_m"] - shift) @ transform
        fixed_components = saved["fixed_initializer_components"].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    meshes = {"polyhedron": seed.reshape(3, count, 3).copy(), "CAD": seed.reshape(3, count, 3).copy()}
    rows = []
    seed_integrals = mesh_integrals(seed, seed_tets, geometry, continuum)
    for level in range(1, args.maximum_level + 1):
        if level > 1:
            old_count = len(unit)
            unit, faces, edges = refine_surface(unit, faces)
            parameter, phi = unit[old_count:, 2], np.arctan2(unit[old_count:, 1], unit[old_count:, 0])
            new_layers = []
            for depth in np.linspace(0, geometry.thickness_m, 3):
                radius, height, _, _, _, _ = meridian(geometry, parameter, depth)
                new_layers.append(np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height)))
            old_poly, old_cad = meshes["polyhedron"], meshes["CAD"]
            meshes["polyhedron"] = np.concatenate((old_poly, old_poly[:, edges].mean(axis=2)), axis=1)
            meshes["CAD"] = np.concatenate((old_cad, np.stack(new_layers)), axis=1)
            np.testing.assert_array_equal(meshes["polyhedron"][:, :old_count], old_poly)
            np.testing.assert_array_equal(meshes["CAD"][:, :old_count], old_cad)
        tetrahedra = prism_tetrahedra(np.sort(faces, axis=1), len(unit), 2)
        for mode, layered in meshes.items():
            vertices = layered.reshape(-1, 3)
            path = args.output / f"{mode.lower()}-level{level}.npz"
            np.savez_compressed(path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
            integral = mesh_integrals(vertices, tetrahedra, geometry, continuum)
            if mode == "polyhedron":
                for key in ["volume_m3", "outer_area_m2", "inner_area_m2", "inertia_about_com_kg_m2"]:
                    np.testing.assert_allclose(integral[key], seed_integrals[key], rtol=1e-11, atol=1e-18)
                np.testing.assert_allclose(integral["com_m"], seed_integrals["com_m"], rtol=0, atol=1e-14)
            tet_edges = np.unique(
                np.sort(tetrahedra[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)), axis=0
            )
            p2_nodes = np.concatenate((vertices, vertices[tet_edges].mean(axis=1)))
            distance, _ = cKDTree(p2_nodes).query(fixed)
            rows.append(
                {
                    "mode": mode,
                    "level": level,
                    "mesh": str(path),
                    "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **integral,
                    "original_initializer_nearest_P2_node_distances_m": distance.tolist(),
                    "original_initializer_all_anchors_retained": bool(np.all(distance < 1e-12)),
                }
            )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "original_initializer_points_in_author_frame_m": fixed.tolist(),
        "original_initializer_components_in_world_frame": fixed_components.tolist(),
        "both_branches_keep_identical_connectivity_per_level": True,
        "old_vertices_retained_exactly_within_each_branch": True,
        "goal_completion": False,
        "limitations": "These are separate declared geometry-model families at fixed density and material. CAD refinement changes represented volume/mass; the polyhedron branch preserves the original boundaries and integrals. Connectivity matching does not isolate every contact or FEM discretization error. Initializer diagnostics compare physical P2 node positions only; they do not authorize changing supports or actuation. Actual reference-contact solves must preserve compatible physical anchors or explicitly qualify a common initializer. No stress, contact-history, physical convergence or warmed-cost acceptance follows from these geometry assets.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"meshes": len(rows), "mass_errors": [row["relative_volume_and_mass_error_from_continuum"] for row in rows]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
