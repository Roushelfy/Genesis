"""Refine conforming paired egg surfaces near fixed CAD contact anchors with unchanged shell parameters."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from skfem import MeshTri

from .audit_egg_continuum_geometry import EggGeometry, continuum_integrals, meridian, mesh_integrals
from .audit_franka_reference import rotation
from .refine_egg_thickness import prism_tetrahedra
from .summarize_fine_references import closed_source


def support_direction(geometry, axis):
    radial = np.linalg.norm(axis[:2])
    phi = np.arctan2(axis[1], axis[0])
    optimum = minimize_scalar(
        lambda s, radial=radial, axis_z=axis[2]: (
            -(
                radial * geometry.equatorial_radius_m * (1 - geometry.taper * s) * np.sqrt(1 - s * s)
                + axis_z * geometry.polar_radius_m * s
            )
        ),
        bounds=(-1, 1),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert optimum.success
    s = optimum.x
    direction = np.array([np.sqrt(1 - s * s) * np.cos(phi), np.sqrt(1 - s * s) * np.sin(phi), s])
    radius, height, _, _, _, _ = meridian(geometry, s, 0)
    return direction, np.array([radius * np.cos(phi), radius * np.sin(phi), height])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--seed-level", type=int, choices=[2, 3], default=3)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--observation-run", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--target-edge-lengths", type=float, nargs=4, default=[1e-4, 5e-5, 2.5e-5, 1.25e-5])
    parser.add_argument("--contact-audit", type=Path, nargs="+", default=[])
    parser.add_argument("--marking-radius", type=float, default=5e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.marking_radius < 0.01
    assert sorted(set(args.target_edge_lengths), reverse=True) == args.target_edge_lengths
    assert args.target_edge_lengths[-1] > 0
    for run in [args.geometry_run, args.reference_run, args.observation_run, args.capture]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    geometry = EggGeometry()
    continuum = continuum_integrals(geometry, 256, 4)
    paths = [args.geometry_run / f"results/{mode}-level{args.seed_level}.npz" for mode in ["cad", "polyhedron"]]
    paths.extend(
        [
            args.reference_run / "results/reference_geometry.npz",
            args.observation_run / "results/summary.json",
            args.capture / "results/summary.json",
        ]
    )
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with np.load(paths[0]) as saved:
        vertices, faces = saved["vertices"].copy(), saved["surface_triangles"].copy()
    count = len(vertices) // 3
    parameter = vertices[:count, 2] / geometry.polar_radius_m
    unit = np.column_stack(
        (vertices[:count, :2] / (geometry.equatorial_radius_m * (1 - geometry.taper * parameter[:, None])), parameter)
    )
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=0, atol=1e-14)
    meshes = {"CAD": vertices.reshape(3, count, 3)}
    with np.load(paths[1]) as saved:
        meshes["polyhedron"] = saved["vertices"].reshape(3, count, 3).copy()
        np.testing.assert_array_equal(saved["surface_triangles"], faces)
        seed_poly_tets = saved["tetrahedra"].copy()
    seed_poly_integrals = mesh_integrals(meshes["polyhedron"].reshape(-1, 3), seed_poly_tets, geometry, continuum)
    frame = [row for row in json.loads(paths[-1].read_text())["captures"] if row["mode"] == "stationary_lift_hold"][-1]
    matrix = rotation(frame["egg_quaternion_wxyz"])
    shift = np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
    with np.load(paths[2]) as saved:
        fixed = (saved["fixed_initializer_points_m"] - shift) @ matrix
    anchors = []
    support_positions = []
    for finger in frame["fingers"]:
        axis = matrix.T @ rotation(finger["world_quaternion_wxyz"]) @ np.array([0, 1, 0])
        direction, position = support_direction(geometry, axis)
        anchors.append(direction)
        support_positions.append(position.tolist())
    observations = json.loads(paths[3].read_text())["actual_state_observations"]
    for observed in observations:
        if Path(observed["state"]).name != "closure_state.npz":
            continue
        source = Path(observed["source_run"])
        closed_source(source)
        contact_path = source / "results/contact_geometry.npz"
        inputs[str(contact_path)] = hashlib.sha256(contact_path.read_bytes()).hexdigest()
        with np.load(contact_path) as saved:
            rest, groups = saved["rest_positions"], saved["groups"]
            for row in observed["stencils"]:
                ids = np.array(row["vertex_ids"])
                point = (rest[ids[groups[ids] == 0]].mean(axis=0) - shift) @ matrix
                s = point[2] / geometry.polar_radius_m
                direction = np.r_[point[:2] / (geometry.equatorial_radius_m * (1 - geometry.taper * s)), s]
                anchors.append(direction / np.linalg.norm(direction))
    for audit_run in args.contact_audit:
        closed_source(audit_run)
        assert int((audit_run / "returncode.txt").read_text()) == 0
        report_path = audit_run / "results/summary.json"
        report = json.loads(report_path.read_text())
        source = Path(report["run"])
        closed_source(source)
        assert int((source / "returncode.txt").read_text()) == 0
        inputs[str(report_path)] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        contact_path = source / "results/contact_geometry.npz"
        inputs[str(contact_path)] = hashlib.sha256(contact_path.read_bytes()).hexdigest()
        with np.load(contact_path) as saved:
            rest = saved["rest_positions"].copy()
        for row in report["rows"]:
            assert row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
            field_path = audit_run / f"results/step{row['step']:04d}.npz"
            inputs[str(field_path)] = hashlib.sha256(field_path.read_bytes()).hexdigest()
            with np.load(field_path) as saved:
                active = saved["active_body_contact_nodes"].copy()
                forces = saved["normal_surface_force_N"][active].copy()
            points = (rest[active] - shift) @ matrix
            for point, force in zip(points, forces, strict=True):
                parameter = point[2] / geometry.polar_radius_m
                direction = np.r_[
                    point[:2] / (geometry.equatorial_radius_m * (1 - geometry.taper * parameter)), parameter
                ]
                anchors.append(direction / np.linalg.norm(direction))
                axis = -force @ matrix
                assert np.linalg.norm(axis) > 0
                axis /= np.linalg.norm(axis)
                direction, position = support_direction(geometry, axis)
                anchors.append(direction)
                support_positions.append(position.tolist())
    anchors = np.array(anchors)
    cap_angle = args.marking_radius / min(geometry.equatorial_radius_m * (1 - geometry.taper), geometry.polar_radius_m)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, refinement = [], []
    for level, target in enumerate(args.target_edge_lengths, 1):
        for _iteration in range(64):
            corners = unit[faces]
            centres = corners.mean(axis=1)
            centres /= np.linalg.norm(centres, axis=1)[:, None]
            radius = np.arccos(np.clip(np.einsum("fvi,fi->fv", corners, centres), -1, 1)).max(axis=1)
            in_cap = np.any(centres @ anchors.T >= np.cos(radius[:, None] + cap_angle), axis=1)
            physical = meshes["CAD"][0, faces]
            edge_lengths = np.linalg.norm(physical[:, [1, 2, 0]] - physical, axis=2).max(axis=1)
            marked = np.flatnonzero(in_cap & (edge_lengths > target))
            if not len(marked):
                maximum_local_edge = float(edge_lengths[in_cap].max())
                assert maximum_local_edge <= target
                break
            mesh = MeshTri(unit.T, faces.T, sort_t=False)
            edges = mesh.facets.T
            parent_midpoints = unit[edges].mean(axis=1)
            refined = mesh.refined(marked)
            new_raw = refined.p.T[len(unit) :]
            distance, parents = cKDTree(parent_midpoints).query(new_raw)
            assert len(new_raw) > 0 and distance.max() <= 1e-14
            np.testing.assert_array_equal(refined.p.T[: len(unit)], unit)
            new_unit = new_raw / np.linalg.norm(new_raw, axis=1)[:, None]
            old_count = len(unit)
            unit = np.concatenate((unit, new_unit))
            faces = refined.t.T
            new_layers = []
            phi = np.arctan2(new_unit[:, 1], new_unit[:, 0])
            for depth in np.linspace(0, geometry.thickness_m, 3):
                radius, height, _, _, _, _ = meridian(geometry, new_unit[:, 2], depth)
                new_layers.append(np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height)))
            old_poly, old_cad = meshes["polyhedron"], meshes["CAD"]
            meshes["polyhedron"] = np.concatenate((old_poly, old_poly[:, edges[parents]].mean(axis=2)), axis=1)
            meshes["CAD"] = np.concatenate((old_cad, np.stack(new_layers)), axis=1)
            np.testing.assert_array_equal(meshes["polyhedron"][:, :old_count], old_poly)
            np.testing.assert_array_equal(meshes["CAD"][:, :old_count], old_cad)
            refinement.append(
                {
                    "iteration": len(refinement) + 1,
                    "target_edge_length_m": target,
                    "marked_triangles": len(marked),
                    "surface_vertices": len(unit),
                    "surface_triangles": len(faces),
                    "maximum_parent_midpoint_match_error": float(distance.max()),
                }
            )

            (args.output / "progress.json").write_text(json.dumps(refinement, indent=2) + "\n")
        else:
            raise RuntimeError("Conforming local refinement did not reach the declared physical edge length")
        tets = prism_tetrahedra(np.sort(faces, axis=1), len(unit), 2)
        for mode, layered in meshes.items():
            vertices = layered.reshape(-1, 3)
            integral = mesh_integrals(vertices, tets, geometry, continuum)
            if mode == "polyhedron":
                for key in ["volume_m3", "outer_area_m2", "inner_area_m2", "inertia_about_com_kg_m2"]:
                    np.testing.assert_allclose(integral[key], seed_poly_integrals[key], rtol=1e-11, atol=1e-18)
                np.testing.assert_allclose(integral["com_m"], seed_poly_integrals["com_m"], rtol=0, atol=1e-14)
            tet_edges = np.unique(
                np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)), axis=0
            )
            p2 = np.concatenate((vertices, vertices[tet_edges].mean(axis=1)))
            distances, _nodes = cKDTree(p2).query(fixed)
            assert distances.max() < 1e-12
            path = args.output / f"{mode.lower()}-level{level}.npz"
            np.savez_compressed(path, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
            rows.append(
                {
                    "mode": mode,
                    "level": level,
                    "local_refinement_iterations": len(refinement),
                    "target_CAD_surface_edge_length_m": target,
                    "maximum_local_CAD_surface_edge_length_m": maximum_local_edge,
                    "mesh": str(path),
                    "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **integral,
                    "original_initializer_all_anchors_retained": True,
                    "original_initializer_nearest_P2_node_distances_m": distances.tolist(),
                }
            )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "rows": rows,
        "refinement": refinement,
        "seed_global_level": args.seed_level,
        "declared_geometry": vars(geometry),
        "mesh_marking_radius_m": args.marking_radius,
        "additional_contact_audits": [str(run) for run in args.contact_audit],
        "conservative_parameter_cap_angle_rad": cap_angle,
        "fixed_unit_sphere_marking_anchors": anchors.tolist(),
        "analytic_support_positions_author_m": support_positions,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Surface bisection and radial projection generate conforming CAD approximations; paired polyhedron vertices use identical midpoint prolongation without projection. Mesh marking anchors/radius are numerical choices and do not constrain collision locations or change any physical contact footprint, law or force. Global CAD approximation outside marked caps remains at its seed level. Geometry and represented mass change explicitly at fixed density; two thickness layers remain. All six original physical initializer points are retained. Geometry identity and small local elements do not establish stress/contact convergence, model/feedback accuracy, complete application/history or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"meshes": len(rows), "tetrahedra": [row["tetrahedra"] for row in rows]}), flush=True)


if __name__ == "__main__":
    main()
