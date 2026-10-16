"""Mesh the unchanged U CAD body with every original physical initialization anchor as a used vertex."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import gmsh
import numpy as np
from scipy.spatial import cKDTree

from .summarize_fine_references import closed_source
from .validation_meshes import GEOMETRIES, build_geometry


def cad_integrals(volumes):
    mass = np.array([gmsh.model.occ.getMass(dim, tag) for dim, tag in volumes])
    centres = np.array([gmsh.model.occ.getCenterOfMass(dim, tag) for dim, tag in volumes])
    return float(mass.sum()), mass @ centres / mass.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--qualified-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.size > 0
    closed_source(args.qualified_run)
    assert int((args.qualified_run / "returncode.txt").read_text()) == 0
    old_anchor = args.qualified_run / "solution/initialization_anchors.json"
    original = json.loads(old_anchor.read_text())
    anchor_sha = hashlib.sha256(args.anchors.read_bytes()).hexdigest()
    assert original["sha256"] == anchor_sha
    with np.load(args.anchors) as saved:
        points, components = saved["positions_m"].copy(), saved["components"].copy()
    np.testing.assert_array_equal(points, original["positions_m"])
    np.testing.assert_array_equal(components, original["components"])
    unique_points = np.unique(points, axis=0)
    args.output.mkdir(parents=True, exist_ok=False)
    gmsh.initialize()
    try:
        for key, value in {
            "General.NumThreads": 1,
            "Mesh.ElementOrder": 1,
            "Mesh.Algorithm3D": 1,
            "Mesh.MeshSizeMin": args.size / 2,
            "Mesh.MeshSizeMax": args.size,
            "Mesh.MeshSizeFromCurvature": 0,
            "Mesh.MeshSizeExtendFromBoundary": 1,
            "Mesh.MeshSizeFromPoints": 0,
        }.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.add("u_fixed_physical_anchors")
        build_geometry("u", 2)
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        original_volume, original_com = cad_integrals(volumes)
        surfaces = gmsh.model.getBoundary(volumes, oriented=False)
        cad_distances = []
        for point in unique_points:
            distances = [
                np.linalg.norm(np.array(gmsh.model.getClosestPoint(dim, tag, point.tolist())[0]) - point)
                for dim, tag in surfaces
            ]
            cad_distances.append(float(min(distances)))
        assert max(cad_distances) <= 1e-12
        entities = [(0, gmsh.model.occ.addPoint(*point)) for point in unique_points]
        gmsh.model.occ.fragment(volumes, entities)
        gmsh.model.occ.synchronize()
        new_volume, new_com = cad_integrals(gmsh.model.getEntities(3))
        np.testing.assert_allclose(new_volume, original_volume, rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(new_com, original_com, rtol=0, atol=1e-14)
        field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(field, "F", str(args.size))
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, element_nodes = gmsh.model.mesh.getElementsByType(4)
        assert len(element_nodes) > 0
        order = np.argsort(tags)
        indices = np.searchsorted(tags[order], element_nodes)
        used, inverse = np.unique(indices, return_inverse=True)
        vertices = coordinates.reshape(-1, 3)[order][used]
        tetrahedra = inverse.reshape(-1, 4)
        determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
        assert np.all(determinant != 0)
        flipped = determinant < 0
        tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
        local_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape(-1, 3)), axis=0, return_counts=True)
        assert counts.max() <= 2
        boundary = faces[counts == 1]
        distance, nodes = cKDTree(vertices).query(points)
        assert distance.max() <= 1e-12, distance
        assert len(np.unique(3 * nodes + components)) == 6
        path = args.output / "mesh.npz"
        np.savez_compressed(
            path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary, cad_coordinates=vertices
        )
        gmsh.write(str(args.output / "mesh.msh"))
        definition = next(item for item in GEOMETRIES if item.name == "u")
        metadata = {
            "geometry": asdict(definition),
            "gmsh_version": gmsh.__version__,
            "mesh_size": args.size,
            "mesh_size_units": "m",
            "tetrahedra": len(tetrahedra),
            "nodes": len(vertices),
            "boundary_triangles": len(boundary),
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "minimum_volume_m3": float(abs(determinant).min() / 6),
            "total_volume_m3": float(abs(determinant).sum() / 6),
            "same_U_CAD_constructor": True,
            "CAD_volume_before_m3": original_volume,
            "CAD_volume_after_m3": new_volume,
            "CAD_COM_before_m": original_com.tolist(),
            "CAD_COM_after_m": new_com.tolist(),
            "original_anchor_sha256": anchor_sha,
            "original_anchor_positions_m": points.tolist(),
            "original_anchor_components": components.tolist(),
            "original_anchor_used_mesh_vertex_distances_m": distance.tolist(),
            "original_anchor_selected_vertex_ids": nodes.tolist(),
            "original_anchor_CAD_surface_distances_m": cad_distances,
            "new_equilibrium_states_solved": 0,
            "goal_completion": False,
            "limitations": "OpenCASCADE point fragmentation constrains the new mesh to original physical initialization coordinates while preserving the original CAD volume/COM at the declared checks. Physical geometry/material/forces/anchors/tolerances are unchanged; polyhedral approximation and tetrahedral/contact trace discretization differ and require complete new history/field audits. Actual used volume-mesh vertices must meet the original1e-12m anchor check. Mesh generation is not a new contact equilibrium or convergence result. Documentation: https://www.gmsh.info/doc/texinfo/gmsh.html and https://gmsh.info/doc/cookbook/geometry/embed-point.html",
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary = {
            "scope": __doc__,
            "rows": [metadata],
            "input_sha256": {
                str(args.anchors): anchor_sha,
                str(old_anchor): hashlib.sha256(old_anchor.read_bytes()).hexdigest(),
            },
            "new_equilibrium_states_solved": 0,
            "goal_completion": False,
        }
        assert hashlib.sha256(args.anchors.read_bytes()).hexdigest() == anchor_sha
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
