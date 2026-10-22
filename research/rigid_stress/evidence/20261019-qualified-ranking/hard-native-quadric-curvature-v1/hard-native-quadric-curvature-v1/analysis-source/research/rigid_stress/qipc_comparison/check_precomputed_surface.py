"""Compare fixed-geometry queries against the original closest-face calculation."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.curvature import (
    StressCurvature,
    StressCurvatureInfo,
    func_closest_surface,
    func_surface_curvature,
)
from genesis.engine.solvers.rigid.stress.pair_geometry import func_opponent_boundary
from genesis.utils import array_class, geom

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def kernel_compare_surface(
    i_g: int,
    queries: qd.types.ndarray(),
    errors: qd.types.ndarray(),
    faces: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_q, i_b in qd.ndrange(queries.shape[0], errors.shape[1]):
        local = qd.Vector([queries[i_q, i] for i in qd.static(range(3))])
        position = geom.qd_transform_by_trans_quat(local, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])
        expected, normal, gap, face, status = func_opponent_boundary(i_g, i_b, position, dyn_state, dyn_info, epsilon)
        current, current_face = func_closest_surface(i_g, i_b, position, dyn_state, dyn_info, curvature, epsilon)
        current_normal, current_tensor, current_status = func_surface_curvature(
            i_g, current_face, i_b, current, dyn_state, dyn_info, curvature, epsilon
        )
        # Original barycentric inversion, independent of the cached transform.
        expected_local = geom.qd_inv_transform_by_trans_quat(
            expected, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
        )
        indices = dyn_info.faces.verts_idx[face]
        i_a, i_c, i_d = indices[0], indices[1], indices[2]
        first = dyn_info.verts.init_pos[i_a]
        edges = qd.Matrix.rows([dyn_info.verts.init_pos[i_c] - first, dyn_info.verts.init_pos[i_d] - first])
        coordinates = (edges @ edges.transpose()).inverse() @ (edges @ (expected_local - first))
        weights = qd.Vector([1.0 - coordinates.sum(), coordinates[0], coordinates[1]])
        expected_normal = qd.Vector.zero(gs.qd_float, 3)
        expected_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            i_v = indices[i]
            expected_normal += weights[i] * dyn_info.verts.init_normal[i_v]
            expected_tensor += weights[i] * curvature.vertices_tensor[i_v]
        expected_normal = expected_normal.normalized()
        projection = qd.Matrix.identity(gs.qd_float, 3) - expected_normal.outer_product(expected_normal)
        rotation = geom.qd_quat_to_R(dyn_state.geoms.quat[i_g, i_b], epsilon)
        expected_tensor = rotation @ projection @ expected_tensor @ projection @ rotation.transpose()
        expected_normal = rotation @ expected_normal
        errors[i_q, i_b, 0] = (current - expected).norm()
        errors[i_q, i_b, 1] = (current_normal - expected_normal).norm()
        errors[i_q, i_b, 2] = (current_tensor - expected_tensor).norm()
        faces[i_q, i_b, 0], faces[i_q, i_b, 1] = face, current_face
        faces[i_q, i_b, 2], faces[i_q, i_b, 3] = status, current_status


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(), "backend": args.backend})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    mesh.vertices *= [1, 0.75, 0.5]
    shifted = mesh.copy()
    shifted.vertices += [0.06, 0, 0.004]
    # One collision geom with disconnected components sharing normal directions.
    combined = trimesh.util.concatenate([mesh, shifted])
    path = args.output / "disconnected.obj"
    combined.export(path)
    entity = scene.add_entity(
        morph=gs.morphs.Mesh(file=path, convexify=False, decimate=False, fixed=True, batch_fixed_verts=True)
    )
    scene.build(n_envs=3)
    entity.set_pos([[0, 0, 0], [0.13, -0.07, 0.19], [2, -1, 3]])
    angles = np.array([[0, 0, 0], [0.3, -0.7, 1.2], [-0.8, 0.6, 0.2]])
    entity.set_quat(geom.xyz_to_quat(angles))
    geometry = entity.base_link.geoms[0]
    assert len(entity.base_link.geoms) == 1
    vertices = geometry.init_verts[geometry.init_faces]
    centroids = vertices.mean(axis=1)
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    queries = np.concatenate(
        [
            centroids,
            centroids + 0.001 * normals,
            centroids - 0.001 * normals,
            vertices[:, 0],
            0.5 * (vertices[:, 0] + vertices[:, 1]),
            [[0.03, 0, 0.002], [0.03, 0.03, 0.002]],
        ]
    )
    solver = scene.rigid_solver
    epsilon = float(np.finfo(gs.np_float).eps)
    cache = StressCurvature(solver.dyn_info, epsilon)
    errors = np.empty((len(queries), 3, 3))
    faces = np.empty((len(queries), 3, 4), dtype=gs.np_int)
    kernel_compare_surface(geometry.idx, queries, errors, faces, solver.dyn_state, solver.dyn_info, cache.info, epsilon)
    assert np.isfinite(errors).all()
    limits = np.array([1e-12, 1e-10, 1e-8])
    assert (errors <= limits).all(), errors.max(axis=(0, 1))
    np.testing.assert_array_equal(faces[..., 3], 1)
    output = args.output / "fields.npz"
    np.savez_compressed(output, queries=queries, errors=errors, faces=faces)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "queries": len(queries) * 3,
            "maximum_errors_position_normal_curvature": errors.max(axis=(0, 1)).tolist(),
            "limits": limits.tolist(),
            "changed_face_choices": int(np.count_nonzero(faces[..., 0] != faces[..., 1])),
            "arrays": file_id(output),
            "scope": "Matched nearest-surface and curvature queries on one disconnected two-component mesh in three rigid frames; face interiors, edges, vertices, inward/outward points and competing components. Implementation conformance only.",
        },
    )


if __name__ == "__main__":
    main()
