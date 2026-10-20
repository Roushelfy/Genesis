"""Check native current curvature, physical minima and source-splitting invariance."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.curvature import (
    CurvatureStatus,
    StressCurvature,
    StressCurvatureInfo,
    func_surface_curvature,
)
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, create_sources
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def kernel_queries(
    geoms: qd.types.ndarray(),
    faces: qd.types.ndarray(),
    positions: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    tensors: qd.types.ndarray(),
    statuses: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i in range(geoms.shape[0]):
        position = qd.Vector([positions[i, j] for j in qd.static(range(3))])
        normal, tensor, status = func_surface_curvature(
            geoms[i], faces[i], 0, position, dyn_state, dyn_info, curvature, epsilon
        )
        for j in qd.static(range(3)):
            normals[i, j] = normal[j]
            for k in qd.static(range(3)):
                tensors[i, j, k] = tensor[j, k]
        statuses[i] = status


@qd.kernel
def kernel_sources(
    own: qd.types.ndarray(),
    points: qd.types.ndarray(),
    forces: qd.types.ndarray(),
    translation: qd.types.ndarray(),
    quaternion: qd.types.ndarray(),
    contacts: StressContactState,
    sources: StressSourceState,
    collider: array_class.ColliderState,
    own_link: int,
    other_link: int,
    other_geom: int,
):
    for env in range(contacts.valid.shape[1]):
        contacts.frame_position[env] = qd.Vector([translation[j] for j in qd.static(range(3))])
        contacts.frame_quaternion[env] = qd.Vector([quaternion[j] for j in qd.static(range(4))])
        collider.n_contacts[env] = points.shape[0]
    for i, env in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        contacts.valid[i, env] = i < points.shape[0]
        contacts.force[i, env] = qd.Vector.zero(gs.qd_float, 3)
        if i < points.shape[0]:
            point = qd.Vector([points[i, j] for j in qd.static(range(3))])
            contacts.position[i, env] = point
            contacts.force[i, env] = qd.Vector([forces[env, i, j] for j in qd.static(range(3))])
            sources.geom[i, env], sources.pair[i, env] = own[i], own[i]
            sources.witness[i, env] = point
            sources.free_moment[i, env] = qd.Vector.zero(gs.qd_float, 3)
            collider.contact_sort_idx[i, env] = i
            collider.contact_data.geom_a[i, env] = own[i]
            collider.contact_data.geom_b[i, env] = other_geom
            collider.contact_data.link_a[i, env] = own_link
            collider.contact_data.link_b[i, env] = other_link


def array(field):
    return qd_to_numpy(field, transpose=True, copy=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"), "mesh_curvature_rms_target": 0.05},
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    epsilon = float(np.finfo(gs.np_float).eps)
    xml = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(xml, "worldbody"), "body", pos="0 0 0.01")
    ET.SubElement(body, "freejoint")
    for x in (-0.03, 0.03):
        ET.SubElement(body, "geom", type="sphere", size="0.01", pos=f"{x} 0 0", mass="0.01")
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(morph=gs.morphs.Plane())
    target = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(xml, encoding="unicode")))
    meshes = []
    for name, scale in (("sphere", [0.02, 0.02, 0.02]), ("ellipsoid", [0.02, 0.015, 0.01])):
        for level in (1, 3):
            mesh = trimesh.creation.icosphere(subdivisions=level)
            mesh.vertices *= scale
            path = args.output / f"{name}-{level}.obj"
            mesh.export(path)
            entity = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=path,
                    pos=(0.2 * (len(meshes) + 1), 0, 0),
                    fixed=True,
                    convexify=False,
                    decimate=False,
                    preserve_mesh_precision=True,
                )
            )
            meshes.append((name, level, np.array(scale), entity))
    scene.build(n_envs=3)
    solver = scene.rigid_solver
    cache = StressCurvature(solver.dyn_info, epsilon)
    own_geoms = np.array([g.idx for link in target.links for g in link.geoms], dtype=gs.np_int)
    plane_geom = plane.base_link.geoms[0].idx
    centers = np.array([[-0.03, 0, 0.01], [0.03, 0, 0.01]])
    point = centers + [0, 0, -0.01]
    normal, tensor, status = np.empty((3, 3)), np.empty((3, 3, 3)), np.empty(3, dtype=gs.np_int)
    kernel_queries(
        np.r_[own_geoms, plane_geom].astype(gs.np_int),
        np.full(3, -1, dtype=gs.np_int),
        np.r_[point, [[0, 0, 0]]],
        normal,
        tensor,
        status,
        solver.dyn_state,
        solver.dyn_info,
        cache.info,
        epsilon,
    )
    np.testing.assert_array_equal(status, CurvatureStatus.ANALYTIC)
    np.testing.assert_allclose(normal, [[0, 0, -1], [0, 0, -1], [0, 0, 1]], atol=1e-14, rtol=0)
    np.testing.assert_allclose(tensor[:2], np.broadcast_to(np.diag([100, 100, 0]), (2, 3, 3)), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(tensor[2], 0)
    curvature_rows, saved_fields = [], {}
    for name, level, scale, entity in meshes:
        geometry = entity.base_link.geoms[0]
        positions = geometry.init_verts[geometry.init_faces].mean(axis=1)
        translation = tensor_to_array(entity.get_pos())[0]
        positions += translation
        count = len(positions)
        normal, tensor, status = np.empty((count, 3)), np.empty((count, 3, 3)), np.empty(count, dtype=gs.np_int)
        kernel_queries(
            np.full(count, geometry.idx, dtype=gs.np_int),
            np.arange(count, dtype=gs.np_int) + geometry.face_start,
            positions,
            normal,
            tensor,
            status,
            solver.dyn_state,
            solver.dyn_info,
            cache.info,
            epsilon,
        )
        np.testing.assert_array_equal(status, CurvatureStatus.SMOOTH_NORMAL_ESTIMATE)
        local = positions - translation
        local /= np.sqrt(np.sum((local / scale) ** 2, axis=1))[:, None]
        gradient = local / scale**2
        expected_normal = gradient / np.linalg.norm(gradient, axis=1)[:, None]
        projection = np.eye(3) - expected_normal[:, :, None] * expected_normal[:, None, :]
        exact = projection @ np.diag(1 / scale**2) @ projection / np.linalg.norm(gradient, axis=1)[:, None, None]
        error = np.linalg.norm(tensor - exact, axis=(1, 2)) / np.linalg.norm(exact, axis=(1, 2))
        curvature_rows.append(
            {
                "shape": name,
                "level": level,
                "faces": count,
                "rms_tensor_relative": float(np.sqrt(np.mean(error**2))),
                "maximum_tensor_relative": float(error.max()),
                "rms_target_met": bool(np.sqrt(np.mean(error**2)) <= 0.05),
            }
        )
        saved_fields[f"{name}_{level}_tensor"] = tensor
        saved_fields[f"{name}_{level}_reference"] = exact
    contacts, sources = create_contacts(8, 3, 1.0, cooperative=False, cooperative_scatter=False), create_sources(8, 3)
    regions = StressHertzRegions(contacts)
    owner = np.repeat(own_geoms, 2)
    offsets = np.array([[-0.001, 0, 0], [0.001, 0, 0], [0, -0.0015, 0], [0, 0.0015, 0]])
    points = np.repeat(centers, 2, axis=0) + offsets
    points[:, 2] -= np.sqrt(0.01**2 - (offsets[:, :2] ** 2).sum(axis=1))
    forces = np.array([[0.005, 0, 0.05], [0.015, 0, 0.15], [-0.01, 0, 0.1], [-0.02, 0, 0.2]])
    forces = np.array([1.0, 1.5, 0.7])[:, None, None] * forces
    translation, quaternion = np.zeros(3), np.array([1.0, 0, 0, 0])
    for case in ("original", "permuted", "split", "release", "rotated"):
        indices = np.arange(4)
        if case == "permuted":
            indices = np.array([2, 0, 3, 1])
        elif case == "split":
            indices = np.repeat(indices, 2)
        current_force = forces[:, indices].copy() / (2 if case == "split" else 1)
        if case == "release":
            current_force[1] = 0
        if case == "rotated":
            rotation = Rotation.from_euler("xyz", [21, -16, 33], degrees=True)
            quaternion = rotation.as_quat()[[3, 0, 1, 2]]
            translation = np.array([0.04, -0.03, 0.02])
            plane.set_pos(translation)
            plane.set_quat(quaternion)
            target.set_pos(rotation.apply([0, 0, 0.01]) + translation)
            target.set_quat(quaternion)
        kernel_sources(
            owner[indices],
            points[indices],
            current_force,
            translation,
            quaternion,
            contacts,
            sources,
            solver.collider.collider_state,
            target.base_link.idx,
            plane.base_link.idx,
            plane_geom,
        )
        regions.update(
            target.base_link.idx,
            contacts,
            sources,
            solver.dyn_state,
            solver.collider.collider_state,
            solver.dyn_info,
            cache.info,
            epsilon,
        )
        data = {
            "center": array(regions.state.center),
            "normal": array(regions.state.normal),
            "curvature": array(regions.state.curvature),
            "status": array(regions.state.status),
            "force": array(regions.state.force),
            "moment": array(regions.state.moment),
            "source_count": array(regions.state.source_count),
            "is_valid": array(regions.state.is_valid),
        }
        np.savez_compressed(args.output / f"{case}.npz", **data)
        assert data["is_valid"].all(), (case, data["status"])
        for env in range(3):
            selected = np.flatnonzero(data["source_count"][env])
            if case == "release" and env == 1:
                assert not len(selected)
                np.testing.assert_array_equal(data["force"][env], 0)
                continue
            assert len(selected) == 2, (case, selected, data["status"])
            expected_centers = np.array([[-0.03, 0, 0], [0.03, 0, 0]])
            expected_force = np.array([forces[env, :2].sum(axis=0), forces[env, 2:].sum(axis=0)])
            expected_moment = np.array(
                [
                    np.cross(points[:2] - expected_centers[0], forces[env, :2]).sum(axis=0),
                    np.cross(points[2:] - expected_centers[1], forces[env, 2:]).sum(axis=0),
                ]
            )
            if case == "rotated":
                expected_centers = rotation.apply(expected_centers) + translation
                expected_force, expected_moment = rotation.apply(expected_force), rotation.apply(expected_moment)
            order = np.argmin(
                np.linalg.norm(data["center"][env, selected, None] - expected_centers[None], axis=2), axis=0
            )
            selected = selected[order]
            np.testing.assert_allclose(data["center"][env, selected], expected_centers, rtol=0, atol=2e-9)
            np.testing.assert_allclose(data["force"][env, selected], expected_force, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(data["moment"][env, selected], expected_moment, rtol=1e-6, atol=1e-9)
    arrays = args.output / "curvature-fields.npz"
    np.savez_compressed(arrays, **saved_fields)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "curvature": curvature_rows,
            "arrays": file_id(arrays),
            "region_cases": ["original", "permuted", "split", "release", "rotated"],
            "scope": "Geometry and complete region-wrench controls. Sources are synthetic over current actual shapes, not admitted physical contact pressures. Smooth mesh estimates are compared to analytical curvature; target flags are distinct from numerical validity.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
