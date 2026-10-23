"""Compare ordered geometry searches and exhaustive vertex-binding decisions."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.curvature import StressCurvature, StressCurvatureInfo, func_closest_surface
from genesis.engine.solvers.rigid.stress.normal_roots import func_normal_root
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, create_sources
from genesis.engine.solvers.rigid.stress import source_vertices
from genesis.utils import array_class
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


@qd.kernel
def kernel_queries(
    i_geom: int,
    directions: qd.types.ndarray(),
    seeds: qd.types.ndarray(),
    centers: qd.types.ndarray(),
    tensors: qd.types.ndarray(),
    root_faces: qd.types.ndarray(),
    residuals: qd.types.ndarray(),
    statuses: qd.types.ndarray(),
    closest: qd.types.ndarray(),
    closest_faces: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_q in range(directions.shape[0]):
        direction = qd.Vector([directions[i_q, j] for j in qd.static(range(3))])
        seed = qd.Vector([seeds[i_q, j] for j in qd.static(range(3))])
        center, tensor, i_root, residual, status = func_normal_root(
            i_geom, 0, direction, seed, dyn_state, dyn_info, curvature, epsilon
        )
        point, i_face = func_closest_surface(i_geom, 0, seed, dyn_state, dyn_info, curvature, epsilon)
        for j in qd.static(range(3)):
            centers[i_q, j], closest[i_q, j] = center[j], point[j]
            for k in qd.static(range(3)):
                tensors[i_q, j, k] = tensor[j, k]
        root_faces[i_q], residuals[i_q], statuses[i_q], closest_faces[i_q] = i_root, residual, status, i_face


@qd.kernel
def kernel_exhaustive_matches(
    matches: qd.types.ndarray(),
    contacts: StressContactState,
    sources: StressSourceState,
    vertices: source_vertices.StressSourceVertexInfo,
    epsilon: float,
):
    for i_q, i_b, i_v in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1], vertices.positions.shape[0]):
        point = sources.witness[i_q, i_b]
        tolerance = 64.0 * epsilon * (point.norm() + contacts.frame_position[i_b].norm())
        matches[i_b, i_q, i_v] = (
            contacts.valid[i_q, i_b]
            and vertices.geom[i_v] == sources.geom[i_q, i_b]
            and (point - vertices.positions[i_v]).norm() <= tolerance
        )


def check_vertices(output, epsilon):
    count = 194
    positions = np.arange(count)[:, None] * np.array([[0.1, 0.2, 0.3]])
    positions[[0, 64, 128]] = 0
    positions[193] = positions[192]
    geoms = np.zeros(count, dtype=gs.np_int)
    geoms[128] = 1
    status = np.zeros(count, dtype=gs.np_int)
    status[129], status[130] = 1, 2
    references = np.arange(count, dtype=gs.np_int)
    references[193] = references[192]
    vertices = source_vertices.StressSourceVertexInfo(
        geom=V(dtype=gs.qd_int, shape=(count,)),
        positions=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
        reference_vertex=V(dtype=gs.qd_int, shape=(count,)),
        reference_face=V(dtype=gs.qd_int, shape=(count,)),
        barycentric=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
        offset=V(dtype=gs.qd_float, shape=(count,)),
        status=V(dtype=gs.qd_int, shape=(count,)),
        search_min=V_VEC(3, dtype=gs.qd_float, shape=((count + 63) // 64,)),
        search_max=V_VEC(3, dtype=gs.qd_float, shape=((count + 63) // 64,)),
    )
    vertices.geom.from_numpy(geoms)
    vertices.positions.from_numpy(positions)
    vertices.reference_vertex.from_numpy(references)
    vertices.reference_face.from_numpy(np.arange(count, dtype=gs.np_int))
    vertices.barycentric.from_numpy(np.tile([0.2, 0.3, 0.5], (count, 1)))
    vertices.status.from_numpy(status)
    source_vertices.kernel_source_vertex_bounds(vertices)
    random = np.random.default_rng(730016)
    direction = random.uniform(0.01, 1, size=(64, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    threshold = 64 * epsilon / (1 - 64 * epsilon)
    scales = [threshold]
    for _ in range(4):
        scales.insert(0, np.nextafter(scales[0], 0))
        scales.append(np.nextafter(scales[-1], np.inf))
    boundary = (-np.array(scales)[:, None, None] * direction[None]).reshape(-1, 3)
    points = np.concatenate([positions, positions + 1e-8, boundary])
    contacts = create_contacts(len(points), 2, 1.0, cooperative=False, cooperative_scatter=False)
    sources = create_sources(len(points), 2)
    contacts.valid.fill(True)
    contacts.frame_position.from_numpy(np.array([[1.0, 0, 0], [1e12, 0, 0]]))
    sources.witness.from_numpy(np.repeat(points[:, None], 2, axis=1))
    sources.geom.from_numpy(
        np.repeat(np.r_[geoms, geoms, np.zeros(len(boundary), dtype=gs.np_int)][:, None], 2, axis=1)
    )
    sources.witness_face.fill(-1)
    sources.witness_barycentric.fill(np.nan)
    matches = np.empty((2, len(points), count), dtype=gs.np_int)
    kernel_exhaustive_matches(matches, contacts, sources, vertices, epsilon)
    source_vertices.kernel_bind_source_vertices(contacts, sources, vertices, epsilon)
    expected_face = np.full((2, len(points)), -1, dtype=gs.np_int)
    expected_ambiguity = np.zeros((2, len(points)), dtype=bool)
    for env, row in np.ndindex(expected_face.shape):
        selected = np.flatnonzero(matches[env, row])
        mapped = selected[status[selected] == 0]
        expected_ambiguity[env, row] = np.any(status[selected] == 2) or len(np.unique(references[mapped])) > 1
        if len(mapped) and not expected_ambiguity[env, row]:
            expected_face[env, row] = mapped[-1]
    actual_face, actual_ambiguity = array(sources.witness_face), array(sources.has_vertex_ambiguity)
    mismatch = (actual_face != expected_face) | (actual_ambiguity != expected_ambiguity)
    np.savez_compressed(
        output / "vertex-binding.npz",
        points=points,
        matches=matches,
        expected_face=expected_face,
        expected_ambiguity=expected_ambiguity,
        actual_face=actual_face,
        actual_ambiguity=actual_ambiguity,
    )
    return {
        "passed": not mismatch.any(),
        "queries": int(mismatch.size),
        "mismatches": np.argwhere(mismatch).tolist(),
        "scope": "Exhaustive same-backend distance membership, duplicate coordinates across blocks, conflicting/same feature ids, unmapped/ambiguous vertices, two world-coordinate scales and 576 roundoff-boundary queries.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--compare", type=Path)
    cli.add_argument("--source-controls", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "baseline": file_id(args.compare) if args.compare else None,
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    epsilon = float(np.finfo(gs.np_float).eps)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
    second = sphere.copy()
    second.vertices += [0.05, 0, 0]
    disconnected = trimesh.util.concatenate([sphere, second])
    angles = np.arange(32) * 2 * np.pi / 32
    circles = np.arange(16) * 2 * np.pi / 16
    u, v = np.meshgrid(angles, circles, indexing="ij")
    torus_points = np.stack(
        [(0.03 + 0.01 * np.cos(v)) * np.cos(u), (0.03 + 0.01 * np.cos(v)) * np.sin(u), 0.01 * np.sin(v)], axis=-1
    ).reshape(-1, 3)
    first = np.arange(32 * 16).reshape(32, 16)
    following = np.roll(first, -1, axis=0)
    torus_faces = np.stack(
        [
            np.stack([first, following, np.roll(first, -1, axis=1)], axis=-1),
            np.stack([following, np.roll(following, -1, axis=1), np.roll(first, -1, axis=1)], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 3)
    meshes = [disconnected, trimesh.Trimesh(vertices=torus_points, faces=torus_faces, process=False)]
    scene = gs.Scene(show_viewer=False)
    entities = []
    for index, mesh in enumerate(meshes):
        path = args.output / f"mesh{index}.obj"
        mesh.export(path)
        entities.append(scene.add_entity(morph=gs.morphs.Mesh(file=path, convexify=False, decimate=False, fixed=True)))
    scene.build()
    solver = scene.rigid_solver
    cache = StressCurvature(solver.dyn_info, epsilon)
    random = np.random.default_rng(730016)
    saved, rows = {}, []
    for index, entity in enumerate(entities):
        geometry = entity.base_link.geoms[0]
        normals = array(solver.dyn_info.verts.init_normal)
        faces = array(solver.dyn_info.faces.verts_idx)[geometry.face_start : geometry.face_end]
        positions = array(solver.dyn_info.verts.init_pos)
        saved[f"mesh{index}_faces"] = array(solver.dyn_info.faces.verts_idx)
        selected = np.linspace(0, len(faces) - 1, 96, dtype=int)
        directions = np.concatenate(
            [normals[faces[selected, 0]], normals[faces[selected]].mean(axis=1), random.normal(size=(96, 3))]
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        seeds = np.tile(positions[faces[selected]].mean(axis=1), (3, 1))
        seeds[192:] += random.normal(size=(96, 3)) * 0.02
        for rotation_index, angles in enumerate([[0, 0, 0], [0.2, -0.5, 0.4]]):
            rotation = Rotation.from_rotvec(angles)
            translation = np.array([0.7, -0.3, 0.2]) * rotation_index
            xyzw = rotation.as_quat()
            entity.set_pos(translation)
            entity.set_quat(xyzw[[3, 0, 1, 2]])
            world_directions, world_seeds = rotation.apply(directions), rotation.apply(seeds) + translation
            centers, closest = np.empty_like(seeds), np.empty_like(seeds)
            tensors = np.empty((len(seeds), 3, 3))
            root_faces, closest_faces, statuses = [np.empty(len(seeds), dtype=gs.np_int) for _ in range(3)]
            residuals = np.empty(len(seeds))
            kernel_queries(
                geometry.idx,
                world_directions,
                world_seeds,
                centers,
                tensors,
                root_faces,
                residuals,
                statuses,
                closest,
                closest_faces,
                solver.dyn_state,
                solver.dyn_info,
                cache.info,
                epsilon,
            )
            prefix = f"mesh{index}_rotation{rotation_index}_"
            saved[prefix + "world_vertices"] = rotation.apply(positions) + translation
            for name, value in [
                ("centers", centers),
                ("tensors", tensors),
                ("root_faces", root_faces),
                ("statuses", statuses),
                ("residuals", residuals),
                ("closest", closest),
                ("closest_faces", closest_faces),
            ]:
                saved[prefix + name] = value
            rows.append(
                {
                    "mesh": index,
                    "rotation": rotation_index,
                    "queries": len(seeds),
                    "successful_roots": int((statuses == 1).sum()),
                }
            )
    np.savez_compressed(args.output / "geometry.npz", **saved)
    aliases = []
    if args.compare:
        with np.load(args.compare) as baseline:
            assert set(saved) == set(baseline.files)
            for name, value in saved.items():
                if name.endswith("_root_faces"):
                    before = baseline[name]
                    prefix = name[: -len("root_faces")]
                    mesh_name = name.split("_", 1)[0]
                    faces = saved[mesh_name + "_faces"]
                    vertices = saved[prefix + "world_vertices"]
                    centers = saved[prefix + "centers"]
                    for index in np.flatnonzero(value != before):
                        assert value[index] >= 0 and before[index] >= 0
                        common = np.intersect1d(faces[value[index]], faces[before[index]])
                        assert len(common), (name, index, value[index], before[index])
                        points = vertices[common]
                        distance = np.linalg.norm(points - centers[index], axis=1).min()
                        for first in range(len(points)):
                            for second in range(first + 1, len(points)):
                                edge = points[second] - points[first]
                                fraction = np.clip(
                                    np.dot(centers[index] - points[first], edge) / np.dot(edge, edge), 0, 1
                                )
                                distance = min(
                                    distance, np.linalg.norm(centers[index] - points[first] - fraction * edge)
                                )
                        assert distance < 1e-12, (name, index, distance)
                        aliases.append(
                            {
                                "field": name,
                                "query": int(index),
                                "shared_vertices": common.tolist(),
                                "distance_to_shared_feature_m": float(distance),
                            }
                        )
                elif value.dtype.kind in "iu":
                    np.testing.assert_array_equal(value, baseline[name], err_msg=name)
                else:
                    np.testing.assert_allclose(value, baseline[name], rtol=1e-10, atol=1e-12, err_msg=name)
    source_result = check_vertices(args.output, epsilon) if args.source_controls else None
    passed = source_result is None or source_result["passed"]
    write_json(
        args.output / "result.json",
        {
            "passed": bool(passed),
            "rows": rows,
            "source_binding": source_result,
            "original_source_conformance": args.compare is not None,
            "shared_feature_face_aliases": aliases,
            "whole_goal_complete": False,
        },
    )
    assert passed, source_result


if __name__ == "__main__":
    main()
