"""Check small Hertz footprints against independent boundary projection and P2 work transfer."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .native_hertz import NativeHertz
from .native_hertz_transfer import HertzSurfaceTransfer, func_projected_surface_walk
from .p2_elastic import FreeElasticP2


@qd.kernel
def kernel_walk_queries(
    anchors: qd.types.ndarray(),
    centers: qd.types.ndarray(),
    targets: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    faces: qd.types.ndarray(),
    points: qd.types.ndarray(),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    for i in range(anchors.shape[0]):
        center = qd.Vector([centers[i, a] for a in qd.static(range(3))])
        target = qd.Vector([targets[i, a] for a in qd.static(range(3))])
        normal = qd.Vector([normals[i, a] for a in qd.static(range(3))])
        face, _, point = func_projected_surface_walk(
            anchors[i], center, target - center, normal, stress_info, surface_info, epsilon
        )
        faces[i] = face
        for a in qd.static(range(3)):
            points[i, a] = point[a]


def voxel_mesh(kind):
    shape = (3, 3, 3) if kind == "cavity" else (3, 3, 1) if kind == "u" else (2, 2, 1)
    cells = np.indices(shape).reshape(3, -1).T
    if kind == "cavity":
        cells = cells[(cells != 1).any(axis=1)]
    elif kind == "u":
        cells = cells[(cells[:, 0] != 1) | (cells[:, 1] == 0)]
    steps = np.eye(3, dtype=int)[np.array(list(permutations(range(3))))]
    offsets = np.concatenate((np.zeros((6, 1, 3), dtype=int), steps.cumsum(axis=1)), axis=1)
    coordinates = cells[:, None, None] + offsets
    grid_shape = tuple(size + 1 for size in shape)
    tets = np.ravel_multi_index(np.moveaxis(coordinates, -1, 0), grid_shape).reshape(-1, 4)
    used, inverse = np.unique(tets, return_inverse=True)
    vertices = np.indices(grid_shape).reshape(3, -1).T[used] * 0.001
    tets = inverse.reshape(-1, 4)
    faces, counts = np.unique(
        np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1),
        axis=0,
        return_counts=True,
    )
    return vertices, tets, faces[counts == 1]


def project_oracle(center, offsets, normal, nodes, coordinates, normals):
    """All-triangle line intersections, independent of the native adjacency walk.

    Controls select the nearest facing height to the known witness. This oracle
    is deliberately not used to admit jumps across the U-shaped material gap.
    """
    origin = coordinates[nodes[:, 0]]
    edges = coordinates[nodes[:, 1:3]] - origin[:, None]
    dual = np.linalg.solve(edges @ edges.swapaxes(1, 2), edges)
    denominator = normals @ normal
    safe = np.where(denominator > 1e-10, denominator, 1)
    faces, barycentric, positions = [], [], []
    for start in range(0, len(offsets), 256):
        targets = center + offsets[start : start + 256]
        height = np.einsum("fi,sfi->sf", normals, origin[None] - targets[:, None]) / safe[None]
        projected = targets[:, None] + height[..., None] * normal
        local = np.einsum("fij,sfj->sfi", dual, projected - origin)
        bary = np.concatenate((1 - local.sum(axis=-1, keepdims=True), local), axis=-1)
        score = np.where((bary >= -1e-10).all(axis=-1) & (denominator > 1e-10)[None], np.abs(height), np.inf)
        selected = score.argmin(axis=1)
        assert np.isfinite(score[np.arange(len(targets)), selected]).all()
        faces.extend(selected)
        barycentric.extend(bary[np.arange(len(targets)), selected])
        positions.extend(projected[np.arange(len(targets)), selected])
    return np.array(faces), np.array(barycentric), np.array(positions)


def scatter_oracle(faces, bary, weights, force, nodes, coordinates):
    shape = np.column_stack(
        (bary * (2 * bary - 1), 4 * bary[:, 0] * bary[:, 1], 4 * bary[:, 0] * bary[:, 2], 4 * bary[:, 1] * bary[:, 2])
    )
    loads = np.zeros_like(coordinates)
    np.add.at(loads, nodes[faces], shape[..., None] * weights[:, None, None] * force)
    return loads


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--egg", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"), "egg": file_id(args.egg)},
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    epsilon = float(np.finfo(gs.np_float).eps)
    rows = []
    for name in ("face", "edge", "vertex", "cavity", "u", "egg", "rotated_egg"):
        if name in ("egg", "rotated_egg"):
            with np.load(args.egg) as saved:
                vertices, tets, triangles = (saved[k] for k in ("vertices", "tetrahedra", "surface_triangles"))
            vertex = np.argmax(vertices[:, 2])
            center = vertices[vertex].copy()
            owned = np.any(triangles == vertex, axis=1)
            raw = np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            )
            raw *= np.where(np.einsum("fi,fi->f", raw, vertices[triangles].mean(axis=1)) > 0, 1, -1)[:, None]
            normal = raw[owned].sum(axis=0)
            normal /= np.linalg.norm(normal)
            tangent = np.cross(normal, [1.0, 0, 0])
            tangent /= np.linalg.norm(tangent)
        else:
            vertices, tets, triangles = voxel_mesh(name)
            center = np.array(
                {
                    "face": [0.0003, 0.0012, 0.001],
                    "u": [0.0003, 0.0012, 0.001],
                    "edge": [0.0005, 0.0015, 0.001],
                    "vertex": [0.001, 0.001, 0.001],
                    "cavity": [0.001, 0.0015, 0.0015],
                }[name]
            )
            normal = np.array([1.0, 0, 0] if name == "cavity" else [0.0, 0, 1])
            tangent = np.array([0.0, 1, 0] if name == "cavity" else [1.0, 0, 0])
        rotation = np.eye(3)
        if name == "rotated_egg":
            rotation = Rotation.from_euler("xyz", [21, -37, 19], degrees=True).as_matrix()
            vertices = vertices @ rotation.T + [0.03, -0.02, 0.01]
            center = center @ rotation.T + [0.03, -0.02, 0.01]
            normal, tangent = normal @ rotation.T, tangent @ rotation.T
        mesh = args.output / f"{name}.npz"
        np.savez_compressed(mesh, vertices=vertices, tetrahedra=tets, surface_triangles=triangles)
        model = StressModel(RigidStressOptions(mesh=mesh, method="direct"))
        surface = StressSurface(3, model.info)
        coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
        nodes = qd_to_numpy(model.info.surface_nodes, transpose=True, copy=True).T
        normals = qd_to_numpy(surface.info.face_normal, transpose=True, copy=True)
        anchor, _, _ = project_oracle(center, np.zeros((1, 3)), normal, nodes, coordinates, normals)
        state = model.create_state(3, "full")
        omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
        omega.fill(0)
        native = NativeHertz(3)
        native.state.force.from_numpy(np.array([0.1, 0.2, 0.1]))
        native.state.modulus.fill(1e10 / (1 - 0.3**2))
        native.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (3, 2, 2)).copy())
        native.update()
        transfer = HertzSurfaceTransfer(3, 3, angles=32)
        patch = transfer.patches
        patch.environment.from_numpy(np.arange(3, dtype=gs.np_int))
        patch.anchor.fill(int(anchor[0]))
        for field, value in ((patch.center, center), (patch.normal, normal), (patch.tangent, tangent)):
            field.from_numpy(np.broadcast_to(value, (3, 3)).copy())
        traction = -normal + (0.2 * tangent if name not in ("egg", "rotated_egg") else 0)
        forces = np.array([0.1, 0.2, 0.1])[:, None] * traction
        patch.force.from_numpy(forces)
        patch.moment.fill(0)
        transfer.scatter(native.state, state, model.info, surface.info, epsilon)
        status = qd_to_numpy(patch.status, transpose=True, copy=True)
        assert np.all(status == 0), (name, status)
        actual = qd_to_numpy(state.force, transpose=True, copy=True)
        points = qd_to_numpy(transfer.samples.position, transpose=True, copy=True)
        samples = qd_to_numpy(transfer.samples.face, transpose=True, copy=True)
        weights = qd_to_numpy(transfer.samples.weight, transpose=True, copy=True)
        for env in range(3):
            offsets = points[env] - center
            offsets -= (offsets @ normal)[:, None] * normal
            faces, bary, expected_points = project_oracle(center, offsets, normal, nodes, coordinates, normals)
            np.testing.assert_allclose(points[env], expected_points, atol=1e-12, rtol=0)
            expected = scatter_oracle(faces, bary, weights, forces[env], nodes, coordinates)
            np.testing.assert_allclose(actual[env], expected, atol=1e-13, rtol=1e-10)
        np.testing.assert_allclose(actual.sum(axis=1), forces, atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.cross(coordinates - center, actual).sum(axis=1), 0, atol=1e-14, rtol=0)
        # A globally quadratic test displacement has an independent closed-form
        # Hertz expectation on a planar patch, even when it crosses mesh edges.
        axes = qd_to_numpy(native.state.axes, transpose=True, copy=True)
        if name not in ("egg", "rotated_egg"):
            local = (coordinates - center) @ tangent
            work = np.einsum("bni,ni->b", actual, local[:, None] ** 2 * traction)
            exact = np.array([0.1, 0.2, 0.1]) * (traction @ traction) * axes[:, 0] ** 2 / 5
            np.testing.assert_allclose(work, exact, rtol=1e-9, atol=1e-20)
        model.recover(omega, state, surface_load=True)
        native_tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
        cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
        permutation = cpu.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
        load = np.empty_like(actual[0])
        load[permutation] = actual[0]
        expected_tensor, _, _, residual = cpu.solve(load)
        tensor_error = np.linalg.norm(native_tensor[0] - expected_tensor) / np.linalg.norm(expected_tensor)
        assert tensor_error < 1e-8 and np.linalg.norm(residual) / np.linalg.norm(load) < 1e-7
        # Zero one environment, then inject an unrepresented free moment into
        # another. Each operation must clear old loads without changing env0.
        native.state.force.from_numpy(np.array([0.1, 0.2, 0.0]))
        native.update()
        forces[2] = 0
        patch.force.from_numpy(forces)
        transfer.scatter(native.state, state, model.info, surface.info, epsilon)
        released = qd_to_numpy(state.force, transpose=True, copy=True)
        np.testing.assert_array_equal(released[2], 0)
        np.testing.assert_allclose(released[:2], actual[:2], atol=1e-14, rtol=1e-12)
        moment = np.zeros((3, 3))
        moment[1] = normal * 1e-5
        patch.moment.from_numpy(moment)
        transfer.scatter(native.state, state, model.info, surface.info, epsilon)
        np.testing.assert_array_equal(qd_to_numpy(patch.status, transpose=True), [0, 4, 1])
        rejected = qd_to_numpy(state.force, transpose=True, copy=True)
        np.testing.assert_array_equal(rejected[1:], 0)
        np.testing.assert_allclose(rejected[0], actual[0], atol=1e-14, rtol=1e-12)
        if name == "u":
            targets = np.array([[0.0008, 0.0025, 0.001], [0.0025, 0.002, 0.001]])
            result_faces = np.empty(2, dtype=gs.np_int)
            result_points = np.empty((2, 3))
            kernel_walk_queries(
                np.repeat(anchor.astype(gs.np_int), 2),
                np.repeat(center[None], 2, axis=0),
                targets,
                np.repeat(normal[None], 2, axis=0),
                result_faces,
                result_points,
                model.info,
                surface.info,
                epsilon,
            )
            assert result_faces[0] >= 0 and result_faces[1] == -1
            np.testing.assert_allclose(result_points[0], targets[0], atol=1e-12, rtol=0)
        if name == "egg":
            original_load, original_tensor = actual[0], native_tensor[0]
        elif name == "rotated_egg":
            np.testing.assert_allclose(actual[0], original_load @ rotation.T, atol=1e-12, rtol=1e-8)
            np.testing.assert_allclose(native_tensor[0], rotation @ original_tensor @ rotation.T, atol=1e-3, rtol=1e-7)
        path = args.output / f"{name}-fields.npz"
        np.savez_compressed(
            path,
            native_force_n=actual,
            native_stress_pa=native_tensor,
            independent_stress_pa=expected_tensor,
            sample_points_m=points,
            sample_faces=samples,
        )
        rows.append(
            {
                "case": name,
                "tensor_relative_error": float(tensor_error),
                "sample_count": len(weights),
                "regions": 3,
                "arrays": file_id(path),
                "mesh": file_id(mesh),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "rows": rows,
            "scope": "Known supplied Hertz regions lifted along witnessed triangle adjacency. Face/edge/vertex/cavity/coarse egg, work/wrench, release/failure isolation and independent same-load P2 fields. Not automatic source grouping, curvature prediction, continuous footprint admission, friction physics or shell Hertz accuracy.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
