"""Check native finite-domain Hertz recovery with independent P2 readouts."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation
from skfem import FacetBasis, asm

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import halfspace_stress
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import elliptic_stress, field_difference, supplied_traction, tensor_at_points


def array(value):
    return qd_to_numpy(value, transpose=True, copy=True)


def matrix(components):
    result = np.empty((*components.shape[:-1], 3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = np.moveaxis(components[..., :3], -1, 0)
    result[..., 0, 1] = result[..., 1, 0] = components[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = components[..., 4]
    result[..., 1, 2] = result[..., 2, 1] = components[..., 5]
    return result


def material_points(points, center, basis):
    local = (points - center) @ basis
    assert local[:, 2].min() >= -1e-12
    local[:, 2] = np.maximum(local[:, 2], 0)
    return local


def remote(cpu, center, basis, load, axes, peak, hertz, order):
    mesh = cpu.reference.mesh
    facets = mesh.boundary_facets()
    vertices = mesh.p[:, mesh.facets[:, facets]].transpose(2, 1, 0)
    depth = (vertices - center) @ basis[:, 2]
    facets = facets[~np.all(np.abs(depth) < 1e-12, axis=1)]
    boundary = FacetBasis(mesh, cpu.reference.element, facets=facets, intorder=order)
    points = boundary.global_coordinates().transpose(1, 2, 0)
    local = material_points(points.reshape(-1, 3), center, basis)
    if hertz:
        stress = elliptic_stress(local, axes, peak=peak, order=96)
    else:
        stress = halfspace_stress(local, np.zeros((1, 2)), np.array([load]))
    stress = (basis @ stress @ basis.T).reshape(*points.shape[:2], 3, 3)
    traction = -np.einsum("fqij,jfq->ifq", stress, boundary.normals)
    return asm(supplied_traction, boundary, traction=traction).reshape(-1, 3)


def configure(module, centers, frames, anchors, positions, elements, barycentric):
    module.frames.center.from_numpy(centers)
    module.frames.basis.from_numpy(frames)
    module.frames.anchor.from_numpy(anchors.astype(gs.np_int))
    module.queries.position.from_numpy(positions)
    module.queries.element.from_numpy(elements.astype(gs.np_int))
    module.queries.barycentric.from_numpy(barycentric)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--rotated", action="store_true")
    cli.add_argument("--benchmark", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"same_load_tensor": 1e-7, "point_rhs": 1e-4, "full_hertz_volume": 1e-3, "max_full": 1e-7}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "reference": file_id(args.reference),
            "rotated": args.rotated,
            "targets": targets,
            "scope": "Normal flat-patch finite-body correction inside the existing rigid StressModel, with two region slots and four independently loaded environments. Prescribed Hertz pressures, independent P2 elasticity and higher quadrature; not independent pressure prediction, arbitrary curved/nonconvex contact, friction memory or a moving scene.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(args.reference) as saved:
        vertices, tets = saved["vertices_m"], saved["tetrahedra"]
        volume_points, weights = saved["common_points_m"], saved["common_weights_m3"]
    major = vertices[:, 0].max() / 20
    forces = np.array([0.2, 0.12, 0.18, 0, 0, 0.05, 0.11, 0])
    centers = major * np.array(
        [[0, 0, 0], [-2, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [2, -1, 0], [1, -1, 20], [0, 0, 0]]
    )
    frames = np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
    frames[6] = np.diag([1.0, -1.0, -1.0])
    rigid_rotation = Rotation.from_rotvec([0.31, -0.47, 0.22]).as_matrix() if args.rotated else np.eye(3)
    translation = np.array([0.003, -0.002, 0.007]) if args.rotated else np.zeros(3)
    vertices, volume_points = vertices @ rigid_rotation.T + translation, volume_points @ rigid_rotation.T + translation
    centers, frames = centers @ rigid_rotation.T + translation, rigid_rotation @ frames
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    mesh_path = args.output / "mesh.npz"
    np.savez_compressed(
        mesh_path,
        vertices=vertices,
        tetrahedra=tets,
        surface_triangles=cpu.reference.mesh.facets[:, cpu.reference.mesh.boundary_facets()].T,
    )
    begun = perf_counter()
    model = StressModel(RigidStressOptions(mesh=mesh_path, method="auto"))
    surface = StressSurface(4, model.info)
    hertz = StressHertz(8)
    hertz.state.force.from_numpy(forces)
    hertz.state.modulus.fill(1e10 / 0.91)
    angle = np.arange(8) * 0.17
    rotation2 = np.stack(
        [np.stack([np.cos(angle), -np.sin(angle)], axis=1), np.stack([np.sin(angle), np.cos(angle)], axis=1)], axis=1
    )
    curvature = rotation2 @ np.diag([50.0, 80.0]) @ rotation2.transpose(0, 2, 1)
    hertz.state.curvature.from_numpy(curvature)
    hertz.update()
    axes, peak, directions = array(hertz.state.axes), array(hertz.state.peak_pressure), array(hertz.state.direction)
    points = [volume_points]
    for i in np.flatnonzero(forces):
        theta = np.arange(32) * 2 * np.pi / 32
        xy = np.column_stack((axes[i, 0] * np.cos(theta), axes[i, 1] * np.sin(theta), np.zeros(32)))
        depth = np.column_stack((np.zeros((25, 2)), np.r_[0, np.geomspace(0.01, 4, 24)] * axes[i, 0]))
        q = np.array([[directions[i, 0], -directions[i, 1], 0], [directions[i, 1], directions[i, 0], 0], [0, 0, 1]])
        frames[i] = frames[i] @ q
        points.append(np.concatenate([xy, depth]) @ frames[i].T + centers[i])
    positions = np.concatenate(points)
    # The module rotates its supplied tangent frame by the Hertz principal angle.
    input_frames = rigid_rotation @ np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
    input_frames[6] = rigid_rotation @ np.diag([1.0, -1.0, -1.0])
    origins, duals, normals = (
        array(surface.info.face_origin),
        array(surface.info.face_dual),
        array(surface.info.face_normal),
    )
    local = np.einsum("fij,pfj->pfi", duals, centers[:, None] - origins)
    bary_face = np.concatenate([1 - local.sum(axis=2, keepdims=True), local], axis=2)
    plane = np.abs(np.einsum("pfj,fj->pf", centers[:, None] - origins, normals)) < 1e-12
    valid = (
        (bary_face.min(axis=2) >= -1e-10)
        & plane
        & (np.einsum("pj,fj->pf", input_frames[:, :, 2], normals) < -1 + 1e-10)
    )
    assert np.all(valid[forces > 0].any(axis=1))
    anchors = valid.argmax(axis=1)
    elements = cpu.reference.mesh.element_finder()(*positions.T)
    local = cpu.reference.basis.mapping.invF(positions.T[:, :, None], tind=elements)[:, :, 0]
    barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
    module = StressHertzRecovery(4, 2, len(positions), "full")
    configure(module, centers, input_frames, anchors, positions, elements, barycentric)
    state = model.create_state(4, "full")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(4,)), V(dtype=gs.qd_int, shape=(4,))
    omega.fill(0)
    errno.fill(0)
    kernel_begin_step(state)
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    assert np.array_equal(array(module.state.status), [0, 0, 0, 1]), array(module.state.status)
    assert np.all(array(state.valid)) and np.all(array(state.step_valid)) and not np.any(array(errno))
    field, corners = matrix(array(module.state.tensor)), matrix(array(state.stress_tensor))
    native_loads = array(state.force)
    permutation = cpu.reference.native_dof_permutation(array(model.info.vertices)).reshape(-1, 3)[:, 0] // 3
    rows, expected_fields = [], []
    for env in range(4):
        applied = np.empty_like(native_loads[env])
        applied[permutation] = native_loads[env]
        regular, displacement, _, residual = cpu.solve(applied)
        expected = tensor_at_points(cpu, displacement, positions)
        point_rhs, hertz_rhs = np.zeros_like(applied), np.zeros_like(applied)
        particular = np.zeros_like(expected)
        corner_particular = np.zeros_like(regular)
        for i in [env, env + 4]:
            if forces[i] > 0:
                point_rhs += remote(cpu, centers[i], frames[i], forces[i], axes[i], peak[i], False, 12)
                hertz_rhs += remote(cpu, centers[i], frames[i], forces[i], axes[i], peak[i], True, 12)
                part = elliptic_stress(
                    material_points(positions, centers[i], frames[i]), axes[i], peak=peak[i], order=16
                )
                particular += frames[i] @ part @ frames[i].T
                part = elliptic_stress(
                    material_points(cpu.reference.corner_positions.reshape(-1, 3), centers[i], frames[i]),
                    axes[i],
                    peak=peak[i],
                    order=16,
                ).reshape(*regular.shape)
                corner_particular += frames[i] @ part @ frames[i].T
        expected += particular
        expected_fields.append(expected)
        scale = max(np.linalg.norm(expected), 1)
        error = float(np.linalg.norm(field[env] - expected) / scale)
        corner_error = float(
            np.linalg.norm(corners[env] - regular - corner_particular)
            / max(np.linalg.norm(regular + corner_particular), 1)
        )
        load_error = float(np.linalg.norm(applied - point_rhs) / max(np.linalg.norm(point_rhs), 1e-15))
        assert (
            error < targets["same_load_tensor"]
            and corner_error < targets["same_load_tensor"]
            and load_error < targets["point_rhs"]
        ), (env, error, corner_error, load_error)
        _, physical_u, _, _ = cpu.solve(hertz_rhs)
        physical_volume = tensor_at_points(cpu, physical_u, volume_points)
        for i in [env, env + 4]:
            if forces[i] > 0:
                part = elliptic_stress(
                    material_points(volume_points, centers[i], frames[i]), axes[i], peak=peak[i], order=96
                )
                physical_volume += frames[i] @ part @ frames[i].T
        volume_error = (
            float(field_difference(field[env, : len(volume_points)], physical_volume, weights))
            if forces[[env, env + 4]].sum()
            else 0.0
        )
        assert volume_error < targets["full_hertz_volume"], (env, volume_error)
        combined = np.concatenate([corners[env].reshape(-1, 3, 3), field[env]])
        tension, vm = stress_measures(combined)
        np.testing.assert_allclose(array(state.peak)[env], vm.max(), rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(array(module.state.tensile_peak)[env], tension.max(), rtol=1e-12, atol=1e-8)
        rows.append(
            {
                "env": env,
                "same_load_field_relative": error,
                "same_load_corner_relative": corner_error,
                "higher_rule_point_rhs_relative": load_error,
                "full_hertz_volume_relative": volume_error,
                "wrench_error_relative": float(array(module.state.wrench_error)[env]),
                "vm_peak_pa": float(vm.max()),
                "tensile_peak_pa": float(tension.max()),
                "independent_elastic_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(applied), 1e-15)),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(
        args.output / "fields.npz",
        positions_m=positions,
        volume_weights_m3=weights,
        tensor_pa=field,
        expected_tensor_pa=np.stack(expected_fields),
        corner_tensor_pa=corners,
        nodal_force_n=native_loads,
        centers_m=centers,
        frames=input_frames,
        hertz_axes_m=axes,
        hertz_pressure_pa=peak,
    )
    max_module = StressHertzRecovery(4, 2, len(positions), "max")
    configure(max_module, centers, input_frames, anchors, positions, elements, barycentric)
    max_state = model.create_state(4, "max")
    max_module.recover(omega, hertz.state, max_state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_allclose(array(max_state.peak), array(state.peak), rtol=targets["max_full"], atol=1e-8)
    np.testing.assert_allclose(
        array(max_module.state.tensile_peak), array(module.state.tensile_peak), rtol=targets["max_full"], atol=1e-8
    )
    bad = anchors.copy()
    bad[1] = -1
    module.frames.anchor.from_numpy(bad.astype(gs.np_int))
    kernel_begin_step(state)
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_array_equal(array(module.state.status), [0, 2, 0, 1])
    assert np.isnan(array(state.stress_tensor)[1]).all() and np.isnan(array(module.state.tensor)[1]).all()
    np.testing.assert_allclose(matrix(array(module.state.tensor))[0], field[0], rtol=1e-7, atol=1e-5)
    module.frames.anchor.from_numpy(anchors.astype(gs.np_int))
    errno.fill(0)
    kernel_begin_step(state)
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_array_equal(array(module.state.status), [0, 0, 0, 1])
    assert np.all(array(state.step_valid)) and not np.any(array(errno))
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "build_and_control_seconds": perf_counter() - begun,
            "targets": targets,
            "max_full_and_anchor_failure_recovery_passed": True,
            "independent_contact_pressure_qualified": False,
            "general_contact_friction_or_scene_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
