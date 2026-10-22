"""Validate moving material probes against independent P2/Hertz fields."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_probes import func_hertz_probe_location
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array, discrete_particular, material_points, matrix, remote
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import elliptic_stress


@qd.kernel
def kernel_locations(starts: qd.Tensor, origins: qd.Tensor, points: qd.Tensor, result: qd.Tensor, info: StressInfo):
    for i in range(starts.shape[0]):
        element, _ = func_hertz_probe_location(starts[i], origins[i], points[i], info, 64)
        result[i] = element


def anchors_at(centers, frames, surface):
    origins, duals, normals = (
        array(surface.info.face_origin),
        array(surface.info.face_dual),
        array(surface.info.face_normal),
    )
    local = np.einsum("fij,pfj->pfi", duals, centers[:, None] - origins)
    bary = np.concatenate([1 - local.sum(axis=2, keepdims=True), local], axis=2)
    valid = (
        (bary.min(axis=2) >= -1e-10)
        & (np.abs(np.einsum("pfj,fj->pf", centers[:, None] - origins, normals)) < 1e-12)
        & (np.einsum("pj,fj->pf", frames[:, :, 2], normals) < -1 + 1e-10)
    )
    assert valid.any(axis=1).all()
    return valid.argmax(axis=1).astype(gs.np_int)


def disconnected_control(vertices, tets, surface, output):
    shift = np.array([3 * np.ptp(vertices[:, 0]), 0.0, 0.0])
    mesh = output / "disconnected.npz"
    np.savez_compressed(
        mesh,
        vertices=np.concatenate([vertices, vertices + shift]),
        tetrahedra=np.concatenate([tets, tets + len(vertices)]),
        surface_triangles=np.concatenate([surface, surface + len(vertices)]),
    )
    # Only geometry/location is used. No elastic solution on disconnected material is claimed.
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    count = len(tets)
    centers = vertices[tets].mean(axis=1)
    starts = V(dtype=gs.qd_int, shape=(3,))
    origins, points = V_VEC(3, dtype=gs.qd_float, shape=(3,)), V_VEC(3, dtype=gs.qd_float, shape=(3,))
    result = V(dtype=gs.qd_int, shape=(3,))
    starts.from_numpy(np.array([0, count, 0], dtype=gs.np_int))
    origins.from_numpy(np.stack([centers[0], centers[0] + shift, centers[0]]))
    points.from_numpy(np.stack([centers[-1], centers[-1] + shift, centers[0] + shift]))
    kernel_locations(starts, origins, points, result, model.info)
    located = array(result)
    assert 0 <= located[0] < count and count <= located[1] < 2 * count and located[2] == -1, located
    return {"passed": True, "located_elements": located.tolist(), "global_reassociation_used": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"same_load_field": 1e-7, "higher_rule_field": 1e-3, "higher_rule_peak": 1e-2, "max_full": 1e-7}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "mesh": file_id(args.control / "mesh.npz"),
            "inputs": file_id(args.control / "fields.npz"),
            "targets": targets,
            "scope": "Moving supplied normal regions and source-relative material probes, not automatic collision association or a live rigid scene. Same-coarse-mesh prescribed-load elasticity and higher quadrature remain distinct from pressure/continuum qualification.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(args.control / "mesh.npz") as saved:
        vertices, tets, triangles = saved["vertices"], saved["tetrahedra"], saved["surface_triangles"]
    with np.load(args.control / "fields.npz") as saved:
        base_centers, frames = saved["centers_m"], saved["frames"]
    major = np.ptp(vertices @ frames[0], axis=0).max() / 20
    model = StressModel(RigidStressOptions(mesh=args.control / "mesh.npz", method="auto"))
    surface = StressSurface(8, model.info)
    hertz = StressHertz(8)
    hertz.state.modulus.fill(1e10 / 0.91)
    module = StressHertzRecovery(4, 2, output_mode="full", moving_probes=True)
    max_module = StressHertzRecovery(4, 2, output_mode="max", moving_probes=True)
    state, max_state = model.create_state(4, "full"), model.create_state(4, "max")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(4,)), V(dtype=gs.qd_int, shape=(4,))
    omega.fill(0)
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    permutation = cpu.reference.native_dof_permutation(array(model.info.vertices)).reshape(-1, 3)[:, 0] // 3
    nominal = np.array([0.2, 0.12, 0.18, 0.0, 0.0, 0.05, 0.11, 0.0])
    rows, first_reference = [], None
    theta = np.arange(32) * 2 * np.pi / 32
    offsets = np.concatenate(
        [
            np.column_stack([np.cos(theta), np.sin(theta), np.zeros(32)]),
            0.5 * np.column_stack([np.cos(theta), np.sin(theta), np.zeros(32)]),
            np.column_stack([np.zeros((25, 2)), np.r_[0, np.geomspace(0.01, 4, 24)]]),
        ]
    )
    for step in range(8):
        force = nominal * (1 + 0.2 * np.sin(0.7 * step + np.arange(8)))
        force[0] = nominal[0]
        if step in [3, 4]:
            force[[1, 5]] = 0
        if step >= 2:
            force[3] = 0.06
        centers = base_centers.copy()
        for patch in [1, 2, 3, 5, 6]:
            tangent = major * np.array([2.5 * np.sin(0.8 * step + patch), 2 * np.cos(0.6 * step + patch), 0])
            centers[patch] += frames[patch] @ tangent
        anchors = anchors_at(centers, frames, surface)
        angle = np.arange(8) * 0.17 + step * 0.037
        angle[0] = 0
        rotation2 = np.stack(
            [
                np.stack([np.cos(angle), -np.sin(angle)], axis=1),
                np.stack([np.sin(angle), np.cos(angle)], axis=1),
            ],
            axis=1,
        )
        curvature = rotation2 @ np.diag([50.0, 80.0]) @ rotation2.transpose(0, 2, 1)
        hertz.state.force.from_numpy(force)
        hertz.state.curvature.from_numpy(curvature)
        hertz.update()
        errno.fill(0)
        for selected, selected_state in [(module, state), (max_module, max_state)]:
            selected.frames.center.from_numpy(centers)
            selected.frames.basis.from_numpy(frames)
            selected.frames.anchor.from_numpy(anchors)
            kernel_begin_step(selected_state)
            selected.recover(omega, hertz.state, selected_state, model, surface.info, 1e-12, errno)
        status = array(module.state.status)
        expected_status = np.where(force.reshape(2, 4).sum(axis=0) > 0, 0, 1)
        assert np.array_equal(status, expected_status), (step, status, array(module.state.wrench_error))
        assert array(state.step_valid).all() and array(max_state.step_valid).all() and not array(errno).any()
        axes, peak, direction = array(hertz.state.axes), array(hertz.state.peak_pressure), array(hertz.state.direction)
        bases = frames.copy()
        for patch in np.flatnonzero(force):
            x, y = direction[patch]
            bases[patch] = frames[patch] @ np.array([[x, -y, 0], [y, x, 0], [0, 0, 1]])
        positions, elements, bary = (
            array(module.probes.position),
            array(module.probes.element),
            array(module.probes.barycentric),
        )
        fields, corners, loads = (
            matrix(array(module.probes.tensor)),
            matrix(array(state.stress_tensor)),
            array(state.force),
        )
        expected_fields, reference_fields = np.zeros_like(fields), np.zeros_like(fields)
        expected_corners, reference_corners, frame_rows = [], [], []
        for patch in np.flatnonzero(force):
            expected_positions = centers[patch] + (offsets * np.r_[axes[patch], axes[patch, 0]]) @ bases[patch].T
            np.testing.assert_allclose(positions[patch], expected_positions, rtol=0, atol=1e-13)
            assert (elements[patch] >= 0).all()
            located = np.einsum("qc,qci->qi", bary[patch], vertices[tets[elements[patch]]])
            np.testing.assert_allclose(located, positions[patch], rtol=0, atol=1e-13)
        for env in range(4):
            applied = np.empty_like(loads[env])
            applied[permutation] = loads[env]
            regular, _, _, _ = cpu.solve(applied)
            active = [patch for patch in [env, env + 4] if force[patch] > 0]
            query_points = np.concatenate([positions[patch] for patch in active]) if active else np.empty((0, 3))
            query_elements = np.concatenate([elements[patch] for patch in active]) if active else np.empty(0, dtype=int)
            query_bary = np.concatenate([bary[patch] for patch in active]) if active else np.empty((0, 4))
            expected = np.einsum("nc,ncij->nij", query_bary, regular[query_elements])
            corner = regular.copy()
            refined_rhs = np.zeros_like(applied)
            particular = np.zeros_like(expected)
            corner_high_part = np.zeros_like(regular)
            for patch in active:
                q = discrete_particular(
                    material_points(query_points, centers[patch], bases[patch]),
                    axes[patch],
                    peak[patch],
                    force[patch],
                    True,
                )
                expected += bases[patch] @ q @ bases[patch].T
                q = discrete_particular(
                    material_points(cpu.reference.corner_positions.reshape(-1, 3), centers[patch], bases[patch]),
                    axes[patch],
                    peak[patch],
                    force[patch],
                    True,
                )
                corner += (bases[patch] @ q @ bases[patch].T).reshape(corner.shape)
                refined_rhs += remote(
                    cpu, centers[patch], bases[patch], force[patch], axes[patch], peak[patch], True, 12
                )
                q = elliptic_stress(
                    material_points(query_points, centers[patch], bases[patch]), axes[patch], peak=peak[patch], order=96
                )
                particular += bases[patch] @ q @ bases[patch].T
                q = elliptic_stress(
                    material_points(cpu.reference.corner_positions.reshape(-1, 3), centers[patch], bases[patch]),
                    axes[patch],
                    peak=peak[patch],
                    order=96,
                )
                corner_high_part += (bases[patch] @ q @ bases[patch].T).reshape(corner_high_part.shape)
            higher_regular, _, _, _ = cpu.solve(refined_rhs)
            higher = np.einsum("nc,ncij->nij", query_bary, higher_regular[query_elements]) + particular
            higher_corner = higher_regular + corner_high_part
            native = np.concatenate([fields[patch] for patch in active]) if active else np.empty((0, 3, 3))
            field_error = np.linalg.norm(native - expected) / max(np.linalg.norm(expected), 1)
            corner_error = np.linalg.norm(corners[env] - corner) / max(np.linalg.norm(corner), 1)
            reference_error = np.linalg.norm(native - higher) / max(np.linalg.norm(higher), 1)
            assert max(field_error, corner_error) <= targets["same_load_field"], (step, env, field_error, corner_error)
            assert reference_error <= targets["higher_rule_field"], (step, env, reference_error)
            combined = np.concatenate([corners[env].reshape(-1, 3, 3), native])
            tension, vm = stress_measures(combined)
            np.testing.assert_allclose(array(state.peak)[env], vm.max(), rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(array(module.state.tensile_peak)[env], tension.max(), rtol=1e-12, atol=1e-8)
            if active:
                higher_t, higher_vm = stress_measures(np.concatenate([higher_corner.reshape(-1, 3, 3), higher]))
                assert abs(vm.max() - higher_vm.max()) / max(higher_vm.max(), 1) < targets["higher_rule_peak"]
                assert abs(tension.max() - higher_t.max()) / max(abs(higher_t.max()), 1) < targets["higher_rule_peak"]
            for index, patch in enumerate(active):
                expected_fields[patch] = expected[index * 89 : (index + 1) * 89]
                reference_fields[patch] = higher[index * 89 : (index + 1) * 89]
            expected_corners.append(corner)
            reference_corners.append(higher_corner)
            frame_rows.append(
                {
                    "env": env,
                    "active_regions": len(active),
                    "probe_field_relative": float(field_error),
                    "corner_field_relative": float(corner_error),
                    "higher_rule_field_relative": float(reference_error),
                }
            )
        if first_reference is None:
            first_reference = (fields[0].copy(), corners[0].copy())
        np.testing.assert_allclose(fields[0], first_reference[0], rtol=1e-12, atol=1e-5)
        np.testing.assert_allclose(corners[0], first_reference[1], rtol=1e-12, atol=1e-5)
        np.testing.assert_allclose(array(max_state.peak), array(state.peak), rtol=targets["max_full"], atol=1e-8)
        np.testing.assert_allclose(
            array(max_module.state.tensile_peak), array(module.state.tensile_peak), rtol=targets["max_full"], atol=1e-8
        )
        np.savez_compressed(
            args.output / f"frame{step}.npz",
            positions_m=positions,
            elements=elements,
            barycentric=bary,
            tensor_pa=fields,
            corners_pa=corners,
            centers_m=centers,
            frames=frames,
            anchors=anchors,
            forces_n=force,
            status=status,
            vm_pa=array(state.peak),
            tension_pa=array(module.state.tensile_peak),
            regular_force_n=loads,
            same_load_probe_tensor_pa=expected_fields,
            same_load_corners_pa=np.stack(expected_corners),
            higher_rule_probe_tensor_pa=reference_fields,
            higher_rule_corners_pa=np.stack(reference_corners),
        )
        rows.append({"step": step, "rows": frame_rows})
        write_json(args.output / "progress.json", {"rows": rows})
    # The probe walk cannot hide a query failure or recover earlier in the same step.
    module.probe_walk_steps = 1
    kernel_begin_step(state)
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    failed = array(module.state.status) == 6
    assert failed.any() and np.isnan(array(module.probes.tensor).reshape(2, 4, 89, 6)[:, failed]).all()
    invalid_counts = array(state.invalid_steps)
    module.probe_walk_steps = 64
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    np.testing.assert_array_equal(array(state.invalid_steps), invalid_counts)
    assert np.isnan(array(state.peak)[failed]).all()
    kernel_begin_step(state)
    errno.fill(0)
    module.recover(omega, hertz.state, state, model, surface.info, 1e-12, errno)
    assert array(state.step_valid).all() and not array(errno).any()
    disconnected = disconnected_control(vertices, tets, triangles, args.output)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "targets": targets,
            "rows": rows,
            "probe_count_per_region": 89,
            "changed_contacts_release_isolation_and_max_full_passed": True,
            "query_failure_and_step_invalidation_passed": True,
            "disconnected_material_location": disconnected,
            "automatic_collision_input_or_scene_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
