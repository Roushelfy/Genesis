"""Check native Hertz input centers for curved witnesses offset along a known load line."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.hard_binding import StressHardBinding
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_binding import StressFrictionRegions
from genesis.engine.solvers.rigid.stress.hertz_contacts import (
    StressHertzContactState,
    kernel_hertz_contact_inputs,
    kernel_hertz_contact_accept,
)
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzFrames
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_MAT, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--curved", type=Path, required=True)
    cli.add_argument("--expect-baseline-failure", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "input": file_id(args.curved),
            "scope": "Given independently evaluated P2 normal roots/curvature, source witnesses are20/80um from the boundary along the known resultant line. Native source resolution supplies the nearest material anchor; contact input projection must return the original physical load point and preserve its moment. Four independently rotated/transformed environments have tangential/normal ratios0,0.2,0.35,0.2. No root pressure/frame model is changed. A parallel free-couple negative control must remain refused.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions(mesh=args.curved, contact_model="hertz", method="direct"))
    surface = StressSurface(4, model.info)
    vertices, indices = array(model.info.vertices), array(model.info.surface_nodes).T
    face_normals = array(surface.info.face_normal)
    face_centers = vertices[indices[:, :3]].mean(axis=1)
    outer = np.flatnonzero(np.einsum("fi,fi->f", face_centers, face_normals) > 0)
    face_ids = outer[np.linspace(0, len(outer) - 1, 8, dtype=int)]
    contacts, sources = create_contacts(8, 4, 0.001), create_sources(8, 4)
    regions, law = StressHertzRegions(contacts).state, StressHertz(32)
    hard = StressHardBinding(contacts, model.info)
    hard.state.is_active.fill(0)
    friction_regions = StressFrictionRegions(
        force=V_VEC(3, dtype=gs.qd_float, shape=(8, 4)),
        source_count=V(dtype=gs.qd_int, shape=(8, 4)),
        leader=V(dtype=gs.qd_int, shape=(8, 4)),
        profile=V(dtype=gs.qd_int, shape=(8, 4)),
    )
    frames = StressHertzFrames(
        center=V_VEC(3, dtype=gs.qd_float, shape=(32,)),
        basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(32,)),
        anchor=V(dtype=gs.qd_int, shape=(32,)),
    )
    state = StressHertzContactState(
        status=V(dtype=gs.qd_int, shape=(4,)),
        force_error=V(dtype=gs.qd_float, shape=(8, 4)),
        moment_error=V(dtype=gs.qd_float, shape=(8, 4)),
        step_principal=V(dtype=gs.qd_float, shape=(4,)),
        step_sampled_vm=V(dtype=gs.qd_float, shape=(4,)),
        observed_step=V(dtype=gs.qd_int, shape=(4,)),
        sampled_active=V(dtype=gs.qd_bool, shape=(8, 4)),
        input_is_valid=V(dtype=gs.qd_bool, shape=(4,)),
    )
    errno = V(dtype=gs.qd_int, shape=(4,))
    rotations = Rotation.from_rotvec([[0, 0, 0], [0.3, -0.4, 0.2], [-0.2, 0.1, 0.5], [0.4, 0.2, -0.1]])
    origins = np.array([[0, 0, 0], [0.04, -0.02, 0.01], [-0.03, 0.02, 0.04], [0.02, 0.01, -0.03]])
    contacts.frame_position.from_numpy(origins)
    contacts.frame_quaternion.from_numpy(rotations.as_quat()[:, [3, 0, 1, 2]])
    contacts.valid.fill(True)
    sources.geom.fill(7)
    sources.pair.from_numpy(np.broadcast_to(np.arange(8, dtype=gs.np_int)[:, None], (8, 4)).copy())
    regions.is_valid.fill(True)
    regions.source_count.fill(1)
    regions.leader.from_numpy(np.broadcast_to(np.arange(8, dtype=gs.np_int)[:, None], (8, 4)).copy())
    rows = []
    for step in range(4):
        bary = np.tile([0.4 - 0.01 * step, 0.3 + 0.01 * step, 0.3], (8, 1))
        values, gradients, hessian = quadratic_basis(bary)
        nodes = vertices[indices[face_ids]]
        center = np.einsum("qn,qni->qi", values, nodes)
        derivative = np.einsum("qna,qni->qia", gradients, nodes)
        normal = np.cross(derivative[:, :, 0], derivative[:, :, 1])
        normal *= np.sign(np.einsum("qi,qi->q", normal, face_normals[face_ids]))[:, None]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal), axis=1)])
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        basis = np.stack([tangent, np.cross(normal, tangent)], axis=2)
        second = np.einsum("nab,qni->qiab", hessian, nodes)
        metric_inverse = np.linalg.inv(derivative.transpose(0, 2, 1) @ derivative)
        operator = (
            -derivative
            @ metric_inverse
            @ np.einsum("qi,qiab->qab", normal, second)
            @ metric_inverse
            @ derivative.transpose(0, 2, 1)
        )
        curvature = basis.transpose(0, 2, 1) @ operator @ basis
        force = 0.1 * (-normal[:, None, :] + np.array([0.0, 0.2, 0.35, 0.2])[None, :, None] * tangent[:, None, :])
        distance = 2e-5 if step % 2 == 0 else 8e-5
        raw = center[:, None, :] - distance * force / np.linalg.norm(force, axis=2)[:, :, None]
        contacts.position.from_numpy(raw)
        contacts.force.from_numpy(force)
        sources.witness.from_numpy(raw)
        sources.free_moment.fill(0)
        kernel_resolve_sources(1e-4, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, surface.info)
        assert np.all(array(sources.status) == 0)
        world_center = np.stack([rotations[i].apply(center) + origins[i] for i in range(4)], axis=1)
        world_normal = np.stack([rotations[i].apply(normal) for i in range(4)], axis=1)
        world_tangent = np.stack([rotations[i].apply(tangent) for i in range(4)], axis=1)
        world_force = np.stack([rotations[i].apply(force[:, i]) for i in range(4)], axis=1)
        moment = np.cross(raw - center[:, None, :], force)
        world_moment = np.stack([rotations[i].apply(moment[:, i]) for i in range(4)], axis=1)
        regions.center.from_numpy(world_center)
        regions.normal.from_numpy(world_normal)
        regions.tangent.from_numpy(world_tangent)
        regions.curvature.from_numpy(np.broadcast_to(curvature[:, None], (8, 4, 2, 2)).copy())
        regions.force.from_numpy(world_force)
        regions.moment.from_numpy(world_moment)
        state.input_is_valid.fill(True)
        errno.fill(0)
        kernel_hertz_contact_inputs(
            contacts,
            sources,
            regions,
            law.state,
            frames,
            state,
            model.info,
            surface.info,
            1e10 / 0.91,
            1e-4,
            float(np.finfo(gs.np_float).eps),
        )
        law.update()
        kernel_hertz_contact_accept(
            contacts,
            regions,
            hard.state,
            hard.allocation.state.original,
            hard.hertz.state,
            hard.frames,
            law.state,
            frames,
            friction_regions,
            state,
            errno,
        )
        projected = array(frames.center).reshape(8, 4, 3)
        error = np.linalg.norm(projected - center[:, None, :], axis=2)
        source_centers = array(contacts.center).transpose(1, 0, 2)
        residual = array(state.moment_error).T
        scale = (array(law.state.force) * array(law.state.axes)[:, 0]).reshape(8, 4)
        is_passed = bool(array(state.input_is_valid).all() and error.max() < 1e-12)
        rows.append(
            {
                "step": step,
                "witness_distance_m": distance,
                "input_status": array(state.status).tolist(),
                "maximum_center_error_m": float(error.max()),
                "maximum_moment_error_nm": float(residual.max()),
                "maximum_moment_over_Fn_a": float(np.max(residual / scale)),
                "source_to_root_maximum_m": float(np.linalg.norm(source_centers - center[:, None, :], axis=2).max()),
                "passed": is_passed,
            }
        )
        np.savez_compressed(
            args.output / f"step{step}.npz",
            expected_centers_m=center,
            projected_centers_m=projected,
            raw_positions_m=raw,
            forces_n=force,
            source_centers_m=source_centers,
            moment_error_nm=residual,
            input_status=array(state.status),
            contact_status=array(contacts.status),
        )
        if not args.expect_baseline_failure:
            assert is_passed, rows[-1]
            np.testing.assert_allclose(array(law.state.force), 0.1, rtol=1e-12, atol=0)
    if args.expect_baseline_failure:
        assert any(not row["passed"] for row in rows)
    else:
        world_moment[0, 1] += world_force[0, 1] / np.linalg.norm(world_force[0, 1]) * 1e-8
        regions.moment.from_numpy(world_moment)
        state.input_is_valid.fill(True)
        kernel_hertz_contact_inputs(
            contacts,
            sources,
            regions,
            law.state,
            frames,
            state,
            model.info,
            surface.info,
            1e10 / 0.91,
            1e-4,
            float(np.finfo(gs.np_float).eps),
        )
        law.update()
        kernel_hertz_contact_accept(
            contacts,
            regions,
            hard.state,
            hard.allocation.state.original,
            hard.hertz.state,
            hard.frames,
            law.state,
            frames,
            friction_regions,
            state,
            errno,
        )
        np.testing.assert_array_equal(array(state.input_is_valid), [True, False, True, True])
        assert array(contacts.status)[1, 0] == 53
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "baseline_failure_expected": args.expect_baseline_failure,
            "free_couple_isolated_refusal_checked": not args.expect_baseline_failure,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
