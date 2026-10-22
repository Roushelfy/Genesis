"""Track native radial friction memory across adjacent curved material faces."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_binding import (
    StressFrictionBindingState,
    kernel_friction_bind,
    kernel_friction_commit,
)
from genesis.engine.solvers.rigid.stress.hertz_friction import StressFrictionHistory
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzFrames
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.hertz_tangential import StressHertzShear
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_MAT, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .hertz_history_cone import ConeFrictionHistory
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
            "scope": "Supplied curved contact frames and forces, two material regions, four environments, slot permutation and one release/recontact. Check geometry binding and minimal-normal tangent transport against independent P2 evaluation, SciPy rotation and radial CPU history. This verifies the declared memory approximation, not moving-contact pressure or physical stress.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions(mesh=args.curved, contact_model="hertz", method="direct"))
    surface = StressSurface(4, model.info)
    contacts, sources = create_contacts(2, 4, 0.001), create_sources(2, 4)
    regions, law, friction = StressHertzRegions(contacts).state, StressHertz(8).state, StressFrictionHistory(8)
    frames = StressHertzFrames(
        center=V_VEC(3, dtype=gs.qd_float, shape=(8,)),
        basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(8,)),
        anchor=V(dtype=gs.qd_int, shape=(8,)),
    )
    binding = StressFrictionBindingState(
        geom=V(dtype=gs.qd_int, shape=(8,)),
        pair=V(dtype=gs.qd_int, shape=(8,)),
        center=V_VEC(3, dtype=gs.qd_float, shape=(8,)),
        basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(8,)),
        anchor=V(dtype=gs.qd_int, shape=(8,)),
        radius=V(dtype=gs.qd_float, shape=(8,)),
        is_lost=V(dtype=gs.qd_bool, shape=(4,)),
    )
    shear = StressHertzShear(
        friction=V(dtype=gs.qd_float, shape=(8,)),
        force=V_VEC(2, dtype=gs.qd_float, shape=(8,)),
        coefficients=friction.state.coefficients,
    )
    status, valid, errno = (
        V(dtype=gs.qd_int, shape=(4,)),
        V(dtype=gs.qd_bool, shape=(4,)),
        V(dtype=gs.qd_int, shape=(4,)),
    )
    valid.fill(True)
    contacts.frame_position.fill(0)
    contacts.frame_quaternion.from_numpy(np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))
    contacts.friction.fill(0.4)
    sources.geom.fill(7)
    law.status.fill(0)
    law.direction.from_numpy(np.tile([1.0, 0.0], (8, 1)))
    vertices, indices = array(model.info.vertices), array(model.info.surface_nodes).T
    neighbors, face_normals = array(model.info.surface_neighbors).T, array(surface.info.face_normal)
    references = [[ConeFrictionHistory(16) for _ in range(2)] for _ in range(4)]
    previous_basis = np.broadcast_to(np.eye(3), (4, 2, 3, 3)).copy()
    has_previous = np.zeros((4, 2), dtype=bool)
    rows = []
    for step in range(12):
        centers, bases, anchors = np.zeros((8, 3)), np.zeros((8, 3, 3)), np.zeros(8, dtype=gs.np_int)
        forces, normal_load, radii, expected_angles = np.zeros((2, 4, 3)), np.zeros(8), np.zeros(8), np.zeros(8)
        labels, active = np.zeros((2, 4), dtype=gs.np_int), np.ones((2, 4), dtype=bool)
        if step == 4:
            active[:, 0] = False
        for env in range(4):
            for identity, face in enumerate([46, 81]):
                slot = (identity + step + env) % 2
                patch = slot * 4 + env
                labels[slot, env] = identity
                shared = indices[face, :2]
                if step >= 6:
                    face = neighbors[face, 0]
                fraction = (abs(step - 5.5) + 0.1 * env) * 0.0001
                bary = np.full(3, fraction)
                bary[np.isin(indices[face, :3], shared)] = (1 - fraction) / 2
                values, gradients, _ = quadratic_basis(bary[None])
                nodes = vertices[indices[face]]
                point = (values @ nodes)[0]
                derivatives = np.einsum("qna,ni->qia", gradients, nodes)[0]
                normal = np.cross(derivatives[:, 0], derivatives[:, 1])
                normal *= np.sign(normal @ face_normals[face])
                normal /= np.linalg.norm(normal)
                tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
                tangent /= np.linalg.norm(tangent)
                basis = np.column_stack((tangent, -np.cross(normal, tangent), -normal))
                phase = 0.23 * step + 0.17 * env + 0.2 * identity
                load = 0.006 * (1 + 0.2 * np.sin(phase))
                radius = 3e-5 * (load / 0.006) ** (1 / 3)
                tangential = load * 0.2 * np.array([np.cos(phase), np.sin(phase)])
                centers[patch], bases[patch], anchors[patch] = point, basis, face
                if active[slot, env]:
                    forces[slot, env] = basis @ np.r_[tangential, load]
                    normal_load[patch], radii[patch] = load, radius
                    angle = 0.0
                    if has_previous[env, identity]:
                        old = previous_basis[env, identity]
                        axis = np.cross(old[:, 2], basis[:, 2])
                        magnitude = np.linalg.norm(axis)
                        transport = (
                            Rotation.identity()
                            if magnitude == 0
                            else Rotation.from_rotvec(axis / magnitude * np.arctan2(magnitude, old[:, 2] @ basis[:, 2]))
                        )
                        rotated = transport.apply(old[:, 0])
                        angle = np.arctan2(rotated @ basis[:, 1], rotated @ basis[:, 0])
                    expected_angles[patch] = angle
                    references[env][identity].update(radius, load, 0.4, tangential, angle)
                    previous_basis[env, identity], has_previous[env, identity] = basis, True
                else:
                    references[env][identity] = ConeFrictionHistory(16)
                    has_previous[env, identity] = False
        contacts.valid.from_numpy(active)
        contacts.force.from_numpy(forces)
        sources.pair.from_numpy(labels)
        regions.force.from_numpy(forces)
        regions.source_count.from_numpy(active.astype(gs.np_int))
        regions.leader.from_numpy(np.broadcast_to(np.arange(2, dtype=gs.np_int)[:, None], (2, 4)).copy())
        frames.center.from_numpy(centers)
        frames.basis.from_numpy(bases)
        frames.anchor.from_numpy(anchors)
        law.force.from_numpy(normal_load)
        law.axes.from_numpy(np.repeat(radii[:, None], 2, axis=1))
        kernel_friction_bind(
            contacts,
            sources,
            regions,
            law,
            frames,
            binding,
            friction.input,
            shear,
            status,
            valid,
            model.info,
            surface.info,
            float(np.finfo(gs.np_float).eps),
        )
        friction.update()
        kernel_friction_commit(
            contacts, sources, regions, law, frames, binding, friction.input, friction.state, status, valid, errno
        )
        is_passed = bool(array(valid).all() and not array(binding.is_lost).any())
        row = {"step": step, "input_valid": array(valid).tolist(), "input_status": array(status).tolist()}
        if args.expect_baseline_failure and not is_passed:
            write_json(
                args.output / "result.json",
                {
                    "passed": True,
                    "expected_baseline_failure_step": step,
                    "rows": [*rows, row],
                    "whole_goal_complete": False,
                },
            )
            return
        assert is_passed, row
        np.testing.assert_allclose(array(friction.input.angle), expected_angles, rtol=0, atol=1e-12)
        line = array(friction.state.line)
        error = 0.0
        for env in range(4):
            for identity in range(2):
                slot = (identity + step + env) % 2
                error = max(
                    error, float(np.max(np.linalg.norm(line[slot * 4 + env] - references[env][identity].line, axis=1)))
                )
        assert error < 1e-5, (step, error)
        row["maximum_line_error"] = error
        rows.append(row)
    assert not args.expect_baseline_failure
    np.savez_compressed(
        args.output / "final.npz",
        line=line,
        centers_m=centers,
        bases=bases,
        anchors=anchors,
        previous_index=array(friction.input.previous_index),
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "crossed_shared_edges": 8,
            "release_and_recontact_environment": 0,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
