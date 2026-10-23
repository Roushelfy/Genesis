"""Check material binding independently of collider slot order and observation."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.hertz import StressHertzState
from genesis.engine.solvers.rigid.stress.hertz_binding import (
    StressFrictionRegions,
    kernel_friction_bind,
    kernel_friction_commit,
)
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzFrames
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.surface_walk import func_projected_surface_walk
from genesis.utils.array_class import V, V_VEC

from .artifacts import runtime_identity, write_json
from .check_live_hertz_normal import build_fixture
from .check_native_hertz_recovery import array
from .hertz_history_cone import ConeFrictionHistory


@qd.kernel
def kernel_inputs(
    i_step: int,
    i_anchor: int,
    centers: qd.Tensor,
    labels: qd.Tensor,
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressFrictionRegions,
    law: StressHertzState,
    frames: StressHertzFrames,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
    use_edge: bool,
):
    for i_b in range(contacts.valid.shape[1]):
        contacts.frame_quaternion[i_b], contacts.frame_position[i_b] = (
            qd.Vector([1.0, 0.0, 0.0, 0.0]),
            qd.Vector.zero(gs.qd_float, 3),
        )
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        center = centers[0]
        i_p = i_c * contacts.valid.shape[1] + i_b
        identity = labels[i_c, i_b]
        contacts.valid[i_c, i_b] = identity >= 0
        contacts.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        regions.force[i_c, i_b], regions.source_count[i_c, i_b], regions.leader[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            0,
            -1,
        )
        regions.profile[i_c, i_b] = 0
        law.force[i_p], law.status[i_p], law.axes[i_p] = 0.0, 1, qd.Vector.zero(gs.qd_float, 2)
        frames.anchor[i_p], frames.basis[i_p] = -1, qd.Matrix.identity(gs.qd_float, 3)
        if identity >= 0:
            phase = i_step * 0.23 + i_b * 0.17
            normal = 0.006 * (1 + 0.2 * qd.sin(phase) + 0.03 * identity)
            radius = 3e-5 * (normal / 0.006) ** (1 / 3)
            angle = 0.1 * i_step + 0.2 * identity
            force = normal * qd.Vector([0.2 * qd.cos(phase), 0.2 * qd.sin(phase), 1.0])
            position = qd.Vector([(2 * identity - 1) * 0.0008, i_step * 1e-7, -0.002])
            anchor, barycentric, projected = func_projected_surface_walk(
                i_anchor, center, position - center, qd.Vector([0.0, 0.0, -1.0]), info, surface, epsilon
            )
            contacts.force[i_c, i_b], contacts.friction[i_c, i_b] = force, 0.4
            sources.geom[i_c, i_b], sources.pair[i_c, i_b] = 0, identity
            regions.force[i_c, i_b], regions.source_count[i_c, i_b], regions.leader[i_c, i_b] = force, 1, i_c
            regions.profile[i_c, i_b] = 2 if use_edge and i_b == 0 and identity == 0 and 4 <= i_step < 8 else 1
            law.force[i_p], law.status[i_p], law.axes[i_p] = normal, 0, qd.Vector([radius, radius])
            law.direction[i_p] = qd.Vector([qd.cos(angle), qd.sin(angle)])
            frames.center[i_p], frames.anchor[i_p] = projected, anchor


@qd.kernel
def kernel_duplicate(
    i_first: int,
    i_second: int,
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressFrictionRegions,
    law: StressHertzState,
    frames: StressHertzFrames,
):
    for i_b in range(1):
        i_p, i_other = i_first * contacts.valid.shape[1], i_second * contacts.valid.shape[1]
        sources.geom[i_second, i_b], sources.pair[i_second, i_b] = (
            sources.geom[i_first, i_b],
            sources.pair[i_first, i_b],
        )
        frames.center[i_other], frames.anchor[i_other] = frames.center[i_p], frames.anchor[i_p]
        law.direction[i_other], law.axes[i_other], law.force[i_other] = (
            law.direction[i_p],
            law.axes[i_p],
            law.force[i_p],
        )
        regions.force[i_second, i_b] = regions.force[i_first, i_b]


@qd.kernel
def kernel_release(contacts: StressContactState, regions: StressFrictionRegions, law: StressHertzState):
    for i_c in range(contacts.valid.shape[0]):
        i_p = i_c * contacts.valid.shape[1]
        contacts.valid[i_c, 0], contacts.force[i_c, 0] = False, qd.Vector.zero(gs.qd_float, 3)
        regions.source_count[i_c, 0], regions.force[i_c, 0] = 0, qd.Vector.zero(gs.qd_float, 3)
        law.force[i_p], law.status[i_p] = 0.0, 1


def advance(fixture):
    entry = fixture.entry
    assert entry is not None and entry.hertz is not None
    native = entry.hertz
    native.state.status.fill(0)
    native.state.input_is_valid.fill(True)
    kernel_friction_bind(
        entry.contacts,
        native.sources,
        native.friction_regions,
        native.law.state,
        native.recovery.frames,
        native.binding,
        native.friction.input,
        native.shear,
        native.state.status,
        native.state.input_is_valid,
        entry.model.info,
        entry.surface.info,
        2.220446049250313e-16,
    )
    native.friction.update()
    kernel_friction_commit(
        entry.contacts,
        native.sources,
        native.friction_regions,
        native.law.state,
        native.recovery.frames,
        native.binding,
        native.friction.input,
        native.friction.state,
        native.state.status,
        native.state.input_is_valid,
        fixture.scene.rigid_solver._errno,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--edge-transition", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "edge_transition": args.edge_transition,
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    fixture = build_fixture(path, friction=0.4)
    fixture.scene.step()
    entry = fixture.entry
    assert entry is not None and entry.hertz is not None
    native = entry.hertz
    loaded = np.flatnonzero(array(native.law.state.force) > 0)
    anchor = int(array(native.recovery.frames.anchor)[loaded[0]])
    center = array(native.recovery.frames.center)[loaded[0]]
    native.reset(fixture.scene._sanitize_envs_idx(None), entry.contacts)
    shape = entry.contacts.valid.shape
    labels = V(dtype=gs.qd_int, shape=shape)
    centers = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    centers.from_numpy(center[None])
    slots = np.full(shape, -1, dtype=gs.np_int)
    references = [[ConeFrictionHistory(16) for _ in range(2)] for _ in range(4)]
    rows = []
    for step in range(16):
        slots.fill(-1)
        order = [0, 1] if step % 2 == 0 else [1, 0]
        slots[step % shape[0]] = order[0]
        slots[(step + 2) % shape[0]] = order[1]
        labels.from_numpy(slots)
        kernel_inputs(
            step,
            anchor,
            centers,
            labels,
            entry.contacts,
            native.sources,
            native.friction_regions,
            native.law.state,
            native.recovery.frames,
            entry.model.info,
            entry.surface.info,
            2.220446049250313e-16,
            args.edge_transition,
        )
        advance(fixture)
        assert array(native.state.input_is_valid).all() and not array(native.binding.is_lost).any()
        line, angles = array(native.friction.state.line), array(native.friction.input.angle)
        friction_status, binding_radius, shear_force = (
            array(native.friction.state.status),
            array(native.binding.radius),
            array(native.shear.force),
        )
        maximum = 0.0
        for env in range(4):
            for identity in range(2):
                slot = int(np.flatnonzero(slots[:, env] == identity)[0]) * 4 + env
                phase, angle = step * 0.23 + env * 0.17, 0.1 * step + 0.2 * identity
                normal = 0.006 * (1 + 0.2 * np.sin(phase) + 0.03 * identity)
                radius = 3e-5 * (normal / 0.006) ** (1 / 3)
                force = normal * 0.2 * np.array([np.cos(phase - angle), np.sin(phase - angle)])
                is_edge = args.edge_transition and env == 0 and identity == 0 and 4 <= step < 8
                is_new_smooth = args.edge_transition and env == 0 and identity == 0 and step == 8
                transport = -0.1 if step and not is_edge and not is_new_smooth else 0.0
                np.testing.assert_allclose(angles[slot], transport, rtol=0, atol=1e-13)
                if is_edge:
                    references[env][identity] = ConeFrictionHistory(16)
                    assert friction_status[slot] == 1
                    assert not binding_radius[slot] > 0
                    np.testing.assert_array_equal(line[slot], 0)
                else:
                    references[env][identity].update(radius, normal, 0.4, force, transport)
                np.testing.assert_allclose(shear_force[slot], force, rtol=1e-12, atol=1e-14)
                maximum = max(
                    maximum, float(np.max(np.linalg.norm(line[slot] - references[env][identity].line, axis=1)))
                )
        assert maximum < 1e-5, (step, maximum)
        rows.append({"step": step, "maximum_line": maximum})
    np.savez_compressed(
        args.output / "final.npz",
        line=line,
        labels=slots,
        previous_index=array(native.friction.input.previous_index),
        angles=angles,
    )
    active_slots = np.flatnonzero(slots[:, 0] >= 0)
    kernel_duplicate(
        int(active_slots[0]),
        int(active_slots[1]),
        entry.contacts,
        native.sources,
        native.friction_regions,
        native.law.state,
        native.recovery.frames,
    )
    advance(fixture)
    np.testing.assert_array_equal(array(native.binding.is_lost), [True, False, False, False])
    assert not array(native.state.input_is_valid)[0]
    advance(fixture)
    assert array(native.binding.is_lost)[0] and not array(native.state.input_is_valid)[0]
    kernel_release(entry.contacts, native.friction_regions, native.law.state)
    advance(fixture)
    assert not array(native.binding.is_lost).any()
    np.testing.assert_array_equal(array(native.friction.state.line).reshape(shape[0], 4, 17, 2)[:, 0], 0)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "duplicate_correspondence_loss_latch_and_release_passed": True,
            "rows": rows,
            "scope": "Native material correspondence and tangent-frame transport under synthetic region inputs, moving centers, changing normal/vector force, two regions, arbitrary slot permutation and four environments. Independent CPU radial law. This does not certify the physical moving-footprint transport approximation or collider region construction.",
            "edge_transition": args.edge_transition,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
