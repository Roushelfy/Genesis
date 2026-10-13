from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step, kernel_reset
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_tractions
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, DataKind, V, iter_data
from genesis.utils.misc import qd_to_numpy
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
def test_shared_support_arrays_participate_in_native_data_enumeration():
    items = tuple(iter_data(create_tractions(1, 1, 1, shared_support=True), "stress.tractions"))
    assert items and all(item.kind is DataKind.SCRATCH for item in items)
    assert any(item.name == "stress.tractions.support.friction" for item in items)


@pytest.mark.required
@pytest.mark.precision("64")
def test_complete_pair_streaming_release_reset_and_source_failure_isolation(record_property):
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    with np.load(Path(__file__).with_name("assets") / "nonconvex_plane_pair_tick710.npz") as saved:
        points, anchors, forces = (saved[name] for name in ("position", "anchor", "force"))
        radii, friction = saved["radius"], saved["friction"]
    options = RigidStressOptions(
        mesh=assets / "elastic.npz", contact_model="intrinsic", intrinsic_support="pair_union", method="direct"
    )
    model, n_envs = StressModel(options), 3
    surface = StressSurface(10, model.info)
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=assets / "egg_shell.urdf",
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
        )
    )
    scene.build(n_envs=n_envs)
    raw = create_contacts(12, n_envs, 0.006)
    # Interleave six actual source pairs in the raw input. The packer must
    # gather each complete pair before it places any workspace boundary.
    original = np.repeat(np.arange(2), 6)
    pairs = np.tile(np.arange(6), 2)
    scale = (pairs + 1) / 7
    raw.position.from_numpy(np.broadcast_to(points[original, None], (12, n_envs, 3)).copy())
    raw.force.from_numpy(np.broadcast_to(forces[original, None] * scale[:, None, None], (12, n_envs, 3)).copy())
    raw.radius.from_numpy(np.broadcast_to(radii[original, None], (12, n_envs)).copy())
    raw.friction.from_numpy(np.broadcast_to(friction[original, None], (12, n_envs)).copy())
    initial = np.column_stack((np.ones(12), pairs % 2 == 1, pairs == 5)).astype(gs.np_bool)
    raw.valid.from_numpy(initial)
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)), V(dtype=gs.qd_int, shape=(n_envs,))
    omega.fill(0)
    epsilon = float(np.finfo(gs.np_float).eps)
    reference = None
    isolation_differences = []
    for block_size in (12, 2, 7):
        settings = options.model_copy(update={"contact_block_size": block_size})
        streamed = IntrinsicContacts(entity.base_link, raw, model.info, surface.info, settings, None)
        assert tuple(streamed.data("stress.intrinsic"))
        streamed.raw_sources.geom.fill(entity.base_link.geoms[0].idx)
        streamed.raw_sources.pair.from_numpy(np.broadcast_to(pairs[:, None] + 100, (12, n_envs)).copy())
        streamed.raw_sources.witness.from_numpy(np.broadcast_to(anchors[original, None], (12, n_envs, 3)).copy())
        state = model.create_state(n_envs, "full")
        control = IntrinsicContacts(entity.base_link, raw, model.info, surface.info, settings, None)
        control.raw_sources.geom.fill(entity.base_link.geoms[0].idx)
        control.raw_sources.pair.from_numpy(np.broadcast_to(pairs[:, None] + 100, (12, n_envs)).copy())
        control.raw_sources.witness.from_numpy(np.broadcast_to(anchors[original, None], (12, n_envs, 3)).copy())
        control_state = model.create_state(n_envs, "full")
        raw.valid.from_numpy(initial)
        for active in (initial, initial & [False, True, True], initial):
            raw.valid.from_numpy(active)
            errno.fill(0)
            kernel_begin_step(state)
            streamed.scatter(raw, state, model.info, surface.info, settings, epsilon, errno)
            assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
            assert_equal(qd_to_numpy(errno, transpose=True), 0)
            model.recover(omega, state)
            kernel_accept(state, raw, errno)
            assert qd_to_numpy(state.valid, transpose=True).all()
            assert qd_to_numpy(state.step_valid, transpose=True).all()
            snapshot = (
                qd_to_numpy(state.force, transpose=True, copy=True),
                qd_to_numpy(state.stress_tensor, transpose=True, copy=True),
                qd_to_numpy(state.von_mises, transpose=True, copy=True),
                qd_to_numpy(state.peak, transpose=True, copy=True),
            )
            if reference is None:
                reference = snapshot
            for actual, expected in zip(snapshot, reference):
                if active[:, 0].any():
                    assert_allclose(actual, expected, rtol=1e-4, atol=1e-10 if actual is snapshot[0] else 1e-3)
                else:
                    assert_equal(actual[0], 0)
                    assert_allclose(actual[1:], expected[1:], rtol=1e-4, atol=1e-10 if actual is snapshot[0] else 1e-3)
            # Isolation compares equal solver histories with only env 0 changed.
            # The cold reference above separately checks physical consistency.
            raw.valid.from_numpy(initial)
            errno.fill(0)
            kernel_begin_step(control_state)
            control.scatter(raw, control_state, model.info, surface.info, settings, epsilon, errno)
            assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
            assert_equal(qd_to_numpy(errno, transpose=True), 0)
            model.recover(omega, control_state)
            kernel_accept(control_state, raw, errno)
            for actual, field in zip(
                snapshot, (control_state.force, control_state.stress_tensor, control_state.von_mises, control_state.peak)
            ):
                expected = qd_to_numpy(field, transpose=True, copy=True)
                isolation_differences.append(float(np.max(np.abs(actual[1:] - expected[1:]))))
                assert_allclose(actual[1:], expected[1:], rtol=1e-12, atol=1e-15 if field is control_state.force else 1e-8)
            raw.valid.from_numpy(active)
        # Partial reset clears only its observation. The next numerical pass
        # rebuilds all group metadata and the original environment returns.
        kernel_reset(np.array([0], dtype=gs.np_int), state)
        assert qd_to_numpy(state.valid, transpose=True)[0]
        assert_equal(qd_to_numpy(state.peak, transpose=True)[0], 0)
        assert np.isnan(qd_to_numpy(state.stress_tensor, transpose=True)[0]).all()
        assert_equal(qd_to_numpy(state.stress_tensor, transpose=True)[1:], snapshot[1][1:])
        errno.fill(0)
        kernel_begin_step(state)
        streamed.scatter(raw, state, model.info, surface.info, settings, epsilon, errno)
        model.recover(omega, state)
        kernel_accept(state, raw, errno)
        assert_allclose(qd_to_numpy(state.peak, transpose=True), reference[3], rtol=1e-4, atol=1e-3)
        # A zero-load source remains geometric support. A bad provenance in
        # only env 0 then invalidates its complete pair, never another env.
        zero_force = np.broadcast_to(forces[original, None] * scale[:, None, None], (12, n_envs, 3)).copy()
        zero_force[6, 0] = 0
        raw.force.from_numpy(zero_force)
        errno.fill(0)
        kernel_begin_step(state)
        streamed.scatter(raw, state, model.info, surface.info, settings, epsilon, errno)
        assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
        assert_equal(qd_to_numpy(errno, transpose=True), 0)
        witness = np.broadcast_to(anchors[original, None], (12, n_envs, 3)).copy()
        witness[6, 0] = [0.0, 0.0, 1.0]
        streamed.raw_sources.witness.from_numpy(witness)
        errno.fill(0)
        kernel_begin_step(state)
        streamed.scatter(raw, state, model.info, surface.info, settings, epsilon, errno)
        statuses = qd_to_numpy(raw.status, transpose=True)
        assert statuses[0, 0] != 0 and statuses[0, 6] != 0
        assert_equal(statuses[1:], 0)
        assert qd_to_numpy(errno, transpose=True)[0] != 0
        assert_equal(qd_to_numpy(errno, transpose=True)[1:], 0)
        model.recover(omega, state)
        kernel_accept(state, raw, errno)
        assert not qd_to_numpy(state.step_valid, transpose=True)[0]
        assert np.isnan(qd_to_numpy(state.step_peak, transpose=True)[0])
        assert np.isnan(qd_to_numpy(state.stress_tensor, transpose=True)[0]).all()
        assert qd_to_numpy(state.step_valid, transpose=True)[1:].all()
        raw.force.from_numpy(np.broadcast_to(forces[original, None] * scale[:, None, None], (12, n_envs, 3)).copy())
        streamed.raw_sources.witness.from_numpy(np.broadcast_to(anchors[original, None], (12, n_envs, 3)).copy())
        kernel_reset(np.array([0], dtype=gs.np_int), state)
        errno.fill(0)
        kernel_begin_step(state)
        streamed.scatter(raw, state, model.info, surface.info, settings, epsilon, errno)
        model.recover(omega, state)
        kernel_accept(state, raw, errno)
        assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
        assert_equal(qd_to_numpy(errno, transpose=True), 0)
        assert qd_to_numpy(state.valid, transpose=True).all()
        assert qd_to_numpy(state.step_valid, transpose=True).all()
        assert_allclose(qd_to_numpy(state.peak, transpose=True), reference[3], rtol=1e-4, atol=1e-3)
    record_property("unaffected_environment_field_differences", isolation_differences)
