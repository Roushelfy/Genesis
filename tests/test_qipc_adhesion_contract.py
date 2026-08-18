"""CPU-only contracts for authored QIPC distance-bond batches."""

from types import SimpleNamespace

import numpy as np
import pytest


def _manager_with_scene(*, fem_global_offset: int = 20):
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=fem_global_offset))
    calls: list[tuple[np.ndarray, float]] = []
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))
    return manager, calls


def test_named_rigid_seed_batches_map_and_seed_as_one_transaction():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    hub = object()
    table = object()
    internal = manager.add_bond_seed_request(
        tape,
        name="internal",
        rigid_entity=hub,
        topologies=np.array([[0, 3, 4, 5], [3, 4, 5, 6]], dtype=np.int32),
        source_fem_global_offset=3,
        rest_height=0.0,
        strict_rigid_mapping=True,
    )
    attached = manager.add_bond_seed_request(
        tape,
        name="table",
        rigid_entity=table,
        topologies=np.array([[2, 4, 5, 6], [4, 6, 7, 8]], dtype=np.int32),
        source_fem_global_offset=4,
        rest_height=0.0,
        strict_rigid_mapping=True,
    )
    calls: list[tuple[np.ndarray, float]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))

    manager.apply_bond_seed_requests(
        {tape: (5, 8)},
        {
            hub: np.array([100, 101, 102], dtype=np.int32),
            table: np.array([200, 201, 202, 203], dtype=np.int32),
        },
    )

    expected_internal = np.array([[100, 25, 26, 27], [25, 26, 27, 28]], dtype=np.int32)
    expected_table = np.array([[202, 25, 26, 27], [25, 27, 28, 29]], dtype=np.int32)
    assert internal.key == 0
    assert internal.name == "internal"
    assert internal.fem_entity is tape
    assert internal.rigid_entity is hub
    assert internal.source_fem_global_offset == 3
    assert internal.rest_height == 0.0
    assert internal.strict_rigid_mapping is True
    assert attached.key == 1
    assert manager.get_bond_seed_handle(tape, name="internal") is internal
    assert manager.get_bond_seed_handle(tape, name="table") is attached
    assert manager.get_bond_seed_result(internal) == (2, 0)
    assert manager.get_bond_seed_result(attached) == (2, 0)
    np.testing.assert_array_equal(manager.get_bond_seed_topologies(internal), expected_internal)
    np.testing.assert_array_equal(manager.get_bond_seed_topologies(attached), expected_table)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.concatenate((expected_internal, expected_table)))
    assert calls[0][1] == 0.0

    with pytest.raises(Exception, match="ambiguous.*'internal'.*'table'"):
        manager.get_bond_seed_result(tape)
    with pytest.raises(Exception, match="ambiguous.*'internal'.*'table'"):
        manager.get_bond_seed_topologies(tape)

    mapped = manager.get_bond_seed_topologies(internal)
    assert mapped is not None
    mapped.fill(-1)
    np.testing.assert_array_equal(manager.get_bond_seed_topologies(internal), expected_internal)

    manager.restore_seeded_bonds()
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], calls[0][0])

    manager.mark_bond_state_captured_in_reset()
    manager.restore_seeded_bonds()
    assert len(calls) == 2


def test_named_frozen_state_batches_map_and_restore_as_one_transaction():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    hub = object()
    table = object()
    internal_Dm_inv = np.stack((np.eye(3), 2.0 * np.eye(3))).reshape(2, 9)
    internal_V0 = np.array([1.0 / 6.0, 1.0 / 48.0])
    internal = manager.add_bond_state_request(
        tape,
        name="internal",
        rigid_source=hub,
        topologies=np.array([[0, 3, 4, 5], [3, 4, 5, 6]], dtype=np.int32),
        source_fem_global_offset=3,
        Dm_inv=internal_Dm_inv,
        V0=internal_V0,
        d_rest=np.array([3.0, 4.0]),
        kappa=np.array([5.0, 6.0]),
        release_force=np.array([7.0, 8.0]),
        release_strain=np.array([9.0, 10.0]),
        release_gap=np.array([11.0, 12.0]),
        release_slip=np.array([13.0, 14.0]),
        age=np.array([15, 16], dtype=np.int32),
    )
    attached = manager.add_bond_state_request(
        tape,
        name="table",
        rigid_source=table,
        topologies=np.array([[2, 4, 5, 6]], dtype=np.int32),
        source_fem_global_offset=4,
        Dm_inv=(3.0 * np.eye(3)).reshape(1, 9),
        V0=np.array([1.0 / 162.0]),
        d_rest=np.array([18.0]),
        kappa=np.array([19.0]),
        release_force=np.array([20.0]),
        release_strain=np.array([21.0]),
        release_gap=np.array([22.0]),
        release_slip=np.array([23.0]),
        age=np.array([24], dtype=np.int32),
    )
    internal_Dm_inv.fill(-1.0)
    internal_V0.fill(-1.0)
    calls: list[dict[str, np.ndarray]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._restore_bond_state_batch = lambda batch: calls.append(
        {name: values.copy() for name, values in batch.items()}
    )

    manager.apply_bond_seed_requests(
        {tape: (5, 8)},
        {
            hub: np.array([100, 101, 102], dtype=np.int32),
            table: np.array([200, 201, 202, 203], dtype=np.int32),
        },
    )

    expected_internal = np.array([[100, 25, 26, 27], [25, 26, 27, 28]], dtype=np.int32)
    expected_table = np.array([[202, 25, 26, 27]], dtype=np.int32)
    assert internal.key == 0
    assert internal.name == "internal"
    assert internal.fem_entity is tape
    assert internal.rigid_entity is hub
    assert internal.source_fem_global_offset == 3
    assert internal.strict_rigid_mapping is True
    assert attached.key == 1
    assert manager.get_bond_state_handle(tape, name="internal") is internal
    assert manager.get_bond_state_handle(tape, name="table") is attached
    assert manager.get_bond_state_result(internal) == (2, 0)
    assert manager.get_bond_state_result(attached) == (1, 0)
    assert manager.get_bond_seed_result(internal) == (2, 0)
    np.testing.assert_array_equal(manager.get_bond_state_topologies(internal), expected_internal)
    np.testing.assert_array_equal(manager.get_bond_seed_topologies(attached), expected_table)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["topo"], np.concatenate((expected_internal, expected_table)))
    np.testing.assert_array_equal(
        calls[0]["Dm_inv"],
        np.stack((np.eye(3), 2.0 * np.eye(3), 3.0 * np.eye(3))).reshape(3, 9),
    )
    np.testing.assert_array_equal(calls[0]["V0"], np.array([1.0 / 6.0, 1.0 / 48.0, 1.0 / 162.0]))
    np.testing.assert_array_equal(calls[0]["release_force"], np.array([7.0, 8.0, 20.0]))
    np.testing.assert_array_equal(calls[0]["age"], np.array([15, 16, 24], dtype=np.int32))

    with pytest.raises(Exception, match="ambiguous.*'internal'.*'table'"):
        manager.get_bond_state_result(tape)
    with pytest.raises(Exception, match="ambiguous.*'internal'.*'table'"):
        manager.get_bond_seed_topologies(tape)

    mapped = manager.get_bond_state_topologies(internal)
    assert mapped is not None
    mapped.fill(-1)
    np.testing.assert_array_equal(manager.get_bond_state_topologies(internal), expected_internal)

    manager.restore_seeded_bonds()
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1]["topo"], calls[0]["topo"])

    manager.mark_bond_state_captured_in_reset()
    manager.restore_seeded_bonds()
    assert len(calls) == 2


def test_frozen_state_registration_is_atomic_owns_inputs_and_rejects_empty_batches():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    hub = object()
    with pytest.raises(Exception, match=r"V0 must have shape \(1,\)"):
        manager.add_bond_state_request(
            tape,
            name="retry",
            rigid_source=hub,
            topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
            source_fem_global_offset=1,
            Dm_inv=np.eye(3).reshape(1, 9),
            V0=np.empty(0, dtype=np.float64),
            d_rest=np.array([0.0]),
            kappa=np.array([3.0e7]),
            release_force=np.array([8.0]),
            release_strain=np.array([9.0]),
            release_gap=np.array([10.0]),
            release_slip=np.array([11.0]),
            age=np.array([12], dtype=np.int32),
        )
    assert manager.get_bond_state_handle(tape, name="retry") is None
    assert manager._bond_state_requests == []

    topologies = np.array([[0, 1, 2, 3]], dtype=np.int32)
    Dm_inv = np.eye(3).reshape(1, 9)
    V0 = np.array([1.0 / 6.0])
    handle = manager.add_bond_state_request(
        tape,
        name="retry",
        rigid_source=hub,
        topologies=topologies,
        source_fem_global_offset=1,
        Dm_inv=Dm_inv,
        V0=V0,
        d_rest=np.array([0.0]),
        kappa=np.array([3.0e7]),
        release_force=np.array([8.0]),
        release_strain=np.array([9.0]),
        release_gap=np.array([10.0]),
        release_slip=np.array([11.0]),
        age=np.array([12], dtype=np.int32),
    )
    topologies.fill(99)
    Dm_inv.fill(99.0)
    V0.fill(99.0)

    empty = np.empty(0, dtype=np.float64)
    with pytest.raises(Exception, match="topologies must not be empty"):
        manager.add_bond_state_request(
            object(),
            rigid_source=object(),
            topologies=np.empty((0, 4), dtype=np.int32),
            source_fem_global_offset=1,
            Dm_inv=np.empty((0, 9), dtype=np.float64),
            V0=empty,
            d_rest=empty,
            kappa=empty,
            release_force=empty,
            release_strain=empty,
            release_gap=empty,
            release_slip=empty,
            age=np.empty(0, dtype=np.int32),
        )

    calls: list[dict[str, np.ndarray]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=10))
    manager._restore_bond_state_batch = lambda batch: calls.append(
        {name: values.copy() for name, values in batch.items()}
    )
    manager.apply_bond_seed_requests(
        {tape: (0, 4)},
        {hub: np.array([42], dtype=np.int32)},
    )

    assert manager.get_bond_state_handle(tape, name="retry") is handle
    np.testing.assert_array_equal(calls[0]["topo"], [[42, 10, 11, 12]])
    np.testing.assert_array_equal(calls[0]["Dm_inv"], np.eye(3).reshape(1, 9))
    np.testing.assert_array_equal(calls[0]["V0"], [1.0 / 6.0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Dm_inv", np.zeros((1, 9)), "Dm_inv matrix must be invertible"),
        ("V0", np.array([0.0]), "V0 must be positive"),
        ("V0", np.array([0.2]), r"V0 must equal abs\(1 / det\(Dm_inv\)\) / 6"),
        ("kappa", np.array([0.0]), "kappa must be positive"),
        ("release_force", np.array([-1.0]), "release_force must be non-negative"),
        ("release_strain", np.array([-1.0]), "release_strain must be non-negative"),
        ("release_gap", np.array([-1.0]), "release_gap must be non-negative"),
        ("release_slip", np.array([-1.0]), "release_slip must be non-negative"),
        ("age", np.array([-1], dtype=np.int32), "age must be non-negative"),
    ],
)
def test_frozen_state_registration_rejects_invalid_solver_domains(field, value, message):
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    state = {
        "Dm_inv": np.eye(3).reshape(1, 9),
        "V0": np.array([1.0 / 6.0]),
        "d_rest": np.array([0.0]),
        "kappa": np.array([3.0e7]),
        "release_force": np.array([1.0]),
        "release_strain": np.array([1.0]),
        "release_gap": np.array([1.0]),
        "release_slip": np.array([1.0]),
        "age": np.array([0], dtype=np.int32),
    }
    state[field] = value

    with pytest.raises(Exception, match=message):
        manager.add_bond_state_request(
            object(),
            rigid_source=object(),
            topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
            source_fem_global_offset=1,
            **state,
        )


@pytest.mark.parametrize(
    ("topologies", "message"),
    [
        (np.array([[0, 1, 2, 3]], dtype=np.int32), "must reference a FEM vertex"),
        (
            np.array([[4, 0, 1, 5]], dtype=np.int32),
            "triangle must belong entirely to FEM or rigid vertices",
        ),
    ],
)
def test_frozen_state_registration_rejects_invalid_pt_owners(topologies, message):
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    with pytest.raises(Exception, match=message):
        manager.add_bond_state_request(
            object(),
            rigid_source=object(),
            topologies=topologies,
            source_fem_global_offset=4,
            Dm_inv=np.eye(3).reshape(1, 9),
            V0=np.array([1.0 / 6.0]),
            d_rest=np.array([0.0]),
            kappa=np.array([3.0e7]),
            release_force=np.array([1.0]),
            release_strain=np.array([1.0]),
            release_gap=np.array([1.0]),
            release_slip=np.array([1.0]),
            age=np.array([0], dtype=np.int32),
        )


@pytest.mark.parametrize("source_fem_global_offset", [True, 1.5])
def test_frozen_state_registration_rejects_non_integer_source_offset(source_fem_global_offset):
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    with pytest.raises(Exception, match="source_fem_global_offset must be an integer"):
        manager.add_bond_state_request(
            object(),
            rigid_source=object(),
            topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
            source_fem_global_offset=source_fem_global_offset,
            Dm_inv=np.eye(3).reshape(1, 9),
            V0=np.array([1.0 / 6.0]),
            d_rest=np.array([0.0]),
            kappa=np.array([3.0e7]),
            release_force=np.array([1.0]),
            release_strain=np.array([1.0]),
            release_gap=np.array([1.0]),
            release_slip=np.array([1.0]),
            age=np.array([0], dtype=np.int32),
        )


def test_strict_rigid_mapping_fails_before_native_seed():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    table = object()
    manager.add_bond_seed_request(
        tape,
        name="table",
        rigid_entity=table,
        topologies=np.array([[0, 4, 5, 6]], dtype=np.int32),
        source_fem_global_offset=4,
        rest_height=0.0,
        strict_rigid_mapping=True,
    )
    calls: list[tuple[np.ndarray, float]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))

    with pytest.raises(Exception, match="batch 'table'.*4 source vertices.*missing"):
        manager.apply_bond_seed_requests({tape: (0, 8)}, {})
    assert calls == []


def test_non_strict_rigid_mapping_reports_dropped_rows_and_keeps_fem_rows():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    handle = manager.add_bond_seed_request(
        tape,
        topologies=np.array([[0, 4, 5, 6], [4, 5, 6, 7]], dtype=np.int32),
        source_fem_global_offset=4,
        rest_height=0.0,
    )
    calls: list[tuple[np.ndarray, float]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))

    manager.apply_bond_seed_requests({tape: (2, 8)}, {})

    expected = np.array([[22, 23, 24, 25]], dtype=np.int32)
    assert manager.get_bond_seed_result(handle) == (1, 1)
    assert manager.get_bond_seed_result(tape) == (1, 1)
    np.testing.assert_array_equal(manager.get_bond_seed_topologies(handle), expected)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], expected)


def test_incompatible_rest_heights_and_duplicate_topologies_fail_before_seed():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    tape = object()
    manager = QIPCAdhesionManager(SimpleNamespace())
    manager.add_bond_seed_request(
        tape,
        name="internal",
        topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=0.0,
    )
    manager.add_bond_seed_request(
        tape,
        name="table",
        topologies=np.array([[1, 2, 3, 4]], dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=1.0e-4,
    )
    calls: list[tuple[np.ndarray, float]] = []
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))

    with pytest.raises(Exception, match="compatible rest_height.*internal.*table"):
        manager.apply_bond_seed_requests({tape: (0, 8)}, {})
    assert calls == []

    manager = QIPCAdhesionManager(SimpleNamespace())
    manager.add_bond_seed_request(
        tape,
        name="internal",
        topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=0.0,
    )
    manager.add_bond_seed_request(
        tape,
        name="table",
        topologies=np.array([[0, 3, 2, 1]], dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=0.0,
    )
    manager._scene = SimpleNamespace(affine_body=SimpleNamespace(n_verts=20))
    manager._seed_bond_batch = lambda topologies, rest_height: calls.append((topologies.copy(), rest_height))

    with pytest.raises(Exception, match="more than once.*internal.*table"):
        manager.apply_bond_seed_requests({tape: (0, 8)}, {})
    assert calls == []


def test_seed_validation_single_manual_transaction_and_reset_snapshot_ownership():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    manager = QIPCAdhesionManager(SimpleNamespace())
    tape = object()
    handle = manager.add_bond_seed_request(
        tape,
        name="internal",
        topologies=np.array([[0, 1, 2, 3]], dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=0.0,
    )
    assert handle.rest_height == 0.0
    with pytest.raises(Exception, match="name must be a non-empty string"):
        manager.add_bond_seed_request(
            object(),
            name="",
            topologies=np.empty((0, 4), dtype=np.int32),
            source_fem_global_offset=None,
            rest_height=0.0,
        )
    with pytest.raises(Exception, match="already registered"):
        manager.add_bond_seed_request(
            tape,
            name="internal",
            topologies=np.empty((0, 4), dtype=np.int32),
            source_fem_global_offset=None,
            rest_height=0.0,
        )
    with pytest.raises(Exception, match="finite and non-negative"):
        manager.add_bond_seed_request(
            object(),
            topologies=np.empty((0, 4), dtype=np.int32),
            source_fem_global_offset=None,
            rest_height=-1.0,
        )

    manual, calls = _manager_with_scene()
    topologies = np.array([[0, 1, 2, 3]], dtype=np.int32)
    manual.mark_bond_state_captured_in_reset()
    manual.seed_bonds(topologies, 0.0)
    assert len(calls) == 1
    with pytest.raises(Exception, match="already installed"):
        manual.seed_bonds(np.array([[4, 5, 6, 7]], dtype=np.int32), 0.0)
    manual.restore_seeded_bonds()
    assert len(calls) == 2

    manual.mark_bond_state_captured_in_reset()
    manual.restore_seeded_bonds()
    assert len(calls) == 2


def test_handle_from_another_manager_is_rejected():
    from genesis.engine.couplers.qipc_coupler.adhesion import QIPCAdhesionManager

    first = QIPCAdhesionManager(SimpleNamespace())
    second = QIPCAdhesionManager(SimpleNamespace())
    handle = first.add_bond_seed_request(
        object(),
        topologies=np.empty((0, 4), dtype=np.int32),
        source_fem_global_offset=None,
        rest_height=0.0,
    )

    with pytest.raises(Exception, match="different adhesion manager"):
        second.get_bond_seed_result(handle)
