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
