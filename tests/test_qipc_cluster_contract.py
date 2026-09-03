"""CPU-only contracts for queued QIPC FEM clusters (affine and rigid proxies)."""

from types import SimpleNamespace

import numpy as np
import pytest


class _CpuScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def cpu(self) -> np.ndarray:
        return np.array([self._value], dtype=np.int32)


class _Collection:
    def __init__(self, events: list[tuple], body_index: int) -> None:
        self.bodies = [body_index]
        self.n_member_elems = [2]
        self.events = events

    def join(self, **kwargs) -> None:
        self.events.append(("join", kwargs))

    def detach(self, **kwargs) -> None:
        self.events.append(("detach", kwargs))


class _Scene:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.collection = _Collection(events, body_index=12)
        self.rigid_collection = _Collection(events, body_index=0)

    def add_affine_cluster(self, **kwargs):
        self.events.append(("declare", kwargs))
        return self.collection

    def add_rigid_cluster(self, **kwargs):
        self.events.append(("declare_rigid", kwargs))
        return self.rigid_collection

    def reset(self) -> None:
        self.events.append(("scene.reset", {}))


def _triangle_slot(*, offset: int = 4, n_vertices: int = 6, n_triangles: int = 4):
    geometry = SimpleNamespace(
        meta={"fem_vert_offset": _CpuScalar(offset), "fem_tri_offset": _CpuScalar(0)},
        vertices=SimpleNamespace(size=n_vertices),
        edges=SimpleNamespace(size=0),
        triangles=SimpleNamespace(size=n_triangles),
        tetrahedra=SimpleNamespace(size=0),
    )
    return SimpleNamespace(id=3, geometry=geometry)


def _cluster_module():
    from genesis.engine.couplers.qipc_coupler import cluster

    return cluster


def test_capability_contract_reports_missing_state_surface():
    cluster_module = _cluster_module()

    class IncompleteScene:
        def reset(self):
            pass

    class IncompleteManager:
        def drstate_proxy_member(self):
            pass

    missing = cluster_module._missing_cluster_capabilities(IncompleteScene(), IncompleteManager)

    assert missing == (
        "Scene.add_affine_cluster",
        "Scene.add_rigid_cluster",
        "FEMClusterManager.drstate_fem_masses",
        "FEMClusterManager.drstate_abd_mass_inv",
        "FEMClusterManager.drstate_global_body_id",
        "FEMClusterManager.restore_drstate",
    )


def test_queued_handle_binds_and_replays_initial_membership(monkeypatch):
    cluster_module = _cluster_module()

    monkeypatch.setattr(cluster_module, "_require_cluster_capability", lambda _scene: None)
    events: list[tuple] = []
    scene = _Scene(events)
    fem_entity = object()
    fem_slot = _triangle_slot()
    manager = cluster_module.QIPCClusterManager()
    handle = manager.add_request(
        fem_entity,
        proxy=cluster_module.AffineClusterProxy(kappa=2.5e7),
        fixed=False,
        initial_tris=np.array([1, 3], dtype=np.int64),
    )

    with pytest.raises(Exception, match="only available after scene.build"):
        _ = handle.fem_vertex_range

    manager.declare(
        scene,
        fem_slots={fem_entity: fem_slot},
        resolve_proxy_slot=lambda _entity, _link, _proxy: None,
    )
    assert events[0][0] == "declare"
    assert events[0][1]["fem"] is fem_slot
    assert events[0][1]["proxy"] is None
    assert events[0][1]["kappa"] == 2.5e7
    assert events[0][1]["fixed"] is False

    manager.initialize()
    assert handle.fem_vertex_range == range(4, 10)
    assert handle.proxy_body_index == 12
    assert handle.member_count == 2
    assert events[1] == (
        "join",
        {"edges": None, "tris": (1, 3), "tets": None},
    )

    handle.detach(tris=3)
    assert events[2] == (
        "detach",
        {"edges": None, "tris": (3,), "tets": None},
    )
    manager.replay_initial_membership()
    assert events[3] == events[1]


def test_rigid_request_declares_a_rigid_cluster_without_stiffness(monkeypatch):
    cluster_module = _cluster_module()

    monkeypatch.setattr(cluster_module, "_require_cluster_capability", lambda _scene: None)
    events: list[tuple] = []
    scene = _Scene(events)
    fem_entity = object()
    resolved: list[tuple] = []
    manager = cluster_module.QIPCClusterManager()
    handle = manager.add_request(
        fem_entity,
        proxy=cluster_module.RigidClusterProxy(),
        fixed=False,
        initial_tris=[0, 2],
    )
    assert handle.proxy == cluster_module.RigidClusterProxy()

    def resolve_proxy_slot(entity, link, proxy):
        resolved.append((entity, link, proxy))
        return None

    manager.declare(scene, fem_slots={fem_entity: _triangle_slot()}, resolve_proxy_slot=resolve_proxy_slot)
    # A ghost proxy never consults the resolver; the declaration carries no kappa.
    assert resolved == []
    assert events[0] == (
        "declare_rigid",
        {"name": "genesis_rigid_cluster_0", "proxy": None, "fem": events[0][1]["fem"], "fixed": False},
    )

    manager.initialize()
    assert handle.proxy_body_index == 0
    assert events[1] == ("join", {"edges": None, "tris": (0, 2), "tets": None})


def test_omitted_initial_membership_is_empty_but_runtime_omission_targets_all(monkeypatch):
    cluster_module = _cluster_module()

    monkeypatch.setattr(cluster_module, "_require_cluster_capability", lambda _scene: None)
    events: list[tuple] = []
    fem_entity = object()
    manager = cluster_module.QIPCClusterManager()
    handle = manager.add_request(fem_entity, proxy=cluster_module.AffineClusterProxy(kappa=1e8), fixed=False)
    manager.declare(
        _Scene(events),
        fem_slots={fem_entity: _triangle_slot()},
        resolve_proxy_slot=lambda _entity, _link, _proxy: None,
    )

    manager.initialize()
    manager.replay_initial_membership()
    assert [event for event in events if event[0] == "join"] == []

    handle.join()
    assert events[-1] == ("join", {"edges": None, "tris": None, "tets": None})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kappa": 0.0}, "kappa"),
        ({"kappa": "invalid"}, "kappa"),
        ({"fixed": 1}, "fixed"),
        ({"proxy_link": "tool"}, "proxy_link requires"),
        ({"initial_tris": [0, 0]}, "duplicate"),
        ({"initial_tris": [-1]}, "non-negative"),
        ({"initial_tris": [0.5]}, "integer"),
        ({"initial_tris": [[0, 1]]}, "one-dimensional"),
    ],
)
def test_declaration_parameter_validation(kwargs, message):
    cluster_module = _cluster_module()

    params = {"kappa": 1e8, "fixed": False, **kwargs}
    proxy = cluster_module.AffineClusterProxy(kappa=params.pop("kappa"))
    with pytest.raises(Exception, match=message):
        cluster_module.QIPCClusterManager().add_request(object(), proxy=proxy, **params)


def test_initial_membership_range_is_validated_before_any_join(monkeypatch):
    cluster_module = _cluster_module()

    monkeypatch.setattr(cluster_module, "_require_cluster_capability", lambda _scene: None)
    events: list[tuple] = []
    scene = _Scene(events)
    fem_entity = object()
    manager = cluster_module.QIPCClusterManager()
    manager.add_request(
        fem_entity,
        proxy=cluster_module.AffineClusterProxy(kappa=1e8),
        fixed=False,
        initial_tris=[4],
    )
    manager.declare(
        scene,
        fem_slots={fem_entity: _triangle_slot(n_triangles=4)},
        resolve_proxy_slot=lambda _entity, _link, _proxy: None,
    )

    with pytest.raises(Exception, match=r"out of range \[0, 4\)"):
        manager.initialize()
    assert [event for event in events if event[0] == "join"] == []


def test_coupler_rejects_foreign_entities_links_and_late_declarations():
    from genesis.engine.couplers.qipc_coupler.cluster import QIPCClusterManager
    from genesis.engine.couplers.qipc_coupler.coupler import QIPCCoupler

    fem_entity = object()
    owned_link = SimpleNamespace(name="owned", idx_local=0, geoms=[])
    proxy_entity = SimpleNamespace(links=[owned_link])
    coupler = object.__new__(QIPCCoupler)
    coupler._scene = None
    coupler._sim = SimpleNamespace(
        fem_solver=SimpleNamespace(entities=[fem_entity]),
        rigid_solver=SimpleNamespace(entities=[proxy_entity]),
    )
    coupler._clusters = QIPCClusterManager()

    with pytest.raises(Exception, match="add_affine_cluster: fem_entity is not a FEM entity owned"):
        coupler.add_affine_cluster(object())
    with pytest.raises(Exception, match="add_rigid_cluster: fem_entity is not a FEM entity owned"):
        coupler.add_rigid_cluster(object())
    with pytest.raises(Exception, match="proxy_entity is not a rigid entity owned"):
        coupler.add_affine_cluster(fem_entity, proxy_entity=object())
    with pytest.raises(Exception, match="selected link does not belong"):
        coupler.add_rigid_cluster(
            fem_entity,
            proxy_entity=proxy_entity,
            proxy_link=SimpleNamespace(name="foreign", idx_local=1, geoms=[]),
        )

    coupler._scene = object()
    with pytest.raises(Exception, match=r"add_rigid_cluster must be called before scene\.build"):
        coupler.add_rigid_cluster(fem_entity)


def test_reset_replays_bonds_before_initial_membership():
    from genesis.engine.couplers.qipc_coupler.coupler import QIPCCoupler

    calls: list[str] = []
    coupler = object.__new__(QIPCCoupler)
    coupler._scene = SimpleNamespace(reset=lambda: calls.append("scene.reset"))
    coupler._adhesion = SimpleNamespace(restore_seeded_bonds=lambda: calls.append("bonds"))
    coupler._clusters = SimpleNamespace(replay_initial_membership=lambda: calls.append("membership"))
    coupler._sealed_gas_bag_by_entity = {}
    coupler._sealed_gas_reset_state = {}
    coupler._rigid_reset_state = None
    coupler._writeback_state = lambda: calls.append("rigid writeback")
    coupler._writeback_fem_state = lambda frame: calls.append(f"fem writeback {frame}")

    coupler.reset()

    assert calls == [
        "scene.reset",
        "bonds",
        "membership",
        "rigid writeback",
        "fem writeback 0",
    ]
