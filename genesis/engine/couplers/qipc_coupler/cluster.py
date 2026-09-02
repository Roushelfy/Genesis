"""Queued FEM-cluster declarations for the QIPC coupler.

Genesis entities exist before the QIPC scene does, while QIPC requires cluster
proxies to be declared before ``Scene.init()`` and membership to change only
after init. This module bridges that lifecycle for both proxy kinds QIPC offers
-- the 12-DOF affine body of ``Scene.add_affine_cluster`` and the exact 6-DOF
rigid body of ``Scene.add_rigid_cluster`` -- without exposing QIPC objects as
the public handle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

import genesis as gs


@dataclass(frozen=True)
class AffineClusterProxy:
    """Carry the clustered interior on a 12-DOF affine body.

    ``kappa`` scales the orthogonality potential of an implicit (ghost) proxy;
    an explicit rigid-entity proxy keeps its own ``qipc_abd_kappa``.
    """

    kappa: float


@dataclass(frozen=True)
class RigidClusterProxy:
    """Carry the clustered interior on an exact 6-DOF rigid body.

    Rigidity comes from the quaternion parameterization, so there is no
    stiffness knob. An explicit proxy must be a ``qipc_rigid_body`` entity.
    QIPC cannot ``Scene.reset()`` a scene that declares a rigid cluster yet.
    """


ClusterProxy = AffineClusterProxy | RigidClusterProxy


def cluster_entry_point(proxy: ClusterProxy) -> str:
    """Name of the coupler method that queues a cluster of this proxy kind."""
    if isinstance(proxy, AffineClusterProxy):
        return "add_affine_cluster"
    if isinstance(proxy, RigidClusterProxy):
        return "add_rigid_cluster"
    gs.raise_exception(f"QIPCCoupler: unsupported cluster proxy {proxy!r}.")


class _ElementSelection(NamedTuple):
    edges: tuple[int, ...] | None
    tris: tuple[int, ...] | None
    tets: tuple[int, ...] | None

    @property
    def is_explicit(self) -> bool:
        return self.edges is not None or self.tris is not None or self.tets is not None


class _ClusterRequest(NamedTuple):
    fem_entity: object
    proxy_entity: object | None
    proxy_link: object | str | None
    proxy: ClusterProxy
    fixed: bool
    initial: _ElementSelection


@dataclass
class _ClusterBinding:
    collection: object
    fem_slot: object
    fem_vertex_range: range | None = None
    proxy_body_index: int | None = None


_SCENE_CAPABILITIES = ("add_affine_cluster", "add_rigid_cluster", "reset")
_MANAGER_CAPABILITIES = (
    "drstate_proxy_member",
    "drstate_fem_masses",
    "drstate_abd_mass_inv",
    "drstate_global_body_id",
    "restore_drstate",
)


def _missing_cluster_capabilities(scene, native_manager_type: type) -> tuple[str, ...]:
    """Return missing QIPC API names without touching CUDA state."""
    missing = [f"Scene.{name}" for name in _SCENE_CAPABILITIES if not hasattr(scene, name)]
    missing.extend(
        f"FEMClusterManager.{name}" for name in _MANAGER_CAPABILITIES if not hasattr(native_manager_type, name)
    )
    return tuple(missing)


def _require_cluster_capability(scene) -> None:
    try:
        from qipc._src.native.solver import FEMClusterManager
    except ImportError as error:
        gs.raise_exception_from(
            "QIPCCoupler FEM clusters require a current cuda-graph-qipc build (FEMClusterManager with "
            "rigid clusters). Install the Genesis 'qipc' extra in its supported Python 3.12 environment.",
            error,
        )

    missing = _missing_cluster_capabilities(scene, FEMClusterManager)
    if missing:
        gs.raise_exception(
            "QIPCCoupler FEM clusters require QIPC affine+rigid clusters with membership reset/restore "
            f"support (missing: {', '.join(missing)}). Rebuild cuda-graph-qipc from the supported branch."
        )


def _normalize_element_indices(values, *, name: str, entry: str) -> tuple[int, ...] | None:
    """Normalize one entity-local element selection to an immutable tuple."""
    if values is None:
        return None
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        gs.raise_exception(f"QIPCCoupler.{entry}: {name} must be a one-dimensional index array.")
    if array.size == 0:
        return ()
    if array.dtype.kind not in "iu":
        gs.raise_exception(f"QIPCCoupler.{entry}: {name} must contain integer indices.")
    if int(array.min()) < 0:
        gs.raise_exception(f"QIPCCoupler.{entry}: {name} indices must be non-negative.")
    if int(array.max()) > np.iinfo(np.int32).max:
        gs.raise_exception(f"QIPCCoupler.{entry}: {name} indices must fit in int32.")
    normalized = tuple(int(index) for index in array)
    if len(set(normalized)) != len(normalized):
        gs.raise_exception(f"QIPCCoupler.{entry}: {name} must not contain duplicate indices.")
    return normalized


def _selection(*, edges=None, tris=None, tets=None, entry: str) -> _ElementSelection:
    return _ElementSelection(
        edges=_normalize_element_indices(edges, name="edges", entry=entry),
        tris=_normalize_element_indices(tris, name="tris", entry=entry),
        tets=_normalize_element_indices(tets, name="tets", entry=entry),
    )


class QIPCCluster:
    """Stable Genesis handle for one queued QIPC FEM cluster (affine or rigid).

    ``join`` and ``detach`` consume entity-local primitive indices. Passing no
    selection at runtime targets every FEM primitive owned by the entity.
    """

    __slots__ = ("_manager", "_request_index")

    def __init__(self, manager: QIPCClusterManager, request_index: int) -> None:
        self._manager = manager
        self._request_index = request_index

    @property
    def proxy(self) -> ClusterProxy:
        """The proxy kind this cluster was queued with."""
        return self._manager._requests[self._request_index].proxy

    @property
    def fem_vertex_range(self) -> range:
        """Entity vertices in QIPC's FEM-local vertex buffer, as ``[start, stop)``."""
        return self._manager._fem_vertex_range(self._request_index)

    @property
    def proxy_body_index(self) -> int:
        """Proxy row in QIPC's affine-body (affine) or rigid-body (rigid) state arrays."""
        return self._manager._proxy_body_index(self._request_index)

    @property
    def member_count(self) -> int:
        """Number of FEM elements currently represented by the cluster."""
        return self._manager._member_count(self._request_index)

    @property
    def qipc_collection(self):
        """The bound qipc ``AffineClusterCollection`` / ``RigidClusterCollection`` (post-build).

        Release policies (``qipc.cluster_release_driver``) drive membership
        through this raw handle; driver-internal element indices skip the
        manager's user-input validation by design.
        """
        return self._manager._binding(self._request_index).collection

    def join(self, *, edges=None, tris=None, tets=None) -> None:
        """Join entity-local FEM elements to this cluster between steps."""
        self._manager._membership_event(
            self._request_index,
            join=True,
            selection=_selection(edges=edges, tris=tris, tets=tets, entry=cluster_entry_point(self.proxy)),
        )

    def detach(self, *, edges=None, tris=None, tets=None) -> None:
        """Detach entity-local FEM elements from this cluster between steps."""
        self._manager._membership_event(
            self._request_index,
            join=False,
            selection=_selection(edges=edges, tris=tris, tets=tets, entry=cluster_entry_point(self.proxy)),
        )


class QIPCClusterManager:
    """Own queued declarations, QIPC bindings, and authored reset replay."""

    def __init__(self) -> None:
        self._requests: list[_ClusterRequest] = []
        self._bindings: list[_ClusterBinding] = []
        self._scene = None
        self._initialized = False

    @property
    def has_rigid_proxy(self) -> bool:
        """True when any queued cluster rides a rigid body; QIPC cannot reset such a scene yet."""
        return any(isinstance(request.proxy, RigidClusterProxy) for request in self._requests)

    def add_request(
        self,
        fem_entity,
        *,
        proxy: ClusterProxy,
        proxy_entity=None,
        proxy_link=None,
        fixed: bool,
        initial_edges=None,
        initial_tris=None,
        initial_tets=None,
    ) -> QIPCCluster:
        entry = cluster_entry_point(proxy)
        if self._scene is not None:
            gs.raise_exception(f"QIPCCoupler.{entry} must be called before scene.build().")
        if isinstance(proxy, AffineClusterProxy):
            try:
                kappa_scalar = float(proxy.kappa)
            except (OverflowError, TypeError, ValueError):
                kappa_scalar = float("nan")
            if isinstance(proxy.kappa, (bool, np.bool_)) or not np.isfinite(kappa_scalar) or kappa_scalar <= 0.0:
                gs.raise_exception(f"QIPCCoupler.{entry}: kappa must be finite and positive.")
            proxy = AffineClusterProxy(kappa=kappa_scalar)
        if not isinstance(fixed, (bool, np.bool_)):
            gs.raise_exception(f"QIPCCoupler.{entry}: fixed must be a bool.")
        if proxy_entity is None and proxy_link is not None:
            gs.raise_exception(f"QIPCCoupler.{entry}: proxy_link requires proxy_entity.")
        if proxy_entity is not None and fixed:
            gs.raise_exception(
                f"QIPCCoupler.{entry}: fixed applies only to an implicit cluster proxy; "
                "configure an explicit rigid proxy through its own fixed-base settings."
            )

        request_index = len(self._requests)
        self._requests.append(
            _ClusterRequest(
                fem_entity=fem_entity,
                proxy_entity=proxy_entity,
                proxy_link=proxy_link,
                proxy=proxy,
                fixed=bool(fixed),
                initial=_selection(edges=initial_edges, tris=initial_tris, tets=initial_tets, entry=entry),
            )
        )
        return QIPCCluster(self, request_index)

    def declare(
        self,
        scene,
        *,
        fem_slots: dict[object, object],
        resolve_proxy_slot: Callable[[object, object | str | None, ClusterProxy], object],
    ) -> None:
        """Declare every queued proxy before QIPC ``Scene.init``."""
        if not self._requests:
            return
        _require_cluster_capability(scene)
        self._scene = scene
        used_proxy_slot_ids: set[int] = set()
        for index, request in enumerate(self._requests):
            entry = cluster_entry_point(request.proxy)
            fem_slot = fem_slots.get(request.fem_entity)
            if fem_slot is None:
                gs.raise_exception(f"QIPCCoupler.{entry}: FEM entity is not coupled in this scene.")
            proxy_slot = None
            if request.proxy_entity is not None:
                proxy_slot = resolve_proxy_slot(request.proxy_entity, request.proxy_link, request.proxy)
                proxy_slot_id = int(proxy_slot.id)
                if proxy_slot_id in used_proxy_slot_ids:
                    gs.raise_exception(f"QIPCCoupler.{entry}: one rigid body cannot proxy multiple clusters.")
                used_proxy_slot_ids.add(proxy_slot_id)
            if isinstance(request.proxy, AffineClusterProxy):
                collection = scene.add_affine_cluster(
                    name=f"genesis_affine_cluster_{index}",
                    proxy=proxy_slot,
                    fem=fem_slot,
                    kappa=request.proxy.kappa,
                    fixed=request.fixed,
                )
            else:
                collection = scene.add_rigid_cluster(
                    name=f"genesis_rigid_cluster_{index}",
                    proxy=proxy_slot,
                    fem=fem_slot,
                    fixed=request.fixed,
                )
            self._bindings.append(_ClusterBinding(collection=collection, fem_slot=fem_slot))

    def initialize(self) -> None:
        """Resolve post-init layout and apply authored initial membership."""
        if not self._requests:
            self._initialized = True
            return
        if self._scene is None or len(self._bindings) != len(self._requests):
            gs.raise_exception("QIPCCoupler FEM clusters were not declared before QIPC init.")

        for request, binding in zip(self._requests, self._bindings, strict=True):
            geometry = binding.fem_slot.geometry
            offset = int(geometry.meta["fem_vert_offset"].cpu()[0])
            binding.fem_vertex_range = range(offset, offset + int(geometry.vertices.size))
            # Each collection kind resolves `bodies` in its own index space:
            # ABD rows for affine proxies, rigid rows for rigid proxies.
            binding.proxy_body_index = int(binding.collection.bodies[0])
            self._validate_selection(binding, request.initial)

        for request, binding in zip(self._requests, self._bindings, strict=True):
            if request.initial.is_explicit:
                binding.collection.join(
                    edges=request.initial.edges,
                    tris=request.initial.tris,
                    tets=request.initial.tets,
                )
        self._initialized = True

    def replay_initial_membership(self) -> None:
        """Replay authored membership after raw QIPC reset clears every cluster."""
        if not self._requests:
            return
        self._require_initialized()
        for request, binding in zip(self._requests, self._bindings, strict=True):
            if request.initial.is_explicit:
                binding.collection.join(
                    edges=request.initial.edges,
                    tris=request.initial.tris,
                    tets=request.initial.tets,
                )

    def _membership_event(self, request_index: int, *, join: bool, selection: _ElementSelection) -> None:
        self._require_initialized()
        binding = self._bindings[request_index]
        self._validate_selection(binding, selection)
        method = binding.collection.join if join else binding.collection.detach
        method(edges=selection.edges, tris=selection.tris, tets=selection.tets)

    def _validate_selection(self, binding: _ClusterBinding, selection: _ElementSelection) -> None:
        geometry = binding.fem_slot.geometry
        layouts = (
            ("edges", selection.edges, "fem_edge_offset", int(geometry.edges.size)),
            ("tris", selection.tris, "fem_tri_offset", int(geometry.triangles.size)),
            ("tets", selection.tets, "fem_tet_offset", int(geometry.tetrahedra.size)),
        )
        if not selection.is_explicit:
            if not any(meta_name in geometry.meta for _, _, meta_name, _ in layouts):
                gs.raise_exception("QIPCCoupler FEM cluster entity has no supported elements.")
            return
        for name, indices, meta_name, count in layouts:
            if indices is None:
                continue
            if meta_name not in geometry.meta:
                gs.raise_exception(f"QIPCCoupler FEM cluster entity has no QIPC {name} elements.")
            if indices and max(indices) >= count:
                gs.raise_exception(f"QIPCCoupler FEM cluster {name} index {max(indices)} is out of range [0, {count}).")

    def _require_initialized(self) -> None:
        if not self._initialized:
            gs.raise_exception("QIPC FEM-cluster runtime state is only available after scene.build().")

    def _binding(self, request_index: int) -> _ClusterBinding:
        self._require_initialized()
        return self._bindings[request_index]

    def _collection(self, request_index: int):
        return self._binding(request_index).collection

    def _fem_vertex_range(self, request_index: int) -> range:
        value = self._binding(request_index).fem_vertex_range
        assert value is not None
        return value

    def _proxy_body_index(self, request_index: int) -> int:
        value = self._binding(request_index).proxy_body_index
        assert value is not None
        return value

    def _member_count(self, request_index: int) -> int:
        return int(self._collection(request_index).n_member_elems[0])
