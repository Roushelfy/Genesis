"""Queued per-part contact material declarations for the QIPC coupler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import genesis as gs
from genesis.utils.misc import geometric_mean, harmonic_mean

from .rigid_attachment import QIPCRigidAttachment


@dataclass(frozen=True)
class _ContactRegionRequest:
    name: str
    friction: float
    resistance: float


@dataclass(frozen=True)
class _ContactAssignment:
    region: QIPCContactRegion
    entity: object
    link_local: int | None
    attachment: QIPCRigidAttachment | None


@dataclass(frozen=True)
class _ContactPairRequest:
    first: object
    second: object
    enabled: bool
    friction: float | None
    resistance: float | None


class QIPCContactRegion:
    """Stable handle for a contact material assigned to part of a rigid body."""

    __slots__ = ("_manager", "_request_index")

    def __init__(self, manager: QIPCContactManager, request_index: int) -> None:
        self._manager = manager
        self._request_index = request_index

    def __hash__(self) -> int:
        return hash((id(self._manager), self._request_index))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QIPCContactRegion)
            and other._manager is self._manager
            and other._request_index == self._request_index
        )

    @property
    def name(self) -> str:
        return self._manager._regions[self._request_index].name


class QIPCContactManager:
    """Own contact regions, part assignments, and explicit collision rows."""

    def __init__(self) -> None:
        self._regions: list[_ContactRegionRequest] = []
        self._assignments: list[_ContactAssignment] = []
        self._pairs: list[_ContactPairRequest] = []
        self._built = False

    def add_region(self, *, name: str, friction: float, resistance: float) -> QIPCContactRegion:
        self._require_unbuilt("add_contact_region")
        if not name:
            gs.raise_exception("QIPCCoupler.add_contact_region: name must not be empty.")
        if any(request.name == name for request in self._regions):
            gs.raise_exception(f"QIPCCoupler.add_contact_region: duplicate name '{name}'.")
        friction = float(friction)
        resistance = float(resistance)
        if not np.isfinite(friction) or friction < 0.0:
            gs.raise_exception("QIPCCoupler.add_contact_region: friction must be finite and non-negative.")
        if not np.isfinite(resistance) or resistance <= 0.0:
            gs.raise_exception("QIPCCoupler.add_contact_region: resistance must be finite and positive.")
        index = len(self._regions)
        self._regions.append(_ContactRegionRequest(name=name, friction=friction, resistance=resistance))
        return QIPCContactRegion(self, index)

    def assign_region(
        self,
        region: QIPCContactRegion,
        entity,
        *,
        link_local: int | None,
        attachment: QIPCRigidAttachment | None,
    ) -> None:
        self._require_unbuilt("assign_contact_region")
        self._validate_region(region)
        if (link_local is None) == (attachment is None):
            gs.raise_exception("QIPCCoupler.assign_contact_region requires exactly one of link or attachment.")
        if attachment is not None:
            request = attachment._manager.request(attachment)
            if request.entity is not entity:
                gs.raise_exception(
                    "QIPCCoupler.assign_contact_region: attachment does not belong to the selected entity."
                )
        self._assignments.append(
            _ContactAssignment(
                region=region,
                entity=entity,
                link_local=link_local,
                attachment=attachment,
            )
        )

    def set_pair(
        self,
        first,
        second,
        *,
        enabled: bool,
        friction: float | None,
        resistance: float | None,
    ) -> None:
        self._require_unbuilt("set_contact_pair")
        if isinstance(first, QIPCContactRegion):
            self._validate_region(first)
        if isinstance(second, QIPCContactRegion):
            self._validate_region(second)
        if friction is not None and (not np.isfinite(friction) or friction < 0.0):
            gs.raise_exception("QIPCCoupler.set_contact_pair: friction must be finite and non-negative.")
        if resistance is not None and (not np.isfinite(resistance) or resistance <= 0.0):
            gs.raise_exception("QIPCCoupler.set_contact_pair: resistance must be finite and positive.")
        self._pairs.append(
            _ContactPairRequest(
                first=first,
                second=second,
                enabled=bool(enabled),
                friction=None if friction is None else float(friction),
                resistance=None if resistance is None else float(resistance),
            )
        )

    def create_regions(self, tab) -> dict[QIPCContactRegion, tuple[object, float, float]]:
        self._built = True
        infos: dict[QIPCContactRegion, tuple[object, float, float]] = {}
        for index, request in enumerate(self._regions):
            region = QIPCContactRegion(self, index)
            infos[region] = (tab.create(request.name), request.friction, request.resistance)
        return infos

    def apply_assignments(self, pre_by_entity: dict, info_by_endpoint: dict) -> None:
        assigned_by_slot: dict[int, list[range]] = {}
        for assignment in self._assignments:
            pre = pre_by_entity.get(assignment.entity)
            if pre is None:
                gs.raise_exception("QIPCCoupler.assign_contact_region: entity is not a coupled QIPC rigid entity.")
            source = assignment.attachment if assignment.attachment is not None else assignment.link_local
            location = pre.vertex_ranges.get(source)
            if location is None:
                gs.raise_exception(
                    "QIPCCoupler.assign_contact_region: selected link or attachment has no QIPC collision vertices."
                )
            rep, vertex_range = location
            slot = pre.group_slots[rep]
            elem = info_by_endpoint[assignment.region][0]
            slot_id = int(slot.id)
            ranges = assigned_by_slot.setdefault(slot_id, [])
            if any(
                max(vertex_range.start, existing.start) < min(vertex_range.stop, existing.stop) for existing in ranges
            ):
                gs.raise_exception("QIPCCoupler.assign_contact_region: collision vertex ranges overlap.")
            ranges.append(vertex_range)

            geo = slot.geometry
            if "contact_element_id" not in geo.vertices:
                geo.vertices.create("contact_element_id", np.int32, default=-1)
                element_ids = np.full(geo.vertices.size, -1, dtype=np.int32)
            else:
                element_ids = np.asarray(geo.vertices["contact_element_id"].cpu(), dtype=np.int32).copy()
            element_ids[vertex_range.start : vertex_range.stop] = np.int32(elem.id)
            geo.vertices["contact_element_id"] = element_ids

    def insert_explicit_pairs(self, tab, info_by_endpoint: dict) -> None:
        for request in self._pairs:
            first = info_by_endpoint.get(request.first)
            second = info_by_endpoint.get(request.second)
            if first is None or second is None:
                gs.raise_exception("QIPCCoupler.set_contact_pair: endpoint is not a coupled entity or contact region.")
            elem_a, mu_a, res_a = first
            elem_b, mu_b, res_b = second
            friction = geometric_mean(mu_a, mu_b) if request.friction is None else request.friction
            resistance = harmonic_mean(res_a, res_b) if request.resistance is None else request.resistance
            tab.insert(
                elem_a,
                elem_b,
                friction_rate=float(friction),
                resistance=float(resistance),
                enable=request.enabled,
                adhesion=None,
                bond=None,
            )

    def _validate_region(self, region: QIPCContactRegion) -> None:
        if region._manager is not self:
            gs.raise_exception("QIPCCoupler: contact region belongs to a different scene.")

    def _require_unbuilt(self, method: str) -> None:
        if self._built:
            gs.raise_exception(f"QIPCCoupler.{method} must be called before scene.build().")
