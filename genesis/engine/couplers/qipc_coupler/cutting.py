"""Pressure-driven shell cutting with a serrated part of a coupled rigid tool."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs

if TYPE_CHECKING:
    from qipc.geometry.geometry_slot import GeometrySlot
    from qipc.kerf.system import CutEvent, KerfCutter

    from genesis.engine.couplers.qipc_coupler.coupler import AbdEntityPreInit, QIPCCoupler
    from genesis.engine.entities.fem_entity import FEMEntity
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity


class QIPCShellCutter:
    """A shell target and its serrated tool; live vertex IDs change after a cut.

    Register callbacks to remap application-owned vertex handles. Constraint
    records owned by the coupler and the rendered mesh are updated internally.
    """

    def __init__(
        self,
        coupler: QIPCCoupler,
        target: FEMEntity,
        tool: RigidEntity,
        link: str,
        strength: float,
        kerf_width: float | None,
    ) -> None:
        if not np.isfinite(strength) or strength <= 0:
            gs.raise_exception("Cut strength must be finite and positive.")
        if kerf_width is not None and (not np.isfinite(kerf_width) or kerf_width <= 0):
            gs.raise_exception("Kerf width must be finite and positive.")
        self.target = target
        self._coupler = coupler
        self._tool = tool
        self._link = tool.get_link(link).idx_local
        self._strength = strength
        self._kerf_width = kerf_width
        self._listeners: list[Callable[[CutEvent], None]] = []
        self._cutter: KerfCutter | None = None
        self.epoch = 0

    def on_cut(self, callback: Callable[[CutEvent], None]) -> None:
        self._listeners.append(callback)

    @property
    def events(self) -> tuple[CutEvent, ...]:
        return () if self._cutter is None else tuple(self._cutter.events)

    def declare(
        self, rigid_entries: list[AbdEntityPreInit], fem_entries: list[tuple[FEMEntity, GeometrySlot, bool]]
    ) -> None:
        from qipc.constitution import BreakableShell
        from qipc.geometry.sharp_edges import mark_serrated_edges

        slot = next((slot for entity, slot, _ in fem_entries if entity is self.target), None)
        if slot is None or slot.geometry.dim != 2:
            gs.raise_exception("A kerf target must be a coupled cloth shell.")
        if len(self.target.vgeoms) != 1:
            gs.raise_exception("Shell cutting currently requires one visual mesh per entity.")
        BreakableShell(strength=self._strength, debounce_frames=2, cooldown_frames=3).apply_to(slot.geometry)
        pre = next((entry for entry in rigid_entries if entry.entity is self._tool), None)
        if pre is None or self._link not in pre.vertex_ranges:
            gs.raise_exception("The serrated tool link has no coupled collision mesh.")
        representative, vertices = pre.vertex_ranges[self._link]
        mark_serrated_edges(pre.group_slots[representative].geometry, np.asarray(vertices, dtype=np.int64))

    def initialize(self) -> None:
        from qipc.kerf.system import KerfConfig, KerfCutter

        self._cutter = KerfCutter(
            self._coupler._scene,
            KerfConfig(kerf_width=self._kerf_width, backend="cpu", settle_frames=2),
        )
        self.epoch += 1

    def update(self) -> None:
        assert self._cutter is not None
        event = self._cutter.update()
        if event is None or event.rejected:
            return
        self._coupler._remap_after_cut(event)
        for callback in self._listeners:
            callback(event)
        self.epoch += 1
