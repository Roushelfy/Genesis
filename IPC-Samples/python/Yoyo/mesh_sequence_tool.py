from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from uipc import view


class MeshSequenceTool:
    """Export registered mesh state to per-object JSON sequence.

    API:
        tool = MeshSequenceTool(output_dir, file_prefix)
        tool.transform_output_slot(name, geo_slot)
        tool.position_output_slot(geo_slot, key='string_positions')
        tool.save(frame)
    """

    def __init__(self, output_dir: str | Path, file_prefix: str = "sequence"):
        self.output_dir = Path(output_dir)
        self.file_prefix = str(file_prefix)
        self._transform_slots: dict[str, Any] = {}
        self._position_slots: dict[str, tuple[Any, str]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def transform_output_slot(self, name: str, geo_slot: Any) -> None:
        self._transform_slots[name] = geo_slot

    def position_output_slot(
        self,
        geo_slot: Any,
        key: str = "string_positions",
        name: str | None = None,
    ) -> None:
        slot_name = key if name is None else str(name)
        self._position_slots[slot_name] = (geo_slot, key)

    def _slot_transform(self, geo_slot: Any) -> list[list[float]]:
        geo = geo_slot.geometry()
        xform = np.array(view(geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
        return xform.tolist()

    def save(self, frame: int) -> None:
        f = int(frame)
        for name, slot in self._transform_slots.items():
            obj_dir = self.output_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "frame": f,
                "name": name,
                "transform": self._slot_transform(slot),
            }
            out_file = obj_dir / f"{self.file_prefix}.{f}.json"
            out_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

        for name, (slot, key) in self._position_slots.items():
            obj_dir = self.output_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            positions = np.array(
                view(slot.geometry().positions()),
                copy=False,
            ).reshape(-1, 3)
            payload = {
                "frame": f,
                "name": name,
                key: positions.tolist(),
            }
            out_file = obj_dir / f"{self.file_prefix}.{f}.json"
            out_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
