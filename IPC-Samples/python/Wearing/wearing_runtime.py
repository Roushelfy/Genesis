from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uipc import Vector3, builtin, view
from uipc.core import Engine, FiniteElementStateAccessorFeature, SanityCheckResult, Scene, World
from uipc.geometry import SimplicialComplexIO as GeometryIO
from uipc.gui import SceneGUI

from wearing_core import WearingCore, build_capsule_mesh
from wearing_urdf import UrdfForwardAdapter


def _match_vector_attr_layout(attr_view: np.ndarray, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if attr_view.ndim == 3 and attr_view.shape[-1] == 1 and arr.ndim == 2:
        return arr[..., None]
    if attr_view.ndim == 2 and arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


class WearingRuntimeAPI:
    def __init__(
        self, engine: Engine, core: WearingCore, world: World, scene: Scene, sgui: SceneGUI | None, output_dir: Path
    ) -> None:
        self.engine = engine
        self.core = core
        self.world = world
        self.scene = scene
        self.sgui = sgui
        self.output_dir = output_dir
        self.run_simulation = False
        self.global_scale = 1.0
        self.target_global_scale = 1.0
        self.bone_scales = [1.0 for _ in self.core.proxies]
        self.target_bone_scales = [1.0 for _ in self.core.proxies]
        self.bone_display_names = [proxy.config.name for proxy in self.core.proxies]
        self.bone_length_start_offsets = [0.0 for _ in self.core.proxies]
        self.bone_length_end_offsets = [0.0 for _ in self.core.proxies]
        self._bone_base_starts = [np.asarray(proxy.config.start, dtype=np.float64).reshape(3).copy() for proxy in self.core.proxies]
        self._bone_base_ends = [np.asarray(proxy.config.end, dtype=np.float64).reshape(3).copy() for proxy in self.core.proxies]
        self.max_global_scale_delta_per_frame = 0.02
        self.max_bone_scale_delta_per_frame = 0.02
        self.checkpoint_interval = 20
        self.did_first_frame_sanity_check = False
        self.last_valid_frame = 0
        self.latest_sanity = None
        self.latest_message = "Ready"
        self.enable_auto_recover = True
        self.io = GeometryIO()
        self.urdf_adapter: UrdfForwardAdapter | None = None
        self.urdf_edges: np.ndarray | None = None
        self.proxy_edge_indices: list[int] = []
        self.urdf_bone_count = 0
        self.has_started_simulation = False
        self.world_initialized = False
        self._fem_accessor: FiniteElementStateAccessorFeature | None = None
        self._proxy_state_geos: list = []
        self._proxy_state_position_attrs: list = []
        self._has_unchecked_teleport = False
        self._teleport_guard_frame: int | None = None

    def toggle_run(self) -> None:
        if self.run_simulation:
            self.run_simulation = False
            return
        if not self._ensure_world_initialized():
            return
        if not self._validate_teleport_before_resume():
            return
        self.run_simulation = True

    def pause(self) -> None:
        self.run_simulation = False

    def _ensure_world_initialized(self) -> bool:
        if self.world_initialized:
            return True
        self.world.init(self.scene)
        self.world.retrieve()
        if self.sgui is not None:
            self.sgui.update()
        if not self.world.is_valid():
            self.latest_message = "World init failed sanity checks."
            return False
        self.world.dump()
        self.last_valid_frame = int(self.world.frame())
        self.world_initialized = True
        self._init_proxy_state_accessor()
        self.latest_message = "World initialized. Ready to run."
        return True

    def set_global_inflation_scale(self, scale: float) -> None:
        self.target_global_scale = float(max(scale, 0.0))
        self._apply_inflation_step()
        self._apply_proxy_pose_update()

    def set_bone_inflation_scale(self, bone_index: int, scale: float) -> None:
        self.target_bone_scales[bone_index] = float(max(scale, 0.0))
        self._apply_inflation_step()
        self._apply_proxy_pose_update()

    def set_bone_length_offsets(self, bone_index: int, start_offset: float, end_offset: float) -> None:
        self.bone_length_start_offsets[bone_index] = float(start_offset)
        self.bone_length_end_offsets[bone_index] = float(end_offset)
        self._apply_proxy_pose_update()

    def attach_urdf_adapter(
        self,
        urdf_adapter: UrdfForwardAdapter,
        urdf_edges: np.ndarray,
        proxy_edge_indices: list[int],
        urdf_edge_names: list[str] | None = None,
    ) -> None:
        self.urdf_adapter = urdf_adapter
        self.urdf_edges = urdf_edges
        self.proxy_edge_indices = list(proxy_edge_indices)
        self.urdf_bone_count = int(urdf_edges.shape[0])
        if urdf_edge_names is not None:
            names: list[str] = []
            for proxy_i, edge_i in enumerate(self.proxy_edge_indices):
                if 0 <= edge_i < len(urdf_edge_names):
                    name = str(urdf_edge_names[edge_i])
                else:
                    name = f"bone_{proxy_i}"
                names.append(name if len(name) > 0 else f"bone_{proxy_i}")
            if len(names) == len(self.bone_display_names):
                self.bone_display_names = names

    def set_joint_angle(self, joint_name: str, angle: float) -> None:
        if self.urdf_adapter is None:
            return
        self.urdf_adapter.set_joint_angle(joint_name, angle)
        self.urdf_adapter.sync_visual_mesh()
        self.sync_urdf_to_proxy_segments()
        self._apply_proxy_pose_update()

    def sync_urdf_to_proxy_segments(self) -> None:
        if self.urdf_adapter is None or self.urdf_edges is None:
            return
        self.urdf_adapter.sync_visual_mesh()
        vertices, edges, _, _, _, cap_starts, cap_ends, _ = self.urdf_adapter.line_skeleton()
        if vertices.shape[0] == 0 or edges.shape[0] == 0:
            return
        used = min(len(self.proxy_edge_indices), len(self.core.proxies))
        for proxy_idx in range(used):
            edge_idx = self.proxy_edge_indices[proxy_idx]
            if edge_idx < 0 or edge_idx >= int(edges.shape[0]):
                continue
            self._bone_base_starts[proxy_idx] = np.asarray(cap_starts[edge_idx], dtype=np.float64).reshape(3).copy()
            self._bone_base_ends[proxy_idx] = np.asarray(cap_ends[edge_idx], dtype=np.float64).reshape(3).copy()
        self._apply_capsule_length_offsets()

    def _vertex_count(self, geo) -> int:
        return int(np.asarray(view(geo.positions())).reshape(-1, 3).shape[0])

    def _init_proxy_state_accessor(self) -> None:
        self._fem_accessor = self.world.features().find(FiniteElementStateAccessorFeature)
        self._proxy_state_geos = []
        self._proxy_state_position_attrs = []
        if self._fem_accessor is None:
            return

        entries: list[tuple[int, int, str, int]] = []
        for i, target in enumerate(self.core.targets):
            geo = target.geo_slot.geometry()
            dof_offset_attr = geo.meta().find(builtin.dof_offset)
            if dof_offset_attr is None:
                continue
            dof_offset = int(np.asarray(view(dof_offset_attr)).reshape(-1)[0])
            vcount = self._vertex_count(geo)
            entries.append((dof_offset, vcount, "target", i))
        for i, proxy in enumerate(self.core.proxies):
            geo = proxy.geo_slot.geometry()
            dof_offset_attr = geo.meta().find(builtin.dof_offset)
            if dof_offset_attr is None:
                continue
            dof_offset = int(np.asarray(view(dof_offset_attr)).reshape(-1)[0])
            vcount = self._vertex_count(geo)
            entries.append((dof_offset, vcount, "proxy", i))

        if len(entries) == 0:
            return
        entries.sort(key=lambda x: x[0])

        compact_offsets: dict[int, int] = {}
        running = 0
        for _, vcount, kind, idx in entries:
            if kind == "proxy":
                compact_offsets[idx] = running
            running += int(vcount)

        total_vertices = int(self._fem_accessor.vertex_count())
        for i, proxy in enumerate(self.core.proxies):
            if i not in compact_offsets:
                continue
            vertex_offset = int(compact_offsets[i])
            vcount = self._vertex_count(proxy.geo_slot.geometry())
            if vertex_offset < 0 or vertex_offset + vcount > total_vertices:
                continue
            state_geo = self._fem_accessor.create_geometry(vertex_offset, vcount)
            pos_attr = state_geo.vertices().create(builtin.position, Vector3.Zero())
            self._proxy_state_geos.append(state_geo)
            self._proxy_state_position_attrs.append(pos_attr)

    def _can_accessor_teleport_proxy(self) -> bool:
        return (
            (self._fem_accessor is not None)
            and self.has_started_simulation
            and (not self.run_simulation)
            and (len(self._proxy_state_geos) == len(self.core.proxies))
        )

    def _proxy_vertices_for_index(self, proxy_index: int) -> np.ndarray:
        proxy = self.core.proxies[proxy_index]
        cfg = proxy.config
        radius = cfg.base_radius * cfg.radial_scale * self.global_scale
        vertices, _ = build_capsule_mesh(
            start=cfg.start,
            end=cfg.end,
            radius=radius,
            radial_resolution=proxy.radial_resolution,
            lat_resolution=proxy.lat_resolution,
        )
        return np.asarray(vertices, dtype=np.float64)

    def _teleport_proxies_with_state_accessor(self) -> bool:
        if not self._can_accessor_teleport_proxy():
            return False
        if not self._has_unchecked_teleport and self.world.is_valid():
            self.world.dump()
            self._teleport_guard_frame = int(self.world.frame())
            self._has_unchecked_teleport = True
        assert self._fem_accessor is not None
        for i in range(len(self.core.proxies)):
            state_geo = self._proxy_state_geos[i]
            pos_attr = self._proxy_state_position_attrs[i]
            self._fem_accessor.copy_to(state_geo)
            pos_view = view(pos_attr)
            target_vertices = _match_vector_attr_layout(pos_view, self._proxy_vertices_for_index(i))
            pos_view[:] = target_vertices
            self._fem_accessor.copy_from(state_geo)
        return True

    def _validate_teleport_before_resume(self) -> bool:
        if not self._has_unchecked_teleport:
            return True
        summary = self.core.run_sanity_check(auto_report=True)
        self.latest_sanity = summary
        if summary.result == SanityCheckResult.Success:
            self._has_unchecked_teleport = False
            self._teleport_guard_frame = None
            self.last_valid_frame = int(self.world.frame())
            self.latest_message = "Teleport sanity check: Success"
            return True

        recover_ok = False
        if self._teleport_guard_frame is not None:
            recover_ok = self.core.recover_to_frame(self._teleport_guard_frame)
        self._has_unchecked_teleport = False
        self._teleport_guard_frame = None
        if recover_ok:
            if self.sgui is not None:
                self.sgui.update()
            self.latest_message = (
                f"Teleport sanity check failed: {len(summary.errors)} errors, {len(summary.warnings)} warnings"
                " | Recovered to state before teleport."
            )
        else:
            self.latest_message = (
                f"Teleport sanity check failed: {len(summary.errors)} errors, {len(summary.warnings)} warnings"
                " | Recovery failed."
            )
        return False

    def _apply_proxy_pose_update(self) -> None:
        self._apply_capsule_length_offsets()
        self.core.sync_proxy_targets(immediate_apply=False)
        if self._teleport_proxies_with_state_accessor():
            self.refresh_view(retrieve_backend=True)
            return
        self.core.sync_proxy_targets(immediate_apply=not self.run_simulation)
        self.refresh_view(retrieve_backend=self.run_simulation)

    def _apply_capsule_length_offsets(self) -> None:
        used = min(
            len(self.core.proxies),
            len(self.bone_length_start_offsets),
            len(self.bone_length_end_offsets),
            len(self._bone_base_starts),
            len(self._bone_base_ends),
        )
        for i in range(used):
            start = np.asarray(self._bone_base_starts[i], dtype=np.float64).reshape(3)
            end = np.asarray(self._bone_base_ends[i], dtype=np.float64).reshape(3)
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length < 1e-9:
                self.core.set_bone_segment(i, start, end)
                continue
            direction = axis / length
            start_offset = float(self.bone_length_start_offsets[i])
            end_offset = float(self.bone_length_end_offsets[i])
            # Positive start_offset extends capsule backward from start.
            # Positive end_offset extends capsule forward to end direction.
            new_start = start - direction * start_offset
            new_end = end + direction * end_offset
            self.core.set_bone_segment(i, new_start, new_end)

    def run_sanity_check(self) -> bool:
        if not self.world_initialized:
            self.latest_message = "World is not initialized yet."
            return False
        summary = self.core.run_sanity_check(auto_report=True)
        self.latest_sanity = summary
        if summary.result == SanityCheckResult.Success:
            self.latest_message = "Sanity check: Success"
            return True
        self.latest_message = f"Sanity check failed: {len(summary.errors)} errors, {len(summary.warnings)} warnings"
        if self.enable_auto_recover:
            recovered = self.core.recover_to_frame(self.last_valid_frame)
            if recovered:
                if self.sgui is not None:
                    self.sgui.update()
                self.latest_message += f" | Recovered to frame {self.last_valid_frame}"
        return False

    def export_shapes(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = self.core.export_target_shapes(out_dir, self.io, prefix="wearing")
        self.latest_message = f"Exported rest-shape and init-shape to {out_dir}."
        return paths

    def save_joint_json(self, file_path: str | Path) -> str | None:
        if self.urdf_adapter is None:
            self.latest_message = "No URDF adapter found. Joint file is not saved."
            return None
        joint_state = self.urdf_adapter.joint_state()
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = int(self.world.frame()) if self.world_initialized else 0
        payload = {
            "frame": frame,
            "joint_angles": joint_state,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.latest_message = f"Saved joint pose to {output_path}"
        return str(output_path)

    def load_joint_json(self, file_path: str | Path) -> dict[str, float] | None:
        if self.urdf_adapter is None:
            self.latest_message = "No URDF adapter found. Joint file is not loaded."
            return None
        input_path = Path(file_path)
        if not input_path.exists():
            self.latest_message = f"Joint file does not exist: {input_path}"
            return None
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_joint_angles = payload.get("joint_angles", payload)
        if not isinstance(raw_joint_angles, dict):
            self.latest_message = f"Invalid joint file format: {input_path}"
            return None
        for joint_name, angle in raw_joint_angles.items():
            try:
                self.urdf_adapter.set_joint_angle(str(joint_name), float(angle))
            except Exception:
                continue
        self.urdf_adapter.sync_visual_mesh()
        self.sync_urdf_to_proxy_segments()
        self._apply_proxy_pose_update()
        state = self.urdf_adapter.joint_state()
        self.latest_message = f"Loaded joint pose from {input_path}"
        return state

    def save_line_mesh_obj(self, file_path: str | Path) -> str | None:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertices, edges = self._current_line_mesh()
        if vertices.shape[0] == 0 or edges.shape[0] == 0:
            self.latest_message = "No line mesh data available."
            return None
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# line mesh exported by 38_wearing\n")
            for v in vertices:
                f.write(f"v {float(v[0]):.9g} {float(v[1]):.9g} {float(v[2]):.9g}\n")
            for e in edges:
                a = int(e[0]) + 1
                b = int(e[1]) + 1
                f.write(f"l {a} {b}\n")
        self.latest_message = f"Saved line mesh to {output_path}"
        return str(output_path)

    def save_inflation_json(self, file_path: str | Path) -> str:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = int(self.world.frame()) if self.world_initialized else 0
        payload = {
            "frame": frame,
            "global_scale": float(self.target_global_scale),
            "bone_names": [str(v) for v in self.bone_display_names],
            "bone_scale_by_name": {
                str(self.bone_display_names[i]): float(self.target_bone_scales[i])
                for i in range(min(len(self.bone_display_names), len(self.target_bone_scales)))
            },
            "bone_length_start_offset_by_name": {
                str(self.bone_display_names[i]): float(self.bone_length_start_offsets[i])
                for i in range(min(len(self.bone_display_names), len(self.bone_length_start_offsets)))
            },
            "bone_length_end_offset_by_name": {
                str(self.bone_display_names[i]): float(self.bone_length_end_offsets[i])
                for i in range(min(len(self.bone_display_names), len(self.bone_length_end_offsets)))
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.latest_message = f"Saved inflation scaling to {output_path}"
        return str(output_path)

    def load_inflation_json(self, file_path: str | Path) -> bool:
        input_path = Path(file_path)
        if not input_path.exists():
            self.latest_message = f"Inflation file does not exist: {input_path}"
            return False
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        bone_by_name = payload.get("bone_scale_by_name", {})
        loaded_scales: list[float] = [1.0 for _ in self.target_bone_scales]
        has_scale_by_name = isinstance(bone_by_name, dict) and len(bone_by_name) > 0
        if has_scale_by_name:
            for i, name in enumerate(self.bone_display_names):
                if name in bone_by_name:
                    loaded_scales[i] = float(max(0.0, bone_by_name[name]))
        else:
            # Backward compatibility: legacy index-based list.
            raw_bone_scales = payload.get("bone_scales", None)
            if isinstance(raw_bone_scales, list):
                for i in range(min(len(loaded_scales), len(raw_bone_scales))):
                    loaded_scales[i] = float(max(0.0, raw_bone_scales[i]))

        loaded_start_offsets: list[float] = [0.0 for _ in self.bone_length_start_offsets]
        loaded_end_offsets: list[float] = [0.0 for _ in self.bone_length_end_offsets]
        start_by_name = payload.get("bone_length_start_offset_by_name", {})
        end_by_name = payload.get("bone_length_end_offset_by_name", {})
        has_start_by_name = isinstance(start_by_name, dict) and len(start_by_name) > 0
        has_end_by_name = isinstance(end_by_name, dict) and len(end_by_name) > 0
        if has_start_by_name:
            for i, name in enumerate(self.bone_display_names):
                if i < len(loaded_start_offsets) and name in start_by_name:
                    loaded_start_offsets[i] = float(start_by_name[name])
        if has_end_by_name:
            for i, name in enumerate(self.bone_display_names):
                if i < len(loaded_end_offsets) and name in end_by_name:
                    loaded_end_offsets[i] = float(end_by_name[name])
        if not has_start_by_name:
            # Backward compatibility: legacy index-based list.
            raw_start_offsets = payload.get("bone_length_start_offsets", None)
            if isinstance(raw_start_offsets, list):
                for i in range(min(len(loaded_start_offsets), len(raw_start_offsets))):
                    loaded_start_offsets[i] = float(raw_start_offsets[i])
        if not has_end_by_name:
            # Backward compatibility: legacy index-based list.
            raw_end_offsets = payload.get("bone_length_end_offsets", None)
            if isinstance(raw_end_offsets, list):
                for i in range(min(len(loaded_end_offsets), len(raw_end_offsets))):
                    loaded_end_offsets[i] = float(raw_end_offsets[i])

        loaded_global = float(max(0.0, payload.get("global_scale", self.target_global_scale)))
        self.global_scale = loaded_global
        self.target_global_scale = loaded_global
        self.bone_scales = loaded_scales.copy()
        self.target_bone_scales = loaded_scales.copy()
        self.bone_length_start_offsets = loaded_start_offsets.copy()
        self.bone_length_end_offsets = loaded_end_offsets.copy()
        self._apply_scale_state_to_core()
        self._apply_proxy_pose_update()
        self.latest_message = f"Loaded inflation scaling from {input_path}"
        return True

    def save_checkpoint(self) -> None:
        if not self.world_initialized or (not self.world.is_valid()):
            return
        self.world.dump()
        self.last_valid_frame = int(self.world.frame())

    def step_once(self, force_sanity_check: bool = False) -> None:
        if not self.world_initialized:
            if not self._ensure_world_initialized():
                return
        if not self.world.is_valid():
            self.latest_message = "World is invalid. Skip simulation step."
            return
        if self._has_unchecked_teleport and (not self.run_simulation):
            if not self._validate_teleport_before_resume():
                return
        self.sync_urdf_to_proxy_segments()
        self._apply_inflation_step()
        self.core.sync_proxy_targets(immediate_apply=False)
        self.world.advance()
        self.has_started_simulation = True
        self.refresh_view(retrieve_backend=True)
        frame = int(self.world.frame())

        if frame % self.checkpoint_interval == 0:
            self.save_checkpoint()

        if force_sanity_check or (frame == 1 and not self.did_first_frame_sanity_check):
            if self.run_sanity_check():
                self.last_valid_frame = frame
            self.did_first_frame_sanity_check = True

    def refresh_view(self, retrieve_backend: bool = True) -> None:
        if retrieve_backend:
            self.world.retrieve()
        if self.sgui is not None:
            self.sgui.update()

    def _step_towards(self, current: float, target: float, max_delta: float) -> float:
        delta = target - current
        if abs(delta) <= max_delta:
            return target
        return current + np.sign(delta) * max_delta

    def _apply_inflation_step(self) -> None:
        self.global_scale = self._step_towards(
            self.global_scale, self.target_global_scale, self.max_global_scale_delta_per_frame
        )
        for i in range(len(self.bone_scales)):
            self.bone_scales[i] = self._step_towards(
                self.bone_scales[i], self.target_bone_scales[i], self.max_bone_scale_delta_per_frame
            )
        self._apply_scale_state_to_core()

    def _apply_scale_state_to_core(self) -> None:
        self.core.set_global_inflation_scale(self.global_scale)
        for i in range(min(len(self.bone_scales), len(self.core.proxies))):
            self.core.set_bone_inflation_scale(i, self.bone_scales[i])

    def _current_line_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.core.proxies) > 0:
            vertices: list[np.ndarray] = []
            edges: list[list[int]] = []
            for proxy in self.core.proxies:
                start_idx = len(vertices)
                vertices.append(np.asarray(proxy.config.start, dtype=np.float64).reshape(3))
                end_idx = len(vertices)
                vertices.append(np.asarray(proxy.config.end, dtype=np.float64).reshape(3))
                edges.append([start_idx, end_idx])
            return np.asarray(vertices, dtype=np.float64), np.asarray(edges, dtype=np.int32)

        if self.urdf_adapter is not None:
            vertices, edges, _, _, _, _, _, _ = self.urdf_adapter.line_skeleton()
            if vertices.shape[0] > 0 and edges.shape[0] > 0:
                return vertices, edges

        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.int32)
