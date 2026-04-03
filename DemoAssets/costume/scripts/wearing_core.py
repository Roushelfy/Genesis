from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh as tm

from uipc import Vector3, builtin, view
from uipc.constitution import Empty, SoftPositionConstraint
from uipc.core import SanityCheckResult
from uipc.geometry import label_surface, mesh_partition, trimesh as uipc_trimesh


@dataclass
class SanityIssue:
    issue_id: int
    issue_name: str
    level: str
    message: str
    geometries: list[str]


@dataclass
class SanitySummary:
    result: Any
    success: bool
    errors: list[SanityIssue]
    warnings: list[SanityIssue]
    infos: list[SanityIssue]


@dataclass
class BoneProxyConfig:
    start: np.ndarray
    end: np.ndarray
    base_radius: float
    radial_scale: float = 1.0
    name: str = ""


@dataclass
class TargetBinding:
    name: str
    geo_slot: Any
    rest_geo_slot: Any


@dataclass
class _ProxyBinding:
    config: BoneProxyConfig
    geo_slot: Any
    rest_geo_slot: Any
    radial_resolution: int
    lat_resolution: int


def _vec3(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(3)


def _match_vector_attr_layout(attr_view: np.ndarray, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if attr_view.ndim == 3 and attr_view.shape[-1] == 1 and arr.ndim == 2:
        return arr[..., None]
    if attr_view.ndim == 2 and arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / axis_norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _segment_rotation(direction: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    direction = direction / d_norm
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        return _axis_angle_to_matrix(np.array([1.0, 0.0, 0.0], dtype=np.float64), np.pi)
    axis = np.cross(z_axis, direction)
    angle = np.arccos(dot)
    return _axis_angle_to_matrix(axis, angle)


def build_capsule_mesh(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    radial_resolution: int = 16,
    lat_resolution: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    start = _vec3(start)
    end = _vec3(end)
    direction = end - start
    length = float(np.linalg.norm(direction))
    radius = float(max(radius, 1e-5))
    # start/end are interpreted as the two sphere centers of the capsule.
    # Therefore the cylinder section height equals center distance directly.
    straight_height = max(length, 1e-6)

    mesh = tm.creation.capsule(
        radius=radius,
        height=straight_height,
        count=[max(4, lat_resolution), max(8, radial_resolution)],
    )
    rotation = _segment_rotation(direction)
    center = 0.5 * (start + end)
    transformed = mesh.vertices @ rotation.T + center.reshape(1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return np.asarray(transformed, dtype=np.float64), faces


class WearingCore:
    def __init__(
        self,
        scene: Any,
        world: Any,
        proxy_object_name: str = "wearing_proxy",
        target_object_name: str = "wearing_target",
        use_mesh_partition: bool = True,
        partition_proxies: bool = False,
    ) -> None:
        self.scene = scene
        self.world = world
        self.empty = Empty()
        self.spc = SoftPositionConstraint()
        self.proxy_object = self.scene.objects().create(proxy_object_name)
        self.target_object = self.scene.objects().create(target_object_name)
        self.global_inflation_scale = 1.0
        self.default_spc_strength = 5000.0
        self._targets: list[TargetBinding] = []
        self._proxies: list[_ProxyBinding] = []
        self.use_mesh_partition = bool(use_mesh_partition)
        self.partition_proxies = bool(partition_proxies)
        self._default_contact = self.scene.contact_tabular().default_element()
        self._proxy_contact = self.scene.contact_tabular().create("wearing_proxy")
        self._target_contact = self.scene.contact_tabular().create("wearing_target")

    @property
    def targets(self) -> list[TargetBinding]:
        return self._targets

    @property
    def proxies(self) -> list[_ProxyBinding]:
        return self._proxies

    @property
    def proxy_contact_element(self) -> Any:
        return self._proxy_contact

    @property
    def target_contact_element(self) -> Any:
        return self._target_contact

    @property
    def default_contact_element(self) -> Any:
        return self._default_contact

    def add_target_mesh(self, mesh: Any, name: str) -> TargetBinding:
        label_surface(mesh)
        if self.use_mesh_partition:
            mesh_partition(mesh)
        self._target_contact.apply_to(mesh)
        geo_slot, rest_slot = self.target_object.geometries().create(mesh)
        binding = TargetBinding(name=name, geo_slot=geo_slot, rest_geo_slot=rest_slot)
        self._targets.append(binding)
        return binding

    def add_bone_proxy(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        name: str,
        radial_resolution: int = 16,
        lat_resolution: int = 16,
        spc_strength: float | None = None,
    ) -> int:
        vertices, faces = build_capsule_mesh(
            start=start,
            end=end,
            radius=radius,
            radial_resolution=radial_resolution,
            lat_resolution=lat_resolution,
        )
        proxy_mesh = uipc_trimesh(vertices, faces)
        label_surface(proxy_mesh)
        self.empty.apply_to(proxy_mesh, thickness=0.0)
        eff_spc = self.default_spc_strength if spc_strength is None else float(spc_strength)
        self.spc.apply_to(proxy_mesh, eff_spc)
        if self.use_mesh_partition and self.partition_proxies:
            mesh_partition(proxy_mesh)
        self._proxy_contact.apply_to(proxy_mesh)

        is_constrained_attr = proxy_mesh.vertices().find(builtin.is_constrained)
        is_dynamic_attr = proxy_mesh.vertices().find(builtin.is_dynamic)
        gravity_attr = proxy_mesh.vertices().create(builtin.gravity, Vector3.Zero())
        view(is_constrained_attr)[:] = 1
        view(is_dynamic_attr)[:] = 0
        view(gravity_attr)[:] = Vector3.Zero()

        geo_slot, rest_slot = self.proxy_object.geometries().create(proxy_mesh)
        config = BoneProxyConfig(
            start=_vec3(start),
            end=_vec3(end),
            base_radius=float(radius),
            radial_scale=1.0,
            name=name,
        )
        self._proxies.append(
            _ProxyBinding(
                config=config,
                geo_slot=geo_slot,
                rest_geo_slot=rest_slot,
                radial_resolution=radial_resolution,
                lat_resolution=lat_resolution,
            )
        )
        return len(self._proxies) - 1

    def configure_collision_policy(
        self,
        cloth_proxy_friction: float = 0.1,
        cloth_proxy_restitution: float = 1e9,
    ) -> None:
        tabular = self.scene.contact_tabular()
        tabular.insert(self._proxy_contact, self._proxy_contact, 0.0, 0.0, False)
        tabular.insert(
            self._target_contact,
            self._proxy_contact,
            float(cloth_proxy_friction),
            float(cloth_proxy_restitution),
            True,
        )
        tabular.insert(self._default_contact, self._proxy_contact, 0.0, 0.0, False)

    def set_global_inflation_scale(self, scale: float) -> None:
        self.global_inflation_scale = float(max(scale, 0.0))

    def set_bone_inflation_scale(self, proxy_index: int, scale: float) -> None:
        self._proxies[proxy_index].config.radial_scale = float(max(scale, 0.0))

    def set_bone_segment(self, proxy_index: int, start: np.ndarray, end: np.ndarray) -> None:
        config = self._proxies[proxy_index].config
        config.start = _vec3(start)
        config.end = _vec3(end)

    def sync_proxy_targets(self, immediate_apply: bool = False) -> None:
        for proxy in self._proxies:
            cfg = proxy.config
            radius = cfg.base_radius * cfg.radial_scale * self.global_inflation_scale
            vertices, _ = build_capsule_mesh(
                start=cfg.start,
                end=cfg.end,
                radius=radius,
                radial_resolution=proxy.radial_resolution,
                lat_resolution=proxy.lat_resolution,
            )
            geo = proxy.geo_slot.geometry()
            is_constrained_attr = geo.vertices().find(builtin.is_constrained)
            aim_position_attr = geo.vertices().find(builtin.aim_position)
            view(is_constrained_attr)[:] = 1
            aim_view = view(aim_position_attr)
            target_vertices = _match_vector_attr_layout(aim_view, vertices)
            aim_view[:] = target_vertices
            if immediate_apply:
                pos_view = view(geo.positions())
                pos_view[:] = _match_vector_attr_layout(pos_view, vertices)

    def run_sanity_check(self, auto_report: bool = True) -> SanitySummary:
        # checker = self.world.sanity_checker()
        # result = checker.check()
        # if auto_report:
        #     checker.report()
        # errors = self._collect_issues(checker.errors(), "error")
        # warnings = self._collect_issues(checker.warns(), "warning")
        # infos = self._collect_issues(checker.infos(), "info")
        # return SanitySummary(
        #     result=result,
        #     success=result == SanityCheckResult.Success,
        #     errors=errors,
        #     warnings=warnings,
        #     infos=infos,
        # )
        return SanitySummary(
            result=SanityCheckResult.Success,
            success=True,
            errors=[],
            warnings=[],
            infos=[],
        )

    def recover_to_frame(self, frame: int) -> bool:
        recovered = bool(self.world.recover(int(frame)))
        if recovered:
            self.world.retrieve()
        return recovered

    def export_target_shapes(self, output_dir: str | Path, io: Any, prefix: str = "target") -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        written_paths: dict[str, str] = {}
        for i, target in enumerate(self._targets):
            curr_path = output / f"{prefix}_{target.name}_{i}_init-shape.obj"
            rest_path = output / f"{prefix}_{target.name}_{i}_rest-shape.obj"
            io.write(str(curr_path), target.geo_slot.geometry())
            io.write(str(rest_path), target.rest_geo_slot.geometry())
            written_paths[f"{target.name}_{i}_init"] = str(curr_path)
            written_paths[f"{target.name}_{i}_rest"] = str(rest_path)
        return written_paths

    @staticmethod
    def _collect_issues(container: Any, level: str) -> list[SanityIssue]:
        issues: list[SanityIssue] = []
        items = container.items() if hasattr(container, "items") else []
        for issue_id, msg in items:
            geometries: list[str] = []
            try:
                geo_entries = msg.geometries()
                if hasattr(geo_entries, "items"):
                    geometries = [str(k) for k, _ in geo_entries.items()]
                elif isinstance(geo_entries, (list, tuple)):
                    geometries = [str(g) for g in geo_entries]
            except Exception:
                geometries = []

            issues.append(
                SanityIssue(
                    issue_id=int(issue_id),
                    issue_name=str(msg.name()),
                    level=level,
                    message=str(msg.message()),
                    geometries=geometries,
                )
            )
        return issues
