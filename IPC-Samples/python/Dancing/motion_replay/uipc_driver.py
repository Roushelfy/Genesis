from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from typing import Any

import numpy as np

from uipc import builtin, view
from uipc.constitution import AffineBodyShell, SoftTransformConstraint
from uipc.geometry import label_surface, trimesh
from uipc.unit import MPa

from .urdf_kinematics import MeshNode


@dataclass
class BodyBinding:
    node_name: str
    object_name: str
    obj: Any
    geo_slot: Any


@dataclass
class DirectTransformBinding:
    node_name: str
    object_name: str
    local_vertices: np.ndarray
    faces: np.ndarray
    world_transform: np.ndarray
    ps_mesh: Any | None = None


class AffineStcDriver:
    def __init__(
        self,
        scene: Any,
        object_prefix: str = "robot_link",
        kappa_mpa: float = 80.0,
        stc_strength: np.ndarray | None = None,
        shell_thickness: float = 0.0001,
    ) -> None:
        self.scene = scene
        self.object_prefix = object_prefix
        self.abs = AffineBodyShell()
        self.stc = SoftTransformConstraint()
        self.default_contact = self.scene.contact_tabular().default_element()
        self.robot_contact = self.scene.contact_tabular().create(f"{self.object_prefix}_contact")
        self.kappa = float(kappa_mpa) * MPa
        self.shell_thickness = float(shell_thickness)
        self.stc_strength = (
            np.asarray(stc_strength, dtype=np.float64)
            if stc_strength is not None
            else np.array([120.0, 120.0], dtype=np.float64)
        )
        self.bindings: list[BodyBinding] = []
        self.direct_bindings: list[DirectTransformBinding] = []

    @staticmethod
    def _is_ignored(node_name: str, ignore_link_patterns: list[str]) -> bool:
        if not ignore_link_patterns:
            return False
        # Match both raw node name and basename before first dot.
        raw = str(node_name)
        base = raw.split(".", 1)[0]
        for pattern in ignore_link_patterns:
            pat = str(pattern).strip()
            if not pat:
                continue
            if fnmatch.fnmatch(raw, pat) or fnmatch.fnmatch(base, pat):
                return True
        return False

    def create_bodies(self, mesh_nodes: list[MeshNode], ignore_link_patterns: list[str] | None = None) -> None:
        ignore_patterns = [] if ignore_link_patterns is None else list(ignore_link_patterns)
        self.bindings.clear()
        self.direct_bindings.clear()
        for i, node in enumerate(mesh_nodes):
            object_name = f"{self.object_prefix}_{i}_{node.node_name}"

            if self._is_ignored(node.node_name, ignore_patterns):
                # Keep ignored links as transform-driven visuals only (outside IPC).
                self.direct_bindings.append(
                    DirectTransformBinding(
                        node_name=node.node_name,
                        object_name=object_name,
                        local_vertices=np.array(node.local_vertices, copy=True),
                        faces=np.array(node.faces, copy=True),
                        world_transform=np.eye(4, dtype=np.float64),
                    )
                )
                continue

            sc = trimesh(node.local_vertices, node.faces)
            label_surface(sc)
            self.abs.apply_to(sc, self.kappa, thickness=self.shell_thickness)
            self.stc.apply_to(sc, self.stc_strength)
            self.robot_contact.apply_to(sc)
            obj = self.scene.objects().create(object_name)
            geo_slot, _ = obj.geometries().create(sc)
            self.bindings.append(
                BodyBinding(
                    node_name=node.node_name,
                    object_name=object_name,
                    obj=obj,
                    geo_slot=geo_slot,
                )
            )

    def apply_target_transforms(
        self, node_world_transforms: dict[str, np.ndarray], snap_to_pose: bool = False
    ) -> None:
        for binding in self.bindings:
            if binding.node_name not in node_world_transforms:
                continue
            geo = binding.geo_slot.geometry()
            is_constrained = geo.instances().find(builtin.is_constrained)
            aim_transform = geo.instances().find(builtin.aim_transform)
            world_tf = np.asarray(node_world_transforms[binding.node_name], dtype=np.float64)
            view(is_constrained)[0] = 1
            view(aim_transform)[0] = world_tf
            if snap_to_pose:
                view(geo.transforms())[0] = world_tf

        for binding in self.direct_bindings:
            if binding.node_name not in node_world_transforms:
                continue
            world_tf = np.asarray(node_world_transforms[binding.node_name], dtype=np.float64)
            binding.world_transform = world_tf
            if binding.ps_mesh is not None:
                world_vertices = binding.local_vertices @ world_tf[:3, :3].T + world_tf[:3, 3]
                binding.ps_mesh.update_vertex_positions(world_vertices)

    def register_ignored_polyscope_meshes(
        self,
        *,
        name_prefix: str | None = None,
        transparency: float = 0.45,
        edge_width: float = 0.0,
    ) -> None:
        if not self.direct_bindings:
            return
        import polyscope as ps

        prefix = name_prefix if name_prefix is not None else f"{self.object_prefix}_ignored"
        for i, binding in enumerate(self.direct_bindings):
            world_tf = binding.world_transform
            world_vertices = binding.local_vertices @ world_tf[:3, :3].T + world_tf[:3, 3]
            ps_name = f"{prefix}_{i}_{binding.node_name}"
            mesh = ps.register_surface_mesh(ps_name, world_vertices, binding.faces)
            mesh.set_transparency(float(transparency))
            mesh.set_edge_width(float(edge_width))
            binding.ps_mesh = mesh

    def configure_contact_policy(
        self,
        *,
        enable_robot_robot_contact: bool = False,
        enable_default_robot_contact: bool = True,
        default_robot_friction: float = 0.5,
        default_robot_resistance: float = 1.0e9,
    ) -> None:
        tabular = self.scene.contact_tabular()
        tabular.insert(
            self.robot_contact,
            self.robot_contact,
            float(default_robot_friction),
            float(default_robot_resistance),
            bool(enable_robot_robot_contact),
        )
        tabular.insert(
            self.default_contact,
            self.robot_contact,
            float(default_robot_friction),
            float(default_robot_resistance),
            bool(enable_default_robot_contact),
        )
