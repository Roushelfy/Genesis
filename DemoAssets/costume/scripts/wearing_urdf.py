from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from yourdfpy import URDF


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: np.ndarray
    lower_limit: float | None
    upper_limit: float | None


@dataclass
class _UrdfMeshNode:
    node_name: str
    local_vertices: np.ndarray
    faces: np.ndarray
    ps_name: str = ""
    ps_mesh: Any | None = None


class UrdfForwardAdapter:
    def __init__(
        self,
        scene: Any,
        urdf_path: str | Path,
        mesh_source: str = "collision",
        contact_element_name: str = "urdf_robot",
    ) -> None:
        self.scene = scene
        self.urdf_path = Path(urdf_path)
        self.mesh_source = mesh_source.lower().strip()
        if self.mesh_source not in ("collision", "visual"):
            raise ValueError("mesh_source must be either 'collision' or 'visual'.")

        load_collision = self.mesh_source == "collision"
        self.urdf = URDF.load(
            str(self.urdf_path),
            build_scene_graph=not load_collision,
            build_collision_scene_graph=load_collision,
            load_meshes=not load_collision,
            load_collision_meshes=load_collision,
        )
        self.mesh_scene = self.urdf.collision_scene if load_collision else self.urdf.scene
        if self.mesh_scene is None:
            raise RuntimeError(f"Failed to build {self.mesh_source} scene from URDF: {self.urdf_path}")

        self._root_transform = np.eye(4, dtype=np.float64)
        self._joint_state: dict[str, float] = {name: 0.0 for name in self.urdf.actuated_joint_names}
        self._joint_specs = self._collect_joint_specs()
        self._parent_of_link = {str(j.child): str(j.parent) for j in self.urdf.joint_map.values()}
        self._root_link = str(self.urdf.base_link)
        self._mesh_nodes: list[_UrdfMeshNode] = []

        self._collect_mesh_nodes()
        self.sync_visual_mesh()

    def _collect_joint_specs(self) -> dict[str, JointSpec]:
        specs: dict[str, JointSpec] = {}
        for name, joint in self.urdf.joint_map.items():
            axis = np.asarray(joint.axis if joint.axis is not None else [0.0, 0.0, 1.0], dtype=np.float64)
            lower = None if joint.limit is None else float(joint.limit.lower)
            upper = None if joint.limit is None else float(joint.limit.upper)
            specs[name] = JointSpec(
                name=name,
                joint_type=str(joint.type),
                parent=str(joint.parent),
                child=str(joint.child),
                axis=axis,
                lower_limit=lower,
                upper_limit=upper,
            )
        return specs

    def joint_names(self) -> list[str]:
        return list(self.urdf.actuated_joint_names)

    def joint_state(self) -> dict[str, float]:
        return dict(self._joint_state)

    def set_joint_angle(self, joint_name: str, angle: float) -> None:
        if joint_name not in self._joint_state:
            return
        spec = self._joint_specs.get(joint_name)
        value = float(angle)
        if spec is not None:
            if spec.lower_limit is not None:
                value = max(value, spec.lower_limit)
            if spec.upper_limit is not None:
                value = min(value, spec.upper_limit)
        self._joint_state[joint_name] = value
        self.urdf.update_cfg(self._joint_state)

    def move_root(self, xyz: np.ndarray, rpy: np.ndarray) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
        self._root_transform = np.eye(4, dtype=np.float64)
        self._root_transform[:3, :3] = _rpy_matrix(rpy)
        self._root_transform[:3, 3] = xyz

    def sync_visual_mesh(self) -> None:
        for mesh_node in self._mesh_nodes:
            node_name = mesh_node.node_name
            node_tf, _ = self.mesh_scene.graph.get(node_name)
            world_tf = self._root_transform @ np.asarray(node_tf, dtype=np.float64)
            world_vertices = mesh_node.local_vertices @ world_tf[:3, :3].T + world_tf[:3, 3]
            if mesh_node.ps_mesh is not None:
                mesh_node.ps_mesh.update_vertex_positions(world_vertices)

    def register_polyscope_meshes(
        self,
        name_prefix: str = "urdf_collision",
        transparency: float = 0.7,
        edge_width: float = 0.0,
    ) -> None:
        import polyscope as ps

        for i, mesh_node in enumerate(self._mesh_nodes):
            ps_name = f"{name_prefix}_{i}_{mesh_node.node_name}"
            node_tf, _ = self.mesh_scene.graph.get(mesh_node.node_name)
            world_tf = self._root_transform @ np.asarray(node_tf, dtype=np.float64)
            world_vertices = mesh_node.local_vertices @ world_tf[:3, :3].T + world_tf[:3, 3]
            ps_mesh = ps.register_surface_mesh(ps_name, world_vertices, mesh_node.faces)
            ps_mesh.set_edge_width(float(edge_width))
            ps_mesh.set_transparency(float(transparency))
            mesh_node.ps_name = ps_name
            mesh_node.ps_mesh = ps_mesh

    def _collect_mesh_nodes(self) -> None:
        for node_name in self.mesh_scene.graph.nodes:
            node_tf, geom_name = self.mesh_scene.graph.get(node_name)
            if geom_name is None or geom_name not in self.mesh_scene.geometry:
                continue

            geom = self.mesh_scene.geometry[geom_name]
            vertices = np.array(geom.vertices, dtype=np.float64, copy=True)
            faces = np.array(geom.faces, dtype=np.int32, copy=True)
            if vertices.size == 0 or faces.size == 0:
                continue

            self._mesh_nodes.append(
                _UrdfMeshNode(
                    node_name=node_name,
                    local_vertices=vertices,
                    faces=faces,
                )
            )

    def _link_transform(self, link_name: str) -> np.ndarray:
        local_tf, _ = self.mesh_scene.graph.get(link_name)
        return self._root_transform @ np.asarray(local_tf, dtype=np.float64)

    def _group_key_for_link(self, link_name: str) -> str:
        current = str(link_name)
        prev = current
        while current in self._parent_of_link:
            parent = self._parent_of_link[current]
            if parent == self._root_link:
                return prev
            prev = parent
            current = parent
        return current

    def _link_mesh_vertices_world(self, link_name: str) -> np.ndarray:
        children = list(self.mesh_scene.graph.transforms.children.get(link_name, []))
        all_vertices: list[np.ndarray] = []
        for node_name in children:
            node_tf, geom_name = self.mesh_scene.graph.get(node_name)
            if geom_name is None or geom_name not in self.mesh_scene.geometry:
                continue
            mesh = self.mesh_scene.geometry[geom_name]
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if verts.size == 0:
                continue
            world_tf = self._root_transform @ np.asarray(node_tf, dtype=np.float64)
            world = verts @ world_tf[:3, :3].T + world_tf[:3, 3]
            all_vertices.append(world)
        if not all_vertices:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(all_vertices, axis=0)

    @staticmethod
    def _capsule_radius_for_segment(start: np.ndarray, end: np.ndarray, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.002
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-9:
            dist = np.linalg.norm(points - start.reshape(1, 3), axis=1)
            return float(max(np.max(dist), 0.002))
        direction = segment / length
        rel = points - start.reshape(1, 3)
        t = rel @ direction
        t_clamped = np.clip(t, 0.0, length)
        closest = start.reshape(1, 3) + t_clamped[:, None] * direction.reshape(1, 3)
        dist = np.linalg.norm(points - closest, axis=1)
        return float(max(np.max(dist), 0.002))

    @staticmethod
    def _fit_bounding_capsule(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if points.shape[0] == 0:
            p = np.zeros((3,), dtype=np.float64)
            return p, p, 0.002
        if points.shape[0] == 1:
            p = points[0].copy()
            return p, p, 0.002

        center = np.mean(points, axis=0)
        centered = points - center.reshape(1, 3)
        cov = centered.T @ centered / max(points.shape[0], 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        axis = eig_vecs[:, int(np.argmax(eig_vals))]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm

        t = centered @ axis
        t_min = float(np.min(t))
        t_max = float(np.max(t))
        start = center + axis * t_min
        end = center + axis * t_max
        radius = UrdfForwardAdapter._capsule_radius_for_segment(start, end, points)
        return start, end, float(max(radius, 0.002))

    @staticmethod
    def _leaf_end_from_mesh(start: np.ndarray, end: np.ndarray, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return end.copy()
        axis = end - start
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-9:
            center = np.mean(points, axis=0)
            centered = points - center.reshape(1, 3)
            cov = centered.T @ centered / max(points.shape[0], 1)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            axis = eig_vecs[:, int(np.argmax(eig_vals))]
            axis_len = float(np.linalg.norm(axis))
            if axis_len < 1e-12:
                return end.copy()
        direction = axis / axis_len
        proj = (points - start.reshape(1, 3)) @ direction
        t_base = float(np.linalg.norm(end - start))
        t_max = float(np.max(proj))
        t_target = max(t_base, t_max, 1e-6)
        return start + direction * t_target

    def line_skeleton(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        link_to_index: dict[str, int] = {}
        vertices: list[np.ndarray] = []
        edges: list[list[int]] = []
        edge_joint_names: list[str] = []
        edge_radii: list[float] = []
        edge_has_geometry: list[bool] = []
        edge_capsule_starts: list[np.ndarray] = []
        edge_capsule_ends: list[np.ndarray] = []
        edge_group_keys: list[str] = []

        parent_links = {str(j.parent) for j in self.urdf.joint_map.values()}
        leaf_links = {ln for ln in self.urdf.link_map.keys() if ln not in parent_links}

        def index_of(link_name: str) -> int:
            if link_name not in link_to_index:
                link_to_index[link_name] = len(vertices)
                tf = self._link_transform(link_name)
                vertices.append(tf[:3, 3].copy())
            return link_to_index[link_name]

        for joint_name, joint in self.urdf.joint_map.items():
            parent = str(joint.parent)
            child = str(joint.child)
            if parent not in self.urdf.link_map or child not in self.urdf.link_map:
                continue
            parent_idx = index_of(parent)
            child_idx = index_of(child)
            edges.append([parent_idx, child_idx])
            edge_joint_names.append(joint_name)
            edge_group_keys.append(self._group_key_for_link(child))

            start = vertices[parent_idx]
            end = vertices[child_idx]
            child_mesh_vertices = self._link_mesh_vertices_world(child)
            edge_has_geometry.append(child_mesh_vertices.shape[0] > 0)

            # Keep capsule sphere centers on skeleton edge endpoints for all links,
            # including terminal links, so adjacent segments connect at joints.
            edge_capsule_starts.append(start.copy())
            edge_capsule_ends.append(end.copy())

            if self.mesh_source == "collision" and child in leaf_links and child_mesh_vertices.shape[0] > 0:
                # Terminal link: keep joint-side sphere center at skeleton start,
                # and extend distal endpoint using collision mesh extent.
                extended_end = self._leaf_end_from_mesh(start, end, child_mesh_vertices)
                edge_capsule_ends[-1] = extended_end.copy()
                cap_radius = self._capsule_radius_for_segment(start, extended_end, child_mesh_vertices)
                edge_radii.append(cap_radius)
            else:
                edge_radii.append(self._capsule_radius_for_segment(start, end, child_mesh_vertices))

        if not vertices:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 2), dtype=np.int32),
                [],
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.bool_),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                [],
            )
        return (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(edges, dtype=np.int32),
            edge_joint_names,
            np.asarray(edge_radii, dtype=np.float64),
            np.asarray(edge_has_geometry, dtype=np.bool_),
            np.asarray(edge_capsule_starts, dtype=np.float64),
            np.asarray(edge_capsule_ends, dtype=np.float64),
            edge_group_keys,
        )
