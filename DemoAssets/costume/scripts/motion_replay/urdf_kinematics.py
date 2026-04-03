from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from yourdfpy import URDF


@dataclass
class MeshNode:
    node_name: str
    geom_name: str
    local_vertices: np.ndarray
    faces: np.ndarray


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class UrdfKinematics:
    def __init__(self, urdf_path: str | Path, mesh_source: str = "collision") -> None:
        if mesh_source not in ("collision", "visual"):
            raise ValueError("mesh_source must be 'collision' or 'visual'.")
        self.urdf_path = Path(urdf_path)
        load_collision = mesh_source == "collision"
        self.urdf = URDF.load(
            str(self.urdf_path),
            build_collision_scene_graph=load_collision,
            load_collision_meshes=load_collision,
            build_scene_graph=not load_collision,
            load_meshes=not load_collision,
        )
        self.mesh_scene = self.urdf.collision_scene if load_collision else self.urdf.scene
        if self.mesh_scene is None:
            raise RuntimeError(f"Failed to load {mesh_source} scene from URDF: {self.urdf_path}")

        self._joint_names = list(self.urdf.actuated_joint_names)
        self._joint_state = {name: 0.0 for name in self._joint_names}
        self._root_transform = np.eye(4, dtype=np.float64)
        self._mesh_nodes = self._collect_mesh_nodes()

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def mesh_nodes(self) -> list[MeshNode]:
        return self._mesh_nodes

    @property
    def root_transform(self) -> np.ndarray:
        return np.array(self._root_transform, copy=True)

    def _collect_mesh_nodes(self) -> list[MeshNode]:
        nodes: list[MeshNode] = []
        for node_name in self.mesh_scene.graph.nodes:
            _, geom_name = self.mesh_scene.graph.get(node_name)
            if geom_name is None or geom_name not in self.mesh_scene.geometry:
                continue
            geom = self.mesh_scene.geometry[geom_name]
            vertices = np.asarray(geom.vertices, dtype=np.float64)
            faces = np.asarray(geom.faces, dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                continue
            nodes.append(
                MeshNode(
                    node_name=str(node_name),
                    geom_name=str(geom_name),
                    local_vertices=np.array(vertices, copy=True),
                    faces=np.array(faces, copy=True),
                )
            )
        if not nodes:
            raise RuntimeError("No collision mesh nodes found in URDF scene graph.")
        return nodes

    def set_pose(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joint_pos: np.ndarray) -> None:
        root_pos = np.asarray(root_pos, dtype=np.float64).reshape(3)
        root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4)
        joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(len(self._joint_names))

        self._root_transform = np.eye(4, dtype=np.float64)
        self._root_transform[:3, :3] = _quat_wxyz_to_matrix(root_quat_wxyz)
        self._root_transform[:3, 3] = root_pos

        for i, name in enumerate(self._joint_names):
            self._joint_state[name] = float(joint_pos[i])
        self.urdf.update_cfg(self._joint_state)

    def node_world_transforms(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for mesh_node in self._mesh_nodes:
            node_tf, _ = self.mesh_scene.graph.get(mesh_node.node_name)
            out[mesh_node.node_name] = self._root_transform @ np.asarray(node_tf, dtype=np.float64)
        return out
