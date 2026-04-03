"""URDFController: load a URDF robot, FK/IK, drive links via SoftTransformConstraint."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ikpy.chain import Chain as IkChain
from ikpy.link import OriginLink, URDFLink
from yourdfpy import URDF

from uipc import view
import uipc.builtin as builtin
from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
from uipc.geometry import label_surface, trimesh
from uipc.geometry import affine_body
from uipc.unit import MPa


def _axis_angle_mat4(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: axis-angle -> 4x4 homogeneous."""
    k = np.asarray(axis, dtype=np.float64).ravel()
    n = np.linalg.norm(k)
    if n < 1e-12:
        return np.eye(4, dtype=np.float64)
    k = k / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    return M


def _translate_mat4(offset: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, 3] = np.asarray(offset, dtype=np.float64).ravel()
    return M


def _mat4_to_rpy(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose 4x4 homogeneous matrix into (translation(3,), rpy(3,)).

    RPY follows the URDF / ikpy convention (static-XYZ Euler angles):
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    t = M[:3, 3].copy()
    R = M[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return t, np.array([roll, pitch, yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------


@dataclass
class JointInfo:
    name: str
    type: str  # revolute / prismatic / fixed / continuous
    parent_link: str
    child_link: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray  # (3,)
    lower: float
    upper: float


@dataclass
class InertialInfo:
    mass: float
    center_of_mass: np.ndarray  # (3,) in link frame
    inertia: np.ndarray  # (3,3) about CoM in link frame


@dataclass
class MeshNode:
    node_name: str
    geom_name: str
    parent_link: str
    local_vertices: np.ndarray
    faces: np.ndarray


@dataclass
class BodyBinding:
    node_name: str
    object_name: str
    parent_link: str
    obj: Any
    geo_slot: Any
    rest_geo_slot: Any = None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class URDFController:
    """Load URDF, FK/IK, drive links via AffineBodyConstitution + SoftTransformConstraint.

    Usage::

        ctrl = URDFController("robot.urdf")
        ctrl.set_joint_positions({"Joint1": 0.5})

        # IPC integration
        ctrl.create_ipc_bodies(scene)
        ctrl.apply_to_scene(snap=True)
        ctrl.bind_animator(scene)

        # IK
        ctrl.solve_ik("Gripper_Tip_R", target_xyz)
    """

    def __init__(self, urdf_path: str | Path, mesh_source: str = "collision") -> None:
        self.urdf_path = Path(urdf_path)
        if mesh_source not in ("collision", "visual"):
            raise ValueError("mesh_source must be 'collision' or 'visual'")

        load_coll = mesh_source == "collision"
        self.urdf = URDF.load(
            str(self.urdf_path),
            build_collision_scene_graph=load_coll,
            load_collision_meshes=load_coll,
            build_scene_graph=not load_coll,
            load_meshes=not load_coll,
        )
        self.mesh_scene = self.urdf.collision_scene if load_coll else self.urdf.scene
        if self.mesh_scene is None:
            raise RuntimeError(f"Failed to load {mesh_source} scene from {self.urdf_path}")

        # Kinematic tree
        self._joints: dict[str, JointInfo] = {}
        self._child_to_joint: dict[str, str] = {}
        self._root_link: str = ""
        self._build_kinematic_tree()

        # Actuated joints
        self._actuated_names: list[str] = list(self.urdf.actuated_joint_names)
        self._joint_state: dict[str, float] = {n: 0.0 for n in self._actuated_names}

        # Mesh geometry nodes (from yourdfpy scene graph)
        self._mesh_nodes: list[MeshNode] = self._collect_mesh_nodes()

        # Link inertial data (mass, CoM, inertia) from URDF
        self._link_inertials: dict[str, InertialInfo] = self._parse_link_inertials()

        # Optional world offset for the entire robot
        self._root_transform = np.eye(4, dtype=np.float64)

        # IPC bindings (populated by create_ipc_bodies)
        self._bindings: list[BodyBinding] = []
        self._robot_contact: Any = None

        # ikpy chains (lazily built per end-effector)
        self._ik_chains: dict[str, tuple[IkChain, list[tuple[int, str]]]] = {}

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_kinematic_tree(self) -> None:
        child_set: set[str] = set()
        for joint in self.urdf.robot.joints:
            origin = np.eye(4, dtype=np.float64)
            if joint.origin is not None:
                origin = np.asarray(joint.origin, dtype=np.float64).reshape(4, 4)
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if joint.axis is not None:
                axis = np.asarray(joint.axis, dtype=np.float64).ravel()
            lo, hi = -np.pi, np.pi
            if joint.limit is not None:
                if joint.limit.lower is not None:
                    lo = float(joint.limit.lower)
                if joint.limit.upper is not None:
                    hi = float(joint.limit.upper)
            self._joints[joint.name] = JointInfo(
                name=joint.name,
                type=joint.type,
                parent_link=joint.parent,
                child_link=joint.child,
                origin=origin,
                axis=axis,
                lower=lo,
                upper=hi,
            )
            self._child_to_joint[joint.child] = joint.name
            child_set.add(joint.child)

        all_links = {link.name for link in self.urdf.robot.links}
        roots = all_links - child_set
        self._root_link = sorted(roots)[0] if roots else self.urdf.robot.links[0].name

    def _collect_mesh_nodes(self) -> list[MeshNode]:
        link_names = {j.parent_link for j in self._joints.values()} | {j.child_link for j in self._joints.values()}
        nodes: list[MeshNode] = []
        for node_name in self.mesh_scene.graph.nodes:
            _, geom_name = self.mesh_scene.graph.get(node_name)
            if geom_name is None or geom_name not in self.mesh_scene.geometry:
                continue
            geom = self.mesh_scene.geometry[geom_name]
            verts = np.asarray(geom.vertices, dtype=np.float64)
            faces = np.asarray(geom.faces, dtype=np.int32)
            if verts.size == 0 or faces.size == 0:
                continue
            # Walk up scene graph to find the owning URDF link
            parent_link = ""
            cur = node_name
            while cur is not None:
                if str(cur) in link_names:
                    parent_link = str(cur)
                    break
                p = self.mesh_scene.graph.transforms.parents.get(cur)
                cur = p if isinstance(p, str) else None
            nodes.append(
                MeshNode(
                    node_name=str(node_name),
                    geom_name=str(geom_name),
                    parent_link=parent_link,
                    local_vertices=verts.copy(),
                    faces=faces.copy(),
                )
            )
        return nodes

    def _parse_link_inertials(self) -> dict[str, InertialInfo]:
        """Extract mass, center-of-mass and inertia tensor from each URDF link."""
        result: dict[str, InertialInfo] = {}
        for link in self.urdf.robot.links:
            if link.inertial is None:
                continue
            m = float(link.inertial.mass)
            origin = np.eye(4, dtype=np.float64)
            if link.inertial.origin is not None:
                origin = np.asarray(link.inertial.origin, dtype=np.float64).reshape(4, 4)
            com = origin[:3, 3].copy()
            I_cm = np.asarray(link.inertial.inertia, dtype=np.float64).reshape(3, 3)
            result[link.name] = InertialInfo(mass=m, center_of_mass=com, inertia=I_cm)
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        return list(self._actuated_names)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {n: (self._joints[n].lower, self._joints[n].upper) for n in self._actuated_names}

    @property
    def mesh_nodes(self) -> list[MeshNode]:
        return self._mesh_nodes

    @property
    def bindings(self) -> list[BodyBinding]:
        return self._bindings

    @property
    def robot_contact(self) -> Any:
        return self._robot_contact

    def find_binding(self, pattern: str) -> BodyBinding | None:
        """Return the first binding whose node_name, object_name, or parent_link matches *pattern*."""
        for b in self._bindings:
            if (
                fnmatch.fnmatch(b.node_name, pattern)
                or fnmatch.fnmatch(b.object_name, pattern)
                or fnmatch.fnmatch(b.parent_link, pattern)
            ):
                return b
        return None

    # ------------------------------------------------------------------
    # Root transform
    # ------------------------------------------------------------------

    def set_root_transform(self, transform: np.ndarray) -> None:
        self._root_transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)

    @property
    def root_transform(self) -> np.ndarray:
        return self._root_transform.copy()

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def set_joint_positions(self, positions: dict[str, float] | np.ndarray) -> None:
        if isinstance(positions, np.ndarray):
            arr = positions.ravel()
            for i, name in enumerate(self._actuated_names):
                if i < len(arr):
                    self._joint_state[name] = float(arr[i])
        else:
            for name, val in positions.items():
                if name in self._joint_state:
                    self._joint_state[name] = float(val)
        self._clamp_joints()
        self.urdf.update_cfg(self._joint_state)

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._joint_state)

    def get_joint_array(self) -> np.ndarray:
        return np.array([self._joint_state[n] for n in self._actuated_names], dtype=np.float64)

    def _clamp_joints(self) -> None:
        for name in self._actuated_names:
            info = self._joints[name]
            self._joint_state[name] = float(np.clip(self._joint_state[name], info.lower, info.upper))

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def _joint_transform(self, joint_name: str) -> np.ndarray:
        """Origin transform + joint-variable motion for one joint."""
        info = self._joints[joint_name]
        T = info.origin.copy()
        q = self._joint_state.get(info.name, 0.0)
        if info.type in ("revolute", "continuous"):
            T = T @ _axis_angle_mat4(info.axis, q)
        elif info.type == "prismatic":
            T = T @ _translate_mat4(q * info.axis)
        return T

    def get_link_transform(self, link_name: str) -> np.ndarray:
        """World 4x4 transform for *any* link (including meshless ones like Gripper_Tip)."""
        chain: list[str] = []
        cur = link_name
        while cur in self._child_to_joint:
            jname = self._child_to_joint[cur]
            chain.append(jname)
            cur = self._joints[jname].parent_link
        T = np.eye(4, dtype=np.float64)
        for jname in reversed(chain):
            T = T @ self._joint_transform(jname)
        return self._root_transform @ T

    def get_link_position(self, link_name: str) -> np.ndarray:
        return self.get_link_transform(link_name)[:3, 3].copy()

    def get_mesh_transforms(self) -> dict[str, np.ndarray]:
        """World transforms for all mesh nodes (via yourdfpy scene graph + root_transform)."""
        out: dict[str, np.ndarray] = {}
        for node in self._mesh_nodes:
            tf, _ = self.mesh_scene.graph.get(node.node_name)
            out[node.node_name] = self._root_transform @ np.asarray(tf, dtype=np.float64)
        return out

    # ------------------------------------------------------------------
    # IPC body creation
    # ------------------------------------------------------------------

    def _node_matches(self, node: MeshNode, patterns: list[str]) -> bool:
        """Match a mesh node against glob patterns (checks node_name and parent_link)."""
        raw = str(node.node_name)
        base = raw.split(".", 1)[0]
        link = node.parent_link
        return any(fnmatch.fnmatch(raw, p) or fnmatch.fnmatch(base, p) or fnmatch.fnmatch(link, p) for p in patterns)

    @staticmethod
    def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
        """Signed volume of a closed triangle mesh (absolute value)."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        return float(abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0))

    def create_ipc_bodies(
        self,
        scene: Any,
        object_prefix: str = "robot_link",
        kappa_mpa: float = 80.0,
        stc_strength: np.ndarray | None = None,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> None:
        """Create one AffineBodyConstitution + SoftTransformConstraint per mesh node.

        Uses rigid-body mass/inertia from the URDF to build the 12x12 ABD
        mass matrix via ``affine_body.from_rigid_body``.  Falls back to
        ``mass_density=1000`` when a link has no ``<inertial>`` tag.

        Args:
            ignore_patterns: Glob patterns for nodes to *skip*.
            include_patterns: If given, *only* nodes matching these patterns
                are included (ignore_patterns still applies on top).
        """
        abd = AffineBodyConstitution()
        stc = SoftTransformConstraint()
        kappa = float(kappa_mpa) * MPa
        strength = (
            np.asarray(stc_strength, dtype=np.float64)
            if stc_strength is not None
            else np.array([120.0, 120.0], dtype=np.float64)
        )
        ign = ignore_patterns or []
        inc = include_patterns

        self._robot_contact = scene.contact_tabular().create(f"{object_prefix}_contact")
        self._bindings.clear()
        self._sim_node_names: set[str] = set()

        for i, node in enumerate(self._mesh_nodes):
            if self._node_matches(node, ign):
                continue
            if inc is not None and not self._node_matches(node, inc):
                continue

            obj_name = f"{object_prefix}_{i}_{node.node_name}"
            sc = trimesh(node.local_vertices, node.faces)
            label_surface(sc)

            inertial = self._link_inertials.get(node.parent_link)
            if inertial is not None:
                mass_12x12 = affine_body.from_rigid_body(inertial.mass, inertial.center_of_mass, inertial.inertia)
                vol = max(self._mesh_volume(node.local_vertices, node.faces), 1e-9)
                abd.apply_to(sc, kappa, mass_12x12, vol)
            else:
                abd.apply_to(sc, kappa, mass_density=1000.0)

            view(sc.instances().find(builtin.is_dynamic))[:] = 0

            stc.apply_to(sc, strength)
            self._robot_contact.apply_to(sc)

            obj = scene.objects().create(obj_name)
            geo_slot, rest_geo_slot = obj.geometries().create(sc)
            self._bindings.append(
                BodyBinding(
                    node_name=node.node_name,
                    object_name=obj_name,
                    parent_link=node.parent_link,
                    obj=obj,
                    geo_slot=geo_slot,
                    rest_geo_slot=rest_geo_slot,
                )
            )
            self._sim_node_names.add(node.node_name)

    def register_visual_only_meshes(
        self,
        transparency: float = 0.45,
        edge_width: float = 0.0,
        color: tuple[float, float, float] = (0.7, 0.7, 0.7),
    ) -> dict[str, Any]:
        """Register non-simulated mesh nodes as polyscope visual-only meshes.

        Must be called after ``create_ipc_bodies()``.  Returns a dict of
        ``{node_name: ps_mesh}`` for later updating via ``update_visual_meshes()``.
        """
        import polyscope as ps

        sim_names = getattr(self, "_sim_node_names", set())
        visual_meshes: dict[str, Any] = {}
        for node in self._mesh_nodes:
            if node.node_name in sim_names:
                continue
            m = ps.register_surface_mesh(f"visual_{node.node_name}", node.local_vertices, node.faces)
            m.set_transparency(transparency)
            m.set_edge_width(edge_width)
            m.set_color(color)
            visual_meshes[node.node_name] = (m, node)
        self._visual_meshes = visual_meshes
        return visual_meshes

    def update_visual_meshes(self) -> None:
        """Push FK transforms to visual-only polyscope meshes."""
        if not hasattr(self, "_visual_meshes"):
            return
        transforms = self.get_mesh_transforms()
        for name, (mesh, node) in self._visual_meshes.items():
            if name not in transforms:
                continue
            tf = transforms[name]
            world_verts = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
            mesh.update_vertex_positions(world_verts)

    def apply_to_scene(self, snap: bool = False) -> None:
        """Push current FK transforms to IPC bodies."""
        transforms = self.get_mesh_transforms()
        for b in self._bindings:
            if b.node_name not in transforms:
                continue
            geo = b.geo_slot.geometry()
            tf = transforms[b.node_name]
            view(geo.instances().find(builtin.is_constrained))[0] = 1
            view(geo.instances().find(builtin.aim_transform))[0] = tf
            if snap:
                view(geo.transforms())[0] = tf

    def bind_animator(
        self,
        scene: Any,
        on_frame: Any | None = None,
    ) -> None:
        """Register an animation callback that applies current transforms each frame.

        Args:
            scene: The UIPC Scene.
            on_frame: Optional ``callable(info, controller)`` invoked each substep
                *before* ``apply_to_scene()``.  Use it to update joint positions
                based on ``info.frame()`` (e.g. keyframe interpolation).
        """
        if not self._bindings:
            raise RuntimeError("No IPC bodies. Call create_ipc_bodies() first.")

        def _animate(info: Any) -> None:
            if on_frame is not None:
                on_frame(info, self)
            self.apply_to_scene()

        scene.animator().insert(self._bindings[0].obj, _animate)

    def configure_contact(
        self,
        scene: Any,
        enable_self: bool = False,
        enable_default: bool = True,
        friction: float = 0.5,
        resistance: float = 1.0e9,
    ) -> None:
        tab = scene.contact_tabular()
        tab.insert(self._robot_contact, self._robot_contact, friction, resistance, enable_self)
        tab.insert(tab.default_element(), self._robot_contact, friction, resistance, enable_default)

    # ------------------------------------------------------------------
    # Inverse kinematics (ikpy)
    # ------------------------------------------------------------------

    def find_arm_joints(self, tip_link: str) -> list[str]:
        """Walk from *tip_link* to root, collecting actuated revolute/continuous joints."""
        joints: list[str] = []
        cur = tip_link
        while cur in self._child_to_joint:
            jname = self._child_to_joint[cur]
            info = self._joints[jname]
            if info.type in ("revolute", "continuous") and jname in self._joint_state:
                joints.append(jname)
            cur = info.parent_link
        joints.reverse()
        return joints

    def find_end_effectors(self) -> list[str]:
        """Leaf links that are never a parent in any joint."""
        parents = {info.parent_link for info in self._joints.values()}
        children = {info.child_link for info in self._joints.values()}
        return sorted(children - parents)

    def _trace_joints_to_root(self, link_name: str) -> list[str]:
        """All joints from *link_name* to root, in tip-to-root order."""
        chain: list[str] = []
        cur = link_name
        while cur in self._child_to_joint:
            jname = self._child_to_joint[cur]
            chain.append(jname)
            cur = self._joints[jname].parent_link
        return chain

    def _build_ik_chain(
        self, tip_link: str, active_joints: set[str] | None = None
    ) -> tuple[IkChain, list[tuple[int, str]]]:
        """Build an ikpy ``Chain`` from the URDF root to *tip_link*.

        Returns ``(chain, joint_map)`` where *joint_map* is a list of
        ``(index_in_chain, joint_name)`` for every active joint so
        that IK results can be written back to ``_joint_state``.
        """
        chain_joints = self._trace_joints_to_root(tip_link)
        chain_joints.reverse()  # root -> tip

        links: list[OriginLink | URDFLink] = [OriginLink()]
        joint_map: list[tuple[int, str]] = []
        active_mask: list[bool] = [False]

        for jname in chain_joints:
            info = self._joints[jname]
            trans, rpy = _mat4_to_rpy(info.origin)

            kwargs: dict[str, Any] = {
                "name": info.child_link,
                "origin_translation": trans,
                "origin_orientation": rpy,
            }

            is_active = False
            if info.type in ("revolute", "continuous"):
                kwargs["joint_type"] = "revolute"
                kwargs["rotation"] = info.axis.tolist()
                is_active = jname in self._joint_state if active_joints is None else jname in active_joints
            elif info.type == "prismatic":
                kwargs["joint_type"] = "prismatic"
                kwargs["translation"] = info.axis.tolist()
                is_active = jname in self._joint_state if active_joints is None else jname in active_joints
            else:
                kwargs["joint_type"] = "fixed"

            if is_active:
                kwargs["bounds"] = (info.lower, info.upper)

            links.append(URDFLink(**kwargs))
            active_mask.append(is_active)
            if is_active:
                joint_map.append((len(links) - 1, jname))

        return IkChain(name=f"chain_{tip_link}", links=links, active_links_mask=active_mask), joint_map

    def _get_ik_chain(
        self, tip_link: str, arm_joints: list[str] | None = None
    ) -> tuple[IkChain, list[tuple[int, str]]]:
        """Return (possibly cached) ikpy chain for *tip_link*."""
        if arm_joints is not None:
            return self._build_ik_chain(tip_link, active_joints=set(arm_joints))
        if tip_link not in self._ik_chains:
            self._ik_chains[tip_link] = self._build_ik_chain(tip_link)
        return self._ik_chains[tip_link]

    def solve_ik(
        self,
        target_link: str,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None = None,
        orientation_mode: str | None = None,
        arm_joints: list[str] | None = None,
    ) -> dict[str, float]:
        """Solve IK via **ikpy** (L-BFGS-B optimisation).

        Parameters
        ----------
        target_link : str
            Name of the URDF link whose position should reach the target
            (e.g. ``"Gripper_Tip_R"``).
        target_position : (3,) array
            Desired world-space XYZ position.
        target_orientation : (3, 3) array, optional
            Desired rotation matrix.  Only used when *orientation_mode*
            is also set.
        orientation_mode : str, optional
            One of ``None``, ``"X"``, ``"Y"``, ``"Z"``, ``"all"``.
            Passed to ``ikpy.Chain.inverse_kinematics_frame``.
        arm_joints : list[str], optional
            Restrict IK to these joints only.  Defaults to all actuated
            joints on the kinematic path from root to *target_link*.

        Returns
        -------
        dict[str, float]
            Updated joint-state dictionary.
        """
        chain, joint_map = self._get_ik_chain(target_link, arm_joints)
        if not joint_map:
            return dict(self._joint_state)

        target_pos = np.asarray(target_position, dtype=np.float64).ravel()[:3]

        # Account for root transform
        if not np.allclose(self._root_transform, np.eye(4)):
            root_inv = np.linalg.inv(self._root_transform)
            target_pos = (root_inv[:3, :3] @ target_pos) + root_inv[:3, 3]

        # Initial seed from current joint state
        initial = np.zeros(len(chain.links), dtype=np.float64)
        for idx, jname in joint_map:
            initial[idx] = self._joint_state.get(jname, 0.0)

        # Solve
        if target_orientation is not None and orientation_mode is not None:
            target_frame = np.eye(4, dtype=np.float64)
            target_frame[:3, 3] = target_pos
            target_frame[:3, :3] = np.asarray(target_orientation, dtype=np.float64).reshape(3, 3)
            result = chain.inverse_kinematics_frame(
                target_frame, initial_position=initial, orientation_mode=orientation_mode
            )
        else:
            result = chain.inverse_kinematics(target_pos, initial_position=initial)

        # Write back
        for idx, jname in joint_map:
            self._joint_state[jname] = float(result[idx])
        self._clamp_joints()
        self.urdf.update_cfg(self._joint_state)
        return dict(self._joint_state)
