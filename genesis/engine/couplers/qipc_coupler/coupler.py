from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis.utils.misc import geometric_mean, harmonic_mean, qd_to_torch, tensor_to_array

from .adhesion import QIPCAdhesionManager
from .affine_cluster import QIPCAffineCluster, QIPCAffineClusterManager


if TYPE_CHECKING:
    from genesis.engine.entities.fem_entity import FEMEntity
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import QIPCCouplerOptions

    from qipc.scene.joint_collection import JointCollection
    from qipc.scene.scene import Scene as QIPCScene


# ---------------------------------------------------------------------------
# Strong-typed data structures (no plain dict)
# ---------------------------------------------------------------------------


class EntityConfig(NamedTuple):
    """Per-entity QIPC configuration derived from material fields."""

    abd_kappa: float
    kappa_pivot: float
    kappa_axis: float
    default_kp: float
    default_kv: float
    home_qpos: tuple[float, ...] | None


class FreeBaseEntry(NamedTuple):
    """Tracks a free-base entity for qpos writeback."""

    entity: RigidEntity
    body_offset: int


class AbdEntityPreInit(NamedTuple):
    """Per-entity pre-init results: geometry/joints created, body offsets unresolved."""

    entity: RigidEntity
    group_slots: dict[int, object]
    link_to_rep: dict[int, int]
    T_world: dict[int, np.ndarray]
    joint_collections: list[JointCollection]
    genesis_dof_indices: list[int]
    genesis_q_indices: list[int]
    is_free_base: bool


class FemEntityEntry(NamedTuple):
    """Per-FEM-entity QIPC bookkeeping (resolved after scene.init())."""

    entity: FEMEntity
    slot: object  # qipc GeometrySlot
    is_cloth: bool
    offset: int  # fem_vert_offset into the global QIPC FEM vertex buffer
    n_verts: int


@dataclass
class FemConstraintRecord:
    """One set_vertex_constraints call on a FEM entity (QIPC backend).

    Static-target constraints are written once at creation; link-following
    constraints are refreshed every step in preprocess().
    """

    entity: FEMEntity
    verts: torch.Tensor  # (n,) int64 cuda, entity-local vertex indices
    is_soft: bool
    link: RigidLink | None = None
    link_offsets: torch.Tensor | None = None  # (n, 3) f64 cuda, in link frame


class QIPCSolverStatistics(NamedTuple):
    """Per-step QIPC solver statistics exposed without leaking the native solver."""

    step_ms: float
    newton_iters: int
    max_pcg_iters: int
    max_line_search_iters: int


class SealedGasState(NamedTuple):
    """Runtime state of one sealed-gas FEM entity."""

    p0: float
    v0: float
    enabled: bool


class _SealedGasResetState(NamedTuple):
    """Initial live state restored by `QIPCCoupler.reset`."""

    p0: float
    v0: float
    v_min: float
    enabled: float


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation: sign-preserving axis-angle to 3x3 matrix.

    Computes sin(theta) directly so negative angles rotate correctly.
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(theta)
    s = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) * c + (1 - c) * np.outer(axis, axis) + s * K


def _perpendicular_direction(axis: np.ndarray) -> np.ndarray:
    """Return a stable unit direction perpendicular to an axis."""
    candidate = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    perpendicular = candidate - np.dot(candidate, axis) * axis
    return perpendicular / np.linalg.norm(perpendicular)


def _triangle_component_count(triangles: npt.NDArray[np.int32]) -> int:
    """Count components connected through shared triangle edges."""
    n_triangles = len(triangles)
    if n_triangles == 0:
        return 0

    parents = list(range(n_triangles))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owner_by_edge: dict[tuple[int, int], int] = {}
    for triangle_index, (i, j, k) in enumerate(triangles):
        for a, b in ((i, j), (j, k), (k, i)):
            a_int = int(a)
            b_int = int(b)
            edge = (min(a_int, b_int), max(a_int, b_int))
            owner = owner_by_edge.setdefault(edge, triangle_index)
            union(triangle_index, owner)
    return len({find(index) for index in range(n_triangles)})


def _positive_gas_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or (torch.is_tensor(value) and value.dtype == torch.bool):
        gs.raise_exception(f"FEMEntity.set_gas_state: {name} must be finite and positive.")
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError):
        gs.raise_exception(f"FEMEntity.set_gas_state: {name} must be finite and positive.")
    if not np.isfinite(scalar) or scalar <= 0.0:
        gs.raise_exception(f"FEMEntity.set_gas_state: {name} must be finite and positive.")
    return scalar


# ---------------------------------------------------------------------------
# Quadrants kernels
# ---------------------------------------------------------------------------


@qd.func
def _func_mat3_to_quat(
    r00: gs.qd_float,
    r01: gs.qd_float,
    r02: gs.qd_float,
    r10: gs.qd_float,
    r11: gs.qd_float,
    r12: gs.qd_float,
    r20: gs.qd_float,
    r21: gs.qd_float,
    r22: gs.qd_float,
):
    """3x3 matrix -> quaternion (w,x,y,z) via Shepperd's method."""
    trace = r00 + r11 + r22
    w = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    if trace > 0.0:
        s = 2.0 * qd.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * qd.sqrt(1.0 + r00 - r11 - r22)
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * qd.sqrt(1.0 + r11 - r00 - r22)
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = 2.0 * qd.sqrt(1.0 + r22 - r00 - r11)
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = qd.sqrt(w * w + x * x + y * y + z * z)
    return w / norm, x / norm, y / norm, z / norm


@qd.func
def _func_q12_to_T(t, R):
    """Build 4x4 transform from translation vector and 3x3 rotation matrix."""
    T = qd.Matrix.identity(gs.qd_float, 4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@qd.func
def _func_R_to_quat(R):
    """3x3 rotation matrix to quaternion (w,x,y,z) via Shepperd's method."""
    return _func_mat3_to_quat(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )


@qd.kernel(fastcache=True)
def _kernel_qipc_writeback(
    abd_q: qd.types.ndarray(),
    body_indices: qd.types.ndarray(),
    link_indices: qd.types.ndarray(),
    rel_transforms: qd.types.ndarray(),
    dofs_pos: qd.types.ndarray(),
    dofs_vel: qd.types.ndarray(),
    dof_indices: qd.types.ndarray(),
    joint_q_indices: qd.types.ndarray(),
    free_base_body_indices: qd.types.ndarray(),
    free_base_link_indices: qd.types.ndarray(),
    free_base_q_starts: qd.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    rigid_info: array_class.RigidInfo,
):
    """Single-kernel writeback: ABD q -> links, DOFs, and generalized positions.

    All state derived from ABD body transforms (first-class truth) in one launch.
    """
    n_links = link_indices.shape[0]
    n_dofs = dof_indices.shape[0]
    n_free = free_base_body_indices.shape[0]

    for i in range(n_links):
        i_b = body_indices[i]
        link_idx = link_indices[i]

        T_body = _func_q12_to_T(
            qd.Vector([abd_q[i_b, 0], abd_q[i_b, 1], abd_q[i_b, 2]]),
            qd.Matrix(
                [
                    [abd_q[i_b, 3], abd_q[i_b, 4], abd_q[i_b, 5]],
                    [abd_q[i_b, 6], abd_q[i_b, 7], abd_q[i_b, 8]],
                    [abd_q[i_b, 9], abd_q[i_b, 10], abd_q[i_b, 11]],
                ]
            ),
        )

        T_rel = _func_q12_to_T(
            qd.Vector([rel_transforms[i, 0], rel_transforms[i, 1], rel_transforms[i, 2]]),
            qd.Matrix(
                [
                    [rel_transforms[i, 3], rel_transforms[i, 4], rel_transforms[i, 5]],
                    [rel_transforms[i, 6], rel_transforms[i, 7], rel_transforms[i, 8]],
                    [rel_transforms[i, 9], rel_transforms[i, 10], rel_transforms[i, 11]],
                ]
            ),
        )

        T_link = T_body @ T_rel

        links_state.pos[link_idx, 0] = T_link[:3, 3]
        w, x, y, z = _func_R_to_quat(T_link[:3, :3])
        links_state.quat[link_idx, 0][0] = w
        links_state.quat[link_idx, 0][1] = x
        links_state.quat[link_idx, 0][2] = y
        links_state.quat[link_idx, 0][3] = z

    for i in range(n_dofs):
        dof_idx = dof_indices[i]
        dofs_state.pos[dof_idx, 0] = dofs_pos[i]
        dofs_state.vel[dof_idx, 0] = dofs_vel[i]
        rigid_info.qpos[joint_q_indices[i], 0] = dofs_pos[i]

    for i in range(n_free):
        i_b = free_base_body_indices[i]
        link_idx = free_base_link_indices[i]
        q_start = free_base_q_starts[i]
        # pos: read from links_state (already written above)
        rigid_info.qpos[q_start, 0] = links_state.pos[link_idx, 0][0]
        rigid_info.qpos[q_start + 1, 0] = links_state.pos[link_idx, 0][1]
        rigid_info.qpos[q_start + 2, 0] = links_state.pos[link_idx, 0][2]
        # quat: read from links_state (already written above)
        rigid_info.qpos[q_start + 3, 0] = links_state.quat[link_idx, 0][0]
        rigid_info.qpos[q_start + 4, 0] = links_state.quat[link_idx, 0][1]
        rigid_info.qpos[q_start + 5, 0] = links_state.quat[link_idx, 0][2]
        rigid_info.qpos[q_start + 6, 0] = links_state.quat[link_idx, 0][3]


# ---------------------------------------------------------------------------
# QIPCCoupler
# ---------------------------------------------------------------------------


class QIPCCoupler(RBC):
    """
    QIPC coupler: uses cuda-graph-qipc as the sole physics engine for rigid/FEM entities.

    Genesis provides all scene data (link meshes, mass/inertia, joint topology);
    QIPC receives structured data and runs the physics. No asset files are loaded by QIPC.

    Design invariant: ABD body transforms are the first-class truth. Joint dof and
    joint/free-base qpos are derived products written back for observation and
    kinematics only.
    """

    def __init__(self, simulator: Simulator, options: QIPCCouplerOptions) -> None:
        self._sim: Simulator = simulator
        self._options: QIPCCouplerOptions = options
        self._adhesion: QIPCAdhesionManager = QIPCAdhesionManager(options)
        self._affine_clusters = QIPCAffineClusterManager()
        self._fem_rest_positions: dict = {}
        self._sealed_gas_bag_by_entity: dict[FEMEntity, int] = {}
        self._sealed_gas_reset_state: dict[FEMEntity, _SealedGasResetState] = {}
        self._stc_requests: list[tuple] = []
        self._stc_geos: dict[tuple, object] = {}

    @property
    def adhesion(self) -> QIPCAdhesionManager:
        """Adhesion declarations and runtime bond/beta state (see adhesion.py)."""
        return self._adhesion

    def add_adhesion(self, source_entity, target_entities=None, **kwargs) -> None:
        """Declare soft adhesion (and per-pair bond overrides) between entities.

        Convenience passthrough to ``self.adhesion.add_request`` — see
        QIPCAdhesionManager.add_request for the full parameter list. Must be
        called before scene.build().
        """
        self._adhesion.add_request(source_entity, target_entities, **kwargs)

    def add_affine_cluster(
        self,
        fem_entity: FEMEntity,
        *,
        proxy_entity: RigidEntity | None = None,
        proxy_link: RigidLink | str | None = None,
        kappa: float = 1e8,
        fixed: bool = False,
        initial_edges=None,
        initial_tris=None,
        initial_tets=None,
    ) -> QIPCAffineCluster:
        """Queue a QIPC affine cluster before ``scene.build()``.

        The returned handle remains stable across build and reset. Element
        selections are local to ``fem_entity``. An omitted ``proxy_entity``
        creates a ghost affine body using ``kappa`` and ``fixed``; an explicit
        rigid proxy resolves ``proxy_link`` to its merged QIPC body, whose own
        material/fixed-base configuration remains authoritative. Omitting all
        ``initial_*`` selections starts with empty membership, while calling
        ``join()`` or ``detach()`` without a selection targets all FEM elements.
        """
        if hasattr(self, "_scene"):
            gs.raise_exception("QIPCCoupler.add_affine_cluster must be called before scene.build().")
        if not any(entity is fem_entity for entity in self._sim.fem_solver.entities):
            gs.raise_exception("QIPCCoupler.add_affine_cluster: fem_entity is not a FEM entity owned by this scene.")
        if proxy_entity is not None:
            if not any(entity is proxy_entity for entity in self._sim.rigid_solver.entities):
                gs.raise_exception(
                    "QIPCCoupler.add_affine_cluster: proxy_entity is not a rigid entity owned by this scene."
                )
            if self._is_plane_entity(proxy_entity):
                gs.raise_exception("QIPCCoupler.add_affine_cluster: a Plane cannot be an affine-cluster proxy.")
            self._resolve_link_local(proxy_entity, proxy_link)
        return self._affine_clusters.add_request(
            fem_entity,
            proxy_entity=proxy_entity,
            proxy_link=proxy_link,
            kappa=kappa,
            fixed=fixed,
            initial_edges=initial_edges,
            initial_tris=initial_tris,
            initial_tets=initial_tets,
        )

    def set_fem_rest_positions(self, entity: FEMEntity, rest_verts) -> None:
        """Give a FEM entity rest positions that differ from its initial ones.

        Must be called before scene.build(). QIPC derives the rest metric,
        rest areas/volumes, lumped MASSES, and (for shells) the rest dihedral
        angles from the rest mesh, while the simulation starts at the entity's
        initial positions — the difference is stored as prestress (e.g. a wound
        tape coil with a flat rest strip). The rest mesh must have identical
        topology/vertex order; its absolute placement is irrelevant.
        """
        if hasattr(self, "_scene"):
            gs.raise_exception("QIPCCoupler.set_fem_rest_positions must be called before scene.build().")
        rest = np.ascontiguousarray(
            tensor_to_array(rest_verts) if torch.is_tensor(rest_verts) else rest_verts, dtype=np.float64
        )
        if rest.ndim != 2 or rest.shape[1] != 3:
            gs.raise_exception("QIPCCoupler.set_fem_rest_positions: rest_verts must have shape (n_verts, 3).")
        self._fem_rest_positions[entity] = rest

    def enable_soft_transform(self, entity: RigidEntity, link=None, strength=(1e3, 1e3)) -> None:
        """Attach a SoftTransformConstraint to a rigid entity's ABD body (pre-build).

        Inert until the first set_soft_transform_target call. ``link`` selects
        which merged body (None = base link's group; a RigidLink or link name
        otherwise). ``strength`` is the (translation, rotation) strength-ratio
        pair of the mass-weighted penalty.
        """
        if hasattr(self, "_scene"):
            gs.raise_exception("QIPCCoupler.enable_soft_transform must be called before scene.build().")
        if isinstance(strength, (int, float)):
            strength = (float(strength), float(strength))
        self._stc_requests.append((entity, link, (float(strength[0]), float(strength[1]))))

    @staticmethod
    def _resolve_link_local(entity: RigidEntity, link) -> int:
        if link is None:
            return entity.links[0].idx_local
        if isinstance(link, str):
            for candidate in entity.links:
                if candidate.name == link:
                    return candidate.idx_local
            gs.raise_exception(f"QIPCCoupler: entity has no link named '{link}'.")
        if not any(candidate is link for candidate in entity.links):
            gs.raise_exception("QIPCCoupler: selected link does not belong to the proxy entity.")
        return link.idx_local

    def _declare_affine_clusters(
        self,
        all_pre_inits: list[AbdEntityPreInit],
        fem_pre_entries: list[tuple[FEMEntity, object, bool]],
    ) -> None:
        """Resolve Genesis entities and declare QIPC proxies before init."""
        fem_slots = {entity: slot for entity, slot, _is_cloth in fem_pre_entries}
        pre_by_entity = {pre.entity: pre for pre in all_pre_inits}

        def resolve_proxy_slot(entity: RigidEntity, link) -> object:
            pre = pre_by_entity.get(entity)
            if pre is None:
                gs.raise_exception("QIPCCoupler.add_affine_cluster: proxy entity has no coupled QIPC rigid body.")
            link_local = self._resolve_link_local(entity, link)
            rep = pre.link_to_rep.get(link_local)
            slot = pre.group_slots.get(rep)
            if slot is None:
                gs.raise_exception(
                    "QIPCCoupler.add_affine_cluster: selected proxy link has no QIPC ABD collision body."
                )
            return slot

        self._affine_clusters.declare(
            self._scene,
            fem_slots=fem_slots,
            resolve_proxy_slot=resolve_proxy_slot,
        )

    def _apply_rigid_extras(self, all_pre_inits: list[AbdEntityPreInit]) -> None:
        """Per-geometry d_hat overrides + queued SoftTransformConstraints (pre-init)."""
        for pre in all_pre_inits:
            d_hat_override = getattr(pre.entity.material, "qipc_d_hat", None)
            if d_hat_override is not None:
                for slot in pre.group_slots.values():
                    geo = slot.geometry
                    if "d_hat" not in geo.meta:
                        geo.meta.create("d_hat", np.float64)
                    geo.meta["d_hat"] = np.array([float(d_hat_override)], dtype=np.float64)

        if not self._stc_requests:
            return
        from qipc.constitution import SoftTransformConstraint

        pre_by_entity = {pre.entity: pre for pre in all_pre_inits}
        for entity, link, strength in self._stc_requests:
            pre = pre_by_entity.get(entity)
            if pre is None:
                gs.raise_exception("QIPCCoupler.enable_soft_transform: entity is not a coupled ABD rigid entity.")
            link_local = self._resolve_link_local(entity, link)
            rep = pre.link_to_rep.get(link_local)
            slot = pre.group_slots.get(rep)
            if slot is None:
                gs.raise_exception(
                    "QIPCCoupler.enable_soft_transform: the selected link's merge group has no ABD body."
                )
            SoftTransformConstraint().apply_to(slot.geometry, strength_ratio=strength)
            key = (entity, None if link is None else str(link) if isinstance(link, str) else link.name)
            self._stc_geos[key] = slot.geometry

    def set_soft_transform_target(self, entity: RigidEntity, pos, quat, enabled: bool = True, link=None) -> None:
        """Drive a SoftTransformConstraint target (graph-safe device writes).

        ``enable_soft_transform`` must have been called for (entity, link)
        before build. ``quat`` is (w, x, y, z).
        """
        key = (entity, None if link is None else str(link) if isinstance(link, str) else link.name)
        geo = self._stc_geos.get(key)
        if geo is None:
            gs.raise_exception(
                "QIPCCoupler.set_soft_transform_target: call enable_soft_transform for this "
                "(entity, link) before scene.build()."
            )
        from qipc.solver.affine_body import transform_to_q

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        T[:3, :3] = gu.quat_to_R(np.asarray(quat, dtype=np.float64).reshape(4))
        q12 = np.asarray(transform_to_q(T), dtype=np.float64).reshape(-1)[:12]
        geo.instances["aim_q"].gpu()[:] = torch.as_tensor(q12, device="cuda").reshape(
            geo.instances["aim_q"].gpu().shape
        )
        geo.instances["is_constrained"].gpu()[:] = 1 if enabled else 0

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def options(self) -> QIPCCouplerOptions:
        return self._options

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self) -> None:
        from qipc import Scene as QIPCSceneCls
        from qipc import trimesh
        from qipc.constitution import AffineBodyConstitution
        from qipc.geometry import ground as qipc_ground
        from qipc.scene.joint_collection import JointCollection

        assert self._sim.n_envs <= 1, "QIPCCoupler: n_envs > 1 not supported"

        # --- Classify entities: plane vs abd ---
        all_entities: list[RigidEntity] = list(self._sim.rigid_solver.entities)
        plane_entities: list[RigidEntity] = []
        abd_entities: list[RigidEntity] = []
        for entity in all_entities:
            if self._is_plane_entity(entity):
                plane_entities.append(entity)
            else:
                abd_entities.append(entity)

        # --- Create QIPC Scene with contact + solver config ---
        scene_config: dict = {
            "contact/enable": self._options.contact_enable,
            "contact/d_hat": self._options.contact_d_hat,
            "contact/init_collision_pair_capacity": self._options.init_collision_pair_capacity,
        }
        solver_passthrough = {
            "newton/velocity_tol": self._options.solver_newton_velocity_tol,
            "newton/max_iter": self._options.solver_newton_max_iter,
            "linear_system/max_iter": self._options.solver_linear_max_iter,
            "linear_system/tol_rate": self._options.solver_linear_tol_rate,
            "linear_system/preconditioner": self._options.solver_linear_preconditioner,
            "linear_system/solver": self._options.solver_linear_solver,
            "line_search/max_iter": self._options.solver_line_search_max_iter,
            "contact/ccd_partition": (
                None if self._options.contact_ccd_partition is None else int(self._options.contact_ccd_partition)
            ),
        }
        scene_config.update({key: value for key, value in solver_passthrough.items() if value is not None})
        self._scene: QIPCScene = QIPCSceneCls(
            dt=self._sim.dt,
            gravity=tuple(self._sim._gravity),
            **scene_config,
        )

        # --- Global default contact model ---
        # Used by ground half-planes and any geometry without a per-entity contact
        # element. NOTE: QIPC only wires the friction machinery when the *default*
        # model's friction_rate > 0, so this must not be zero.
        self._scene.contact_tabular.default_model(
            friction_rate=self._options.contact_friction,
            resistance=self._options.contact_resistance,
        )

        # --- Ground planes ---
        self._ground_planes: list[tuple[np.ndarray, float]] = []
        for entity in plane_entities:
            self._create_ground(entity, qipc_ground)

        # --- ABD entities: phase 1 (pre-init, create geometry/joints) ---
        abd = AffineBodyConstitution()
        all_pre_inits: list[AbdEntityPreInit] = []
        for entity in abd_entities:
            pre = self._build_abd_entity_pre_init(entity, abd, trimesh, JointCollection)
            all_pre_inits.append(pre)

        # --- Rigid extras: per-geometry d_hat overrides + queued STCs ---
        self._apply_rigid_extras(all_pre_inits)

        # --- FEM entities (volumetric + cloth), pre-init ---
        fem_pre_entries: list[tuple[FEMEntity, object, bool]] = self._build_fem_entities()

        # --- Affine clusters: declare proxy/FEM pairings before QIPC init ---
        self._declare_affine_clusters(all_pre_inits, fem_pre_entries)

        # --- Reject vertices flush with / below a ground plane (readable error
        # instead of a device trap inside QIPC's halfplane kernel) ---
        self._preflight_ground_clearance(all_pre_inits, fem_pre_entries)

        # --- Per-entity contact elements + pairwise contact models ---
        self._setup_contact_tabular(all_pre_inits, fem_pre_entries, plane_entities)

        # --- Contact constitution (adhesion) ---
        self._adhesion.apply_constitution(self._scene, has_fem_entities=bool(fem_pre_entries))

        # --- Aggregate JointCollections and dof order (pre-init) ---
        all_jcs: list[JointCollection] = []
        all_genesis_dof_indices: list[int] = []
        all_genesis_q_indices: list[int] = []
        for pre in all_pre_inits:
            all_jcs.extend(pre.joint_collections)
            all_genesis_dof_indices.extend(pre.genesis_dof_indices)
            all_genesis_q_indices.extend(pre.genesis_q_indices)

        self._jc: JointCollection | None = JointCollection.merge(all_jcs) if all_jcs else None
        self._genesis_dof_order: torch.Tensor = torch.tensor(
            all_genesis_dof_indices, dtype=torch.int64, device=gs.device
        )

        # --- Init QIPC (assigns abd_body_offset / fem_vert_offset to each geometry) ---
        self._scene.init()

        # --- FEM post-init: resolve vertex offsets into the global FEM buffer ---
        self._fem_entries: list[FemEntityEntry] = []
        self._fem_entry_by_entity: dict[FEMEntity, FemEntityEntry] = {}
        for fem_entity, slot, is_cloth in fem_pre_entries:
            offset = int(slot.geometry.meta["fem_vert_offset"].cpu()[0])
            entry = FemEntityEntry(fem_entity, slot, is_cloth, offset, fem_entity.n_vertices)
            self._fem_entries.append(entry)
            self._fem_entry_by_entity[fem_entity] = entry
        if self._adhesion.has_bond_seed_requests():
            self._adhesion.apply_bond_seed_requests(
                {entry.entity: (entry.offset, entry.n_verts) for entry in self._fem_entries},
                self._abd_vertex_ids_by_entity(all_pre_inits),
            )
        # Membership is a post-init transaction. Seed bonds first so fully
        # internal bonds become dormant rather than being omitted/re-created.
        self._affine_clusters.initialize()
        self._initialize_sealed_gas_state()
        self._fem_constraints: list[FemConstraintRecord] = []

        # --- Phase 2 (post-init): resolve body offsets, build writeback tensors ---
        all_link_indices: list[int] = []
        all_body_indices: list[int] = []
        all_rel_transforms: list[np.ndarray] = []
        free_base_entries: list[FreeBaseEntry] = []

        for pre in all_pre_inits:
            link_indices, body_indices, rel_transforms, free_entry = self._resolve_post_init(pre)
            self._apply_gravity_compensation(pre.entity, body_indices)
            all_link_indices.extend(link_indices)
            all_body_indices.extend(body_indices)
            all_rel_transforms.extend(rel_transforms)
            if free_entry is not None:
                free_base_entries.append(free_entry)

        self._free_base_entries: list[FreeBaseEntry] = free_base_entries

        # --- Build GPU tensors for writeback ---
        self._link_indices_t: torch.Tensor = torch.tensor(all_link_indices, dtype=torch.int32, device=gs.device)
        self._body_indices_t: torch.Tensor = torch.tensor(all_body_indices, dtype=torch.int64, device=gs.device)

        rel_data = np.zeros((len(all_rel_transforms), 12), dtype=np.float64)
        for i, rt in enumerate(all_rel_transforms):
            rel_data[i] = rt
        self._rel_transforms_t: torch.Tensor = torch.tensor(rel_data, dtype=torch.float64, device=gs.device)

        n_controlled_dofs = len(all_genesis_dof_indices)
        self._dof_indices_t: torch.Tensor = torch.tensor(all_genesis_dof_indices, dtype=torch.int32, device=gs.device)
        self._joint_q_indices_t: torch.Tensor = torch.tensor(all_genesis_q_indices, dtype=torch.int32, device=gs.device)
        self._wb_dofs_pos: torch.Tensor = torch.zeros(n_controlled_dofs, dtype=gs.tc_float, device=gs.device)
        self._wb_dofs_vel: torch.Tensor = torch.zeros(n_controlled_dofs, dtype=gs.tc_float, device=gs.device)
        self._prev_theta: torch.Tensor = torch.zeros(n_controlled_dofs, dtype=torch.float64, device="cuda")

        # Free-base tensors for unified kernel writeback
        fb_body_indices = [e.body_offset for e in free_base_entries]
        fb_link_indices = [e.entity.link_start for e in free_base_entries]
        fb_q_starts = [e.entity.q_start for e in free_base_entries]
        self._free_base_body_indices_t: torch.Tensor = torch.tensor(
            fb_body_indices or [0], dtype=torch.int64, device=gs.device
        )[: len(free_base_entries)]
        self._free_base_link_indices_t: torch.Tensor = torch.tensor(
            fb_link_indices or [0], dtype=torch.int32, device=gs.device
        )[: len(free_base_entries)]
        self._free_base_q_starts_t: torch.Tensor = torch.tensor(
            fb_q_starts or [0], dtype=torch.int32, device=gs.device
        )[: len(free_base_entries)]

        # --- Debug viewer ---
        self._debug_viewer = None
        if self._options.debug_viewer:
            self._debug_viewer = self._scene.viewer
            self._debug_viewer.up_axis = "z"

        # --- Substep tracking ---
        self._substep_count: int = 0
        self._substeps_per_step: int = self._sim._substeps
        self._is_first_step: bool = True

        # --- Initial writeback ---
        self._writeback_state()

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    def reset(self, envs_idx=None) -> None:
        if envs_idx is not None:
            gs.raise_exception("QIPCCoupler.reset does not support partial environment reset.")

        runtime_gas_state = self._snapshot_sealed_gas_state()
        self._restore_sealed_gas_state()
        try:
            self._scene.reset()
        except RuntimeError:
            self._write_sealed_gas_state(runtime_gas_state)
            torch.cuda.synchronize()
            raise
        self._adhesion.restore_seeded_bonds()
        self._affine_clusters.replay_initial_membership()
        self._writeback_state()
        self._writeback_fem_state(0)

    def _joint_rows(self, entity: RigidEntity, dofs_idx_local) -> list[int]:
        local = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        global_indices = local + entity.dof_start
        row_by_dof = {int(dof): row for row, dof in enumerate(self._genesis_dof_order.cpu().tolist())}
        missing = [int(dof) for dof in global_indices if int(dof) not in row_by_dof]
        if missing:
            gs.raise_exception(f"QIPCCoupler: entity DOFs are not controlled by QIPC: {missing}.")
        return [row_by_dof[int(dof)] for dof in global_indices]

    def configure_dofs(
        self,
        entity: RigidEntity,
        dofs_idx_local,
        *,
        kp,
        kv,
        force_lower,
        force_upper,
    ) -> None:
        """Configure QIPC PD gains and force bounds using entity-local DOF indices."""
        if self._jc is None:
            gs.raise_exception("QIPCCoupler.configure_dofs requires a scene with controllable joints.")
        rows = self._joint_rows(entity, dofs_idx_local)
        self._jc[rows].set_dofs_kp(kp)
        self._jc[rows].set_dofs_kv(kv)
        self._jc[rows].set_dofs_force_range(force_lower, force_upper)

    def get_dofs_applied_force(self, entity: RigidEntity, dofs_idx_local) -> torch.Tensor:
        """Return QIPC applied forces for entity-local DOF indices."""
        if self._jc is None:
            gs.raise_exception("QIPCCoupler.get_dofs_applied_force requires controllable joints.")
        return self._jc.get_dofs_applied_force()[self._joint_rows(entity, dofs_idx_local)]

    def get_solver_statistics(self) -> QIPCSolverStatistics:
        """Return the latest solver counters as stable Python scalar values."""
        solver = self._scene.solver
        return QIPCSolverStatistics(
            step_ms=float(solver.step_ms),
            newton_iters=int(solver.newton_iters),
            max_pcg_iters=int(solver.max_pcg_iters),
            max_line_search_iters=int(solver.max_ls_iters),
        )

    def _sealed_gas_data(self):
        from qipc import SealedVolumeGas

        data = self._scene._constitution_data.get(SealedVolumeGas)
        if data is None:
            gs.raise_exception("QIPCCoupler: sealed-gas backend state is unavailable.")
        return data

    def _sealed_gas_bag(self, entity: FEMEntity) -> int:
        bag = self._sealed_gas_bag_by_entity.get(entity)
        if bag is None:
            gs.raise_exception(
                "QIPCCoupler: gas state is only available for FEM entities using gs.materials.FEM.SealedGasShell."
            )
        return bag

    def fem_get_gas_state(self, entity: FEMEntity) -> SealedGasState:
        """Return absolute reference pressure, reference volume, and enable state."""
        bag = self._sealed_gas_bag(entity)
        data = self._sealed_gas_data()
        return SealedGasState(
            p0=float(data["p0"][bag].item()),
            v0=float(data["v0"][bag].item()),
            enabled=bool(data["enabled"][bag].item()),
        )

    def fem_set_gas_state(
        self,
        entity: FEMEntity,
        *,
        p0: float | None,
        v0: float | None,
        enabled: bool | None,
    ) -> None:
        """Update graph-safe live state for one sealed-gas FEM entity.

        `p0` is absolute reference pressure in Pa. The line-search volume floor
        remains tied to the authored initial geometry when `v0` changes.
        """
        p0_scalar = _positive_gas_scalar("p0", p0) if p0 is not None else None
        v0_scalar = _positive_gas_scalar("v0", v0) if v0 is not None else None
        if enabled is not None and not isinstance(enabled, (bool, np.bool_)):
            gs.raise_exception("FEMEntity.set_gas_state: enabled must be a bool.")

        bag = self._sealed_gas_bag(entity)
        data = self._sealed_gas_data()
        if p0_scalar is not None:
            data["p0"][bag] = p0_scalar
        if v0_scalar is not None:
            data["v0"][bag] = v0_scalar
        if enabled is not None:
            data["enabled"][bag] = 1.0 if enabled else 0.0

    def preprocess(self, f: int) -> None:
        """Forward Genesis control targets to QIPC joint controller.

        Unconditionally forwards both position and velocity targets. The
        effective control mode is determined by the gain settings (kp/kv):
        - Position control: kp > 0, target_velocity = 0
        - Velocity control: kp = 0, target_velocity = user target
        - Force control: kp = 0, target_velocity = 0, use control_dofs_force
        """
        self._preprocess_fem()

        if self._jc is None:
            return

        dofs_state = self._sim.rigid_solver.dyn_state.dofs
        ctrl_pos_all: torch.Tensor = qd_to_torch(dofs_state.ctrl_pos)[:, 0].to(torch.float64)
        ctrl_vel_all: torch.Tensor = qd_to_torch(dofs_state.ctrl_vel)[:, 0].to(torch.float64)

        pos_targets: torch.Tensor = ctrl_pos_all[self._genesis_dof_order]
        vel_targets: torch.Tensor = ctrl_vel_all[self._genesis_dof_order]

        self._jc.control_dofs_position(pos_targets)
        self._jc.control_dofs_velocity(vel_targets)

    def _preprocess_fem(self) -> None:
        """Refresh link-following FEM vertex constraint targets (device writes)."""
        for record in self._fem_constraints:
            if record.link is None:
                continue
            link = record.link
            link_pos = link.get_pos().reshape(-1, 3)[0].to(torch.float64)
            link_quat = link.get_quat().reshape(-1, 4)[0].to(torch.float64)
            targets = gu.transform_by_quat(record.link_offsets, link_quat.expand(record.link_offsets.shape[0], 4))
            targets = targets + link_pos
            self._write_fem_constraint_targets(record, targets)

    def couple(self, f: int) -> None:
        self._substep_count += 1
        if self._substep_count < self._substeps_per_step:
            return
        self._substep_count = 0

        if self._is_first_step:
            # Genesis calls couple() once at the end of build() as part of its
            # initialization sequence. QIPC must not step on this call because
            # the scene was just initialized and no user control has been applied.
            self._is_first_step = False
            return

        self._scene.step()

        self._writeback_state()
        self._writeback_fem_state(f + 1)
        self._sim.rigid_solver._func_update_geoms(self._sim._scene._envs_idx)

        gs.logger.debug(
            f"[QIPC] sim={self._scene.solver.step_ms:.2f}ms "
            f"newton={self._scene.solver.newton_iters} "
            f"pcg={self._scene.solver.max_pcg_iters} "
            f"ls={self._scene.solver.max_ls_iters}"
        )

        if self._debug_viewer is not None:
            self._debug_viewer.show()

    def couple_grad(self, f: int) -> None:
        pass

    # -------------------------------------------------------------------------
    # State writeback (transform is first-class truth)
    # -------------------------------------------------------------------------

    def _writeback_state(self) -> None:
        """Write QIPC state -> Genesis buffers in a single kernel launch.

        All state derives from ABD body transforms (first-class truth):
        links_state.pos/quat, dofs_state.pos, dofs_state.vel, and free-base qpos.
        Joint velocity is finite-differenced from theta.
        """
        if self._link_indices_t.numel() == 0 and self._dof_indices_t.numel() == 0:
            # FEM-only scene: no ABD bodies/joints to write back.
            return

        abd_q: torch.Tensor = self._scene.affine_body.q

        if self._jc is not None:
            theta: torch.Tensor = self._jc.get_dofs_position()
            self._wb_dofs_pos[:] = theta.to(gs.tc_float)
            if self._scene.frame == 0:
                self._prev_theta[:] = theta
            # Finite-difference velocity: (theta - theta_prev) / dt
            self._wb_dofs_vel[:] = ((theta - self._prev_theta) / self._sim.dt).to(gs.tc_float)
            self._prev_theta[:] = theta

        _kernel_qipc_writeback(
            abd_q=abd_q,
            body_indices=self._body_indices_t,
            link_indices=self._link_indices_t,
            rel_transforms=self._rel_transforms_t,
            dofs_pos=self._wb_dofs_pos,
            dofs_vel=self._wb_dofs_vel,
            dof_indices=self._dof_indices_t,
            joint_q_indices=self._joint_q_indices_t,
            free_base_body_indices=self._free_base_body_indices_t,
            free_base_link_indices=self._free_base_link_indices_t,
            free_base_q_starts=self._free_base_q_starts_t,
            links_state=self._sim.rigid_solver.dyn_state.links,
            dofs_state=self._sim.rigid_solver.dyn_state.dofs,
            rigid_info=self._sim.rigid_solver.rigid_info,
        )

    # -------------------------------------------------------------------------
    # FEM: build, writeback, constraints (P1/P2)
    # -------------------------------------------------------------------------

    def _build_fem_entities(self) -> list[tuple[FEMEntity, object, bool]]:
        """Create one QIPC FEM geometry per Genesis FEM entity (pre-init).

        Volumetric entities become tetmeshes with StableNeoHookean; Cloth
        entities become trimeshes with the QIPC Cloth preset (Baraff-Witkin
        membrane + strain limiting + quadratic bending). Every FEM geometry
        additionally gets a resident (inert) SoftPositionConstraint so vertex
        constraints can be toggled at runtime via device writes.
        """
        fem_solver = self._sim.fem_solver
        if not fem_solver.is_active:
            return []

        from qipc import tetmesh
        from qipc import trimesh as qipc_trimesh
        from qipc.constitution import Cloth as QipcCloth
        from qipc.constitution import SealedVolumeGas, SoftPositionConstraint, StableNeoHookean
        from qipc.inflation import MeshValidationError, closed_surface_orientation

        from genesis.engine.materials.FEM import Cloth as ClothMaterial
        from genesis.engine.materials.FEM import Muscle as MuscleMaterial
        from genesis.engine.materials.FEM import SealedGasShell

        pre_entries: list[tuple[FEMEntity, object, bool]] = []
        for i_e, entity in enumerate(fem_solver.entities):
            mat = entity.material
            if isinstance(mat, MuscleMaterial):
                gs.raise_exception(
                    "QIPCCoupler: gs.materials.FEM.Muscle (actuation) is not supported by the QIPC backend."
                )
            is_cloth = isinstance(mat, ClothMaterial)
            verts = np.ascontiguousarray(tensor_to_array(entity.init_positions), dtype=np.float64)

            if is_cloth:
                faces = np.ascontiguousarray(entity.surface_triangles, dtype=np.int32)
                geo = qipc_trimesh(verts, faces)
                has_bending = mat.bending_stiffness is not None and float(mat.bending_stiffness) > 0.0
                cloth_kwargs = dict(
                    youngs_modulus=float(mat.E),
                    thickness=float(mat.thickness),
                    mass_density=float(mat.rho),
                    membrane=mat.membrane,
                    bending=mat.bending_model if has_bending else None,
                    bending_youngs_modulus=float(mat.bending_stiffness) if has_bending else None,
                    contact_thickness=(float(mat.contact_thickness) if mat.contact_thickness is not None else None),
                )
                if mat.membrane == "stvk":
                    # Continuum membrane: uses (E, nu) directly (e.g. tape).
                    cloth_kwargs["poissons_ratio"] = float(mat.nu)
                else:
                    # Baraff-Witkin takes an effective (E, G) pair, not a Poisson
                    # ratio; derive the isotropic shear modulus when not given.
                    cloth_kwargs["shear_modulus"] = (
                        float(mat.shear_modulus)
                        if mat.shear_modulus is not None
                        else float(mat.E) / (2.0 * (1.0 + float(mat.nu)))
                    )
                if mat.strain_limit_multiplier is not None:
                    cloth_kwargs["strain_limit_multiplier"] = float(mat.strain_limit_multiplier)
                QipcCloth().apply_to(geo, **cloth_kwargs)
                if isinstance(mat, SealedGasShell):
                    try:
                        closed_surface_orientation(verts, faces, auto_flip=mat.auto_flip)
                    except MeshValidationError as error:
                        gs.raise_exception_from(
                            f"QIPCCoupler: sealed-gas entity '{entity.name}' must use a closed, "
                            f"consistently wound triangle mesh: {error}",
                            error,
                        )
                    component_count = _triangle_component_count(faces)
                    if component_count != 1:
                        gs.raise_exception(
                            f"QIPCCoupler: sealed-gas entity '{entity.name}' has {component_count} disconnected "
                            "closed shells; use one FEM entity per gas bag."
                        )
                    SealedVolumeGas().apply_to(
                        geo,
                        p_gauge0=float(mat.p_gauge0),
                        p_atm=float(mat.p_atm),
                        gamma=float(mat.gamma),
                        v_min_rel=float(mat.v_min_rel),
                        auto_flip=bool(mat.auto_flip),
                    )
            else:
                if mat.model != "stable_neohookean":
                    gs.logger.warning(
                        f"QIPCCoupler: FEM material model '{mat.model}' is not supported by the QIPC "
                        "backend; falling back to stable_neohookean."
                    )
                elems = np.ascontiguousarray(entity.elems, dtype=np.int32)
                geo = tetmesh(verts, elems)
                StableNeoHookean().apply_to(
                    geo,
                    youngs_modulus=float(mat.E),
                    poissons_ratio=float(mat.nu),
                    mass_density=float(mat.rho),
                )

            # Resident soft-constraint channel: inert (is_constrained=0) until
            # FEMEntity.set_vertex_constraints toggles vertices at runtime.
            SoftPositionConstraint().apply_to(geo, strength_ratio=self._options.fem_constraint_strength)

            # Prestress channel: rest metric/masses come from the rest mesh,
            # simulation starts at the (e.g. wound) initial positions.
            rest_verts = self._fem_rest_positions.pop(entity, None)
            if rest_verts is not None:
                if rest_verts.shape != verts.shape:
                    gs.raise_exception(
                        f"QIPCCoupler: rest positions shape {rest_verts.shape} does not match FEM entity "
                        f"vertex array {verts.shape}."
                    )
                if is_cloth:
                    rest_geo = qipc_trimesh(rest_verts, faces)
                else:
                    rest_geo = tetmesh(rest_verts, elems)
                slot = self._scene.geometries.create(f"fem_{i_e}", geo, rest_geometry=rest_geo)
            else:
                slot = self._scene.geometries.create(f"fem_{i_e}", geo)
            pre_entries.append((entity, slot, is_cloth))

        if self._fem_rest_positions:
            gs.raise_exception(
                "QIPCCoupler.set_fem_rest_positions was called for an entity that is not a coupled "
                "FEM entity in this scene."
            )
        return pre_entries

    def _initialize_sealed_gas_state(self) -> None:
        """Resolve each Genesis entity to its native per-bag row after init."""
        from genesis.engine.materials.FEM import SealedGasShell

        sealed_entries = [entry for entry in self._fem_entries if isinstance(entry.entity.material, SealedGasShell)]
        if not sealed_entries:
            return

        data = self._sealed_gas_data()
        vert_index: torch.Tensor = data["vert_index"]
        vert_bag: torch.Tensor = data["vert_bag"]
        if data["p0"].numel() != len(sealed_entries):
            gs.raise_exception(
                f"QIPCCoupler: found {data['p0'].numel()} native gas bags for "
                f"{len(sealed_entries)} sealed-gas FEM entities."
            )
        used_bags: set[int] = set()
        for entry in sealed_entries:
            in_entity = (vert_index >= entry.offset) & (vert_index < entry.offset + entry.n_verts)
            bags = torch.unique(vert_bag[in_entity])
            if bags.numel() != 1:
                gs.raise_exception(
                    f"QIPCCoupler: sealed-gas entity '{entry.entity.name}' mapped to {bags.numel()} gas bags; "
                    "expected exactly one."
                )
            bag = int(bags.item())
            if bag in used_bags:
                gs.raise_exception(f"QIPCCoupler: gas bag row {bag} is shared by multiple FEM entities.")
            used_bags.add(bag)
            self._sealed_gas_bag_by_entity[entry.entity] = bag
        self._sealed_gas_reset_state = self._snapshot_sealed_gas_state()

    def _snapshot_sealed_gas_state(self) -> dict[FEMEntity, _SealedGasResetState]:
        if not self._sealed_gas_bag_by_entity:
            return {}
        data = self._sealed_gas_data()
        return {
            entity: _SealedGasResetState(
                p0=float(data["p0"][bag].item()),
                v0=float(data["v0"][bag].item()),
                v_min=float(data["v_min"][bag].item()),
                enabled=float(data["enabled"][bag].item()),
            )
            for entity, bag in self._sealed_gas_bag_by_entity.items()
        }

    def _write_sealed_gas_state(self, states: dict[FEMEntity, _SealedGasResetState]) -> None:
        if not states:
            return
        data = self._sealed_gas_data()
        for entity, state in states.items():
            bag = self._sealed_gas_bag_by_entity[entity]
            data["p0"][bag] = state.p0
            data["v0"][bag] = state.v0
            data["v_min"][bag] = state.v_min
            data["enabled"][bag] = state.enabled

    def _restore_sealed_gas_state(self) -> None:
        self._write_sealed_gas_state(self._sealed_gas_reset_state)

    def _setup_contact_tabular(
        self,
        all_pre_inits: list[AbdEntityPreInit],
        fem_pre_entries: list[tuple[FEMEntity, object, bool]],
        plane_entities: list[RigidEntity],
    ) -> None:
        """Per-entity contact elements + pairwise contact models.

        friction combines via geometric mean, resistance via harmonic mean
        (matching IPCCoupler). Ground half-planes use the default element; the
        (entity, default) pair carries the plane material's friction if a plane
        exists.
        """
        tab = self._scene.contact_tabular
        fallback_res = self._options.contact_resistance

        self._contact_elem_by_entity: dict[object, tuple[object, float, float]] = {}

        infos: list[tuple[object, float, float]] = []
        # Entities opting out of self-contact (see Rigid.qipc_self_contact).
        no_self_contact: set[int] = set()
        for i, pre in enumerate(all_pre_inits):
            mat = pre.entity.material
            mu = float(mat.coup_friction)
            res = float(mat.contact_resistance) if mat.contact_resistance is not None else fallback_res
            elem = tab.create(f"rigid_contact_{i}")
            for slot in pre.group_slots.values():
                elem.apply_to(slot.geometry)
            if not getattr(mat, "qipc_self_contact", True):
                no_self_contact.add(len(infos))
            infos.append((elem, mu, res))
            self._contact_elem_by_entity[pre.entity] = (elem, mu, res)

        for i, (fem_entity, slot, _is_cloth) in enumerate(fem_pre_entries):
            mat = fem_entity.material
            mu = float(mat.friction_mu)
            res = float(mat.contact_resistance) if mat.contact_resistance is not None else fallback_res
            elem = tab.create(f"fem_contact_{i}")
            elem.apply_to(slot.geometry)
            infos.append((elem, mu, res))
            self._contact_elem_by_entity[fem_entity] = (elem, mu, res)

        # Pairwise models, upper triangle including self-pairs (FEM/cloth
        # self-contact stays enabled).
        for i in range(len(infos)):
            elem_i, mu_i, res_i = infos[i]
            for j in range(i, len(infos)):
                elem_j, mu_j, res_j = infos[j]
                if i == j and i in no_self_contact:
                    # enable=False masks the pair in QIPC's contact system
                    # (ContactTabular is_enabled -> mask_table), sparing a robot
                    # whose collision capsules overlap by construction from
                    # fighting itself every step.
                    tab.insert(
                        elem_i,
                        elem_j,
                        friction_rate=float(mu_i),
                        resistance=float(res_i),
                        enable=False,
                    )
                    continue
                tab.insert(
                    elem_i,
                    elem_j,
                    friction_rate=float(geometric_mean(mu_i, mu_j)),
                    resistance=float(harmonic_mean(res_i, res_j)),
                )

        # Pairs against the default element (ground half-planes, unassigned geoms).
        if plane_entities:
            plane_mat = plane_entities[0].material
            mu_g = float(plane_mat.coup_friction)
            res_g = float(plane_mat.contact_resistance) if plane_mat.contact_resistance is not None else fallback_res
        else:
            mu_g = self._options.contact_friction
            res_g = fallback_res
        default_elem = tab.default_element()
        for elem_i, mu_i, res_i in infos:
            tab.insert(
                elem_i,
                default_elem,
                friction_rate=float(geometric_mean(mu_i, mu_g)),
                resistance=float(harmonic_mean(res_i, res_g)),
            )

        # Adhesion request rows LAST (insert is an upsert, so these override the
        # plain friction/resistance rows written above).
        self._adhesion.insert_tabular_rows(tab, self._contact_elem_by_entity)

    def _writeback_fem_state(self, f: int) -> None:
        """Zero-copy QIPC FEM state -> Genesis elements_v[f] (pos AND vel).

        fe.x / fe.velocities are CUDA float64 torch views over the native FEM
        buffers; per-entity slices feed the quadrants kernels directly with no
        host round trip. Written at frame f+1 of the step so save_ckpt's
        copy_frame(substeps, 0) promotes it to the next frame 0 (renderer and
        entity.get_state() both read frame 0 after the step).
        """
        if not self._fem_entries:
            return
        fe = self._scene.finite_element
        x_all: torch.Tensor = fe.x
        v_all: torch.Tensor = fe.velocities
        for entry in self._fem_entries:
            sl = slice(entry.offset, entry.offset + entry.n_verts)
            entry.entity.set_pos(f, x_all[sl].unsqueeze(0).to(gs.tc_float))
            entry.entity.set_vel(f, v_all[sl].unsqueeze(0).to(gs.tc_float))

    def _fem_entry(self, entity: FEMEntity) -> FemEntityEntry:
        entry = self._fem_entry_by_entity.get(entity)
        if entry is None:
            gs.raise_exception("QIPCCoupler: unknown FEM entity (was it added before scene.build()?).")
        return entry

    def fem_set_vertex_constraints(
        self,
        entity: FEMEntity,
        verts_idx_local,
        target_poss,
        link: RigidLink | None,
        is_soft_constraint: bool,
        stiffness: float,
    ) -> None:
        """QIPC backend for FEMEntity.set_vertex_constraints.

        Soft constraints drive vertices toward targets through the resident
        SoftPositionConstraint (mass-weighted quadratic penalty; `stiffness`
        maps to its strength ratio). Runtime hard constraints are not supported
        by current QIPC because `is_fixed` is init-only; this legacy branch will
        raise when it attempts the write.
        """
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        verts = torch.as_tensor(verts_idx_local, dtype=torch.int64, device="cuda").reshape(-1)

        if target_poss is None:
            targets = fe.x[entry.offset + verts].clone()
        else:
            targets = torch.as_tensor(target_poss, dtype=torch.float64, device="cuda").reshape(-1, 3)

        link_offsets: torch.Tensor | None = None
        if link is not None:
            link_pos = link.get_pos().reshape(-1, 3)[0].to(device="cuda", dtype=torch.float64)
            link_quat = link.get_quat().reshape(-1, 4)[0].to(device="cuda", dtype=torch.float64)
            inv_quat = gu.inv_quat(link_quat)
            link_offsets = gu.transform_by_quat(targets - link_pos, inv_quat.expand(targets.shape[0], 4))

        if is_soft_constraint:
            geo = entry.slot.geometry
            strength = float(stiffness) if stiffness else self._options.fem_constraint_strength
            geo.vertices["strength_ratio"].gpu()[verts] = strength
            geo.vertices["is_constrained"].gpu()[verts] = 1
        else:
            fe.is_fixed[entry.offset + verts] = 1

        record = FemConstraintRecord(
            entity=entity,
            verts=verts,
            is_soft=bool(is_soft_constraint),
            link=link,
            link_offsets=link_offsets,
        )
        self._write_fem_constraint_targets(record, targets)
        self._fem_constraints.append(record)

    def _write_fem_constraint_targets(self, record: FemConstraintRecord, targets: torch.Tensor) -> None:
        entry = self._fem_entry(record.entity)
        if record.is_soft:
            geo = entry.slot.geometry
            geo.vertices["aim_position"].gpu()[record.verts] = targets
        else:
            fe = self._scene.finite_element
            idx = entry.offset + record.verts
            fe.x[idx] = targets
            fe.x_prev[idx] = targets
            fe.velocities[idx] = 0.0

    def fem_update_constraint_targets(self, entity: FEMEntity, verts_idx_local, target_poss) -> None:
        """QIPC backend for FEMEntity.update_constraint_targets."""
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        verts = torch.as_tensor(verts_idx_local, dtype=torch.int64, device="cuda").reshape(-1)
        targets = torch.as_tensor(target_poss, dtype=torch.float64, device="cuda").reshape(-1, 3)

        # Soft channel: aim_position is inert for unconstrained vertices.
        geo = entry.slot.geometry
        geo.vertices["aim_position"].gpu()[verts] = targets

        # Hard channel: only teleport vertices that are actually fixed.
        idx = entry.offset + verts
        fixed_mask = fe.is_fixed[idx] > 0
        if bool(fixed_mask.any()):
            sel = idx[fixed_mask]
            fe.x[sel] = targets[fixed_mask]
            fe.x_prev[sel] = targets[fixed_mask]
            fe.velocities[sel] = 0.0

    def fem_remove_vertex_constraints(self, entity: FEMEntity, verts_idx_local=None) -> None:
        """QIPC backend for FEMEntity.remove_vertex_constraints."""
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        geo = entry.slot.geometry

        if verts_idx_local is None:
            geo.vertices["is_constrained"].gpu()[:] = 0
            fe.is_fixed[entry.offset : entry.offset + entry.n_verts] = 0
            self._fem_constraints = [r for r in self._fem_constraints if r.entity is not entity]
            return

        verts = torch.as_tensor(verts_idx_local, dtype=torch.int64, device="cuda").reshape(-1)
        geo.vertices["is_constrained"].gpu()[verts] = 0
        fe.is_fixed[entry.offset + verts] = 0

        removed = set(int(v) for v in verts.tolist())
        kept_records: list[FemConstraintRecord] = []
        for record in self._fem_constraints:
            if record.entity is not entity:
                kept_records.append(record)
                continue
            keep_mask = torch.tensor(
                [int(v) not in removed for v in record.verts.tolist()],
                dtype=torch.bool,
                device=record.verts.device,
            )
            if bool(keep_mask.all()):
                kept_records.append(record)
            elif bool(keep_mask.any()):
                record.verts = record.verts[keep_mask]
                if record.link_offsets is not None:
                    record.link_offsets = record.link_offsets[keep_mask]
                kept_records.append(record)
        self._fem_constraints = kept_records

    def fem_set_entity_position(self, entity: FEMEntity, pos) -> None:
        """Teleport a FEM entity in QIPC: write x AND x_prev (zero velocity).

        The QIPC predictor only reads x_prev/velocities, so writing x alone
        would be physically undone within one step.
        """
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        p = torch.as_tensor(pos, dtype=torch.float64, device="cuda").reshape(-1, entry.n_verts, 3)[0]
        sl = slice(entry.offset, entry.offset + entry.n_verts)
        fe.x[sl] = p
        fe.x_prev[sl] = p
        fe.velocities[sl] = 0.0

    def fem_set_entity_velocity(self, entity: FEMEntity, vel) -> None:
        """Set per-vertex velocities of a FEM entity in QIPC."""
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        v = torch.as_tensor(vel, dtype=torch.float64, device="cuda").reshape(-1, entry.n_verts, 3)[0]
        fe.velocities[entry.offset : entry.offset + entry.n_verts] = v

    def fem_set_external_acc(self, entity: FEMEntity, acc) -> None:
        """Set an external acceleration (m/s^2) on a FEM entity's vertices.

        Folded into the QIPC predictor as x_tilde = x_prev + dt*v + dt^2*(g + acc).
        ``acc`` may be (3,) (uniform) or (n_verts, 3).
        """
        entry = self._fem_entry(entity)
        fe = self._scene.finite_element
        a = torch.as_tensor(acc, dtype=torch.float64, device="cuda")
        fe.external_acc[entry.offset : entry.offset + entry.n_verts] = a

    # -------------------------------------------------------------------------
    # Build helpers: entity classification
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_plane_entity(entity: RigidEntity) -> bool:
        """Check if any geom in the entity is a plane."""
        for link in entity.links:
            for geom in link.geoms:
                if geom.type == gs.GEOM_TYPE.PLANE:
                    return True
        return False

    @staticmethod
    def _get_entity_config(entity: RigidEntity) -> EntityConfig:
        """Read per-entity QIPC config from material fields."""
        mat = entity.material
        return EntityConfig(
            abd_kappa=mat.qipc_abd_kappa or 1e8,
            kappa_pivot=mat.qipc_kappa_pivot or 1e8,
            kappa_axis=mat.qipc_kappa_axis or 1e8,
            default_kp=mat.qipc_default_kp or 100.0,
            default_kv=mat.qipc_default_kv or 10.0,
            home_qpos=tuple(mat.qipc_home_qpos) if mat.qipc_home_qpos is not None else None,
        )

    # -------------------------------------------------------------------------
    # Build helpers: ground
    # -------------------------------------------------------------------------

    def _create_ground(self, entity: RigidEntity, ground_factory) -> None:
        """Convert a Genesis Plane entity to a QIPC half-plane ground.

        The plane definitions are also recorded for the build-time ground
        clearance preflight (see _preflight_ground_clearance).
        """
        for link in entity.links:
            for geom in link.geoms:
                if geom.type != gs.GEOM_TYPE.PLANE:
                    continue
                local_normal: np.ndarray = geom.data[:3].astype(np.float64, copy=False)
                R_geom: np.ndarray = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                normal: np.ndarray = R_geom @ local_normal
                n_len = np.linalg.norm(normal)
                if n_len < 1e-15:
                    continue
                normal = normal / n_len
                height: float = float(np.dot(np.array(geom.init_pos, dtype=np.float64), normal))
                geo = ground_factory(height=height, N=tuple(normal))
                self._scene.geometries.create(f"ground_{entity.idx}", geo)
                self._ground_planes.append((normal, height))

    def _preflight_ground_clearance(
        self,
        all_pre_inits: list[AbdEntityPreInit],
        fem_pre_entries: list[tuple[FEMEntity, object, bool]],
    ) -> None:
        """Reject vertices lying exactly on (or below) a ground half-plane.

        QIPC's halfplane query kernel asserts d > 0 for every surface vertex
        (halfplane_contact_kernels.cu, since #108); a vertex flush with the
        ground -- e.g. Box(fixed=True) placed with pos_z == half_size -- trips
        a device __trap() that surfaces only as an unreadable "CUDA error:
        unspecified launch failure" at SimEngine::init. Catch it on the host
        with a readable error instead. (Release builds compile the assert out
        and silently skip such pairs -- equally undesirable.)
        """
        if not self._ground_planes or not self._options.contact_enable:
            return

        worst: tuple[float, str] | None = None

        def check(verts_world: np.ndarray, name: str) -> None:
            nonlocal worst
            for normal, height in self._ground_planes:
                d_min = float((verts_world @ normal).min() - height)
                if worst is None or d_min < worst[0]:
                    worst = (d_min, name)

        for pre in all_pre_inits:
            for link in pre.entity.links:
                if not link.geoms:
                    continue
                T_link = pre.T_world[link.idx_local]
                for geom in link.geoms:
                    v = geom.init_verts.astype(np.float64, copy=True)
                    R_geom = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                    v = (R_geom @ v.T).T + geom.init_pos
                    v = (T_link[:3, :3] @ v.T).T + T_link[:3, 3]
                    check(v, f"rigid entity {pre.entity.idx} link '{link.name}'")

        for fem_entity, _slot, _is_cloth in fem_pre_entries:
            verts = tensor_to_array(fem_entity.init_positions).astype(np.float64)
            check(verts, f"FEM entity {fem_entity.idx}")

        if worst is not None and worst[0] <= 0.0:
            d_min, name = worst
            gs.raise_exception(
                f"QIPCCoupler: {name} has vertices at signed distance {d_min:.2e} from a ground "
                "plane (must be strictly positive at build time; QIPC's halfplane kernel traps on "
                "d <= 0). Lift the entity so its lowest vertex clears the ground, ideally by at "
                f"least contact_d_hat ({self._options.contact_d_hat})."
            )

    # -------------------------------------------------------------------------
    # Build helpers: per-entity ABD construction
    # -------------------------------------------------------------------------

    def _build_abd_entity_pre_init(
        self,
        entity: RigidEntity,
        abd: object,
        trimesh_factory: object,
        joint_collection_cls: type,
    ) -> AbdEntityPreInit:
        """Phase 1: create QIPC geometry/joints for one entity (before scene.init)."""
        cfg: EntityConfig = self._get_entity_config(entity)

        T_world: dict[int, np.ndarray] = self._compute_initial_transforms(entity, cfg)
        merge_groups, link_to_rep = self._build_merge_groups(entity)

        group_slots: dict[int, object] = {}

        for rep, members in merge_groups:
            slot = self._create_merged_body(
                entity,
                rep,
                members,
                T_world,
                abd,
                trimesh_factory,
                abd_kappa=cfg.abd_kappa,
            )
            if slot is None:
                continue
            group_slots[rep] = slot

        # --- Classify joints by type ---
        revolute_joints: list[RigidJoint] = []
        prismatic_joints: list[RigidJoint] = []
        is_free_base: bool = False

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FIXED:
                continue
            elif joint.type == gs.JOINT_TYPE.REVOLUTE:
                revolute_joints.append(joint)
            elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                prismatic_joints.append(joint)
            elif joint.type == gs.JOINT_TYPE.FREE:
                is_free_base = True

        # Also detect free-base from non-fixed root link without FREE joint
        if not is_free_base:
            base_link = entity.links[0]
            if not base_link.is_fixed:
                is_free_base = True

        # --- Build joints (unified per type) ---
        per_joint_jcs: list[JointCollection] = []
        genesis_dof_indices: list[int] = []
        genesis_q_indices: list[int] = []

        init_qpos = cfg.home_qpos if cfg.home_qpos is not None else entity.init_qpos

        for joint in revolute_joints:
            jc, dof_idx = self._create_joint(
                entity,
                joint,
                "revolute",
                link_to_rep,
                group_slots,
                T_world,
                cfg,
                init_qpos,
            )
            if jc is not None:
                per_joint_jcs.append(jc)
                genesis_dof_indices.append(dof_idx)
                genesis_q_indices.append(joint.q_start)

        for joint in prismatic_joints:
            jc, dof_idx = self._create_joint(
                entity,
                joint,
                "prismatic",
                link_to_rep,
                group_slots,
                T_world,
                cfg,
                init_qpos,
            )
            if jc is not None:
                per_joint_jcs.append(jc)
                genesis_dof_indices.append(dof_idx)
                genesis_q_indices.append(joint.q_start)

        return AbdEntityPreInit(
            entity=entity,
            group_slots=group_slots,
            link_to_rep=link_to_rep,
            T_world=T_world,
            joint_collections=per_joint_jcs,
            genesis_dof_indices=genesis_dof_indices,
            genesis_q_indices=genesis_q_indices,
            is_free_base=is_free_base,
        )

    @staticmethod
    def _resolve_post_init(
        pre: AbdEntityPreInit,
    ) -> tuple[list[int], list[int], list[np.ndarray], FreeBaseEntry | None]:
        """Phase 2: after scene.init(), resolve abd_body_offset and build writeback arrays."""
        entity = pre.entity
        group_slots = pre.group_slots
        link_to_rep = pre.link_to_rep
        T_world = pre.T_world

        # Resolve link -> body offset mapping (abd_body_offset exists after init)
        link_body_indices: dict[int, int] = {}
        for link in entity.links:
            rep = link_to_rep[link.idx_local]
            if rep not in group_slots:
                continue
            slot = group_slots[rep]
            body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
            link_body_indices[link.idx_local] = body_offset

        # Free-base entry
        free_entry: FreeBaseEntry | None = None
        if pre.is_free_base:
            base_link = entity.links[0]
            base_rep = link_to_rep.get(base_link.idx_local, base_link.idx_local)
            if base_rep in group_slots:
                slot = group_slots[base_rep]
                body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
                free_entry = FreeBaseEntry(entity=entity, body_offset=body_offset)

        # Build global-index arrays for writeback
        active_links: list[RigidLink] = [link for link in entity.links if link.idx_local in link_body_indices]
        link_indices: list[int] = []
        body_indices: list[int] = []
        rel_transforms_list: list[np.ndarray] = []

        for link in active_links:
            global_link_idx = entity.link_start + link.idx_local
            link_indices.append(global_link_idx)
            body_indices.append(link_body_indices[link.idx_local])

            rep = link_to_rep[link.idx_local]
            if link.idx_local == rep:
                rt = np.zeros(12, dtype=np.float64)
                rt[3] = 1.0
                rt[7] = 1.0
                rt[11] = 1.0
            else:
                T_rep = T_world[rep]
                T_member = T_world[link.idx_local]
                T_rel = np.linalg.inv(T_rep) @ T_member
                rt = np.zeros(12, dtype=np.float64)
                rt[0:3] = T_rel[:3, 3]
                rt[3:12] = T_rel[:3, :3].flatten()
            rel_transforms_list.append(rt)

        return link_indices, body_indices, rel_transforms_list, free_entry

    def _abd_vertex_ids_by_entity(
        self,
        all_pre_inits: list[AbdEntityPreInit],
    ) -> dict[RigidEntity, np.ndarray]:
        """Map each Genesis rigid entity's authored vertex order to QIPC global ids."""
        body_id = np.asarray(self._scene.affine_body.host("body_id"), dtype=np.int64)
        vertex_ids: dict[RigidEntity, np.ndarray] = {}
        for pre in all_pre_inits:
            entity_body_ids: list[int] = []
            for slot in pre.group_slots.values():
                body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
                entity_body_ids.extend(range(body_offset, body_offset + int(slot.geometry.instances.size)))
            if entity_body_ids:
                vertex_ids[pre.entity] = np.flatnonzero(
                    np.isin(body_id, np.asarray(entity_body_ids, dtype=np.int64))
                ).astype(np.int32)
        return vertex_ids

    def _apply_gravity_compensation(self, entity: RigidEntity, body_indices: list[int]) -> None:
        """Match Genesis rigid-body gravity compensation for one QIPC entity."""
        if not body_indices or entity.gravity_compensation == 0.0:
            return
        gravity = self._scene.affine_body.gravity
        assert gravity is not None
        unique_body_indices = torch.tensor(
            sorted(set(body_indices)),
            dtype=torch.int64,
            device=gravity.device,
        )
        gravity[unique_body_indices] *= 1.0 - entity.gravity_compensation

    def _create_joint(
        self,
        entity: RigidEntity,
        joint: RigidJoint,
        jtype: str,
        link_to_rep: dict[int, int],
        group_slots: dict[int, object],
        T_world: dict[int, np.ndarray],
        cfg: EntityConfig,
        init_qpos: tuple[float, ...] | np.ndarray,
    ) -> tuple[JointCollection | None, int]:
        """Create a single QIPC joint. Returns (JointCollection, global_dof_idx) or (None, -1)."""
        child_link: RigidLink = joint.link
        parent_local: int = child_link.parent_idx - entity.link_start
        child_local: int = child_link.idx_local

        parent_rep: int = link_to_rep[parent_local]
        child_rep: int = link_to_rep[child_local]

        if parent_rep == child_rep:
            return None, -1

        if parent_rep not in group_slots or child_rep not in group_slots:
            gs.logger.warning(f"QIPCCoupler: skipping joint '{joint.name}' -- parent or child body not created.")
            return None, -1

        if jtype == "revolute":
            axis_local: np.ndarray = np.array(joint.dofs_motion_ang[0], dtype=np.float64)
            extra_kwargs = {"kappa_pivot": cfg.kappa_pivot}
        else:
            axis_local = np.array(joint.dofs_motion_vel[0], dtype=np.float64)
            extra_kwargs = {"kappa_lateral": cfg.kappa_pivot}

        T_parent_rep: np.ndarray = T_world[parent_rep]
        T_child_rep: np.ndarray = T_world[child_rep]
        T_parent_link: np.ndarray = T_world[parent_local]

        R_parent_rep_inv: np.ndarray = T_parent_rep[:3, :3].T
        R_child_rep_inv: np.ndarray = T_child_rep[:3, :3].T

        T_joint_world: np.ndarray = T_parent_link @ self._make_link_to_child_T(child_link)
        anchor_world: np.ndarray = T_joint_world[:3, 3]

        anchor_left: np.ndarray = R_parent_rep_inv @ (anchor_world - T_parent_rep[:3, 3])
        anchor_right: np.ndarray = R_child_rep_inv @ (anchor_world - T_child_rep[:3, 3])

        R_child_in_parent: np.ndarray = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
        axis_world: np.ndarray = T_parent_link[:3, :3] @ R_child_in_parent @ axis_local
        axis_world = axis_world / np.linalg.norm(axis_world)

        axis_left: np.ndarray = R_parent_rep_inv @ axis_world
        axis_right: np.ndarray = R_child_rep_inv @ axis_world

        if jtype == "prismatic":
            # Both local references must map to one world direction. Generating
            # them independently in each body frame changes the joint's rest
            # twist whenever the bodies start with a relative rotation.
            n_perp_world = _perpendicular_direction(axis_world)
            extra_kwargs["n_perp_left"] = (R_parent_rep_inv @ n_perp_world).tolist()
            extra_kwargs["n_perp_right"] = (R_child_rep_inv @ n_perp_world).tolist()

        kp, kv = self._resolve_joint_gains(joint, entity)

        dof_local: int = joint.dofs_idx_local[0]
        global_dof_idx: int = entity.dof_start + dof_local
        q_local: int = joint.q_start - entity.q_start
        if not 0 <= q_local < len(init_qpos):
            gs.raise_exception(
                f"QIPCCoupler: joint '{joint.name}' qpos index {q_local} exceeds init_qpos length {len(init_qpos)}."
            )

        jc: JointCollection = self._scene.add_joint(
            joint.name,
            type=jtype,
            left=group_slots[parent_rep],
            right=group_slots[child_rep],
            anchor_left=anchor_left.tolist(),
            anchor_right=anchor_right.tolist(),
            axis_left=axis_left.tolist(),
            axis_right=axis_right.tolist(),
            kappa_axis=cfg.kappa_axis,
            enable_controller=True,
            kp=kp,
            kv=kv,
            theta_lower=float(joint.dofs_limit[0, 0]),
            theta_upper=float(joint.dofs_limit[0, 1]),
            init_theta=float(init_qpos[q_local]),
            **extra_kwargs,
        )
        return jc, global_dof_idx

    # -------------------------------------------------------------------------
    # Build helpers: merge groups, merged body, inertials, FK, joint gains
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_merge_groups(entity: RigidEntity) -> tuple[list[tuple[int, list[int]]], dict[int, int]]:
        """Group links connected by fixed joints.

        Returns (groups, link_to_rep) where groups is a list of (rep, members)
        tuples and link_to_rep maps each idx_local to its group representative.
        """
        from collections import defaultdict

        fixed_adj: dict[int, list[int]] = defaultdict(list)

        for link in entity.links:
            for joint in link.joints:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    if link.parent_idx < 0:
                        # Fixed to the world (e.g. Box(fixed=True)): no intra-entity
                        # merge edge; the body is anchored via link.is_fixed instead.
                        continue
                    parent_local = link.parent_idx - entity.link_start
                    fixed_adj[link.idx_local].append(parent_local)
                    fixed_adj[parent_local].append(link.idx_local)

        for link in entity.links:
            if link.parent_idx >= 0 and len(link.joints) == 0:
                parent_local = link.parent_idx - entity.link_start
                fixed_adj[link.idx_local].append(parent_local)
                fixed_adj[parent_local].append(link.idx_local)

        depth: dict[int, int] = {}
        for link in entity.links:
            if link.parent_idx < 0:
                depth[link.idx_local] = 0
        bfs_queue = [idx for idx in depth]
        while bfs_queue:
            current = bfs_queue.pop(0)
            for link in entity.links:
                parent_local = link.parent_idx - entity.link_start
                if parent_local == current and link.idx_local not in depth:
                    depth[link.idx_local] = depth[current] + 1
                    bfs_queue.append(link.idx_local)

        visited: set[int] = set()
        groups: list[tuple[int, list[int]]] = []

        for link in entity.links:
            if link.idx_local in visited:
                continue
            members: list[int] = []
            queue = [link.idx_local]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                members.append(n)
                for neighbor in fixed_adj.get(n, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            rep = min(members, key=lambda x: depth.get(x, 999))
            groups.append((rep, members))

        link_to_rep: dict[int, int] = {}
        for rep, members in groups:
            for m in members:
                link_to_rep[m] = rep

        return groups, link_to_rep

    def _create_merged_body(
        self,
        entity: RigidEntity,
        rep: int,
        members: list[int],
        T_world: dict[int, np.ndarray],
        abd: object,
        trimesh_factory: object,
        *,
        abd_kappa: float,
    ) -> object | None:
        """Create a single ABD body from a merge group. Returns geometry slot or None."""
        T_rep: np.ndarray = T_world[rep]
        T_rep_inv: np.ndarray = np.linalg.inv(T_rep)
        R_rep: np.ndarray = T_rep[:3, :3]
        t_rep: np.ndarray = T_rep[:3, 3]

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vert_offset: int = 0

        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}

        for m_idx in members:
            link = link_by_idx[m_idx]
            if len(link.geoms) == 0:
                continue

            T_member: np.ndarray = T_world[m_idx]

            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R_geom = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R_geom @ v.T).T + geom.init_pos

                T_geom_in_rep = T_rep_inv @ T_member
                v_h = np.hstack([v, np.ones((len(v), 1))])
                v_rep = (T_geom_in_rep @ v_h.T).T[:, :3]

                all_verts.append(v_rep)
                all_faces.append(geom.init_faces.copy() + vert_offset)
                vert_offset += len(v_rep)

        total_mass, com_world, I_world = self._merge_inertials(entity, members, T_world)
        com_local: np.ndarray = R_rep.T @ (com_world - t_rep)
        I_local: np.ndarray = R_rep.T @ I_world @ R_rep

        is_fixed: bool = any(link_by_idx[m].is_fixed for m in members)

        if all_verts:
            merged_verts: np.ndarray = np.concatenate(all_verts, axis=0)
            merged_faces: np.ndarray = np.concatenate(all_faces, axis=0)

            geo = trimesh_factory(merged_verts, merged_faces)
            geo.instances.resize(1)
            geo.transforms = T_rep.reshape(1, 4, 4)

            # Degenerate inertia (e.g. fixed primitives with zeroed inertials)
            # would make the explicit 12x12 ABD mass matrix singular; fall back
            # to the mesh-integrated density path in that case.
            has_valid_inertia = total_mass > 0 and float(np.linalg.eigvalsh(I_local).min()) > 1e-12 * float(total_mass)
            if not has_valid_inertia and total_mass > 0:
                gs.logger.debug(
                    f"QIPCCoupler: merge group rep={rep} has degenerate inertia; "
                    "using mesh-density ABD mass instead of explicit inertials."
                )

            if has_valid_inertia:
                vol = self._compute_merged_volume(entity, members)
                abd.apply_to(
                    geo,
                    kappa=abd_kappa,
                    mass=total_mass,
                    center_of_mass=com_local,
                    inertia=I_local,
                    volume=vol,
                    is_fixed=is_fixed,
                )
            else:
                vol = self._compute_merged_volume(entity, members)
                density = total_mass / vol if total_mass > 0 else 1e3
                abd.apply_to(geo, kappa=abd_kappa, mass_density=density, is_fixed=is_fixed)

            rep_link = link_by_idx[rep]
            slot = self._scene.geometries.create(rep_link.name, geo)
            return slot
        elif total_mass > 0:
            vol = total_mass / 1e3
            geo = abd.create_proxy(
                kappa=abd_kappa,
                mass=total_mass,
                center_of_mass=com_local,
                inertia=I_local,
                volume=vol,
            )
            if is_fixed:
                geo.instances["is_fixed"] = np.array([1], dtype=np.int32)
            geo.instances["transform"] = T_rep.reshape(1, 4, 4)

            rep_link = link_by_idx[rep]
            slot = self._scene.geometries.create(rep_link.name, geo)
            return slot

        return None

    @staticmethod
    def _merge_inertials(
        entity: RigidEntity,
        members: list[int],
        T_world: dict[int, np.ndarray],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Combine inertials of multiple links using parallel axis theorem.

        Returns (total_mass, com_world, inertia_world_at_com).
        """
        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}

        total_mass: float = 0.0
        weighted_com: np.ndarray = np.zeros(3, dtype=np.float64)
        entries: list[tuple[float, np.ndarray, np.ndarray]] = []

        for m_idx in members:
            link = link_by_idx[m_idx]
            if link.inertial_mass is None or link.inertial_mass <= 0:
                continue

            m = float(link.inertial_mass)
            T_link: np.ndarray = T_world[m_idx]

            inertial_pos: np.ndarray = np.array(link.inertial_pos, dtype=np.float64)
            com_link_h = np.array([*inertial_pos, 1.0], dtype=np.float64)
            com_world_pt: np.ndarray = (T_link @ com_link_h)[:3]

            R_link: np.ndarray = T_link[:3, :3]
            R_inertial: np.ndarray = gu.quat_to_R(np.array(link.inertial_quat, dtype=np.float64))
            R_world_inertial: np.ndarray = R_link @ R_inertial
            I_principal: np.ndarray = np.array(link.inertial_i, dtype=np.float64)
            I_world_link: np.ndarray = R_world_inertial @ I_principal @ R_world_inertial.T

            entries.append((m, com_world_pt, I_world_link))
            total_mass += m
            weighted_com += m * com_world_pt

        if total_mass <= 0:
            return 0.0, np.zeros(3), np.zeros((3, 3))

        com: np.ndarray = weighted_com / total_mass

        I_combined: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        for m, com_i, I_i in entries:
            d = com_i - com
            I_combined += I_i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        return total_mass, com, I_combined

    def _resolve_joint_gains(self, joint: RigidJoint, entity: RigidEntity) -> tuple[float, float]:
        """Resolve kp/kv: material > actuator gains > coupler defaults."""
        mat = entity.material
        mat_kp: float | None = mat.qipc_default_kp
        mat_kv = mat.qipc_default_kv

        if mat_kp is not None:
            kp = float(mat_kp)
        else:
            act_gain = joint.dofs_act_gain
            if act_gain is not None and len(act_gain) > 0 and float(act_gain[0]) > 0:
                kp = float(act_gain[0])
            else:
                kp = 100.0

        if mat_kv is not None and not isinstance(mat_kv, str):
            kv = float(mat_kv)
        else:
            act_bias = joint.dofs_act_bias
            if act_bias is not None and len(act_bias) > 0 and len(act_bias[0]) >= 3 and float(-act_bias[0][2]) > 0:
                kv = float(-act_bias[0][2])
            else:
                kv = 10.0

        return kp, kv

    @staticmethod
    def _compute_merged_volume(entity: RigidEntity, members: list[int]) -> float:
        """Compute total volume for a merge group."""
        from qipc.solver.affine_body import compute_mesh_volume_trimesh

        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}
        total_vol: float = 0.0

        for m_idx in members:
            link = link_by_idx[m_idx]
            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R @ v.T).T + geom.init_pos
                f = geom.init_faces
                total_vol += abs(compute_mesh_volume_trimesh(v, f))

        return max(total_vol, 1e-12)

    @staticmethod
    def _make_link_to_child_T(child_link: RigidLink) -> np.ndarray:
        """Build the 4x4 transform from parent link frame to child link frame origin."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = child_link.pos
        T[:3, :3] = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
        return T

    @staticmethod
    def _compute_initial_transforms(entity: RigidEntity, cfg: EntityConfig) -> dict[int, np.ndarray]:
        """Compute world-frame 4x4 transforms for each link via FK at init_qpos."""
        T_world: dict[int, np.ndarray] = {}

        morph = entity.morph
        T_root = np.eye(4, dtype=np.float64)
        T_root[:3, 3] = np.array(morph.pos, dtype=np.float64)
        if morph.quat is not None:
            T_root[:3, :3] = gu.quat_to_R(np.array(morph.quat, dtype=np.float64))

        init_qpos = cfg.home_qpos if cfg.home_qpos is not None else entity.init_qpos

        for link in entity.links:
            if link.parent_idx < 0:
                T_world[link.idx_local] = T_root.copy()
            else:
                parent_local: int = link.parent_idx - entity.link_start
                T_parent: np.ndarray = T_world[parent_local]

                T_child_in_parent = np.eye(4, dtype=np.float64)
                T_child_in_parent[:3, 3] = link.pos
                T_child_in_parent[:3, :3] = gu.quat_to_R(np.array(link.quat, dtype=np.float64))

                T_joint = np.eye(4, dtype=np.float64)
                for joint in link.joints:
                    for d in range(joint.n_dofs):
                        q_local = joint.q_start - entity.q_start + d
                        if not 0 <= q_local < len(init_qpos):
                            gs.raise_exception(
                                f"QIPCCoupler: joint '{joint.name}' qpos index {q_local} "
                                f"exceeds init_qpos length {len(init_qpos)}."
                            )
                        theta = init_qpos[q_local]
                        if joint.type == gs.JOINT_TYPE.REVOLUTE:
                            axis = np.array(joint.dofs_motion_ang[d], dtype=np.float64)
                            T_joint[:3, :3] = T_joint[:3, :3] @ _rodrigues(axis, float(theta))
                        elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                            axis = joint.dofs_motion_vel[d]
                            T_joint[:3, 3] += axis * theta

                T_world[link.idx_local] = T_parent @ T_child_in_parent @ T_joint

        return T_world
