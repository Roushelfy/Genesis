"""
Data classes for IPC coupler.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs

if TYPE_CHECKING:
    from uipc.geometry import GeometrySlot

    from genesis.engine.entities.rigid_entity import RigidLink


class COUPLING_TYPE(IntEnum):
    TWO_WAY_SOFT_CONSTRAINT = 0
    EXTERNAL_ARTICULATION = 1
    IPC_ONLY = 2
    NONE = 3
    IPC_AUTHORITATIVE = 4


@dataclass
class ABDLinkData:
    """Per-link ABD data across all envs.

    Build-time
    ----------
    slots : list[GeometrySlot]
        IPC geometry slots, one per environment.

    Per-step inputs
    ---------------
    aim_transforms : np.ndarray | None
        (B, 4, 4) — Predicted Genesis link transforms, written as
        SoftTransformConstraint targets via the animator callback.

    Per-step outputs
    ----------------
    ipc_transforms : np.ndarray
        (B, 4, 4) — IPC-resolved link transforms.
    ipc_velocities : np.ndarray
        (B, 4, 4) — IPC-resolved velocity matrices.
    """

    slots: list[GeometrySlot]
    aim_transforms: np.ndarray | None = None
    ipc_transforms: np.ndarray | None = None
    ipc_velocities: np.ndarray | None = None


@dataclass
class ArticulatedEntityData:
    """Per-entity data for external_articulation coupling.

    External articulation always has a fixed base. Joint DOFs are coupled via
    ExternalArticulationConstraint: Genesis sends ``delta_theta_tilde`` (predicted
    joint displacement) to IPC, and reads back ``delta_theta`` (IPC-resolved
    displacement accounting for contacts).

    Build-time
    ----------
    slots : list[GeometrySlot]
        IPC articulation geometry slots, one per environment.
    q_slice : slice
        Slice into the global qpos array for this entity's generalized coordinates.
    dof_slice : slice
        Slice into the global dofs array for this entity's degrees of freedom.
    joints_child_link : list[RigidLink]
        Child link for each articulated joint.
    joints_qs_idx_local : list[int]
        Entity-local qpos index for each articulated joint.

    Per-step inputs
    ---------------
    delta_theta_tilde : np.ndarray | None
        (B, n_joints) — Predicted joint displacement (qpos - qpos_prev),
        sent to IPC as the articulation target.
    prev_qpos : np.ndarray | None
        (B, n_qs) — Entity qpos from the previous timestep, used as the
        baseline for applying IPC's delta_theta output.
    mass_matrix : np.ndarray | None
        (B, n_dofs, n_dofs) — Entity mass matrix, sent to IPC for
        articulation constraint weighting.

    Per-step outputs
    ----------------
    ipc_qpos : np.ndarray | None
        (B, n_qs) — IPC-resolved qpos computed as prev_qpos + delta_theta.
        Written back to the rigid solver after IPC advance.
    """

    slots: list[GeometrySlot]
    q_slice: slice
    dof_slice: slice
    joints_child_link: list["RigidLink"]
    joints_qs_idx_local: list[int]
    delta_theta_tilde: np.ndarray | None = None
    prev_qpos: np.ndarray | None = None
    mass_matrix: np.ndarray | None = None
    ipc_qpos: np.ndarray | None = None


@dataclass
class IpcAuthoritativeEntityData:
    """Per-entity data for ``ipc_authoritative`` coupling (IPC owns the dynamics).

    Unlike external_articulation, Genesis runs no dynamics here: each step the
    coupler pushes a per-joint scalar torque (computed Genesis-side from the
    control law) into IPC via ``AffineBodyRevoluteJointExternalForce``, IPC
    integrates the affine-body articulation, and joint angles are read back by
    signed-angle reconstruction from the parent/child body transforms.

    v1 invariant: fixed base, revolute + fixed joints only, single env.

    Build-time
    ----------
    joint_slots : list[list[GeometrySlot]]
        Per-env list of per-joint linemesh slots (each has AffineBodyRevoluteJoint
        and AffineBodyRevoluteJointExternalForce applied). Indexed [env][joint].
    joints_child_link / joints_parent_link : list[RigidLink]
        ABD parent/child link for each actuated joint (after fixed-joint merge).
    joints_qs_idx_local / joints_dof_idx_local : list[int]
        Entity-local qpos / dof index for each actuated joint (1-DOF joints).
    joints_axis_local : list[np.ndarray]
        Joint motion axis in the child link frame (for angle reconstruction).
    q0 : np.ndarray
        (n_qs,) loader-pose qpos; reconstruction baseline so the per-joint zero
        offset maps IPC's internal angle to Genesis qpos (C4 calibration).

    Per-step
    --------
    torque : np.ndarray
        (B, n_joints) per-joint scalar torque pushed to IPC each step.
    """

    joint_slots: list[list["GeometrySlot"]]
    joints_child_link: list["RigidLink"]
    joints_parent_link: list["RigidLink"]
    joints_qs_idx_local: list[int]
    joints_dof_idx_local: list[int]
    joints_axis_local: list[np.ndarray]
    q0: np.ndarray
    torque: np.ndarray | None = None
