"""Post-init component state overlays for composed QIPC scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import genesis as gs


@dataclass(frozen=True)
class _RigidInitialStateRequest:
    entity: object
    body_q: dict[str, np.ndarray]
    joint_theta: dict[str, float]


class QIPCInitialStateManager:
    """Apply named ABD/joint snapshots after QIPC assigns composed-scene IDs."""

    def __init__(self) -> None:
        self._requests: list[_RigidInitialStateRequest] = []

    def add_rigid_request(self, entity, *, body_q: dict[str, np.ndarray], joint_theta: dict[str, float]) -> None:
        if any(request.entity is entity for request in self._requests):
            gs.raise_exception("QIPCCoupler.set_rigid_initial_state: entity already has an initial state request.")
        normalized_q: dict[str, np.ndarray] = {}
        for link_name, values in body_q.items():
            q = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
            if q.shape != (12,) or not np.isfinite(q).all():
                gs.raise_exception("QIPCCoupler.set_rigid_initial_state: every body q must be finite with shape (12,).")
            normalized_q[str(link_name)] = q.copy()
        if not normalized_q:
            gs.raise_exception("QIPCCoupler.set_rigid_initial_state: body_q must not be empty.")
        normalized_theta = {str(name): float(value) for name, value in joint_theta.items()}
        if not all(np.isfinite(value) for value in normalized_theta.values()):
            gs.raise_exception("QIPCCoupler.set_rigid_initial_state: joint angles must be finite.")
        self._requests.append(
            _RigidInitialStateRequest(
                entity=entity,
                body_q=normalized_q,
                joint_theta=normalized_theta,
            )
        )

    def apply(self, scene, all_pre_inits: list, joint_collection) -> bool:
        if not self._requests:
            return False
        pre_by_entity = {pre.entity: pre for pre in all_pre_inits}
        ab = scene.affine_body
        q = ab.q
        q_prev = ab.q_prev
        q_v = ab.q_v
        assert q is not None and q_prev is not None and q_v is not None

        joint_system = scene.joint_system
        joint_index_by_slot: dict[int, int] = {}
        if joint_collection is not None:
            indices = joint_collection._joint_dof_indices
            if indices is None:
                gs.raise_exception("QIPCCoupler: merged joint collection was not activated after QIPC init.")
            slot_ids = joint_collection._joint_slot_ids
            if len(slot_ids) != len(indices):
                gs.raise_exception("QIPCCoupler: expected one QIPC DOF per merged revolute/prismatic joint slot.")
            joint_index_by_slot = dict(zip(slot_ids, indices.tolist(), strict=True))
        for request in self._requests:
            pre = pre_by_entity.get(request.entity)
            if pre is None:
                gs.raise_exception("QIPCCoupler.set_rigid_initial_state: entity has no coupled QIPC rigid body.")
            links_by_name = {link.name: link for link in request.entity.links}
            used_reps: set[int] = set()
            for link_name, body_q in request.body_q.items():
                link = links_by_name.get(link_name)
                if link is None:
                    gs.raise_exception(f"QIPCCoupler.set_rigid_initial_state: entity has no link named '{link_name}'.")
                rep = pre.link_to_rep[link.idx_local]
                if rep in used_reps:
                    gs.raise_exception(
                        "QIPCCoupler.set_rigid_initial_state: multiple names select the same fixed-merged body."
                    )
                used_reps.add(rep)
                slot = pre.group_slots.get(rep)
                if slot is None:
                    gs.raise_exception(
                        f"QIPCCoupler.set_rigid_initial_state: link '{link_name}' has no QIPC collision body."
                    )
                body_index = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
                q_value = torch.as_tensor(body_q, dtype=torch.float64, device="cuda")
                q[body_index].copy_(q_value)
                q_prev[body_index].copy_(q_value)
                q_v[body_index].zero_()

            if request.joint_theta:
                joint_indices: dict[str, int] = {}
                for collection in pre.joint_collections:
                    names = collection.joint_names
                    slot_ids = collection._joint_slot_ids
                    if len(names) != len(slot_ids):
                        gs.raise_exception("QIPCCoupler: joint collection names do not match its slots.")
                    for name, slot_id in zip(names, slot_ids, strict=True):
                        if name in joint_indices:
                            gs.raise_exception(
                                f"QIPCCoupler.set_rigid_initial_state: entity has duplicate joint name '{name}'."
                            )
                        index = joint_index_by_slot.get(slot_id)
                        if index is None:
                            gs.raise_exception(
                                f"QIPCCoupler: joint slot for '{name}' is absent from the merged collection."
                            )
                        joint_indices[name] = index
                missing = sorted(set(request.joint_theta) - set(joint_indices))
                if missing:
                    gs.raise_exception(
                        "QIPCCoupler.set_rigid_initial_state: unknown or uncoupled joints: " + ", ".join(missing)
                    )
                for name, theta in request.joint_theta.items():
                    index = joint_indices[name]
                    joint_system.theta[index] = theta
                    joint_system.target_theta[index] = theta
                    joint_system.target_velocity[index] = 0.0
                    joint_system.control_force[index] = 0.0
                    joint_system.applied_force[index] = 0.0
        torch.cuda.synchronize()
        return True

    @staticmethod
    def rebuild_and_capture_reset(scene) -> None:
        """Rebuild derived positions/contact state and promote the overlay to reset."""
        if not hasattr(scene, "_capture_reset_state"):
            gs.raise_exception("QIPCCoupler initial-state overlays require QIPC Scene reset-state capture support.")

        # The native reset pipeline recomputes global positions, BVHs, and
        # contact candidates from the overlaid ABD/FEM state without advancing
        # physics. Bond slots have already been restored at this point.
        scene._solver.reset()
        # qipc #294 replaced the tensor-valued _drstate_views() with accessors that
        # resolve a field only when called with a DRInfo.
        from qipc._src.native.solver import DRInfo

        accessors = scene._drstate_accessors()

        def field(key):
            accessor = accessors.get(key)
            return None if accessor is None else accessor(DRInfo())

        positions = field("GlobalVertexManager/positions")
        lagged = field("ContactSystem/lagged_positions")
        if lagged is not None:
            lagged.copy_(positions)

        keys = field("AdhesiveIPCContactConstitution/pair_keys")
        betas = field("AdhesiveIPCContactConstitution/pair_beta")
        seen = field("AdhesiveIPCContactConstitution/pair_seen")
        if keys is not None:
            keys.fill_(-1)
        if betas is not None:
            betas.zero_()
        if seen is not None:
            seen.zero_()
        torch.cuda.synchronize()
        scene._capture_reset_state()
