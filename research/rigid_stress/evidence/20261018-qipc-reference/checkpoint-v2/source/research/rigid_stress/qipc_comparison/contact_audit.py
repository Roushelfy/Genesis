"""Inspect original contact candidates at an unchanged state before attempting a Newton solve."""

import numpy as np
from scipy.spatial.transform import Rotation

from qipc._src.native.solver import GlobalBodyManager, GlobalVertexManager, RigidBodyDynamics

from .artifacts import write_json


def initial_contact_audit(scene, robot, output):
    manager = scene.sim_systems[GlobalVertexManager]
    positions = manager.positions_view().clone()
    body_ids = manager.body_id_view().cpu().numpy()
    body_manager = scene.sim_systems[GlobalBodyManager]
    offset = body_manager.body_offsets_view()[body_manager.body_slot(RigidBodyDynamics.layout_rank())].item()
    names = {}
    vertex_offset = manager.vertex_offsets_view()[manager.vertex_slot(RigidBodyDynamics.layout_rank())].item()
    rigid = scene.rigid_body
    local_ids = rigid.body_id.cpu().numpy()
    local = rigid.x_bar.cpu().numpy() - rigid.com_local.cpu().numpy()[local_ids]
    quaternions = rigid.quat.cpu().numpy()
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    expected_world = np.einsum("vij,vj->vi", rotations[local_ids], local) + rigid.t.cpu().numpy()[local_ids]
    actual_world = positions[vertex_offset : vertex_offset + len(local)].cpu().numpy()
    errors = np.linalg.norm(expected_world - actual_world, axis=1)
    np.savez_compressed(
        output / "initial-rigid-publication.npz",
        global_positions_m=positions.cpu().numpy(),
        global_body_ids=body_ids,
        rigid_vertex_offset=vertex_offset,
        public_rigid_body_ids=local_ids,
        expected_rigid_world_m=expected_world,
        actual_rigid_world_m=actual_world,
    )
    for slot in robot.links:
        local = slot.geometry.meta["rigid_body_offset"].cpu()[0].item()
        names[offset + local] = slot.name
    mechanisms = scene.rigid_body.host("mechanism_of_proxy").reshape(-1)
    for local, mechanism in enumerate(mechanisms):
        if mechanism >= 0:
            names[offset + local] = names[offset + mechanism]
    result = scene.contact_system.recompute(positions, positions)
    records = []
    for channel, table in (
        ("pt", result.pt),
        ("ee", result.ee),
        ("pe", result.pe),
        ("pp", result.pp),
        ("ph", result.ph),
    ):
        stencil = table.stencil.cpu().numpy()
        active = table.active.cpu().numpy() != 0
        distance = np.sqrt(table.d2.cpu().numpy())
        body_pairs = np.sort(np.stack((body_ids[stencil].min(axis=1), body_ids[stencil].max(axis=1)), axis=1), axis=1)
        unique = np.unique(body_pairs[active], axis=0)
        for left, right in unique:
            selected = active & (body_pairs[:, 0] == left) & (body_pairs[:, 1] == right)
            records.append(
                {
                    "channel": channel,
                    "left": names.get(left, f"global-body-{left}"),
                    "right": names.get(right, f"global-body-{right}"),
                    "active_pairs": int(selected.sum()),
                    "minimum_distance_m": float(distance[selected].min()),
                }
            )
        np.savez_compressed(
            output / f"initial-contact-{channel}.npz",
            stencil=stencil,
            active=active,
            distance_m=distance,
            body_pairs=body_pairs,
            gradient=table.grad.cpu().numpy(),
        )
    write_json(
        output / "initial-contact-audit.json",
        {
            "records": records,
            "maximum_rigid_vertex_publication_error_m": errors.max().item(),
            "scope": "Original read-only contact recompute at identical static endpoints; no Newton step, friction or dynamics qualification",
        },
    )
