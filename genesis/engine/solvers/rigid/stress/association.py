"""Associate solved rigid contacts with a stress link in the matching substep frame."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom

from .contact import StressContactState


@qd.kernel
def kernel_associate(
    i_l: int,
    links_offset_pos: qd.types.ndarray(),
    links_offset_quat: qd.types.ndarray(),
    omega: qd.Tensor,
    dyn_state: array_class.DynState,
    contact_state: StressContactState,
    collider_state: array_class.ColliderState,
    batch_offsets: qd.template(),
    enable_constraint: bool,
    step_valid: qd.Tensor,
    errno: qd.Tensor,
):
    func_associate(
        i_l,
        links_offset_pos,
        links_offset_quat,
        omega,
        dyn_state,
        contact_state,
        collider_state,
        batch_offsets,
        enable_constraint,
        step_valid,
        errno,
    )


@qd.func
def func_associate(
    i_l: int,
    links_offset_pos: qd.types.ndarray(),
    links_offset_quat: qd.types.ndarray(),
    omega: qd.Tensor,
    dyn_state: array_class.DynState,
    contact_state: StressContactState,
    collider_state: array_class.ColliderState,
    batch_offsets: qd.template(),
    enable_constraint: bool,
    step_valid: qd.Tensor,
    errno: qd.Tensor,
):
    for i_b in range(omega.shape[0]):
        offset_pos = qd.Vector.zero(gs.qd_float, 3)
        offset_quat = qd.Vector.zero(gs.qd_float, 4)
        for i_a in qd.static(range(3)):
            if qd.static(batch_offsets):
                offset_pos[i_a] = links_offset_pos[i_b, i_l, i_a]
            else:
                offset_pos[i_a] = links_offset_pos[i_l, i_a]
        for i_a in qd.static(range(4)):
            if qd.static(batch_offsets):
                offset_quat[i_a] = links_offset_quat[i_b, i_l, i_a]
            else:
                offset_quat[i_a] = links_offset_quat[i_l, i_a]
        quat = geom.qd_transform_quat_by_quat(geom.qd_inv_quat(offset_quat), dyn_state.links.quat[i_l, i_b])
        contact_state.frame_quaternion[i_b] = quat
        contact_state.frame_position[i_b] = dyn_state.links.pos[i_l, i_b] - geom.qd_transform_by_quat(offset_pos, quat)
        omega[i_b] = geom.qd_inv_transform_by_quat(dyn_state.links.cd_ang[i_l, i_b], quat)
        unsupported = (
            dyn_state.links.cfrc_applied_ang[i_l, i_b].norm() > 0.0
            or dyn_state.links.cfrc_applied_vel[i_l, i_b].norm() > 0.0
            or dyn_state.links.cfrc_coupling_ang[i_l, i_b].norm() > 0.0
            or dyn_state.links.cfrc_coupling_vel[i_l, i_b].norm() > 0.0
            or collider_state.n_contacts[i_b] > contact_state.valid.shape[0]
        )
        if unsupported:
            step_valid[i_b] = False
            qd.atomic_or(errno[i_b], array_class.ErrorCode.INVALID_STRESS_LOAD)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        contact_state.valid[i_c, i_b] = False
        if enable_constraint and i_c < collider_state.n_contacts[i_b]:
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            i_a = collider_state.contact_data.link_a[i_contact, i_b]
            i_c_link = collider_state.contact_data.link_b[i_contact, i_b]
            if i_a == i_l or i_c_link == i_l:
                sign = gs.qd_float(1.0)
                if i_a == i_l:
                    sign = -1.0
                quat = contact_state.frame_quaternion[i_b]
                pos = contact_state.frame_position[i_b]
                contact_state.position[i_c, i_b] = geom.qd_inv_transform_by_quat(
                    collider_state.contact_data.pos[i_contact, i_b] - pos, quat
                )
                contact_state.force[i_c, i_b] = geom.qd_inv_transform_by_quat(
                    sign * collider_state.contact_data.force[i_contact, i_b], quat
                )
                contact_state.normal[i_c, i_b] = geom.qd_inv_transform_by_quat(
                    -sign * collider_state.contact_data.normal[i_contact, i_b], quat
                )
                contact_state.friction[i_c, i_b] = collider_state.contact_data.friction[i_contact, i_b]
                contact_state.valid[i_c, i_b] = True
