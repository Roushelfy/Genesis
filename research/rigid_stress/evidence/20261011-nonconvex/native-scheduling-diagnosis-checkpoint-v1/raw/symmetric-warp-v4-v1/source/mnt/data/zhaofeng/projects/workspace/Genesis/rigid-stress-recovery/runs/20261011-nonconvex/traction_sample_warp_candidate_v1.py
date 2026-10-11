"""Experimental sample-parallel commit and scatter with unchanged original budgets."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState
from genesis.engine.solvers.rigid.stress.intrinsic import (
    StressContactPacking,
    StressWrenchCertificates,
    kernel_commit_intrinsic_contacts as kernel_commit_scalar,
)
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_source_traction,
    func_traction_geometry,
    func_traction_shape,
    func_traction_source_friction,
    func_traction_support_range,
    kernel_scatter_tractions as kernel_scatter_scalar,
)
from genesis.utils import array_class


@qd.kernel
def kernel_commit_warp(
    i_start: int,
    epsilon: float,
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
):
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.valid.shape[0] * contact_state.valid.shape[1] * 32):
        lane = i_thread % 32
        component = i_thread // 32
        i_c, i_b = component // contact_state.valid.shape[1], component % contact_state.valid.shape[1]
        if contact_state.valid[i_c, i_b]:
            start = i_start
            if qd.static(len(packing.block_start.shape) == 2):
                start = packing.block_start[i_start, i_b]
            i_raw = packing.indices[start + i_c, i_b]
            force_scale = contact_state.force[i_c, i_b].norm()
            if force_scale == 0.0:
                force_scale = traction_state.scale[i_c, i_b]
            force_error = traction_state.force_error[i_c, i_b] * force_scale
            moment_error = traction_state.moment_error[i_c, i_b] * force_scale * contact_state.radius[i_c, i_b]
            status = 0
            if source_state.status[i_c, i_b] != 0:
                status = 10 + source_state.status[i_c, i_b]
            elif footprint_state.status[i_c, i_b] != 0:
                status = 20 + footprint_state.status[i_c, i_b]
            elif traction_state.status[i_c, i_b] != 0:
                status = 30 + traction_state.status[i_c, i_b]
            else:
                total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                scale = qd.max(contact_state.force[i_c, i_b].norm(), traction_state.scale[i_c, i_b])
                cone_tolerance = 64.0 * epsilon * scale
                cone_valid = True
                first, last = func_traction_support_range(i_c, i_b, traction_state)
                for i_s in range(first, last):
                    friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
                    count = traction_state.sample_count[i_s, i_b]
                    for chunk in range((count + 31) // 32):
                        i_q = 32 * chunk + lane
                        if i_q < count:
                            force = func_source_traction(i_q, i_s, i_c, i_b, contact_state, traction_state, traction_info)
                            position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                            compression = force.dot(normal)
                            tangent = force - compression * normal
                            cone_valid = cone_valid and compression >= -cone_tolerance
                            cone_valid = cone_valid and tangent.norm() <= friction * compression + cone_tolerance
                            total += force
                            moment += (position - contact_state.position[i_c, i_b]).cross(force)
                for a in qd.static(range(3)):
                    total[a] = qd.simt.subgroup.reduce_all_add(total[a])
                    moment[a] = qd.simt.subgroup.reduce_all_add(moment[a])
                cone_valid = qd.simt.subgroup.all_true(cone_valid)
                force_error = (total - contact_state.force[i_c, i_b]).norm()
                moment_error = (moment - source_state.free_moment[i_c, i_b]).norm()
                budget_scale = contact_state.force[i_c, i_b].norm()
                if budget_scale == 0.0:
                    budget_scale = scale
                budget = 1e-8 * budget_scale
                if not cone_valid or force_error > budget or moment_error > budget * contact_state.radius[i_c, i_b]:
                    status = 40
            if lane == 0:
                raw_contacts.center[i_raw, i_b] = contact_state.center[i_c, i_b]
                raw_contacts.anchor_face[i_raw, i_b] = contact_state.anchor_face[i_c, i_b]
                raw_sources.barycentric[i_raw, i_b] = source_state.barycentric[i_c, i_b]
                raw_sources.anchor_moment[i_raw, i_b] = source_state.anchor_moment[i_c, i_b]
                raw_sources.status[i_raw, i_b] = source_state.status[i_c, i_b]
                raw_contacts.candidate_count[i_raw, i_b] = footprint_state.samples[i_c, i_b]
                raw_contacts.evaluations[i_raw, i_b] = traction_state.evaluations[i_c, i_b]
                certificates.ray[i_raw, i_b] = traction_state.certificate[i_c, i_b]
                certificates.margin[i_raw, i_b] = traction_state.certificate_margin[i_c, i_b]
                raw_contacts.force_error[i_raw, i_b] = force_error
                raw_contacts.moment_error[i_raw, i_b] = moment_error
                raw_contacts.status[i_raw, i_b] = status


@qd.kernel
def kernel_scatter_warp(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    traction_info: StressTractionInfo,
    errno: qd.Tensor,
    reset_force: qd.template(),
):
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        if qd.static(reset_force):
            state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and traction_state.status[i_c, i_b] != StressTractionStatus.SUCCESS:
            state.step_valid[i_b] = False
            qd.atomic_or(errno[i_b], array_class.ErrorCode.INVALID_STRESS_LOAD)
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.valid.shape[0] * contact_state.valid.shape[1] * 32):
        lane = i_thread % 32
        destination = i_thread // 32
        i_s, i_b = destination // contact_state.valid.shape[1], destination % contact_state.valid.shape[1]
        if contact_state.valid[i_s, i_b]:
            n_samples = traction_state.sample_count[i_s, i_b]
            start, end = func_traction_support_range(i_s, i_b, traction_state)
            for chunk in range((n_samples + 31) // 32):
                i_q = 32 * chunk + lane
                if i_q < n_samples and traction_state.profile[i_q, i_s, i_b] > 0.0:
                    force = qd.Vector.zero(gs.qd_float, 3)
                    for i_c in range(start, end):
                        force += func_source_traction(i_q, i_s, i_c, i_b, contact_state, traction_state, traction_info)
                    i_face, basis = func_traction_shape(i_q, i_s, i_b, traction_state, surface_info, traction_info)
                    for a in qd.static(range(6)):
                        i_node = stress_info.surface_nodes[i_face, a]
                        for k in qd.static(range(3)):
                            qd.atomic_add(state.force[i_node, i_b][k], basis[a] * force[k])


def kernel_commit_intrinsic_contacts(*args):
    if gs.backend == gs.cuda:
        return kernel_commit_warp(*args)
    return kernel_commit_scalar(*args)


def kernel_scatter_tractions(*args, **kwargs):
    if gs.backend == gs.cuda:
        return kernel_scatter_warp(*args, **kwargs)
    return kernel_scatter_scalar(*args, **kwargs)
