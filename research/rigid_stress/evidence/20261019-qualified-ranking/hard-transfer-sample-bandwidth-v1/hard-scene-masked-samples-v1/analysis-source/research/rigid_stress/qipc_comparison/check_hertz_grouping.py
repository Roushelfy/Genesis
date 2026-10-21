"""Exact serial grouping oracle and device-side comparison for implementation changes."""

import quadrants as qd
import genesis as gs
from genesis.utils import geom
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegionState, HertzRegionStatus


@qd.kernel
def kernel_serial_grouping(
    contact_state: StressContactState,
    source_state: StressSourceState,
    region_state: StressHertzRegionState,
    reference_state: StressHertzRegionState,
    epsilon: float,
):
    for i_b in range(reference_state.is_valid.shape[0]):
        reference_state.is_valid[i_b] = True
        for i_c in range(region_state.status.shape[0]):
            reference_state.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            reference_state.moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            reference_state.source_count[i_c, i_b] = 0
            reference_state.leader[i_c, i_b] = -1
            status = region_state.status[i_c, i_b]
            if status != HertzRegionStatus.SUCCESS and status != HertzRegionStatus.UNLOADED:
                reference_state.is_valid[i_b] = False
            if status == HertzRegionStatus.SUCCESS:
                i_leader = i_c
                curvature = region_state.curvature[i_c, i_b]
                resolution = 1e-6 * curvature.trace() / curvature.determinant()
                center = region_state.center[i_c, i_b]
                resolution += 128.0 * epsilon * center.norm()
                for j_c in range(i_c):
                    if region_state.status[j_c, i_b] == HertzRegionStatus.SUCCESS:
                        if region_state.opponent_link[j_c, i_b] == region_state.opponent_link[i_c, i_b]:
                            if (region_state.center[j_c, i_b] - center).norm() <= resolution:
                                i_leader = qd.min(i_leader, reference_state.leader[j_c, i_b])
                reference_state.leader[i_c, i_b] = i_leader
        for i_c in range(region_state.status.shape[0]):
            i_leader = reference_state.leader[i_c, i_b]
            if i_leader >= 0 and reference_state.is_valid[i_b]:
                rotation = contact_state.frame_quaternion[i_b]
                force = geom.qd_transform_by_quat(contact_state.force[i_c, i_b], rotation)
                position = geom.qd_transform_by_trans_quat(
                    contact_state.position[i_c, i_b], contact_state.frame_position[i_b], rotation
                )
                moment = (position - region_state.center[i_leader, i_b]).cross(force)
                moment += geom.qd_transform_by_quat(source_state.free_moment[i_c, i_b], rotation)
                reference_state.force[i_leader, i_b] += force
                reference_state.moment[i_leader, i_b] += moment
                reference_state.source_count[i_leader, i_b] += 1


@qd.kernel
def kernel_compare_grouping(
    current: StressHertzRegionState,
    reference: StressHertzRegionState,
    errors: qd.types.ndarray(),
):
    for i_c, i_b in qd.ndrange(current.status.shape[0], current.status.shape[1]):
        wrong = current.is_valid[i_b] != reference.is_valid[i_b]
        wrong = wrong or current.leader[i_c, i_b] != reference.leader[i_c, i_b]
        wrong = wrong or current.source_count[i_c, i_b] != reference.source_count[i_c, i_b]
        qd.atomic_max(errors[i_b, 0], gs.qd_float(wrong))
        qd.atomic_max(errors[i_b, 1], (current.force[i_c, i_b] - reference.force[i_c, i_b]).norm())
        qd.atomic_max(errors[i_b, 2], (current.moment[i_c, i_b] - reference.moment[i_c, i_b]).norm())
