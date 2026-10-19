"""Explicit body-wrench partition with separately retained local force priors."""

import quadrants as qd

import genesis as gs

from .contact import StressContactState
from .provenance import StressSourceState
from .traction import StressTractionState


@qd.kernel
def kernel_snapshot_source_priors(
    contacts: StressContactState, sources: StressSourceState, tractions: StressTractionState
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        tractions.priors.force[i_c, i_b] = contacts.force[i_c, i_b]
        tractions.priors.anchor_moment[i_c, i_b] = sources.anchor_moment[i_c, i_b]
        tractions.priors.group_scale[i_c, i_b] = 0.0
        tractions.priors.force_scale[i_c, i_b] = 0.0
        tractions.priors.moment_scale[i_c, i_b] = 0.0
        weight_sum = gs.qd_float(0.0)
        if contacts.valid[i_c, i_b]:
            for i_q in range(tractions.sample_count[i_c, i_b]):
                weight_sum += tractions.profile[i_q, i_c, i_b]
        tractions.priors.weight_sum[i_c, i_b] = weight_sum


@qd.kernel
def kernel_partition_source_priors(
    contacts: StressContactState, sources: StressSourceState, tractions: StressTractionState
):
    """Replace only auxiliary constraints; every physical source stays valid."""
    for i_b in range(contacts.valid.shape[1]):
        if contacts.valid[0, i_b]:
            anchor = contacts.center[0, i_b]
            total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
            source_scale = gs.qd_float(0.0)
            moment_scale = gs.qd_float(0.0)
            for i_c in range(contacts.valid.shape[0]):
                if contacts.valid[i_c, i_b]:
                    force = tractions.priors.force[i_c, i_b]
                    anchor_moment = tractions.priors.anchor_moment[i_c, i_b]
                    total += force
                    moment += anchor_moment + (contacts.center[i_c, i_b] - anchor).cross(force)
                    radius = contacts.radius[i_c, i_b]
                    if radius > 0.0:
                        source_scale += qd.max(force.norm(), anchor_moment.norm() / radius)
                        # Bound moment summation at the unchanged application point, including each support radius
                        arm = (contacts.center[i_c, i_b] - contacts.position[0, i_b]).norm() + radius
                        moment_scale += anchor_moment.norm() + arm * force.norm()
                    contacts.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    sources.anchor_moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    sources.free_moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            contacts.force[0, i_b] = total
            sources.anchor_moment[0, i_b] = moment
            sources.free_moment[0, i_b] = moment - (contacts.position[0, i_b] - anchor).cross(total)
            tractions.priors.force_scale[0, i_b] = source_scale
            tractions.priors.moment_scale[0, i_b] = moment_scale
            if contacts.radius[0, i_b] > 0.0:
                source_scale = qd.max(source_scale, total.norm(), moment.norm() / contacts.radius[0, i_b])
            tractions.priors.group_scale[0, i_b] = source_scale


@qd.kernel
def kernel_refresh_refined_source_priors(contacts: StressContactState, tractions: StressTractionState):
    """Keep each retained prior normalization consistent with changed local integration."""
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and (tractions.circle.refined[i_c, i_b] or tractions.fan_refined[i_c, i_b]):
            weight_sum = gs.qd_float(0.0)
            for i_q in range(tractions.sample_count[i_c, i_b]):
                weight_sum += tractions.profile[i_q, i_c, i_b]
            tractions.priors.weight_sum[i_c, i_b] = weight_sum
