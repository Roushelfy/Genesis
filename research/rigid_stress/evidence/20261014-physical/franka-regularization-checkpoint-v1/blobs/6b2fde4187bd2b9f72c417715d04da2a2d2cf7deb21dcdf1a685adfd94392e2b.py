"""Native sample-cooperative Coulomb dual fitting with unchanged original-point error budgets."""

import quadrants as qd

import genesis as gs

from .contact import StressContactState
from .provenance import StressSourceState, StressSourceStatus
from .traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_coulomb_projection,
    func_traction_constraint_active,
    func_traction_direction,
    func_traction_geometry,
    func_traction_reference,
    func_traction_source_friction,
    func_traction_support_range,
)

_HESSIAN_FLOAT32 = qd.types.matrix(6, 6, qd.f32)
_HESSIAN_FLOAT64 = qd.types.matrix(6, 6, qd.f64)


@qd.func
def func_dual_warp(
    i_c: int,
    i_b: int,
    lane: int,
    lanes: qd.template(),
    hessian_type: qd.template(),
    dual: qd.types.vector(6),
    force: qd.types.vector(3),
    target: qd.types.vector(6),
    anchor: qd.types.vector(3),
    radius: float,
    friction: float,
    weight_sum: float,
    state: StressTractionState,
    info: StressTractionInfo,
    sample_lanes: qd.template(),
):
    energy = gs.qd_float(0.0)
    resultant = qd.Vector.zero(gs.qd_float, 6)
    hessian = qd.Matrix.zero(hessian_type.dtype, 6, 6)
    start, end = func_traction_support_range(i_c, i_b, state)
    for i_s in range(start, end):
        source_friction = func_traction_source_friction(i_s, i_b, friction, state)
        n_samples = state.sample_count[i_s, i_b]
        for chunk in range((n_samples + sample_lanes - 1) // sample_lanes):
            i_q = chunk * sample_lanes + lane
            if lane < sample_lanes and i_q < n_samples:
                weight = state.profile[i_q, i_s, i_b] / weight_sum
                if weight > 0.0:
                    position, normal = func_traction_geometry(i_q, i_s, i_b, state, info)
                    relative = (position - anchor) / radius
                    reference = func_traction_reference(i_s, i_c, i_b, force, weight_sum, state)
                    value = reference - dual[0:3] - dual[3:6].cross(relative)
                    projected, jacobian = func_coulomb_projection(value, normal, source_friction)
                    moment = relative.cross(projected)
                    wrench = qd.Vector([projected[0], projected[1], projected[2], moment[0], moment[1], moment[2]])
                    operator = qd.Matrix.zero(hessian_type.dtype, 6, 3)
                    for a in qd.static(range(3)):
                        operator[a, a] = 1.0
                    operator[3, 1], operator[3, 2] = -relative[2], relative[1]
                    operator[4, 0], operator[4, 2] = relative[2], -relative[0]
                    operator[5, 0], operator[5, 1] = -relative[1], relative[0]
                    energy += 0.5 * weight * projected.dot(projected)
                    resultant += weight * wrench
                    product = operator @ jacobian.cast(hessian_type.dtype)
                    hessian_weight = qd.cast(weight, hessian_type.dtype)
                    for a, b in qd.static(qd.ndrange(6, 6)):
                        if qd.static(b <= a):
                            hessian[a, b] += hessian_weight * product[a, :].dot(operator[b, :])
    if qd.static(lanes > 1):
        energy = qd.simt.subgroup.reduce_all_add(energy)
        for a in qd.static(range(6)):
            resultant[a] = qd.simt.subgroup.reduce_all_add(resultant[a])
    energy += target.dot(dual)
    for a, b in qd.static(qd.ndrange(6, 6)):
        if qd.static(b <= a):
            if qd.static(lanes > 1):
                hessian[a, b] = qd.simt.subgroup.reduce_all_add(hessian[a, b])
            hessian[b, a] = hessian[a, b]
    return energy, target - resultant, hessian.cast(gs.qd_float)


def kernel_fit_tractions(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    epsilon: float,
    fit_tolerance: float,
    max_evaluations: int,
    warm_start: bool = True,
    is_retry: bool = False,
    sample_lanes: int = 32,
):
    assert sample_lanes in (1, 2, 4, 8, 16, 32)
    # Approximate only the warm Newton search direction. Physical tractions,
    # gradients and final acceptance keep scene precision; cold/retry Hessians
    # retain scene precision, including existing FP32 scenes.
    hessian_type = (
        _HESSIAN_FLOAT32
        if gs.qd_float == qd.f32 or (gs.backend == gs.cuda and warm_start and not is_retry)
        else _HESSIAN_FLOAT64
    )
    kernel_fit_tractions_impl(
        contact_state,
        source_state,
        traction_state,
        traction_info,
        epsilon,
        fit_tolerance,
        max_evaluations,
        warm_start,
        is_retry,
        sample_lanes,
        hessian_type,
    )


@qd.kernel
def kernel_fit_tractions_impl(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    epsilon: float,
    fit_tolerance: float,
    max_evaluations: int,
    warm_start: qd.template(),
    is_retry: qd.template(),
    sample_lanes: qd.template(),
    hessian_type: qd.template(),
):
    lanes = qd.static(32 if gs.backend == gs.cuda else 1)
    n_sample_lanes = qd.static(sample_lanes if gs.backend == gs.cuda else 1)
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.valid.shape[0] * contact_state.valid.shape[1] * lanes):
        lane = i_thread % lanes
        pair = i_thread // lanes
        i_c, i_b = pair // contact_state.valid.shape[1], pair % contact_state.valid.shape[1]
        should_fit, prior_evaluations = True, 0
        if qd.static(is_retry):
            should_fit = traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
            should_fit = should_fit or traction_state.status[i_c, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
            prior_evaluations = traction_state.evaluations[i_c, i_b]
        status = gs.qd_int(StressTractionStatus.SUCCESS)
        current = qd.Vector.zero(gs.qd_float, 6)
        if qd.static(warm_start):  # noqa: SIM102 — keep the cold specialization free of workspace reads.
            # Scratch slots may hold another source: reuse only a numerical
            # initial guess, then validate and fit the current problem fully.
            if traction_state.status[i_c, i_b] == StressTractionStatus.SUCCESS and traction_state.scale[i_c, i_b] > 0.0:
                previous = traction_state.dual[i_c, i_b]
                if qd.abs(previous.sum()) < qd.math.inf and previous.norm() <= 1e6:
                    current = previous
        weight_sum, scale = gs.qd_float(0.0), gs.qd_float(0.0)
        force_error, moment_error = gs.qd_float(0.0), gs.qd_float(0.0)
        evaluations = 0
        certificate = qd.Vector.zero(gs.qd_float, 6)
        certificate_margin = gs.qd_float(0.0)
        if should_fit and contact_state.valid[i_c, i_b] and func_traction_constraint_active(i_c, i_b, traction_state):
            anchor = contact_state.center[i_c, i_b]
            radius, friction = contact_state.radius[i_c, i_b], contact_state.friction[i_c, i_b]
            force, moment = contact_state.force[i_c, i_b], source_state.anchor_moment[i_c, i_b]
            is_valid = source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS
            is_valid = is_valid and radius > 0.0 and friction >= 0.0 and epsilon > 0.0 and fit_tolerance > 0.0
            finite = anchor.sum() + force.sum() + moment.sum() + radius + friction
            is_valid = is_valid and qd.abs(finite) < qd.math.inf
            start, end = func_traction_support_range(i_c, i_b, traction_state)
            is_valid = is_valid and 0 <= start < end <= contact_state.valid.shape[0]
            lipschitz = gs.qd_float(0.0)
            if is_valid:
                for i_s in range(start, end):
                    support_center, support_radius = anchor, radius
                    if qd.static(len(traction_state.support.valid.shape) == 2):
                        is_valid = is_valid and traction_state.support.valid[i_s, i_b]
                        support_center = traction_state.support.center[i_s, i_b]
                        support_radius = traction_state.support.radius[i_s, i_b]
                    source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                    is_valid = is_valid and source_friction >= 0.0 and source_friction < qd.math.inf
                    is_valid = is_valid and support_radius > 0.0 and support_radius < qd.math.inf
                    n_samples = traction_state.sample_count[i_s, i_b]
                    is_valid = is_valid and 0 <= n_samples <= traction_state.profile.shape[0]
                    if qd.static(len(traction_state.priors.force.shape) == 2):
                        source_weight = traction_state.priors.weight_sum[i_s, i_b]
                        original_force = traction_state.priors.force[i_s, i_b]
                        is_valid = is_valid and source_weight > 0.0 and source_weight < qd.math.inf
                        is_valid = is_valid and qd.abs(original_force.sum()) < qd.math.inf
                    for chunk in range((n_samples + n_sample_lanes - 1) // n_sample_lanes):
                        i_q = chunk * n_sample_lanes + lane
                        if lane < n_sample_lanes and i_q < n_samples:
                            weight = traction_state.profile[i_q, i_s, i_b]
                            valid_weight = weight >= 0.0 and weight < qd.math.inf
                            is_valid = is_valid and valid_weight
                            if weight > 0.0 and valid_weight:
                                position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                                relative = (position - anchor) / radius
                                is_valid = is_valid and qd.abs(normal.dot(normal) - 1.0) <= 64.0 * epsilon
                                is_valid = is_valid and (position - support_center).norm() <= support_radius * (
                                    1.0 + 64.0 * epsilon
                                )
                                weight_sum += weight
                                lipschitz += weight * (1.0 + relative.dot(relative))
            if qd.static(lanes > 1):
                is_valid = qd.simt.subgroup.all_true(is_valid)
                weight_sum = qd.simt.subgroup.reduce_all_add(weight_sum)
                lipschitz = qd.simt.subgroup.reduce_all_add(lipschitz)
            if not is_valid:
                status = StressTractionStatus.INVALID_INPUT
                weight_sum = 0.0
            elif weight_sum <= 0.0:
                status = StressTractionStatus.UNRESOLVED_INTEGRATION
            else:
                lipschitz /= weight_sum
                scale = qd.max(force.norm(), moment.norm() / radius)
                if qd.static(len(traction_state.priors.force.shape) == 2):
                    scale = qd.max(scale, traction_state.priors.group_scale[i_c, i_b])
                if scale > 0.0:
                    budget_scale = force.norm()
                    if budget_scale == 0.0:
                        budget_scale = scale
                    force /= scale
                    moment /= radius * scale
                    target = qd.Vector([force[0], force[1], force[2], moment[0], moment[1], moment[2]])
                    proposal, direction = current, current
                    energy = gs.qd_float(0.0)
                    gradient = current
                    gradient_squared = gs.qd_float(float("inf"))
                    alpha, slope = gs.qd_float(1.0), gs.qd_float(0.0)
                    is_trial, is_active = False, True
                    status = StressTractionStatus.NUMERICAL_FAILURE
                    for _ in range(max_evaluations):
                        if is_active:
                            trial_energy, trial_gradient, hessian = func_dual_warp(
                                i_c,
                                i_b,
                                lane,
                                lanes,
                                hessian_type,
                                proposal,
                                force,
                                target,
                                anchor,
                                radius,
                                friction,
                                weight_sum,
                                traction_state,
                                traction_info,
                                n_sample_lanes,
                            )
                            evaluations += 1
                            trial_squared = trial_gradient.dot(trial_gradient)
                            comparison_scale = 1.0 + qd.abs(energy) + qd.abs(trial_energy)
                            comparison_scale += (qd.abs(current) * qd.abs(target)).sum()
                            comparison_scale += (qd.abs(proposal) * qd.abs(target)).sum()
                            allowance = 32.0 * epsilon * comparison_scale
                            accepted = not is_trial or trial_energy <= energy - 1e-4 * alpha * slope
                            accepted = accepted or (
                                trial_energy <= energy + allowance and trial_squared < gradient_squared
                            )
                            if accepted:
                                current, energy, gradient = proposal, trial_energy, trial_gradient
                                gradient_squared = trial_squared
                                force_error = gradient[0:3].norm() * scale / budget_scale
                                offset = (contact_state.position[i_c, i_b] - anchor) / radius
                                moment_error = (
                                    (gradient[3:6] - offset.cross(gradient[0:3])).norm() * scale / budget_scale
                                )
                                if qd.max(force_error, moment_error) <= fit_tolerance:
                                    status = StressTractionStatus.SUCCESS
                                    is_active = False
                                else:
                                    positive = True
                                    if lane == 0:
                                        ridge, retry = 0.0, True
                                        for _attempt in range(2):
                                            if retry:
                                                direction, positive = func_traction_direction(hessian, gradient, ridge)
                                                slope = gradient.dot(direction)
                                                retry = not positive or slope <= 0.0 or not slope < qd.math.inf
                                                ridge = 1e-12 * qd.max(1.0, hessian.trace() / 6.0)
                                    if qd.static(lanes > 1):
                                        for a in qd.static(range(6)):
                                            direction[a] = qd.simt.subgroup.broadcast(direction[a], qd.u32(0))
                                        positive = qd.simt.subgroup.broadcast(qd.u32(positive), qd.u32(0)) != 0
                                    slope = gradient.dot(direction)
                                    if not positive or slope <= 0.0 or not slope < qd.math.inf:
                                        direction = gradient / lipschitz
                                        slope = gradient.dot(direction)
                                    trust_radius = 2.0 * qd.max(1.0, current.norm())
                                    trust_fraction = qd.min(1.0, trust_radius / qd.max(direction.norm(), 1e-30))
                                    direction *= trust_fraction
                                    slope *= trust_fraction
                                    alpha = 1.0
                                    proposal = current - direction
                                    is_trial = True
                            else:
                                alpha *= 0.5
                                if alpha < 1e-12:
                                    direction = gradient / lipschitz
                                    slope = gradient.dot(direction)
                                    alpha = 1.0
                                proposal = current - alpha * direction
                    if status == StressTractionStatus.NUMERICAL_FAILURE and current.norm() > 0.0:
                        ray = -current / current.norm()
                        polar_max = gs.qd_float(-float("inf"))
                        for i_s in range(start, end):
                            source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                            n_samples = traction_state.sample_count[i_s, i_b]
                            for chunk in range((n_samples + n_sample_lanes - 1) // n_sample_lanes):
                                i_q = chunk * n_sample_lanes + lane
                                if (
                                    lane < n_sample_lanes
                                    and i_q < n_samples
                                    and traction_state.profile[i_q, i_s, i_b] > 0.0
                                ):
                                    position, normal = func_traction_geometry(
                                        i_q, i_s, i_b, traction_state, traction_info
                                    )
                                    relative = (position - anchor) / radius
                                    value = ray[0:3] + ray[3:6].cross(relative)
                                    compression = value.dot(normal)
                                    polar = compression + source_friction * (value - compression * normal).norm()
                                    polar_max = qd.max(polar_max, polar)
                        if qd.static(lanes > 1):
                            polar_max = qd.simt.subgroup.reduce_all_max(polar_max)
                        margin = target.dot(ray)
                        if polar_max < -64.0 * epsilon and margin > fit_tolerance:
                            status = StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
                            certificate, certificate_margin = ray, margin
        if lane == 0 and should_fit:
            traction_state.dual[i_c, i_b] = current
            traction_state.weight_sum[i_c, i_b] = weight_sum
            traction_state.scale[i_c, i_b] = scale
            traction_state.status[i_c, i_b] = status
            traction_state.force_error[i_c, i_b] = force_error
            traction_state.moment_error[i_c, i_b] = moment_error
            traction_state.evaluations[i_c, i_b] = prior_evaluations + evaluations
            traction_state.certificate[i_c, i_b] = certificate
            traction_state.certificate_margin[i_c, i_b] = certificate_margin
