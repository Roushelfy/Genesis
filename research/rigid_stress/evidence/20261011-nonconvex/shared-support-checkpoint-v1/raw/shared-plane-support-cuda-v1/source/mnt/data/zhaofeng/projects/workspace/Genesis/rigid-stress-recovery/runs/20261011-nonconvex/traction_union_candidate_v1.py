"""Offline union-support fit candidate; caller verifies every sample's constituent intrinsic disk.

The original radius remains the wrench-conditioning scale and final moment
budget. The single-anchor Euclidean bound is replaced only because support
is explicitly a union of individually verified, unchanged-radius disks.
This is not production behavior or automatic grouping of arbitrary contacts.
"""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, StressSourceStatus
from genesis.engine.solvers.rigid.stress.traction import StressTractionInfo, StressTractionState, StressTractionStatus, func_traction_dual, func_traction_direction, func_traction_geometry

@qd.kernel
def kernel_fit_tractions(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    epsilon: float,
    fit_tolerance: float,
    max_evaluations: int,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        traction_state.dual[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        traction_state.weight_sum[i_c, i_b] = 0.0
        traction_state.scale[i_c, i_b] = 0.0
        traction_state.status[i_c, i_b] = StressTractionStatus.SUCCESS
        traction_state.force_error[i_c, i_b] = 0.0
        traction_state.moment_error[i_c, i_b] = 0.0
        traction_state.evaluations[i_c, i_b] = 0
        traction_state.certificate[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        traction_state.certificate_margin[i_c, i_b] = 0.0
        if contact_state.valid[i_c, i_b]:
            anchor = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            friction = contact_state.friction[i_c, i_b]
            force = contact_state.force[i_c, i_b]
            moment = source_state.anchor_moment[i_c, i_b]
            is_valid = source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS
            is_valid = is_valid and radius > 0.0 and friction >= 0.0 and epsilon > 0.0 and fit_tolerance > 0.0
            finite = anchor.sum() + force.sum() + moment.sum() + radius + friction
            is_valid = is_valid and qd.abs(finite) < qd.math.inf
            weight_sum = gs.qd_float(0.0)
            lipschitz = gs.qd_float(0.0)
            n_samples = traction_state.sample_count[i_c, i_b]
            is_valid = is_valid and 0 <= n_samples <= traction_state.profile.shape[0]
            for i_q in range(n_samples):
                weight = traction_state.profile[i_q, i_c, i_b]
                is_valid = is_valid and weight >= 0.0 and weight < qd.math.inf
                if weight > 0.0 and is_valid:
                    position, normal = func_traction_geometry(i_q, i_c, i_b, traction_state, traction_info)
                    relative = (position - anchor) / radius
                    is_valid = is_valid and qd.abs(normal.dot(normal) - 1.0) <= 64.0 * epsilon
                    # Geometry comes from an explicitly verified union of unchanged-radius intrinsic disks.
                    is_valid = is_valid and qd.abs(relative.norm()) < qd.math.inf
                    weight_sum += weight
                    lipschitz += weight * (1.0 + relative.dot(relative))
            if not is_valid:
                traction_state.status[i_c, i_b] = StressTractionStatus.INVALID_INPUT
            elif weight_sum <= 0.0:
                traction_state.status[i_c, i_b] = StressTractionStatus.UNRESOLVED_INTEGRATION
            else:
                traction_state.weight_sum[i_c, i_b] = weight_sum
                lipschitz /= weight_sum
                scale = qd.max(force.norm(), moment.norm() / radius)
                traction_state.scale[i_c, i_b] = scale
                if scale > 0.0:
                    budget_scale = force.norm()
                    if budget_scale == 0.0:
                        budget_scale = scale
                    force /= scale
                    moment /= radius * scale
                    target = qd.Vector([force[0], force[1], force[2], moment[0], moment[1], moment[2]])
                    current = qd.Vector.zero(gs.qd_float, 6)
                    proposal = current
                    direction = current
                    energy = gs.qd_float(0.0)
                    gradient = current
                    gradient_squared = gs.qd_float(float("inf"))
                    alpha = gs.qd_float(1.0)
                    slope = gs.qd_float(0.0)
                    is_trial, is_active = False, True
                    traction_state.status[i_c, i_b] = StressTractionStatus.NUMERICAL_FAILURE
                    for _ in range(max_evaluations):
                        if is_active:
                            trial_energy, trial_gradient, hessian = func_traction_dual(
                                i_c,
                                i_b,
                                proposal,
                                force,
                                target,
                                anchor,
                                radius,
                                friction,
                                traction_state,
                                traction_info,
                            )
                            traction_state.evaluations[i_c, i_b] += 1
                            trial_squared = trial_gradient.dot(trial_gradient)
                            # The dual's small energy can cancel large wrench
                            # products. Bound comparison roundoff from those
                            # operands, not only the already-cancelled result.
                            comparison_scale = 1.0 + qd.abs(energy) + qd.abs(trial_energy)
                            comparison_scale += (qd.abs(current) * qd.abs(target)).sum()
                            comparison_scale += (qd.abs(proposal) * qd.abs(target)).sum()
                            allowance = 32.0 * epsilon * comparison_scale
                            is_accepted = not is_trial or trial_energy <= energy - 1e-4 * alpha * slope
                            is_accepted = is_accepted or (
                                trial_energy <= energy + allowance and trial_squared < gradient_squared
                            )
                            if is_accepted:
                                current, energy, gradient = proposal, trial_energy, trial_gradient
                                gradient_squared = trial_squared
                                traction_state.dual[i_c, i_b] = current
                                force_error = gradient[0:3].norm() * scale / budget_scale
                                moment_error = gradient[3:6].norm() * scale / budget_scale
                                traction_state.force_error[i_c, i_b] = force_error
                                traction_state.moment_error[i_c, i_b] = moment_error
                                if qd.max(force_error, moment_error) <= fit_tolerance:
                                    traction_state.status[i_c, i_b] = StressTractionStatus.SUCCESS
                                    is_active = False
                                else:
                                    ridge = 1e-12 * qd.max(1.0, hessian.trace() / 6.0)
                                    direction, is_positive = func_traction_direction(hessian, gradient, ridge)
                                    slope = gradient.dot(direction)
                                    if not is_positive or slope <= 0.0 or not slope < qd.math.inf:
                                        direction = gradient / lipschitz
                                        slope = gradient.dot(direction)
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
                    if (
                        traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
                        and current.norm() > 0.0
                    ):
                        # A strictly polar separating ray certifies the sampled cone system only.
                        # Continuous-footprint incompatibility requires a separate geometry certificate.
                        ray = -current / current.norm()
                        polar_max = gs.qd_float(-float("inf"))
                        for i_q in range(n_samples):
                            if traction_state.profile[i_q, i_c, i_b] > 0.0:
                                position, normal = func_traction_geometry(i_q, i_c, i_b, traction_state, traction_info)
                                relative = (position - anchor) / radius
                                value = ray[0:3] + ray[3:6].cross(relative)
                                compression = value.dot(normal)
                                polar = compression + friction * (value - compression * normal).norm()
                                polar_max = qd.max(polar_max, polar)
                        margin = target.dot(ray)
                        if polar_max < -64.0 * epsilon and margin > fit_tolerance:
                            traction_state.status[i_c, i_b] = StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
                            traction_state.certificate[i_c, i_b] = ray
                            traction_state.certificate_margin[i_c, i_b] = margin

