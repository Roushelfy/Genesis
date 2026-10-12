"""Unselected explicit aggregate model retaining every original source-force prior.

Same positive native profiles, radius, friction and topology; aggregate wrench
conservation replaces individual point-component conservation in this experiment.
No production module is patched or changed.
"""

from dataclasses import dataclass
from typing import ClassVar
import quadrants as qd
import genesis as gs
from genesis.utils.array_class import DataKind, V, V_VEC
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, StressSourceStatus
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionState, StressTractionInfo, StressTractionStatus,
    func_traction_geometry, func_traction_source_friction,
    func_traction_support_range, func_coulomb_projection, func_traction_direction,
)

@dataclass(frozen=True)
class StressSourcePriors:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    force: qd.Tensor
    weight_sum: qd.Tensor
    scale: qd.Tensor

def create_source_priors(n_sources, n_envs):
    return StressSourcePriors(
        force=V_VEC(3, dtype=gs.qd_float, shape=(n_sources,n_envs)),
        weight_sum=V(dtype=gs.qd_float, shape=(n_sources,n_envs)),
        scale=V(dtype=gs.qd_float, shape=(n_envs,)),
    )

@qd.kernel
def kernel_snapshot_source_priors(contacts: StressContactState, sources: StressSourceState, tractions: StressTractionState, prior: StressSourcePriors):
    for i_s, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        prior.force[i_s,i_b] = contacts.force[i_s,i_b]
        total = gs.qd_float(0.0)
        for i_q in range(tractions.sample_count[i_s,i_b]):
            total += tractions.profile[i_q,i_s,i_b]
        prior.weight_sum[i_s,i_b] = total
    for i_b in range(contacts.valid.shape[1]):
        scale = gs.qd_float(0.0)
        for i_s in range(contacts.valid.shape[0]):
            scale += qd.max(contacts.force[i_s,i_b].norm(),sources.anchor_moment[i_s,i_b].norm()/contacts.radius[i_s,i_b])
        prior.scale[i_b] = scale

@qd.kernel
def kernel_aggregate_pair(contacts: StressContactState, sources: StressSourceState):
    for i_b in range(contacts.valid.shape[1]):
        anchor = contacts.center[0,i_b]
        point = contacts.position[0,i_b]
        force = contacts.force[0,i_b] + contacts.force[1,i_b]
        moment = sources.anchor_moment[0,i_b] + sources.anchor_moment[1,i_b]
        moment += (contacts.center[1,i_b]-anchor).cross(contacts.force[1,i_b])
        contacts.force[0,i_b] = force
        sources.anchor_moment[0,i_b] = moment
        sources.free_moment[0,i_b] = moment - (point-anchor).cross(force)
        contacts.valid[1,i_b] = False
        contacts.force[1,i_b] = qd.Vector.zero(gs.qd_float,3)
        sources.anchor_moment[1,i_b] = qd.Vector.zero(gs.qd_float,3)
        sources.free_moment[1,i_b] = qd.Vector.zero(gs.qd_float,3)

@qd.func
def func_prior_dual(i_c: int, i_b: int, dual: qd.types.vector(6), force: qd.types.vector(3), target: qd.types.vector(6), anchor: qd.types.vector(3), radius: float, friction: float, traction_state: StressTractionState, traction_info: StressTractionInfo, prior: StressSourcePriors):
    energy = target.dot(dual)
    gradient = target
    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    (start, end) = func_traction_support_range(i_c, i_b, traction_state)
    for i_s in range(start, end):
        source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
        for i_q in range(traction_state.sample_count[i_s, i_b]):
            weight = traction_state.profile[i_q, i_s, i_b] / traction_state.weight_sum[i_c, i_b]
            if weight > 0.0:
                (position, normal) = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                relative = (position - anchor) / radius
                value = prior.force[i_s, i_b] * (traction_state.weight_sum[i_c, i_b] / prior.weight_sum[i_s, i_b]) / traction_state.scale[i_c, i_b] - dual[0:3] - dual[3:6].cross(relative)
                (projected, jacobian) = func_coulomb_projection(value, normal, source_friction)
                moment = relative.cross(projected)
                wrench = qd.Vector([projected[0], projected[1], projected[2], moment[0], moment[1], moment[2]])
                operator = qd.Matrix.zero(gs.qd_float, 6, 3)
                for a in qd.static(range(3)):
                    operator[a, a] = 1.0
                (operator[3, 1], operator[3, 2]) = (-relative[2], relative[1])
                (operator[4, 0], operator[4, 2]) = (relative[2], -relative[0])
                (operator[5, 0], operator[5, 1]) = (-relative[1], relative[0])
                energy += 0.5 * weight * projected.dot(projected)
                gradient -= weight * wrench
                hessian += weight * (operator @ jacobian @ operator.transpose())
    return (energy, gradient, hessian)

@qd.kernel
def kernel_fit_source_priors(contact_state: StressContactState, source_state: StressSourceState, traction_state: StressTractionState, traction_info: StressTractionInfo, epsilon: float, fit_tolerance: float, max_evaluations: int, retry_mode: qd.template(), prior: StressSourcePriors):
    for (i_c, i_b) in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        (should_fit, prior_evaluations) = (True, 0)
        if qd.static(retry_mode > 0):
            should_fit = traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
            should_fit = should_fit or traction_state.status[i_c, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
            prior_evaluations = traction_state.evaluations[i_c, i_b]
        if qd.static(retry_mode == 2):
            should_fit = should_fit and traction_state.circle.refined[i_c, i_b]
        if should_fit:
            traction_state.dual[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
            traction_state.weight_sum[i_c, i_b] = 0.0
            traction_state.scale[i_c, i_b] = 0.0
            traction_state.status[i_c, i_b] = StressTractionStatus.SUCCESS
            traction_state.force_error[i_c, i_b] = 0.0
            traction_state.moment_error[i_c, i_b] = 0.0
            traction_state.evaluations[i_c, i_b] = 0
            traction_state.certificate[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
            traction_state.certificate_margin[i_c, i_b] = 0.0
        if should_fit and contact_state.valid[i_c, i_b]:
            anchor = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            friction = contact_state.friction[i_c, i_b]
            force = contact_state.force[i_c, i_b]
            moment = source_state.anchor_moment[i_c, i_b]
            is_valid = source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS
            is_valid = is_valid and radius > 0.0 and (friction >= 0.0) and (epsilon > 0.0) and (fit_tolerance > 0.0)
            finite = anchor.sum() + force.sum() + moment.sum() + radius + friction
            is_valid = is_valid and qd.abs(finite) < qd.math.inf
            weight_sum = gs.qd_float(0.0)
            lipschitz = gs.qd_float(0.0)
            (start, end) = func_traction_support_range(i_c, i_b, traction_state)
            is_valid = is_valid and 0 <= start < end <= contact_state.valid.shape[0]
            for i_s in range(start, end):
                (support_center, support_radius) = (anchor, radius)
                if qd.static(len(traction_state.support.valid.shape) == 2):
                    is_valid = is_valid and traction_state.support.valid[i_s, i_b]
                    support_center = traction_state.support.center[i_s, i_b]
                    support_radius = traction_state.support.radius[i_s, i_b]
                source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                is_valid = is_valid and source_friction >= 0.0 and (source_friction < qd.math.inf)
                is_valid = is_valid and support_radius > 0.0 and (support_radius < qd.math.inf)
                n_samples = traction_state.sample_count[i_s, i_b]
                is_valid = is_valid and 0 <= n_samples <= traction_state.profile.shape[0]
                for i_q in range(n_samples):
                    weight = traction_state.profile[i_q, i_s, i_b]
                    is_valid = is_valid and weight >= 0.0 and (weight < qd.math.inf)
                    if weight > 0.0 and is_valid:
                        (position, normal) = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                        relative = (position - anchor) / radius
                        is_valid = is_valid and qd.abs(normal.dot(normal) - 1.0) <= 64.0 * epsilon
                        is_valid = is_valid and (position - support_center).norm() <= support_radius * (1.0 + 64.0 * epsilon)
                        weight_sum += weight
                        lipschitz += weight * (1.0 + relative.dot(relative))
            if not is_valid:
                traction_state.status[i_c, i_b] = StressTractionStatus.INVALID_INPUT
            elif weight_sum <= 0.0:
                traction_state.status[i_c, i_b] = StressTractionStatus.UNRESOLVED_INTEGRATION
            else:
                traction_state.weight_sum[i_c, i_b] = weight_sum
                lipschitz /= weight_sum
                scale = qd.max(prior.scale[i_b], force.norm(), moment.norm() / radius)
                traction_state.scale[i_c, i_b] = scale
                if scale > 0.0:
                    budget_scale = prior.scale[i_b]
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
                    gradient_squared = gs.qd_float(float('inf'))
                    alpha = gs.qd_float(1.0)
                    slope = gs.qd_float(0.0)
                    (is_trial, is_active) = (False, True)
                    traction_state.status[i_c, i_b] = StressTractionStatus.NUMERICAL_FAILURE
                    for _ in range(max_evaluations):
                        if is_active:
                            (trial_energy, trial_gradient, hessian) = func_prior_dual(i_c, i_b, proposal, force, target, anchor, radius, friction, traction_state, traction_info, prior)
                            traction_state.evaluations[i_c, i_b] += 1
                            trial_squared = trial_gradient.dot(trial_gradient)
                            comparison_scale = 1.0 + qd.abs(energy) + qd.abs(trial_energy)
                            comparison_scale += (qd.abs(current) * qd.abs(target)).sum()
                            comparison_scale += (qd.abs(proposal) * qd.abs(target)).sum()
                            allowance = 32.0 * epsilon * comparison_scale
                            is_accepted = not is_trial or trial_energy <= energy - 0.0001 * alpha * slope
                            is_accepted = is_accepted or (trial_energy <= energy + allowance and trial_squared < gradient_squared)
                            if is_accepted:
                                (current, energy, gradient) = (proposal, trial_energy, trial_gradient)
                                gradient_squared = trial_squared
                                traction_state.dual[i_c, i_b] = current
                                force_error = gradient[0:3].norm() * scale / budget_scale
                                offset = (contact_state.position[i_c, i_b] - anchor) / radius
                                moment_error = (gradient[3:6] - offset.cross(gradient[0:3])).norm() * scale / budget_scale
                                traction_state.force_error[i_c, i_b] = force_error
                                traction_state.moment_error[i_c, i_b] = moment_error
                                if qd.max(force_error, moment_error) <= fit_tolerance:
                                    traction_state.status[i_c, i_b] = StressTractionStatus.SUCCESS
                                    is_active = False
                                else:
                                    (direction, is_positive, slope) = (gradient, False, 0.0)
                                    ridge = 0.0
                                    for i_attempt in range(2):
                                        if i_attempt == 0 or not is_positive or slope <= 0.0 or (not slope < qd.math.inf):
                                            if i_attempt == 1:
                                                ridge = 1e-12 * qd.max(1.0, hessian.trace() / 6.0)
                                            (direction, is_positive) = func_traction_direction(hessian, gradient, ridge)
                                            slope = gradient.dot(direction)
                                    if not is_positive or slope <= 0.0 or (not slope < qd.math.inf):
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
                    if traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE and current.norm() > 0.0:
                        ray = -current / current.norm()
                        polar_max = gs.qd_float(-float('inf'))
                        for i_s in range(start, end):
                            source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                            for i_q in range(traction_state.sample_count[i_s, i_b]):
                                if traction_state.profile[i_q, i_s, i_b] > 0.0:
                                    (position, normal) = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                                    relative = (position - anchor) / radius
                                    value = ray[0:3] + ray[3:6].cross(relative)
                                    compression = value.dot(normal)
                                    polar = compression + source_friction * (value - compression * normal).norm()
                                    polar_max = qd.max(polar_max, polar)
                        margin = target.dot(ray)
                        if polar_max < -64.0 * epsilon and margin > fit_tolerance:
                            traction_state.status[i_c, i_b] = StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
                            traction_state.certificate[i_c, i_b] = ray
                            traction_state.certificate_margin[i_c, i_b] = margin
        if should_fit:
            traction_state.evaluations[i_c, i_b] += prior_evaluations

@qd.func
def func_prior_traction(i_q: int, i_s: int, i_c: int, i_b: int, contact_state: StressContactState, traction_state: StressTractionState, traction_info: StressTractionInfo, prior: StressSourcePriors):
    """Evaluate an original wrench component at one actual destination source."""
    result = qd.Vector.zero(gs.qd_float, 3)
    if i_q < traction_state.sample_count[i_s, i_b] and traction_state.status[i_c, i_b] == StressTractionStatus.SUCCESS and (traction_state.scale[i_c, i_b] > 0.0):
        weight = traction_state.profile[i_q, i_s, i_b] / traction_state.weight_sum[i_c, i_b]
        if weight > 0.0:
            (position, normal) = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
            relative = (position - contact_state.center[i_c, i_b]) / contact_state.radius[i_c, i_b]
            dual = traction_state.dual[i_c, i_b]
            force = prior.force[i_s, i_b] * (traction_state.weight_sum[i_c, i_b] / prior.weight_sum[i_s, i_b]) / traction_state.scale[i_c, i_b]
            value = force - dual[0:3] - dual[3:6].cross(relative)
            friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
            (projected, _) = func_coulomb_projection(value, normal, friction)
            result = weight * traction_state.scale[i_c, i_b] * projected
    return result


@qd.kernel
def kernel_query_prior_forces(result: qd.Tensor, contacts: StressContactState, tractions: StressTractionState, info: StressTractionInfo, prior: StressSourcePriors):
    for i_q,i_s,i_b in qd.ndrange(result.shape[0], result.shape[1], result.shape[2]):
        result[i_q,i_s,i_b] = func_prior_traction(i_q,i_s,0,i_b,contacts,tractions,info,prior)

@qd.kernel
def kernel_scatter_prior_forces(forces: qd.Tensor, state: StressState, tractions: StressTractionState, info: StressTractionInfo, model: StressInfo, surface: StressSurfaceInfo):
    for i_n,i_b in qd.ndrange(state.force.shape[0],state.force.shape[1]):
        state.force[i_n,i_b] = qd.Vector.zero(gs.qd_float,3)
    for i_q,i_s,i_b in qd.ndrange(forces.shape[0], forces.shape[1], forces.shape[2]):
        if i_q < tractions.sample_count[i_s,i_b] and tractions.profile[i_q,i_s,i_b] > 0.0:
            from_basis = qd.Vector.zero(gs.qd_float,6)
            face = 0
            face, from_basis = func_prior_sample_basis(i_q,i_s,i_b,tractions,surface,info)
            for local in qd.static(range(6)):
                node = model.surface_nodes[face,local]
                for dimension in qd.static(range(3)):
                    qd.atomic_add(state.force[node,i_b][dimension],from_basis[local]*forces[i_q,i_s,i_b][dimension])

from genesis.engine.solvers.rigid.stress.traction import func_traction_shape as func_prior_sample_basis
