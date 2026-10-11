"""Native six-dimensional dual fitting of finite local Coulomb tractions."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import array_class
from genesis.utils.array_class import V_VEC, DataKind, V

from .contact import StressContactState
from .data import StressInfo, StressState
from .provenance import StressSourceState, StressSourceStatus
from .surface import StressSurfaceInfo


class StressTractionStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    UNRESOLVED_INTEGRATION = 2
    NUMERICAL_FAILURE = 3
    SAMPLED_WRENCH_INFEASIBLE = 4
    CONTINUOUS_FORCE_INFEASIBLE = 5


@dataclass(frozen=True)
class StressTractionInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    positions: qd.Tensor
    inward_normals: qd.Tensor


def create_traction_info(surface_info: StressSurfaceInfo) -> StressTractionInfo:
    shape = (surface_info.positions.shape[0],)
    info = StressTractionInfo(
        positions=V_VEC(3, dtype=gs.qd_float, shape=shape),
        inward_normals=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )
    kernel_traction_geometry(surface_info, info)
    return info


@qd.kernel
def kernel_traction_geometry(surface_info: StressSurfaceInfo, traction_info: StressTractionInfo):
    for i_q in range(surface_info.positions.shape[0]):
        i_face = i_q // surface_info.shape.shape[0]
        traction_info.positions[i_q] = surface_info.positions[i_q]
        traction_info.inward_normals[i_q] = -surface_info.face_normal[i_face]


@dataclass(frozen=True)
class StressTractionState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    profile: qd.Tensor
    dual: qd.Tensor
    weight_sum: qd.Tensor
    scale: qd.Tensor
    status: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor
    evaluations: qd.Tensor
    certificate: qd.Tensor
    certificate_margin: qd.Tensor
    sample_positions: qd.Tensor
    sample_normals: qd.Tensor
    sample_faces: qd.Tensor
    sample_shape: qd.Tensor
    sample_count: qd.Tensor
    sample_indices: qd.Tensor


def create_tractions(
    n_samples: int,
    n_contacts: int,
    n_envs: int,
    *,
    local_samples: bool = False,
    local_sample_count: int | None = None,
    packed_samples: bool = False,
) -> StressTractionState:
    """Allocate the exact supplied quadrature subproblem, without a force cloud.

    Profile generation owns intrinsic support and integration accuracy. This
    general subproblem layout is not the selected live scheduling/layout yet.
    """
    shape = (n_contacts, n_envs)
    sample_shape = (
        (n_samples if local_sample_count is None else local_sample_count, n_contacts, n_envs) if local_samples else ()
    )
    face_shape = shape if local_samples and local_sample_count is not None else sample_shape
    state = StressTractionState(
        profile=V(dtype=gs.qd_float, shape=(n_samples, n_contacts, n_envs)),
        dual=V_VEC(6, dtype=gs.qd_float, shape=shape),
        weight_sum=V(dtype=gs.qd_float, shape=shape),
        scale=V(dtype=gs.qd_float, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
        force_error=V(dtype=gs.qd_float, shape=shape),
        moment_error=V(dtype=gs.qd_float, shape=shape),
        evaluations=V(dtype=gs.qd_int, shape=shape),
        certificate=V_VEC(6, dtype=gs.qd_float, shape=shape),
        certificate_margin=V(dtype=gs.qd_float, shape=shape),
        sample_positions=V_VEC(3, dtype=gs.qd_float, shape=sample_shape),
        sample_normals=V_VEC(3, dtype=gs.qd_float, shape=sample_shape),
        sample_faces=V(dtype=gs.qd_int, shape=face_shape),
        sample_shape=V_VEC(6, dtype=gs.qd_float, shape=sample_shape),
        sample_count=V(dtype=gs.qd_int, shape=shape),
        sample_indices=V(dtype=gs.qd_int, shape=(n_samples, n_contacts, n_envs) if packed_samples else ()),
    )
    state.sample_count.fill(0 if packed_samples else n_samples)
    return state


@qd.func
def func_traction_sample_index(i_q: int, i_c: int, i_b: int, traction_state: StressTractionState):
    i_sample = i_q
    if qd.static(len(traction_state.sample_indices.shape) == 3):
        i_sample = traction_state.sample_indices[i_q, i_c, i_b]
    return i_sample


@qd.func
def func_traction_local_index(
    i_sample: int, i_c: int, i_b: int, traction_state: StressTractionState, traction_info: StressTractionInfo
):
    i_local = -1
    if qd.static(len(traction_state.sample_faces.shape) == 2):
        n_quadrature = traction_state.sample_positions.shape[0] // 3
        i_anchor = traction_state.sample_faces[i_c, i_b]
        if i_sample >= traction_info.positions.shape[0]:
            i_local = n_quadrature + i_sample - traction_info.positions.shape[0]
        elif i_sample // n_quadrature == i_anchor:
            i_local = i_sample % n_quadrature
    return i_local


@qd.func
def func_traction_geometry(
    i_q: int, i_c: int, i_b: int, traction_state: StressTractionState, traction_info: StressTractionInfo
):
    position, normal = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    i_sample = func_traction_sample_index(i_q, i_c, i_b, traction_state)
    if qd.static(len(traction_state.sample_faces.shape) == 2):
        i_local = func_traction_local_index(i_sample, i_c, i_b, traction_state, traction_info)
        if i_local >= 0:
            position = traction_state.sample_positions[i_local, i_c, i_b]
            normal = traction_state.sample_normals[i_local, i_c, i_b]
        else:
            position = traction_info.positions[i_sample]
            normal = traction_info.inward_normals[i_sample]
    elif qd.static(len(traction_state.sample_faces.shape) == 3):
        position = traction_state.sample_positions[i_sample, i_c, i_b]
        normal = traction_state.sample_normals[i_sample, i_c, i_b]
    else:
        position = traction_info.positions[i_sample]
        normal = traction_info.inward_normals[i_sample]
    return position, normal


@qd.func
def func_traction_shape(
    i_q: int,
    i_c: int,
    i_b: int,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    traction_info: StressTractionInfo,
):
    i_sample = func_traction_sample_index(i_q, i_c, i_b, traction_state)
    n_quadrature = surface_info.shape.shape[0]
    i_face = i_sample // n_quadrature
    basis = qd.Vector.zero(gs.qd_float, 6)
    if qd.static(len(traction_state.sample_faces.shape) == 2):
        i_local = func_traction_local_index(i_sample, i_c, i_b, traction_state, traction_info)
        if i_local >= 0:
            i_face = traction_state.sample_faces[i_c, i_b]
            basis = traction_state.sample_shape[i_local, i_c, i_b]
        else:
            basis = surface_info.shape[i_sample % n_quadrature]
    elif qd.static(len(traction_state.sample_faces.shape) == 3):
        i_face = traction_state.sample_faces[i_sample, i_c, i_b]
        basis = traction_state.sample_shape[i_sample, i_c, i_b]
    else:
        basis = surface_info.shape[i_sample % n_quadrature]
    return i_face, basis


@dataclass(frozen=True)
class StressTractionSamples:
    """Explicit sample export for diagnostics and independent offline validation."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    positions: qd.Tensor
    normals: qd.Tensor
    faces: qd.Tensor
    shape: qd.Tensor


def create_traction_samples(traction_state: StressTractionState) -> StressTractionSamples:
    shape = traction_state.profile.shape
    return StressTractionSamples(
        positions=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normals=V_VEC(3, dtype=gs.qd_float, shape=shape),
        faces=V(dtype=gs.qd_int, shape=shape),
        shape=V_VEC(6, dtype=gs.qd_float, shape=shape),
    )


@qd.kernel
def kernel_traction_samples(
    samples: StressTractionSamples,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    traction_info: StressTractionInfo,
):
    for i_q, i_c, i_b in qd.ndrange(
        traction_state.profile.shape[0], traction_state.profile.shape[1], traction_state.profile.shape[2]
    ):
        samples.positions[i_q, i_c, i_b] = qd.Vector([float("nan")] * 3)
        samples.normals[i_q, i_c, i_b] = qd.Vector([float("nan")] * 3)
        samples.faces[i_q, i_c, i_b] = -1
        samples.shape[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        if i_q < traction_state.sample_count[i_c, i_b]:
            position, normal = func_traction_geometry(i_q, i_c, i_b, traction_state, traction_info)
            i_face, basis = func_traction_shape(i_q, i_c, i_b, traction_state, surface_info, traction_info)
            samples.positions[i_q, i_c, i_b] = position
            samples.normals[i_q, i_c, i_b] = normal
            samples.faces[i_q, i_c, i_b] = i_face
            samples.shape[i_q, i_c, i_b] = basis


@qd.func
def func_coulomb_projection(value: qd.types.vector(3), normal: qd.types.vector(3), friction: float):
    """Euclidean cone projection and a generalized Jacobian for a unit inward normal."""
    compression = value.dot(normal)
    tangent = value - compression * normal
    magnitude = tangent.norm()
    projected = qd.Vector.zero(gs.qd_float, 3)
    jacobian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if friction == 0.0:
        if compression > 0.0:
            projected = compression * normal
            jacobian = normal.outer_product(normal)
    elif compression >= 0.0 and magnitude <= friction * compression:
        projected = value
        jacobian = qd.Matrix.identity(gs.qd_float, 3)
    elif compression + friction * magnitude > 0.0:
        direction = tangent / magnitude
        axis = normal + friction * direction
        pressure = (compression + friction * magnitude) / (1.0 + friction * friction)
        projected = pressure * axis
        jacobian = axis.outer_product(axis) / (1.0 + friction * friction)
        jacobian += (friction * pressure / magnitude) * (
            qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal) - direction.outer_product(direction)
        )
    return projected, jacobian


@qd.func
def func_traction_dual(
    i_c: int,
    i_b: int,
    dual: qd.types.vector(6),
    force: qd.types.vector(3),
    target: qd.types.vector(6),
    anchor: qd.types.vector(3),
    radius: float,
    friction: float,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    energy = target.dot(dual)
    gradient = target
    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    n_samples = traction_state.sample_count[i_c, i_b]
    for i_q in range(n_samples):
        weight = traction_state.profile[i_q, i_c, i_b] / traction_state.weight_sum[i_c, i_b]
        if weight > 0.0:
            position, normal = func_traction_geometry(i_q, i_c, i_b, traction_state, traction_info)
            relative = (position - anchor) / radius
            value = force - dual[0:3] - dual[3:6].cross(relative)
            projected, jacobian = func_coulomb_projection(value, normal, friction)
            moment = relative.cross(projected)
            wrench = qd.Vector([projected[0], projected[1], projected[2], moment[0], moment[1], moment[2]])
            operator = qd.Matrix.zero(gs.qd_float, 6, 3)
            for a in qd.static(range(3)):
                operator[a, a] = 1.0
            operator[3, 1], operator[3, 2] = -relative[2], relative[1]
            operator[4, 0], operator[4, 2] = relative[2], -relative[0]
            operator[5, 0], operator[5, 1] = -relative[1], relative[0]
            energy += 0.5 * weight * projected.dot(projected)
            gradient -= weight * wrench
            hessian += weight * (operator @ jacobian @ operator.transpose())
    return energy, gradient, hessian


@qd.func
def func_traction_direction(matrix: qd.types.matrix(6, 6), gradient: qd.types.vector(6), ridge: float):
    """Solve the small regularized positive-semidefinite dual system."""
    factor = qd.Matrix.zero(gs.qd_float, 6, 6)
    is_valid = True
    for a in range(6):
        for b in range(a + 1):
            value = matrix[a, b]
            if a == b:
                value += ridge
            for k in range(b):
                value -= factor[a, k] * factor[b, k]
            if a == b:
                is_valid = is_valid and value > 0.0 and value < qd.math.inf
                factor[a, b] = qd.sqrt(qd.max(value, 1e-30))
            else:
                factor[a, b] = value / factor[b, b]
    direction = qd.Vector.zero(gs.qd_float, 6)
    for a in range(6):
        value = gradient[a]
        for b in range(a):
            value -= factor[a, b] * direction[b]
        direction[a] = value / factor[a, a]
    for a_ in range(6):
        a = 5 - a_
        value = direction[a]
        for b in range(a + 1, 6):
            value -= factor[b, a] * direction[b]
        direction[a] = value / factor[a, a]
    return direction, is_valid


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
                    is_valid = is_valid and relative.norm() <= 1.0 + 64.0 * epsilon
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
                            allowance = 32.0 * epsilon * (1.0 + qd.abs(energy))
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


@qd.func
def func_sample_traction(
    i_q: int,
    i_c: int,
    i_b: int,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    result = qd.Vector.zero(gs.qd_float, 3)
    if (
        i_q < traction_state.sample_count[i_c, i_b]
        and traction_state.status[i_c, i_b] == StressTractionStatus.SUCCESS
        and traction_state.scale[i_c, i_b] > 0.0
    ):
        weight = traction_state.profile[i_q, i_c, i_b] / traction_state.weight_sum[i_c, i_b]
        if weight > 0.0:
            position, normal = func_traction_geometry(i_q, i_c, i_b, traction_state, traction_info)
            relative = (position - contact_state.center[i_c, i_b]) / contact_state.radius[i_c, i_b]
            dual = traction_state.dual[i_c, i_b]
            value = contact_state.force[i_c, i_b] / traction_state.scale[i_c, i_b]
            value -= dual[0:3] + dual[3:6].cross(relative)
            projected, _ = func_coulomb_projection(value, normal, contact_state.friction[i_c, i_b])
            result = weight * traction_state.scale[i_c, i_b] * projected
    return result


@qd.kernel
def kernel_scatter_tractions(
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
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b]:
            n_samples = traction_state.sample_count[i_c, i_b]
            for i_q in range(n_samples):
                if traction_state.profile[i_q, i_c, i_b] > 0.0:
                    force = func_sample_traction(i_q, i_c, i_b, contact_state, traction_state, traction_info)
                    i_face, basis = func_traction_shape(i_q, i_c, i_b, traction_state, surface_info, traction_info)
                    for a in qd.static(range(6)):
                        i_node = stress_info.surface_nodes[i_face, a]
                        for k in qd.static(range(3)):
                            qd.atomic_add(state.force[i_node, i_b][k], basis[a] * force[k])
