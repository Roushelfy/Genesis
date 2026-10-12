"""Native six-dimensional dual fitting of finite local Coulomb tractions."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import array_class
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .contact import StressContactState
from .data import StressInfo, StressState
from .provenance import StressSourceState, StressSourceStatus
from .surface import StressSurfaceInfo

_COOPERATIVE_REFINED_FANS = True


class StressTractionStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    UNRESOLVED_INTEGRATION = 2
    NUMERICAL_FAILURE = 3
    SAMPLED_WRENCH_INFEASIBLE = 4
    CONTINUOUS_FORCE_INFEASIBLE = 5
    CONTINUOUS_WRENCH_INFEASIBLE = 6


@dataclass(frozen=True)
class StressTractionInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    positions: qd.Tensor
    inward_normals: qd.Tensor
    face_origin: qd.Tensor = field(default_factory=lambda: V_VEC(3, dtype=gs.qd_float, shape=()))
    face_edges: qd.Tensor = field(default_factory=lambda: V_MAT(2, 3, dtype=gs.qd_float, shape=()))
    face_dual: qd.Tensor = field(default_factory=lambda: V_MAT(2, 3, dtype=gs.qd_float, shape=()))
    gauss: qd.Tensor = field(default_factory=lambda: V(dtype=gs.qd_float, shape=()))


def create_traction_info(surface_info: StressSurfaceInfo) -> StressTractionInfo:
    shape = (surface_info.positions.shape[0],)
    info = StressTractionInfo(
        positions=V_VEC(3, dtype=gs.qd_float, shape=shape),
        inward_normals=V_VEC(3, dtype=gs.qd_float, shape=shape),
        face_origin=surface_info.face_origin,
        face_edges=surface_info.face_edges,
        face_dual=surface_info.face_dual,
        gauss=surface_info.gauss,
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
class StressTractionSupport:
    """Declared source ranges for componentwise fitting on a pair's finite support."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    start: qd.Tensor
    end: qd.Tensor
    valid: qd.Tensor
    center: qd.Tensor
    radius: qd.Tensor
    friction: qd.Tensor


@dataclass(frozen=True)
class StressTractionCircle:
    """Intrinsic disk charts verified to lie within one actual boundary face."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    face: qd.Tensor
    radius: qd.Tensor
    refined: qd.Tensor
    focus: qd.Tensor


@dataclass(frozen=True)
class StressFanFrontier:
    """Stable candidate faces for the unchanged surface fan quadrature."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    indices: qd.Tensor
    count: qd.Tensor
    maximum: qd.Tensor


@dataclass(frozen=True)
class StressTractionPriors:
    """Original local loads and profile integrals for an explicit aggregate law."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    anchor_moment: qd.Tensor
    weight_sum: qd.Tensor
    group_scale: qd.Tensor


@dataclass(frozen=True)
class StressTractionAcceptance:
    """Current-step original-point wrench and cone checks from fused scatter."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    moment: qd.Tensor
    cone_valid: qd.Tensor


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
    fan_barycentric: qd.Tensor
    fan_coordinate_power: qd.Tensor
    fan_refined: qd.Tensor
    circle: StressTractionCircle
    support: StressTractionSupport
    frontier: StressFanFrontier
    priors: StressTractionPriors
    acceptance: StressTractionAcceptance


def create_tractions(
    n_samples: int,
    n_contacts: int,
    n_envs: int,
    *,
    local_samples: bool = False,
    local_sample_count: int | None = None,
    packed_samples: bool = False,
    fan_faces: int = 0,
    fan_coordinate_power: int = 1,
    shared_support: bool = False,
    surface_fan_frontier: bool = True,
    source_priors: bool = False,
    cached_acceptance: bool = False,
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
    frontier_shape = (
        (fan_faces, n_contacts, n_envs) if fan_faces and surface_fan_frontier and gs.backend == gs.cuda else ()
    )
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
        fan_barycentric=V_VEC(3, dtype=gs.qd_float, shape=(fan_faces, n_contacts, n_envs) if fan_faces else ()),
        fan_coordinate_power=V(dtype=gs.qd_int, shape=()),
        fan_refined=V(dtype=gs.qd_bool, shape=shape if fan_faces else ()),
        circle=StressTractionCircle(
            face=V(dtype=gs.qd_int, shape=shape if fan_faces else ()),
            radius=V(dtype=gs.qd_float, shape=shape if fan_faces else ()),
            refined=V(dtype=gs.qd_bool, shape=shape if fan_faces else ()),
            focus=V_VEC(2, dtype=gs.qd_float, shape=shape if fan_faces else ()),
        ),
        support=StressTractionSupport(
            start=V(dtype=gs.qd_int, shape=shape if shared_support else ()),
            end=V(dtype=gs.qd_int, shape=shape if shared_support else ()),
            valid=V(dtype=gs.qd_bool, shape=shape if shared_support else ()),
            center=V_VEC(3, dtype=gs.qd_float, shape=shape if shared_support else ()),
            radius=V(dtype=gs.qd_float, shape=shape if shared_support else ()),
            friction=V(dtype=gs.qd_float, shape=shape if shared_support else ()),
        ),
        frontier=StressFanFrontier(
            indices=V(dtype=gs.qd_int, shape=frontier_shape),
            count=V(dtype=gs.qd_int, shape=shape if frontier_shape else ()),
            maximum=V(dtype=gs.qd_int, shape=()),
        ),
        priors=StressTractionPriors(
            force=V_VEC(3, dtype=gs.qd_float, shape=shape if source_priors else ()),
            anchor_moment=V_VEC(3, dtype=gs.qd_float, shape=shape if source_priors else ()),
            weight_sum=V(dtype=gs.qd_float, shape=shape if source_priors else ()),
            group_scale=V(dtype=gs.qd_float, shape=shape if source_priors else ()),
        ),
        acceptance=StressTractionAcceptance(
            force=V_VEC(3, dtype=gs.qd_float, shape=shape if cached_acceptance else ()),
            moment=V_VEC(3, dtype=gs.qd_float, shape=shape if cached_acceptance else ()),
            cone_valid=V(dtype=gs.qd_bool, shape=shape if cached_acceptance else ()),
        ),
    )
    state.sample_count.fill(0 if packed_samples else n_samples)
    state.fan_coordinate_power.fill(fan_coordinate_power)
    state.fan_refined.fill(False)
    state.circle.face.fill(-1)
    state.circle.refined.fill(False)
    state.circle.focus.fill(0.0)
    return state


@qd.func
def func_traction_support_range(i_c: int, i_b: int, traction_state: StressTractionState):
    start, end = i_c, i_c + 1
    if qd.static(len(traction_state.support.start.shape) == 2):
        start = traction_state.support.start[i_c, i_b]
        end = traction_state.support.end[i_c, i_b]
    return start, end


@qd.func
def func_traction_source_friction(i_s: int, i_b: int, friction: float, traction_state: StressTractionState):
    if qd.static(len(traction_state.support.friction.shape) == 2):
        friction = traction_state.support.friction[i_s, i_b]
    return friction


@qd.func
def func_traction_constraint_active(i_c: int, i_b: int, traction_state: StressTractionState):
    active = True
    if qd.static(len(traction_state.priors.force.shape) == 2):
        active = i_c == traction_state.support.start[i_c, i_b]
    return active


@qd.func
def func_traction_reference(
    i_s: int,
    i_c: int,
    i_b: int,
    force: qd.types.vector(3),
    weight_sum: float,
    traction_state: StressTractionState,
):
    reference = force
    if qd.static(len(traction_state.priors.force.shape) == 2):
        reference = qd.Vector.zero(gs.qd_float, 3)
        source_weight = traction_state.priors.weight_sum[i_s, i_b]
        scale = traction_state.priors.group_scale[i_c, i_b]
        if source_weight > 0.0 and scale > 0.0:
            reference = traction_state.priors.force[i_s, i_b] * (weight_sum / source_weight / scale)
    return reference


@qd.func
def func_circle_panel(
    i_q: int, i_c: int, i_b: int, traction_state: StressTractionState, traction_info: StressTractionInfo
):
    """Return disk coordinates and area Jacobian for a partition adapted to the required compression centroid."""
    fraction, theta = traction_state.circle.focus[i_c, i_b][0], traction_state.circle.focus[i_c, i_b][1]
    half_width = qd.min(0.5 * qd.math.pi, qd.sqrt(8.0 * (1.0 - fraction)))
    radial_edges = qd.Vector(
        [
            0.0,
            qd.max(0.25 * (1.0 - fraction), fraction - 8.0 * (1.0 - fraction)),
            0.5 * (1.0 + fraction),
            1.0,
        ]
    )
    angular_edges = qd.Vector([theta - qd.math.pi, theta - half_width, theta + half_width, theta + qd.math.pi])
    n = traction_info.gauss.shape[0]
    panel, local = i_q // (n * n), i_q % (n * n)
    i_r, i_t = panel // 3, panel % 3
    u, v = traction_info.gauss[local // n], traction_info.gauss[local % n]
    width_r = radial_edges[i_r + 1] - radial_edges[i_r]
    width_t = angular_edges[i_t + 1] - angular_edges[i_t]
    distance = radial_edges[i_r] + width_r * u
    angle = angular_edges[i_t] + width_t * v
    return distance, angle, width_r * width_t


@qd.func
def func_surface_fan_bary(
    i_sample: int, i_c: int, i_b: int, traction_state: StressTractionState, traction_info: StressTractionInfo
):
    n = traction_info.gauss.shape[0]
    n_quadrature = n * n
    i_face = 0
    bary = qd.Vector.zero(gs.qd_float, 3)
    power = traction_state.fan_coordinate_power[None]
    # Circle samples occupy a disjoint index namespace, without storing a
    # per-sample geometry cloud or changing the exact profile allocation.
    circle_start = 3 * traction_info.positions.shape[0]
    if i_sample >= circle_start:
        i_face = traction_state.circle.face[i_c, i_b]
        i_q = i_sample - circle_start
        edges = traction_info.face_edges[i_face]
        normal = traction_info.inward_normals[i_face * n_quadrature]
        length = edges[0, :].norm()
        tangent = edges[0, :] / length
        second = normal.cross(tangent)
        distance, angle = gs.qd_float(0.0), gs.qd_float(0.0)
        if traction_state.circle.refined[i_c, i_b]:
            distance, angle, _ = func_circle_panel(i_q, i_c, i_b, traction_state, traction_info)
            distance *= traction_state.circle.radius[i_c, i_b]
        else:
            u, v = traction_info.gauss[i_q // n], traction_info.gauss[i_q % n]
            distance = traction_state.circle.radius[i_c, i_b] * u**power
            angle = 2.0 * qd.math.pi * v
        dv = distance * qd.sin(angle) / edges[1, :].dot(second)
        du = (distance * qd.cos(angle) - edges[1, :].dot(tangent) * dv) / length
        bary = traction_state.fan_barycentric[i_face, i_c, i_b] + qd.Vector([-du - dv, du, dv])
    else:
        i_face = i_sample // (3 * n_quadrature)
        i_fan = (i_sample // n_quadrature) % 3
        i_q = i_sample % n_quadrature
        u, v = traction_info.gauss[i_q // n], traction_info.gauss[i_q % n]
        u, v = u**power, v**power
        bary = (1.0 - u) * traction_state.fan_barycentric[i_face, i_c, i_b]
        bary[i_fan] += u * (1.0 - v)
        bary[(i_fan + 1) % 3] += u * v
    return i_face, bary


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
    if qd.static(len(traction_state.fan_barycentric.shape) == 3):
        i_face, bary = func_surface_fan_bary(i_sample, i_c, i_b, traction_state, traction_info)
        edges = traction_info.face_edges[i_face]
        position = traction_info.face_origin[i_face] + bary[1] * edges[0, :] + bary[2] * edges[1, :]
        normal = traction_info.inward_normals[i_face * traction_info.gauss.shape[0] ** 2]
    elif qd.static(len(traction_state.sample_faces.shape) == 2):
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
    if qd.static(len(traction_state.fan_barycentric.shape) == 3):
        i_face, bary = func_surface_fan_bary(i_sample, i_c, i_b, traction_state, traction_info)
        basis = qd.Vector(
            [
                bary[0] * (2.0 * bary[0] - 1.0),
                bary[1] * (2.0 * bary[1] - 1.0),
                bary[2] * (2.0 * bary[2] - 1.0),
                4.0 * bary[0] * bary[1],
                4.0 * bary[0] * bary[2],
                4.0 * bary[1] * bary[2],
            ]
        )
    elif qd.static(len(traction_state.sample_faces.shape) == 2):
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
    start, end = func_traction_support_range(i_c, i_b, traction_state)
    for i_s in range(start, end):
        source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
        for i_q in range(traction_state.sample_count[i_s, i_b]):
            weight = traction_state.profile[i_q, i_s, i_b] / traction_state.weight_sum[i_c, i_b]
            if weight > 0.0:
                position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                relative = (position - anchor) / radius
                reference = func_traction_reference(
                    i_s, i_c, i_b, force, traction_state.weight_sum[i_c, i_b], traction_state
                )
                value = reference - dual[0:3] - dual[3:6].cross(relative)
                projected, jacobian = func_coulomb_projection(value, normal, source_friction)
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


def kernel_fit_tractions(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    epsilon: float,
    fit_tolerance: float,
    max_evaluations: int,
    retry_mode: int = 0,
    cooperative: bool = True,
    warm_start: bool = True,
):
    """Fit all supplied tractions or retry only unresolved contacts using the same native numerical path.

    Retry mode zero fits all contacts; one retries numerical or sampled failures; two retries refined disk
    failures; three refits every constraint whose fan support was repartitioned. Retry counts include previous
    evaluations. Geometry changes invalidate successful constraints sharing that support. Mode four continues
    only numerical failures on the unchanged refined fan, retaining their verified finite dual iterate.
    """
    if cooperative and gs.backend == gs.cuda and (retry_mode == 0 or retry_mode == 3 and _COOPERATIVE_REFINED_FANS):
        from .cooperative_fit import kernel_fit_tractions as kernel_fit_cooperative

        if retry_mode == 3:
            # A changed support invalidates successful constraints sharing it.
            # Refit every selected constraint before restricting the serial fallback.
            kernel_fit_cooperative(
                contact_state,
                source_state,
                traction_state,
                traction_info,
                epsilon,
                fit_tolerance,
                max_evaluations,
                warm_start=False,
                sample_lanes=8,
                refit_fans=True,
            )
            kernel_fit_tractions_impl(
                contact_state,
                source_state,
                traction_state,
                traction_info,
                epsilon,
                fit_tolerance,
                max_evaluations,
                retry_mode=5,
            )
        else:
            kernel_fit_cooperative(
                contact_state,
                source_state,
                traction_state,
                traction_info,
                epsilon,
                fit_tolerance,
                max_evaluations,
                warm_start=warm_start,
            )
            # A failed warm iterate can recover cheaply from a cold warp fit.
            # A second reduction order handles remaining stagnation; refined
            # routes retain their serial fallback and continuation.
            kernel_fit_cooperative(
                contact_state,
                source_state,
                traction_state,
                traction_info,
                epsilon,
                fit_tolerance,
                max_evaluations,
                warm_start=False,
                is_retry=True,
            )
            kernel_fit_cooperative(
                contact_state,
                source_state,
                traction_state,
                traction_info,
                epsilon,
                fit_tolerance,
                max_evaluations,
                warm_start=False,
                is_retry=True,
                sample_lanes=8,
            )
    else:
        kernel_fit_tractions_impl(
            contact_state,
            source_state,
            traction_state,
            traction_info,
            epsilon,
            fit_tolerance,
            max_evaluations,
            retry_mode,
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
    retry_mode: qd.template(),
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        should_fit, prior_evaluations = True, 0
        initial_dual = qd.Vector.zero(gs.qd_float, 6)
        if qd.static(retry_mode > 0):
            should_fit = traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
            should_fit = should_fit or traction_state.status[i_c, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
            prior_evaluations = traction_state.evaluations[i_c, i_b]
        if qd.static(retry_mode == 2):
            should_fit = should_fit and traction_state.circle.refined[i_c, i_b]
        if qd.static(retry_mode == 3):
            should_fit = traction_state.fan_refined[i_c, i_b]
        if qd.static(retry_mode == 5):
            should_fit = should_fit and traction_state.fan_refined[i_c, i_b]
        if qd.static(retry_mode == 4):
            should_fit = traction_state.fan_refined[i_c, i_b]
            should_fit = should_fit and traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
            if qd.abs(traction_state.dual[i_c, i_b]).sum() < qd.math.inf:
                initial_dual = traction_state.dual[i_c, i_b]
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
        if should_fit and contact_state.valid[i_c, i_b] and func_traction_constraint_active(i_c, i_b, traction_state):
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
            start, end = func_traction_support_range(i_c, i_b, traction_state)
            is_valid = is_valid and 0 <= start < end <= contact_state.valid.shape[0]
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
                for i_q in range(n_samples):
                    weight = traction_state.profile[i_q, i_s, i_b]
                    is_valid = is_valid and weight >= 0.0 and weight < qd.math.inf
                    if weight > 0.0 and is_valid:
                        position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                        relative = (position - anchor) / radius
                        is_valid = is_valid and qd.abs(normal.dot(normal) - 1.0) <= 64.0 * epsilon
                        # Shared support validates each destination's unchanged disk,
                        # while moments retain the original target radius and anchor.
                        is_valid = is_valid and (position - support_center).norm() <= support_radius * (
                            1.0 + 64.0 * epsilon
                        )
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
                if qd.static(len(traction_state.priors.force.shape) == 2):
                    scale = qd.max(scale, traction_state.priors.group_scale[i_c, i_b])
                traction_state.scale[i_c, i_b] = scale
                if scale > 0.0:
                    budget_scale = force.norm()
                    if budget_scale == 0.0:
                        budget_scale = scale
                    force /= scale
                    moment /= radius * scale
                    target = qd.Vector([force[0], force[1], force[2], moment[0], moment[1], moment[2]])
                    current = initial_dual
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
                                # The constraint is conditioned about the anchor;
                                # acceptance concerns the unchanged original point.
                                offset = (contact_state.position[i_c, i_b] - anchor) / radius
                                moment_error = (
                                    (gradient[3:6] - offset.cross(gradient[0:3])).norm() * scale / budget_scale
                                )
                                traction_state.force_error[i_c, i_b] = force_error
                                traction_state.moment_error[i_c, i_b] = moment_error
                                if qd.max(force_error, moment_error) <= fit_tolerance:
                                    traction_state.status[i_c, i_b] = StressTractionStatus.SUCCESS
                                    is_active = False
                                else:
                                    # Boundary-cone curvature can be much smaller
                                    # than a fixed ridge; damping every step then
                                    # prevents an otherwise feasible wrench fit.
                                    direction, is_positive, slope = gradient, False, 0.0
                                    ridge = 0.0
                                    for i_attempt in range(2):
                                        if i_attempt == 0 or not is_positive or slope <= 0.0 or not slope < qd.math.inf:
                                            if i_attempt == 1:
                                                ridge = 1e-12 * qd.max(1.0, hessian.trace() / 6.0)
                                            direction, is_positive = func_traction_direction(hessian, gradient, ridge)
                                            slope = gradient.dot(direction)
                                    if not is_positive or slope <= 0.0 or not slope < qd.math.inf:
                                        direction = gradient / lipschitz
                                        slope = gradient.dot(direction)
                                    # Near-null curvature can produce a huge
                                    # nominal Newton step before the active cones
                                    # settle. Bound the dual step, not the load.
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
                    if (
                        traction_state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
                        and current.norm() > 0.0
                    ):
                        # A strictly polar separating ray certifies the sampled cone system only.
                        # Continuous-footprint incompatibility requires a separate geometry certificate.
                        ray = -current / current.norm()
                        polar_max = gs.qd_float(-float("inf"))
                        for i_s in range(start, end):
                            source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                            for i_q in range(traction_state.sample_count[i_s, i_b]):
                                if traction_state.profile[i_q, i_s, i_b] > 0.0:
                                    position, normal = func_traction_geometry(
                                        i_q, i_s, i_b, traction_state, traction_info
                                    )
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
def func_sample_traction(
    i_q: int,
    i_c: int,
    i_b: int,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    return func_source_traction(i_q, i_c, i_c, i_b, contact_state, traction_state, traction_info)


@qd.func
def func_source_traction(
    i_q: int,
    i_s: int,
    i_c: int,
    i_b: int,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    """Evaluate an original wrench component at one actual destination source."""
    result = qd.Vector.zero(gs.qd_float, 3)
    if (
        i_q < traction_state.sample_count[i_s, i_b]
        and traction_state.status[i_c, i_b] == StressTractionStatus.SUCCESS
        and traction_state.scale[i_c, i_b] > 0.0
    ):
        weight = traction_state.profile[i_q, i_s, i_b] / traction_state.weight_sum[i_c, i_b]
        if weight > 0.0:
            position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
            relative = (position - contact_state.center[i_c, i_b]) / contact_state.radius[i_c, i_b]
            dual = traction_state.dual[i_c, i_b]
            force = contact_state.force[i_c, i_b] / traction_state.scale[i_c, i_b]
            # Match the dual's subtraction order when large components cancel.
            reference = func_traction_reference(
                i_s, i_c, i_b, force, traction_state.weight_sum[i_c, i_b], traction_state
            )
            value = reference - dual[0:3] - dual[3:6].cross(relative)
            friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
            projected, _ = func_coulomb_projection(value, normal, friction)
            result = weight * traction_state.scale[i_c, i_b] * projected
    return result


def kernel_prepare_cached_tractions(
    epsilon: float,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    traction_info: StressTractionInfo,
    errno: qd.Tensor,
    reset_force: bool = True,
):
    from .cooperative_scatter import kernel_scatter_cached_tractions

    kernel_scatter_cached_tractions(
        epsilon,
        contact_state,
        traction_state,
        state,
        stress_info,
        surface_info,
        traction_info,
        errno,
        reset_force=reset_force,
    )


def kernel_scatter_tractions(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    traction_info: StressTractionInfo,
    errno: qd.Tensor,
    reset_force: bool,
    cooperative: bool = True,
):
    if cooperative and gs.backend == gs.cuda:
        from .cooperative_scatter import kernel_scatter_tractions as kernel_scatter_cooperative

        kernel_scatter_cooperative(
            contact_state, traction_state, state, stress_info, surface_info, traction_info, errno, reset_force
        )
    else:
        kernel_scatter_tractions_impl(
            contact_state, traction_state, state, stress_info, surface_info, traction_info, errno, reset_force
        )


@qd.kernel
def kernel_scatter_tractions_impl(
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
    for i_s, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_s, i_b]:
            n_samples = traction_state.sample_count[i_s, i_b]
            start, end = func_traction_support_range(i_s, i_b, traction_state)
            for i_q in range(n_samples):
                if traction_state.profile[i_q, i_s, i_b] > 0.0:
                    force = qd.Vector.zero(gs.qd_float, 3)
                    for i_c in range(start, end):
                        force += func_source_traction(i_q, i_s, i_c, i_b, contact_state, traction_state, traction_info)
                    i_face, basis = func_traction_shape(i_q, i_s, i_b, traction_state, surface_info, traction_info)
                    for a in qd.static(range(6)):
                        i_node = stress_info.surface_nodes[i_face, a]
                        for k in qd.static(range(3)):
                            qd.atomic_add(state.force[i_node, i_b][k], basis[a] * force[k])
