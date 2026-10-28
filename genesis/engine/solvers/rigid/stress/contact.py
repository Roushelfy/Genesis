"""Native finite, nonnegative pressure preserving each point-contact wrench."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V, of_kind
from genesis.utils.simt import qd_block_min

from .data import StressInfo, StressState, func_scatter_load
from .grid import func_patch_cell, func_patch_grid
from .quadrilateral import func_quad_inverse, func_quad_inverse_prepared, func_quad_section
from .surface import StressSurfaceInfo, func_closest_triangle, func_surface_face_admitted

# The normalized dual gradient bounds relative resultant force and moment/radius.
# Stop at 10% of the independently checked 1e-8 contact wrench budget; the final
# scatter acceptance and complete elastic residual/stress checks remain mandatory.
_PRESSURE_FIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StressPressureAdmission:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    rule: qd.Tensor
    reason: qd.Tensor
    errors: qd.Tensor
    check_mass: qd.Tensor
    attempts: qd.Tensor
    stage_errors: qd.Tensor
    stage_mass: qd.Tensor
    stage_status: qd.Tensor
    stage_reason: qd.Tensor
    stage_evaluations: qd.Tensor
    checker_coefficient: qd.Tensor
    stage_checker_status: qd.Tensor
    stage_checker_evaluations: qd.Tensor
    stage_source_errors: qd.Tensor
    counters: qd.Tensor
    overflow: qd.Tensor


@dataclass(frozen=True)
class StressContactState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    position: qd.Tensor
    center: qd.Tensor
    anchor_face: qd.Tensor
    is_refined: qd.Tensor
    force: qd.Tensor
    normal: qd.Tensor
    friction: qd.Tensor
    radius: qd.Tensor = of_kind(DataKind.INFO)  # noqa: RUF009 - typed array metadata
    radius_status: qd.Tensor
    hertz_profile: qd.Tensor = of_kind(DataKind.CONSTANT)  # noqa: RUF009 - typed array metadata
    valid: qd.Tensor
    coefficient: qd.Tensor
    weight_sum: qd.Tensor
    status: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor
    candidate_count: qd.Tensor
    evaluations: qd.Tensor
    active_pairs: qd.Tensor
    active_count: qd.Tensor
    frame_position: qd.Tensor
    frame_quaternion: qd.Tensor
    admission: StressPressureAdmission


def create_contacts(
    n_contacts: int,
    n_envs: int,
    radius: float,
    cooperative: bool = True,
    cooperative_scatter: bool = True,
    hertz_profile: bool = False,
    adaptive: bool = False,
) -> StressContactState:
    shape = (n_contacts, n_envs)
    state = StressContactState(
        position=V_VEC(3, dtype=gs.qd_float, shape=shape),
        center=V_VEC(3, dtype=gs.qd_float, shape=shape),
        anchor_face=V(dtype=gs.qd_int, shape=shape),
        is_refined=V(dtype=gs.qd_bool, shape=shape),
        force=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        radius=V(dtype=gs.qd_float, shape=shape),
        radius_status=V(dtype=gs.qd_int, shape=shape),
        hertz_profile=V(dtype=gs.qd_bool, shape=()),
        valid=V(dtype=gs.qd_bool, shape=shape),
        coefficient=V_VEC(3, dtype=gs.qd_float, shape=shape),
        weight_sum=V(dtype=gs.qd_float, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
        force_error=V(dtype=gs.qd_float, shape=shape),
        moment_error=V(dtype=gs.qd_float, shape=shape),
        candidate_count=V(dtype=gs.qd_int, shape=shape),
        evaluations=V(dtype=gs.qd_int, shape=shape),
        active_pairs=V_VEC(
            2,
            dtype=gs.qd_int,
            shape=(n_contacts * n_envs if (cooperative or cooperative_scatter) and gs.backend == gs.cuda else 0,),
        ),
        active_count=V(dtype=gs.qd_int, shape=()),
        frame_position=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        frame_quaternion=V_VEC(4, dtype=gs.qd_float, shape=(n_envs,)),
        admission=StressPressureAdmission(
            rule=V_VEC(1, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            reason=V_VEC(1, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            errors=V_VEC(4, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            check_mass=V_VEC(1, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            attempts=V_VEC(1, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            stage_errors=V_MAT(3, 4, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            stage_mass=V_VEC(3, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            stage_status=V_VEC(3, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            stage_reason=V_VEC(3, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            stage_evaluations=V_VEC(3, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            checker_coefficient=V_VEC(3, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            stage_checker_status=V_VEC(3, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            stage_checker_evaluations=V_VEC(3, dtype=gs.qd_int, shape=shape if adaptive else (0, 0)),
            stage_source_errors=V_MAT(3, 4, dtype=gs.qd_float, shape=shape if adaptive else (0, 0)),
            counters=V_VEC(9, dtype=qd.i64, shape=(n_envs if adaptive else 0,)),
            overflow=V_VEC(1, dtype=gs.qd_int, shape=(n_envs if adaptive else 0,)),
        ),
    )
    state.radius.fill(radius)
    state.radius_status.fill(0)
    state.hertz_profile.fill(hertz_profile)
    state.valid.fill(False)
    if adaptive:
        state.admission.rule.fill(1)
        state.admission.reason.fill(0)
        state.admission.errors.fill(0)
        state.admission.check_mass.fill(0)
        state.admission.attempts.fill(0)
        state.admission.stage_errors.fill(0)
        state.admission.stage_mass.fill(0)
        state.admission.stage_status.fill(0)
        state.admission.stage_reason.fill(0)
        state.admission.stage_evaluations.fill(0)
        state.admission.checker_coefficient.fill(0)
        state.admission.stage_checker_status.fill(0)
        state.admission.stage_checker_evaluations.fill(0)
        state.admission.stage_source_errors.fill(0)
        state.admission.counters.fill(0)
        state.admission.overflow.fill(0)
    return state


@qd.func
def func_force_frame(force: qd.types.vector(3)):
    direction = force.normalized()
    axis = 0
    for i_a in range(1, 3):
        if qd.abs(direction[i_a]) < qd.abs(direction[axis]):
            axis = i_a
    unit = qd.Vector.zero(gs.qd_float, 3)
    unit[axis] = 1.0
    first = direction.cross(unit).normalized()
    return first, direction.cross(first)


@qd.func
def func_patch_sample(
    i_q: int,
    center: qd.types.vector(3),
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    radius: float,
    surface_info: StressSurfaceInfo,
):
    relative = (surface_info.positions[i_q] - center) / radius
    distance = relative.dot(relative)
    weight = gs.qd_float(0.0)
    if distance < 1.0:
        weight = qd.exp(-0.5 * distance / (0.45 * 0.45)) * (1.0 - distance) ** 2 * surface_info.weights[i_q]
    coordinates = qd.Vector([1.0, relative.dot(first), relative.dot(second)])
    return coordinates, weight


@qd.func
def func_contact_order(i_c: int, i_b: int, contacts: StressContactState, surface: StressSurfaceInfo):
    order = surface.gauss.shape[0]
    if surface.pressure_rules.orders.shape[0] > 0:
        i_r = contacts.admission.rule[i_c, i_b][0]
        order = surface.pressure_rules.orders[i_r][0]
    return order


@qd.func
def func_contact_gauss(i_q: int, i_c: int, i_b: int, contacts: StressContactState, surface: StressSurfaceInfo):
    node, weight = gs.qd_float(0.0), gs.qd_float(0.0)
    if surface.pressure_rules.orders.shape[0] > 0:
        i_r = contacts.admission.rule[i_c, i_b][0]
        i_start = surface.pressure_rules.offsets[i_r][0]
        node = surface.pressure_rules.nodes[i_start + i_q][0]
        weight = surface.pressure_rules.weights[i_start + i_q][0]
    else:
        node, weight = surface.gauss[i_q], surface.gauss_weights[i_q]
    return node, weight


@qd.func
def func_contact_section(
    i_face: int,
    center: qd.types.vector(3),
    radius: float,
    surface_info: StressSurfaceInfo,
):
    closest = qd.Vector.zero(gs.qd_float, 3)
    remaining, contained = gs.qd_float(0.0), False
    if surface_info.quad_vertices.shape[0] > 0:
        closest, remaining, contained = func_quad_section(
            surface_info.quad_vertices[i_face],
            surface_info.face_origin[i_face],
            surface_info.face_normal[i_face],
            center,
            radius,
        )
    else:
        closest, anchor = func_closest_triangle(i_face, center, surface_info)
        remaining = radius * radius - (closest - center).norm_sqr()
        contained = False
        if remaining > 0.0:
            dual = surface_info.face_dual[i_face]
            gradients = qd.Matrix.rows([-dual[0, :] - dual[1, :], dual[0, :], dual[1, :]])
            contained = True
            for i in qd.static(range(3)):
                contained = contained and anchor[i] > qd.sqrt(remaining) * gradients[i, :].norm()
    return closest, remaining, contained


@qd.func
def func_hertz_disk_moments(
    i_face: int,
    closest: qd.types.vector(3),
    remaining: float,
    center: qd.types.vector(3),
    radius: float,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    surface: StressSurfaceInfo,
):
    """Exact Gram and P2/affine-pressure moments of a contained spherical Hertz section."""
    normal = surface.face_normal[i_face]
    covariance = remaining / 5.0 * (qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal))
    mass = 2.0 * qd.math.pi * remaining * qd.sqrt(remaining) / (3.0 * radius)
    offset = (closest - center) / radius
    mean = qd.Vector([1.0, offset.dot(first), offset.dot(second)])
    directions = qd.Matrix.rows([qd.Vector.zero(gs.qd_float, 3), first / radius, second / radius])
    gram = mass * (mean.outer_product(mean) + directions @ covariance @ directions.transpose())
    dual = surface.face_dual[i_face]
    local = dual @ (closest - surface.face_origin[i_face])
    bary = qd.Vector([1.0 - local.sum(), local[0], local[1]])
    gradients = qd.Matrix.rows([-dual[0, :] - dual[1, :], dual[0, :], dual[1, :]])
    second_moment = bary.outer_product(bary) + gradients @ covariance @ gradients.transpose()
    load = qd.Matrix.zero(gs.qd_float, 6, 3)
    for i in qd.static(range(3)):
        average = 2.0 * second_moment[i, i] - bary[i]
        linear = (4.0 * bary[i] - 1.0) * gradients[i, :]
        load[i, :] = mass * (average * mean + directions @ covariance @ linear)
    for i, j, node in qd.static(((0, 1, 3), (0, 2, 4), (1, 2, 5))):
        linear = 4.0 * (bary[i] * gradients[j, :] + bary[j] * gradients[i, :])
        load[node, :] = mass * (4.0 * second_moment[i, j] * mean + directions @ covariance @ linear)
    return gram, load


@qd.func
def func_contact_ray(
    i_face: int,
    i_triangle: int,
    i_angular: int,
    closest: qd.types.vector(3),
    offset: qd.types.vector(3),
    remaining: float,
    surface: StressSurfaceInfo,
):
    return func_contact_ray_at(i_face, i_triangle, surface.gauss[i_angular], closest, offset, remaining, surface)


@qd.func
def func_contact_ray_selected(
    i_face: int,
    i_triangle: int,
    i_angular: int,
    i_c: int,
    i_b: int,
    closest: qd.types.vector(3),
    offset: qd.types.vector(3),
    remaining: float,
    contacts: StressContactState,
    surface: StressSurfaceInfo,
):
    angular, _weight = func_contact_gauss(i_angular, i_c, i_b, contacts, surface)
    return func_contact_ray_at(i_face, i_triangle, angular, closest, offset, remaining, surface)


@qd.func
def func_contact_ray_at(
    i_face: int,
    i_triangle: int,
    angular: float,
    closest: qd.types.vector(3),
    offset: qd.types.vector(3),
    remaining: float,
    surface: StressSurfaceInfo,
):
    vertices = qd.Matrix.zero(gs.qd_float, 4, 3)
    n_fans = 3
    if surface.quad_vertices.shape[0] > 0:
        n_fans = 4
        normal = surface.face_normal[i_face]
        origin = surface.face_origin[i_face]
        for i in qd.static(range(4)):
            vertex = surface.quad_vertices[i_face][i, :]
            vertices[i, :] = vertex - (vertex - origin).dot(normal) * normal
    else:
        vertices[0, :] = surface.face_origin[i_face]
        vertices[1, :] = surface.face_origin[i_face] + surface.face_edges[i_face][0, :]
        vertices[2, :] = surface.face_origin[i_face] + surface.face_edges[i_face][1, :]
    edge_a = vertices[i_triangle, :] - closest
    edge_b = vertices[(i_triangle + 1) % n_fans, :] - closest
    v = angular
    direction = (1.0 - v) * edge_a + v * edge_b
    cross = edge_a.cross(edge_b)
    area = cross.norm()
    if surface.angular_quadrature[None] and area > 0.0:
        # Uniform edge parameters undersample a tiny disk close to a long edge's
        # endpoint. Uniform fan angle instead has Jacobian angle * ray_length**2.
        normal = cross / area
        tangent = edge_a.normalized()
        angle = qd.atan2(area, edge_a.dot(edge_b))
        unit = qd.cos(v * angle) * tangent + qd.sin(v * angle) * normal.cross(tangent)
        denominator = unit.cross(edge_b - edge_a).dot(normal)
        if denominator > 0.0:
            length = area / denominator
            direction = length * unit
            area = angle * length * length
    squared = direction.norm_sqr()
    limit = gs.qd_float(0.0)
    if squared > 0.0 and area > 0.0 and remaining > 0.0:
        projection = offset.dot(direction)
        discriminant = qd.sqrt(projection * projection + squared * remaining)
        limit = (discriminant - projection) / squared
        if projection >= 0.0:
            limit = remaining / (discriminant + projection)
        limit = qd.min(1.0, limit)
    return direction, area, limit


@qd.func
def func_contact_sample(
    i_q: int,
    i_c: int,
    i_b: int,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
):
    closest = qd.Vector.zero(gs.qd_float, 3)
    remaining, contained = gs.qd_float(0.0), False
    ray = qd.Vector.zero(gs.qd_float, 3)
    area, limit = gs.qd_float(0.0), gs.qd_float(0.0)
    n = func_contact_order(i_c, i_b, contact_state, surface_info)
    n_quadrature = n * n
    if i_q >= surface_info.positions.shape[0]:
        i_virtual = i_q - surface_info.positions.shape[0]
        i_face = i_virtual // (qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n_quadrature)
        closest, remaining, contained = func_contact_section(
            i_face, contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b], surface_info
        )
        if not contained:
            ray, area, limit = func_contact_ray_selected(
                i_face,
                (i_virtual // n_quadrature) % qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3),
                i_virtual % n,
                i_c,
                i_b,
                closest,
                closest - contact_state.center[i_c, i_b],
                remaining,
                contact_state,
                surface_info,
            )
    return func_contact_sample_prepared(
        i_q, i_c, i_b, first, second, contact_state, surface_info, closest, remaining, contained, ray, area, limit
    )


@qd.func
def func_contact_sample_prepared(
    i_q: int,
    i_c: int,
    i_b: int,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
    closest: qd.types.vector(3),
    remaining: float,
    contained: bool,
    ray: qd.types.vector(3),
    ray_area: float,
    ray_limit: float,
):
    """Integrate a sample using the unchanged face/ball section, shared by all samples of a packed face."""
    n = func_contact_order(i_c, i_b, contact_state, surface_info)
    n_quadrature = n * n
    position = qd.Vector.zero(gs.qd_float, 3)
    weight = gs.qd_float(0.0)
    shape = qd.Vector.zero(gs.qd_float, 6)
    u, v, weight_u, weight_v = gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0)
    i_face = i_q // n_quadrature
    i_triangle = 0
    if i_q < surface_info.positions.shape[0]:
        position = surface_info.positions[i_q]
        weight = surface_info.weights[i_q]
        shape = surface_info.shape[i_q % n_quadrature]
        if contact_state.is_refined[i_c, i_b]:
            weight = 0.0
    else:
        i_virtual = i_q - surface_info.positions.shape[0]
        i_face = i_virtual // (qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n_quadrature)
        i_triangle = (i_virtual // n_quadrature) % qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3)
        i_sample = i_virtual % n_quadrature
        u, weight_u = func_contact_gauss(i_sample // n, i_c, i_b, contact_state, surface_info)
        v, weight_v = func_contact_gauss(i_sample % n, i_c, i_b, contact_state, surface_info)
    return func_contact_sample_at(
        i_face,
        i_triangle,
        i_c,
        i_b,
        u,
        v,
        weight_u,
        weight_v,
        position,
        weight,
        shape,
        first,
        second,
        closest,
        remaining,
        contained,
        ray,
        ray_area,
        ray_limit,
        contact_state,
        surface_info,
        i_q >= surface_info.positions.shape[0],
    )


@qd.func
def func_contact_sample_at(
    i_face: int,
    i_triangle: int,
    i_c: int,
    i_b: int,
    u: float,
    v: float,
    weight_u: float,
    weight_v: float,
    position: qd.types.vector(3),
    weight: float,
    shape: qd.types.vector(6),
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    closest: qd.types.vector(3),
    remaining: float,
    contained: bool,
    ray: qd.types.vector(3),
    ray_area: float,
    ray_limit: float,
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
    virtual: bool,
):
    """Evaluate an original physical surface sample with explicit positive integration coordinates."""
    geometry_valid = True
    center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
    if virtual:
        position = closest
        if remaining > 0.0:
            if contained:
                # A full planar section uses a polar rule at the physical contact scale.
                radial = u
                if contact_state.hertz_profile[None]:
                    radial = qd.sqrt(1.0 - u * u)
                angle = 2.0 * qd.math.pi * v
                tangent = surface_info.face_edges[i_face][0, :].normalized()
                normal = surface_info.face_normal[i_face]
                position = closest + qd.sqrt(remaining) * radial * (
                    qd.cos(angle) * tangent + qd.sin(angle) * normal.cross(tangent)
                )
                if i_triangle == 0:
                    weight = 2.0 * qd.math.pi * remaining * u
            else:
                if ray_limit > 0.0:
                    radial, derivative = ray_limit * u, ray_limit
                    if contact_state.hertz_profile[None]:
                        radial, derivative = ray_limit * (1.0 - u * u), 2.0 * ray_limit * u
                    position = closest + radial * ray
                    weight = ray_area * radial * derivative
        weight *= weight_u * weight_v
        if surface_info.quad_vertices.shape[0] > 0:
            if weight > 0.0:
                target = surface_info.face_dual[i_face] @ (position - surface_info.face_origin[i_face])
                nodal, position, _normal, ratio, regular = func_quad_inverse_prepared(
                    surface_info.quad_coefficients[i_face],
                    surface_info.face_origin[i_face],
                    surface_info.face_dual[i_face],
                    target,
                    surface_info.quad_projection[i_face],
                    surface_info.quad_scale[i_face],
                )
                shape = qd.Vector([nodal[0], nodal[1], nodal[2], nodal[3], 0.0, 0.0])
                weight *= ratio
                geometry_valid = regular
        else:
            local = surface_info.face_dual[i_face] @ (position - surface_info.face_origin[i_face])
            bary = qd.Vector([1.0 - local.sum(), local[0], local[1]])
            shape = qd.Vector(
                [
                    bary[0] * (2.0 * bary[0] - 1.0),
                    bary[1] * (2.0 * bary[1] - 1.0),
                    bary[2] * (2.0 * bary[2] - 1.0),
                    4.0 * bary[0] * bary[1],
                    4.0 * bary[0] * bary[2],
                    4.0 * bary[1] * bary[2],
                ]
            )
    relative = (position - center) / radius
    distance = relative.dot(relative)
    if distance < 1.0:
        if contact_state.hertz_profile[None]:
            weight *= qd.sqrt(1.0 - distance)
        else:
            weight *= qd.exp(-0.5 * distance / (0.45 * 0.45)) * (1.0 - distance) ** 2
    else:
        weight = 0.0
    if not geometry_valid:
        weight = float("nan")  # A failed physical inverse must reject pressure fitting, never silently lose area.
    if not func_surface_face_admitted(i_face, i_c, i_b, surface_info):
        weight = 0.0
    coordinates = qd.Vector([1.0, relative.dot(first), relative.dot(second)])
    return coordinates, weight, position, shape, i_face


@qd.func
def func_contact_cell(
    i_cell: int,
    i_c: int,
    i_b: int,
    low: qd.types.vector(3),
    dimensions: qd.types.vector(3),
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
):
    start, end = 0, 0
    if contact_state.is_refined[i_c, i_b]:
        center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
        low_face, high_face = surface_info.face_bounds_low[i_cell], surface_info.face_bounds_high[i_cell]
        distance = qd.max(low_face - center, 0.0) + qd.max(center - high_face, 0.0)
        if distance.norm_sqr() < radius * radius and func_surface_face_admitted(i_cell, i_c, i_b, surface_info):
            _closest, remaining, contained = func_contact_section(i_cell, center, radius, surface_info)
            if remaining > 0.0:
                order = func_contact_order(i_c, i_b, contact_state, surface_info)
                count = order * order
                start = surface_info.positions.shape[0] + i_cell * (
                    qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * count
                )
                # Complete disks assign all weight to fan zero; the other fans contribute exactly zero.
                end = start + qd.select(
                    contained, count, qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * count
                )
    elif i_cell < dimensions[0] * dimensions[1] * dimensions[2]:
        start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
    return start, end


@qd.func
def func_contact_index(i_entry: int, surface_info: StressSurfaceInfo):
    i_q = i_entry
    if i_entry < surface_info.positions.shape[0]:
        i_q = surface_info.grid.samples[i_entry]
    return i_q


def kernel_anchor(
    source_epsilon: float,
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
    cooperative: bool = False,
):
    if cooperative and gs.backend == gs.cuda and contact_state.active_pairs.shape[0] > 0:
        kernel_anchor_pack(source_epsilon, contact_state)
        kernel_anchor_warp(contact_state, surface_info)
    else:
        kernel_anchor_serial(source_epsilon, contact_state, surface_info)


@qd.kernel
def kernel_anchor_serial(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
    func_anchor(source_epsilon, contact_state, surface_info)


@qd.func
def func_contact_force_admissible(
    point: qd.types.vector(3),
    force: qd.types.vector(3),
    normal: qd.types.vector(3),
    friction: float,
    source_epsilon: float,
):
    magnitude = force.norm()
    admitted = normal.norm() > 0.0 and friction >= 0.0
    admitted = admitted and not qd.math.isnan(magnitude) and not qd.math.isinf(magnitude)
    admitted = admitted and not qd.math.isinf(normal.norm()) and not qd.math.isinf(friction)
    for i_a in qd.static(range(3)):
        admitted = admitted and not qd.math.isnan(point[i_a]) and not qd.math.isinf(point[i_a])
    if admitted:
        normal = normal.normalized()
        compression = force.dot(normal)
        tangent = force - compression * normal
        allowance = 16.0 * source_epsilon * magnitude
        admitted = compression >= -allowance and tangent.norm() <= friction * compression + allowance
    return admitted


@qd.func
def func_anchor_source(i_c: int, i_b: int, source_epsilon: float, contact_state: StressContactState):
    contact_state.status[i_c, i_b] = 0
    contact_state.force_error[i_c, i_b] = 0.0
    contact_state.moment_error[i_c, i_b] = 0.0
    contact_state.candidate_count[i_c, i_b] = 0
    contact_state.evaluations[i_c, i_b] = 0
    contact_state.is_refined[i_c, i_b] = False
    loaded = False
    if contact_state.valid[i_c, i_b]:
        force = contact_state.force[i_c, i_b]
        magnitude = force.norm()
        normal = contact_state.normal[i_c, i_b]
        radius = contact_state.radius[i_c, i_b]
        admitted = func_contact_force_admissible(
            contact_state.position[i_c, i_b], force, normal, contact_state.friction[i_c, i_b], source_epsilon
        )
        admitted = admitted and radius > 0.0 and not qd.math.isinf(radius)
        if not admitted:
            contact_state.status[i_c, i_b] = 1
        elif magnitude > 1e-30:
            loaded = True
    return loaded


@qd.func
def func_anchor_frame(force: qd.types.vector(3)):
    magnitude = force.norm()
    direction = -force / magnitude
    line_first, line_second = func_force_frame(direction)
    line_basis = qd.Matrix.rows([line_first, line_second])
    inverse_direction = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(3)):
        if qd.abs(direction[a]) > 0.0:
            inverse_direction[a] = 1.0 / direction[a]
    return direction, line_basis, inverse_direction


@qd.func
def func_anchor_face(
    i_f: int,
    point: qd.types.vector(3),
    direction: qd.types.vector(3),
    line_basis: qd.types.matrix(2, 3),
    inverse_direction: qd.types.vector(3),
    distance: float,
    surface_info: StressSurfaceInfo,
):
    parameter = gs.qd_float(-1e30)
    found = False
    if surface_info.quad_vertices.shape[0] > 0:
        # The convex hull of Q4 corners bounds the complete bilinear physical surface.
        # Test the unchanged infinite force line before inverting a face; padding covers the
        # existing natural-coordinate roundoff allowance, without moving the physical anchor.
        low, high = surface_info.face_bounds_low[i_f], surface_info.face_bounds_high[i_f]
        padding = 4e-12 * (high - low).norm()
        near, far = gs.qd_float(-float("inf")), gs.qd_float(float("inf"))
        intersects = True
        for a in qd.static(range(3)):
            if qd.abs(direction[a]) > 0.0:
                first_hit = (low[a] - padding - point[a]) * inverse_direction[a]
                last_hit = (high[a] + padding - point[a]) * inverse_direction[a]
                near, far = (
                    qd.max(near, qd.min(first_hit, last_hit)),
                    qd.min(far, qd.max(first_hit, last_hit)),
                )
            else:
                intersects = intersects and point[a] >= low[a] - padding and point[a] <= high[a] + padding
        lower_distance = qd.max(near, 0.0) - qd.min(far, 0.0)
        if intersects and near <= far and lower_distance <= distance + padding:
            _shape, hit, outward, _ratio, regular = func_quad_inverse(
                surface_info.quad_coefficients[i_f], point, line_basis, qd.Vector([0.0, 0.0])
            )
            parameter = (hit - point).dot(direction)
            found = regular and outward.dot(direction) > 1e-12 and qd.abs(parameter) < distance
    else:
        denominator = surface_info.face_normal[i_f].dot(direction)
        if denominator > 1e-12:
            parameter = surface_info.face_normal[i_f].dot(surface_info.face_origin[i_f] - point) / denominator
            bary = surface_info.face_dual[i_f] @ (point + parameter * direction - surface_info.face_origin[i_f])
            # The application point identifies the nearest material sheet along the force line.
            found = (
                bary[0] >= -2e-12 and bary[1] >= -2e-12 and bary.sum() <= 1.0 + 2e-12 and qd.abs(parameter) < distance
            )
    return parameter, found


@qd.func
def func_anchor_result(
    i_c: int,
    i_b: int,
    i_f: int,
    point: qd.types.vector(3),
    direction: qd.types.vector(3),
    best: float,
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
):
    if best > -1e30:
        contact_state.anchor_face[i_c, i_b] = i_f
        contact_state.center[i_c, i_b] = point + best * direction
        edges = surface_info.face_edges[i_f]
        diameter = qd.max(edges[0, :].norm(), edges[1, :].norm(), (edges[1, :] - edges[0, :]).norm())
        # A solvable moment fit does not certify integration accuracy when the support is smaller
        # than the face rule's spacing. Integrate these patches at their physical scale immediately.
        contact_state.is_refined[i_c, i_b] = (
            surface_info.quad_vertices.shape[0] > 0
            or surface_info.angular_quadrature[None]
            or contact_state.radius[i_c, i_b] * surface_info.gauss.shape[0] < diameter
        )
    else:
        contact_state.status[i_c, i_b] = 2


@qd.func
def func_anchor(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if func_anchor_source(i_c, i_b, source_epsilon, contact_state):
            point = contact_state.position[i_c, i_b]
            direction, line_basis, inverse_direction = func_anchor_frame(contact_state.force[i_c, i_b])
            best, distance = gs.qd_float(-1e30), gs.qd_float(1e30)
            best_face = surface_info.face_origin.shape[0]
            for i_f in range(surface_info.face_origin.shape[0]):
                parameter, found = func_anchor_face(
                    i_f, point, direction, line_basis, inverse_direction, distance, surface_info
                )
                if found:
                    best_face, best, distance = i_f, parameter, qd.abs(parameter)
            func_anchor_result(i_c, i_b, best_face, point, direction, best, contact_state, surface_info)


@qd.kernel(graph=True)
def kernel_anchor_pack(source_epsilon: float, contact_state: StressContactState):
    for _ in range(1):
        contact_state.active_count[None] = 0
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        # Source validity is recomputed before packing; a previous fit failure must not suppress a new source.
        if func_anchor_source(i_c, i_b, source_epsilon, contact_state):
            slot = qd.atomic_add(contact_state.active_count[None], 1)
            contact_state.active_pairs[slot] = qd.Vector([i_c, i_b])


@qd.kernel
def kernel_anchor_warp(contact_state: StressContactState, surface_info: StressSurfaceInfo):
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.active_count[None] * 32):
        lane, slot = i_thread % 32, i_thread // 32
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        point = contact_state.position[i_c, i_b]
        direction, line_basis, inverse_direction = func_anchor_frame(contact_state.force[i_c, i_b])
        best, distance = gs.qd_float(-1e30), gs.qd_float(1e30)
        n_faces = surface_info.face_origin.shape[0]
        best_face = n_faces
        for i_chunk in range((n_faces + 31) // 32):
            i_f = i_chunk * 32 + lane
            candidate, candidate_distance = gs.qd_float(-1e30), gs.qd_float(1e30)
            found = False
            if i_f < n_faces:
                candidate, found = func_anchor_face(
                    i_f, point, direction, line_basis, inverse_direction, distance, surface_info
                )
                if found:
                    candidate_distance = qd.abs(candidate)
            # All lanes share each chunk's best distance before applying the next chunk's slab pruning.
            minimum = qd_block_min(candidate_distance)
            candidate_face = qd.select(found and candidate_distance == minimum, i_f, n_faces)
            winner = qd_block_min(candidate_face)
            if winner < n_faces:
                # Strict equal-distance ties retain the first face in the original ascending traversal.
                best = qd.simt.subgroup.broadcast(candidate, qd.u32(winner % 32))
                best_face, distance = winner, minimum
        if lane == 0:
            func_anchor_result(i_c, i_b, best_face, point, direction, best, contact_state, surface_info)


@qd.func
def func_pressure_initial(gram: qd.types.matrix(3, 3)):
    weight_sum = gs.qd_float(0.0)
    coefficient = qd.Vector.zero(gs.qd_float, 3)
    accepted = False
    nonsingular = gram[0, 0] > 0.0 and gram.determinant() > 1e-14 * gram.trace() ** 3
    if nonsingular:
        weight_sum = gram[0, 0]
        gram /= weight_sum
        coefficient = gram.inverse() @ qd.Vector([1.0, 0.0, 0.0])
        margin = 1e-12
        if qd.static(gs.qd_float == qd.f32):
            margin = 1e-5
        positive = coefficient[0] - qd.sqrt(coefficient[1] ** 2 + coefficient[2] ** 2)
        accepted = (
            positive > margin * coefficient.norm()
            and (gram @ coefficient - qd.Vector([1.0, 0.0, 0.0])).norm() <= _PRESSURE_FIT_TOLERANCE
        )
        coefficient /= weight_sum
    status = 0 if accepted else (5 if nonsingular else 3)
    return coefficient, weight_sum, status


@qd.func
def func_pressure_step(hessian: qd.types.matrix(3, 3), gradient: qd.types.vector(3)):
    # Normalize dual coordinates before testing/inverting a small active footprint.
    # This leaves the pressure objective, target wrench and convergence budget unchanged.
    scale = qd.Vector.zero(gs.qd_float, 3)
    positive = True
    for a in qd.static(range(3)):
        positive = positive and hessian[a, a] > 0.0
        if hessian[a, a] > 0.0:
            scale[a] = 1.0 / qd.sqrt(hessian[a, a])
    balanced = qd.Matrix.zero(gs.qd_float, 3, 3)
    for a, b in qd.static(qd.ndrange(3, 3)):
        balanced[a, b] = scale[a] * hessian[a, b] * scale[b]
    nonsingular = positive and balanced.determinant() > 1e-14 * balanced.trace() ** 3
    step = qd.Vector.zero(gs.qd_float, 3)
    if nonsingular:
        step = scale * (balanced.inverse() @ (scale * gradient))
    return step, nonsingular


def kernel_pressure(contact_state: StressContactState, surface_info: StressSurfaceInfo, cooperative: bool = True):
    if cooperative and gs.backend == gs.cuda:
        kernel_pack_contacts(contact_state)
        kernel_pressure_warp(contact_state, surface_info, False)
        kernel_pressure_correct_warp(contact_state, surface_info)
        kernel_refine_contacts(contact_state)
        kernel_pressure_warp(contact_state, surface_info, True)
        kernel_pressure_correct_warp(contact_state, surface_info)
    else:
        kernel_pressure_serial(contact_state, surface_info, False)
        kernel_pressure_correct(contact_state, surface_info)
        kernel_refine_contacts(contact_state)
        kernel_pressure_serial(contact_state, surface_info, True)
        kernel_pressure_correct(contact_state, surface_info)


@qd.kernel
def kernel_refine_contacts(contact_state: StressContactState):
    func_refine_contacts(contact_state)


@qd.func
def func_refine_contacts(contact_state: StressContactState):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and contact_state.status[i_c, i_b] == 3
            and not contact_state.is_refined[i_c, i_b]
        ):
            contact_state.is_refined[i_c, i_b] = True
            contact_state.status[i_c, i_b] = 0
            contact_state.candidate_count[i_c, i_b] = 0


@qd.kernel(graph=True)
def kernel_pack_contacts(contact_state: StressContactState):
    func_pack_contacts(contact_state)


@qd.func
def func_pack_contacts(contact_state: StressContactState):
    for _ in range(1):
        contact_state.active_count[None] = 0
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and contact_state.status[i_c, i_b] == 0
            and contact_state.force[i_c, i_b].norm() > 1e-30
        ):
            slot = qd.atomic_add(contact_state.active_count[None], 1)
            contact_state.active_pairs[slot] = qd.Vector([i_c, i_b])


@qd.kernel
def kernel_pressure_warp(contact_state: StressContactState, surface_info: StressSurfaceInfo, refinement: bool):
    func_pressure_warp(contact_state, surface_info, refinement, contact_state.valid, False)


@qd.func
def func_pressure_warp(
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
    refinement: bool,
    processed_refined: qd.Tensor,
    reuse_refined: qd.template(),
    enabled: bool = True,
):
    qd.loop_config(block_dim=128)
    for i_thread in range(qd.select(enabled, contact_state.active_count[None] * 32, 0)):
        lane = i_thread % 32
        slot = i_thread // 32
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        processed = False
        if qd.static(reuse_refined):  # noqa: SIM102 - specialize before testing the runtime index
            if slot < processed_refined.shape[0]:
                processed = processed_refined[slot]
        if refinement == contact_state.is_refined[i_c, i_b] and not processed:
            first, second = func_force_frame(contact_state.force[i_c, i_b])
            center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            low, dimensions = func_patch_grid(center, radius, surface_info.grid)
            gram = qd.Matrix.zero(gs.qd_float, 3, 3)
            candidates = 0
            n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
            if contact_state.is_refined[i_c, i_b]:
                n_cells = surface_info.face_origin.shape[0]
            for i_cell in range(n_cells):
                start, end = func_contact_cell(i_cell, i_c, i_b, low, dimensions, contact_state, surface_info)
                for chunk in range((end - start + 31) // 32):
                    entry = start + chunk * 32 + lane
                    if entry < end:
                        i_q = func_contact_index(entry, surface_info)
                        coordinates, weight, _position, _shape, _face = func_contact_sample(
                            i_q, i_c, i_b, first, second, contact_state, surface_info
                        )
                        gram += weight * coordinates.outer_product(coordinates)
                        candidates += gs.qd_int(weight > 0.0)
            for i, j in qd.static(qd.ndrange(3, 3)):
                gram[i, j] = qd.simt.subgroup.reduce_all_add(gram[i, j])
            candidates = qd.simt.subgroup.reduce_all_add(candidates)
            if lane == 0:
                contact_state.candidate_count[i_c, i_b] += candidates
                coefficient, weight_sum, status = func_pressure_initial(gram)
                contact_state.coefficient[i_c, i_b] = coefficient
                contact_state.weight_sum[i_c, i_b] = weight_sum
                contact_state.status[i_c, i_b] = status


@qd.kernel
def kernel_pressure_correct_warp(contact_state: StressContactState, surface_info: StressSurfaceInfo):
    func_pressure_correct_warp(contact_state, surface_info)


@qd.func
def func_pressure_correct_warp(contact_state: StressContactState, surface_info: StressSurfaceInfo):
    n_warps = qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 1)
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.active_count[None] * 32 * n_warps):
        lane = i_thread % 32
        i_warp = (i_thread // 32) % n_warps
        slot = i_thread // (32 * n_warps)
        sh_fit = qd.simt.block.SharedArray((4, 13), gs.qd_float)
        sh_faces = qd.simt.block.SharedArray((128,), gs.qd_int)
        sh_count = qd.simt.block.SharedArray((1,), gs.qd_int)
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contact_state.status[i_c, i_b] == 5:
            first, second = func_force_frame(contact_state.force[i_c, i_b])
            center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            low, dimensions = func_patch_grid(center, radius, surface_info.grid)
            is_cached = n_warps > 1 and contact_state.is_refined[i_c, i_b]
            n_faces = surface_info.face_origin.shape[0]
            if is_cached:
                if lane == 0 and i_warp == 0:
                    sh_count[0] = 0
                qd.simt.block.sync()
                for i_chunk in range((n_faces + 127) // 128):
                    i_f = i_chunk * 128 + i_warp * 32 + lane
                    if i_f < n_faces:
                        distance = qd.max(surface_info.face_bounds_low[i_f] - center, 0.0)
                        distance += qd.max(center - surface_info.face_bounds_high[i_f], 0.0)
                        if distance.norm_sqr() < radius * radius and func_surface_face_admitted(
                            i_f, i_c, i_b, surface_info
                        ):
                            i_cached = qd.atomic_add(sh_count[0], 1)
                            if i_cached < 128:
                                sh_faces[i_cached] = i_f
                qd.simt.block.sync()
                # Oversized footprints retain the exhaustive face walk and all original samples.
                is_cached = sh_count[0] <= 128
            weight_sum = contact_state.weight_sum[i_c, i_b]
            coefficient = contact_state.coefficient[i_c, i_b] * weight_sum
            accepted, line = False, False
            iterations, line_evaluations = 0, 0
            fraction, reference_objective = gs.qd_float(1.0), gs.qd_float(0.0)
            step = qd.Vector.zero(gs.qd_float, 3)
            slope = gs.qd_float(0.0)
            # At most 80 Newton evaluations, each followed by at most 40 line evaluations.
            # One evaluator body avoids duplicating the integration helper in the kernel.
            for _ in range(80 * 41):
                if accepted or (not line and iterations >= 80):
                    break
                candidate = coefficient
                if line:
                    candidate = coefficient - fraction * step
                objective = gs.qd_float(0.0)
                gradient = qd.Vector.zero(gs.qd_float, 3)
                hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
                if contact_state.is_refined[i_c, i_b]:
                    n_cells = n_faces
                    if is_cached:
                        n_cells = sh_count[0]
                for i_group in range((n_cells + n_warps - 1) // n_warps):
                    i_cell = i_group * n_warps + i_warp
                    start, end = 0, 0
                    if i_cell < n_cells:
                        if is_cached:
                            i_cell = sh_faces[i_cell]
                        start, end = func_contact_cell(i_cell, i_c, i_b, low, dimensions, contact_state, surface_info)
                    closest = qd.Vector.zero(gs.qd_float, 3)
                    remaining, contained = gs.qd_float(0.0), False
                    if contact_state.is_refined[i_c, i_b] and end > start:
                        closest, remaining, contained = func_contact_section(i_cell, center, radius, surface_info)
                    n = func_contact_order(i_c, i_b, contact_state, surface_info)
                    n_quadrature = n * n
                    is_clipped = contact_state.is_refined[i_c, i_b] and not contained and end > start
                    n_rays = qd.select(
                        is_clipped, qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n, end - start
                    )
                    for i_chunk in range((n_rays + 31) // 32):
                        i_ray = i_chunk * 32 + lane
                        if i_ray < n_rays:
                            ray = qd.Vector.zero(gs.qd_float, 3)
                            area, limit = gs.qd_float(0.0), gs.qd_float(0.0)
                            n_radial = 1
                            if is_clipped:
                                ray, area, limit = func_contact_ray_selected(
                                    i_cell,
                                    i_ray // n,
                                    i_ray % n,
                                    i_c,
                                    i_b,
                                    closest,
                                    closest - center,
                                    remaining,
                                    contact_state,
                                    surface_info,
                                )
                                n_radial = qd.select(area > 0.0 and limit > 0.0, n, 0)
                            for i_radial in range(n_radial):
                                i_sample = qd.select(
                                    is_clipped, (i_ray // n) * n_quadrature + i_radial * n + i_ray % n, i_ray
                                )
                                i_q = func_contact_index(start + i_sample, surface_info)
                                coordinates, weight, _position, _shape, _face = func_contact_sample_prepared(
                                    i_q,
                                    i_c,
                                    i_b,
                                    first,
                                    second,
                                    contact_state,
                                    surface_info,
                                    closest,
                                    remaining,
                                    contained,
                                    ray,
                                    area,
                                    limit,
                                )
                                weight /= weight_sum
                                profile = qd.max(0.0, coordinates.dot(candidate))
                                objective -= 0.5 * weight * profile * profile
                                if not line:
                                    gradient += weight * profile * coordinates
                                    if profile > 0.0:
                                        hessian += weight * coordinates.outer_product(coordinates)
                objective = qd.simt.subgroup.reduce_all_add(objective)
                if not line:
                    for a in qd.static(range(3)):
                        gradient[a] = qd.simt.subgroup.reduce_all_add(gradient[a])
                    for a, b in qd.static(qd.ndrange(3, 3)):
                        hessian[a, b] = qd.simt.subgroup.reduce_all_add(hessian[a, b])
                if n_warps > 1:
                    if lane == 0:
                        sh_fit[i_warp, 0] = objective
                        for a in qd.static(range(3)):
                            sh_fit[i_warp, 1 + a] = gradient[a]
                        for a, b in qd.static(qd.ndrange(3, 3)):
                            sh_fit[i_warp, 4 + 3 * a + b] = hessian[a, b]
                    qd.simt.block.sync()
                    objective = 0.0
                    gradient = qd.Vector.zero(gs.qd_float, 3)
                    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for j_warp in qd.static(range(4)):
                        objective += sh_fit[j_warp, 0]
                        for a in qd.static(range(3)):
                            gradient[a] += sh_fit[j_warp, 1 + a]
                        for a, b in qd.static(qd.ndrange(3, 3)):
                            hessian[a, b] += sh_fit[j_warp, 4 + 3 * a + b]
                    qd.simt.block.sync()
                objective += candidate[0]
                if not line:
                    gradient[0] -= 1.0
                    iterations += 1
                    accepted = gradient.norm() <= _PRESSURE_FIT_TOLERANCE
                    proposed_step, nonsingular = func_pressure_step(hessian, gradient)
                    if not accepted and nonsingular:
                        step = proposed_step
                        slope = gradient.dot(step)
                        reference_objective = objective
                        fraction, line = gs.qd_float(1.0), True
                        line_evaluations = 0
                else:
                    line_evaluations += 1
                    if objective >= reference_objective + 1e-4 * fraction * slope - 1e-15:
                        coefficient, line = candidate, False
                    else:
                        fraction *= 0.5
                        if line_evaluations == 40:
                            line = False
            if lane == 0 and i_warp == 0:
                contact_state.coefficient[i_c, i_b] = coefficient / weight_sum
                contact_state.status[i_c, i_b] = 0 if accepted else 3
                contact_state.evaluations[i_c, i_b] += iterations


@qd.kernel(graph=True)
def kernel_pressure_serial(contact_state: StressContactState, surface_info: StressSurfaceInfo, refinement: bool):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and contact_state.status[i_c, i_b] == 0
            and refinement == contact_state.is_refined[i_c, i_b]
        ):
            force = contact_state.force[i_c, i_b]
            if force.norm() > 1e-30:
                first, second = func_force_frame(force)
                center = contact_state.center[i_c, i_b]
                radius = contact_state.radius[i_c, i_b]
                low, dimensions = func_patch_grid(center, radius, surface_info.grid)
                n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
                if contact_state.is_refined[i_c, i_b]:
                    n_cells = surface_info.face_origin.shape[0]
                gram = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i_cell in range(n_cells):
                    start, end = func_contact_cell(i_cell, i_c, i_b, low, dimensions, contact_state, surface_info)
                    for i_entry in range(start, end):
                        i_q = func_contact_index(i_entry, surface_info)
                        coordinates, weight, _position, _shape, _face = func_contact_sample(
                            i_q, i_c, i_b, first, second, contact_state, surface_info
                        )
                        gram += weight * coordinates.outer_product(coordinates)
                        if weight > 0.0:
                            contact_state.candidate_count[i_c, i_b] += 1
                coefficient, weight_sum, status = func_pressure_initial(gram)
                contact_state.coefficient[i_c, i_b] = coefficient
                contact_state.weight_sum[i_c, i_b] = weight_sum
                contact_state.status[i_c, i_b] = status


@qd.kernel(graph=True)
def kernel_pressure_correct(contact_state: StressContactState, surface_info: StressSurfaceInfo):
    # Keep the uncommon constrained solve out of the integration kernel's register footprint.
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and contact_state.status[i_c, i_b] == 5:
            first, second = func_force_frame(contact_state.force[i_c, i_b])
            center = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            low, dimensions = func_patch_grid(center, radius, surface_info.grid)
            n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
            if contact_state.is_refined[i_c, i_b]:
                n_cells = surface_info.face_origin.shape[0]
            weight_sum = contact_state.weight_sum[i_c, i_b]
            coefficient = contact_state.coefficient[i_c, i_b] * weight_sum
            accepted = False
            previous_evaluations = contact_state.evaluations[i_c, i_b]
            for iteration in range(80):
                if accepted:
                    break
                gradient = qd.Vector([-1.0, 0.0, 0.0])
                hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i_cell in range(n_cells):
                    start, end = func_contact_cell(i_cell, i_c, i_b, low, dimensions, contact_state, surface_info)
                    for i_entry in range(start, end):
                        i_q = func_contact_index(i_entry, surface_info)
                        coordinates, weight, _position, _shape, _face = func_contact_sample(
                            i_q, i_c, i_b, first, second, contact_state, surface_info
                        )
                        weight /= weight_sum
                        profile = qd.max(0.0, coordinates.dot(coefficient))
                        gradient += weight * profile * coordinates
                        if profile > 0.0:
                            hessian += weight * coordinates.outer_product(coordinates)
                contact_state.evaluations[i_c, i_b] = previous_evaluations + iteration + 1
                accepted = gradient.norm() <= _PRESSURE_FIT_TOLERANCE
                step, nonsingular = func_pressure_step(hessian, gradient)
                if not accepted and nonsingular:
                    fraction = gs.qd_float(1.0)
                    line_accepted = False
                    for _ in range(40):
                        if line_accepted:
                            break
                        candidate = coefficient - fraction * step
                        difference = candidate - coefficient
                        objective_change = difference[0]
                        for i_cell in range(n_cells):
                            start, end = func_contact_cell(
                                i_cell, i_c, i_b, low, dimensions, contact_state, surface_info
                            )
                            for i_entry in range(start, end):
                                i_q = func_contact_index(i_entry, surface_info)
                                coordinates, weight, _position, _shape, _face = func_contact_sample(
                                    i_q, i_c, i_b, first, second, contact_state, surface_info
                                )
                                weight /= weight_sum
                                profile = qd.max(0.0, coordinates.dot(candidate))
                                reference_profile = qd.max(0.0, coordinates.dot(coefficient))
                                profile_change = profile - reference_profile
                                if profile > 0.0 and reference_profile > 0.0:
                                    profile_change = difference.dot(coordinates)
                                # The energy change retains small ascent steps lost when comparing full objectives.
                                objective_change -= 0.5 * weight * profile_change * (profile + reference_profile)
                        if objective_change >= 1e-4 * fraction * gradient.dot(step) - 1e-15:
                            coefficient = candidate
                            line_accepted = True
                        else:
                            fraction *= 0.5
            contact_state.coefficient[i_c, i_b] = coefficient / weight_sum
            contact_state.status[i_c, i_b] = 0 if accepted else 3


@qd.kernel(graph=True)
def kernel_scatter_serial(
    contact_state: StressContactState,
    stress_state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
):
    for i_n, i_b in qd.ndrange(stress_state.force.shape[0], stress_state.active.shape[0]):
        stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and contact_state.status[i_c, i_b] == 0:
            force = contact_state.force[i_c, i_b]
            magnitude = force.norm()
            if magnitude > 1e-30:
                first, second = func_force_frame(force)
                low, dimensions = func_patch_grid(
                    contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b], surface_info.grid
                )
                n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
                if contact_state.is_refined[i_c, i_b]:
                    n_cells = surface_info.face_origin.shape[0]
                total = qd.Vector.zero(gs.qd_float, 3)
                moment = qd.Vector.zero(gs.qd_float, 3)
                for i_cell in range(n_cells):
                    start, end = func_contact_cell(i_cell, i_c, i_b, low, dimensions, contact_state, surface_info)
                    for i_entry in range(start, end):
                        i_q = func_contact_index(i_entry, surface_info)
                        coordinates, weight, position, shape, i_face = func_contact_sample(
                            i_q, i_c, i_b, first, second, contact_state, surface_info
                        )
                        fraction = weight * qd.max(0.0, coordinates.dot(contact_state.coefficient[i_c, i_b]))
                        sample_force = fraction * force
                        total += sample_force
                        moment += (position - contact_state.position[i_c, i_b]).cross(sample_force)
                        if weight > 0.0:
                            for i_local in range(qd.select(stress_info.load_nodes.shape[0] > 0, 4, 6)):
                                i_n = stress_info.surface_nodes[i_face, i_local]
                                value = shape[i_local] * sample_force
                                func_scatter_load(i_n, i_b, value, stress_state, stress_info)
                contact_state.force_error[i_c, i_b] = (total - force).norm()
                contact_state.moment_error[i_c, i_b] = moment.norm()
                if (total - force).norm() > 1e-8 * magnitude or moment.norm() > 1e-8 * magnitude * contact_state.radius[
                    i_c, i_b
                ]:
                    contact_state.status[i_c, i_b] = 4


def kernel_scatter(
    contact_state: StressContactState,
    stress_state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    cooperative: bool = True,
    cached_bounds: bool = True,
):
    if cooperative and gs.backend == gs.cuda:
        kernel_pack_contacts(contact_state)
        kernel_scatter_warp(contact_state, stress_state, stress_info, surface_info, cached_bounds)
    else:
        kernel_scatter_serial(contact_state, stress_state, stress_info, surface_info)


@qd.kernel(graph=True)
def kernel_scatter_warp(
    contact_state: StressContactState,
    stress_state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    cached_bounds: qd.template(),
):
    func_scatter_warp(contact_state, stress_state, stress_info, surface_info, cached_bounds)


@qd.func
def func_scatter_warp(
    contact_state: StressContactState,
    stress_state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    cached_bounds: qd.template(),
    enabled: bool = True,
):
    for i_n, i_b in qd.ndrange(qd.select(enabled, stress_state.force.shape[0], 0), stress_state.active.shape[0]):
        stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    qd.loop_config(block_dim=128)
    for i_thread in range(qd.select(enabled, contact_state.active_count[None] * 32, 0)):
        lane, slot = i_thread % 32, i_thread // 32
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contact_state.status[i_c, i_b] == 0:
            force = contact_state.force[i_c, i_b]
            first, second = func_force_frame(force)
            center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            total = qd.Vector.zero(gs.qd_float, 3)
            moment = qd.Vector.zero(gs.qd_float, 3)
            for i_f in range(surface_info.face_origin.shape[0]):
                low, high = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                if qd.static(cached_bounds):
                    low, high = surface_info.face_bounds_low[i_f], surface_info.face_bounds_high[i_f]
                else:
                    x = surface_info.face_origin[i_f]
                    a = x + surface_info.face_edges[i_f][0, :]
                    b = x + surface_info.face_edges[i_f][1, :]
                    low, high = qd.min(x, a, b), qd.max(x, a, b)
                    if surface_info.quad_vertices.shape[0] > 0:
                        low, high = surface_info.face_bounds_low[i_f], surface_info.face_bounds_high[i_f]
                distance = qd.max(low - center, 0.0) + qd.max(center - high, 0.0)
                if distance.dot(distance) < radius * radius and func_surface_face_admitted(i_f, i_c, i_b, surface_info):
                    order = func_contact_order(i_c, i_b, contact_state, surface_info)
                    n_q = order * order
                    start, count = i_f * n_q, n_q
                    if contact_state.is_refined[i_c, i_b]:
                        start, count = (
                            surface_info.positions.shape[0]
                            + i_f * qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n_q,
                            qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n_q,
                        )
                    load = qd.Matrix.zero(gs.qd_float, 6, 3)
                    for chunk in range((count + 31) // 32):
                        i_sample = chunk * 32 + lane
                        if i_sample < count:
                            coordinates, weight, position, shape, _face = func_contact_sample(
                                start + i_sample, i_c, i_b, first, second, contact_state, surface_info
                            )
                            sample_force = (
                                weight * qd.max(0.0, coordinates.dot(contact_state.coefficient[i_c, i_b])) * force
                            )
                            total += sample_force
                            moment += (position - contact_state.position[i_c, i_b]).cross(sample_force)
                            load += shape.outer_product(sample_force)
                    for a, b in qd.static(qd.ndrange(6, 3)):
                        load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
                    if lane == 0:
                        for i_local in range(qd.select(stress_info.load_nodes.shape[0] > 0, 4, 6)):
                            i_n = stress_info.surface_nodes[i_f, i_local]
                            func_scatter_load(i_n, i_b, load[i_local, :], stress_state, stress_info)
            for a in qd.static(range(3)):
                total[a] = qd.simt.subgroup.reduce_all_add(total[a])
                moment[a] = qd.simt.subgroup.reduce_all_add(moment[a])
            if lane == 0:
                contact_state.force_error[i_c, i_b] = (total - force).norm()
                contact_state.moment_error[i_c, i_b] = moment.norm()
                if (total - force).norm() > 1e-8 * force.norm() or moment.norm() > 1e-8 * force.norm() * radius:
                    contact_state.status[i_c, i_b] = 4
