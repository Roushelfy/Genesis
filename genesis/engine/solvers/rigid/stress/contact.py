"""Native finite, nonnegative pressure preserving each point-contact wrench."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V, of_kind

from .data import StressInfo, StressState
from .grid import func_patch_cell, func_patch_grid
from .surface import StressSurfaceInfo, func_closest_triangle

# The normalized dual gradient bounds relative resultant force and moment/radius.
# Stop at 10% of the independently checked 1e-8 contact wrench budget; the final
# scatter acceptance and complete elastic residual/stress checks remain mandatory.
_PRESSURE_FIT_TOLERANCE = 1e-9


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
    hertz_profile: qd.Tensor = of_kind(DataKind.CONSTANT)
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


def create_contacts(
    n_contacts: int,
    n_envs: int,
    radius: float,
    cooperative: bool = True,
    cooperative_scatter: bool = True,
    hertz_profile: bool = False,
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
    )
    state.radius.fill(radius)
    state.radius_status.fill(0)
    state.hertz_profile.fill(hertz_profile)
    state.valid.fill(False)
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
def func_contact_section(
    i_face: int,
    center: qd.types.vector(3),
    radius: float,
    surface_info: StressSurfaceInfo,
):
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
def func_contact_sample(
    i_q: int,
    i_c: int,
    i_b: int,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
):
    center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
    n_quadrature = surface_info.shape.shape[0]
    position = qd.Vector.zero(gs.qd_float, 3)
    weight = gs.qd_float(0.0)
    shape = qd.Vector.zero(gs.qd_float, 6)
    i_face = i_q // n_quadrature
    if i_q < surface_info.positions.shape[0]:
        position = surface_info.positions[i_q]
        weight = surface_info.weights[i_q]
        shape = surface_info.shape[i_q % n_quadrature]
        if contact_state.is_refined[i_c, i_b]:
            weight = 0.0
    else:
        i_virtual = i_q - surface_info.positions.shape[0]
        i_face = i_virtual // (3 * n_quadrature)
        i_triangle = (i_virtual // n_quadrature) % 3
        i_sample = i_virtual % n_quadrature
        n = surface_info.gauss.shape[0]
        u, v = surface_info.gauss[i_sample // n], surface_info.gauss[i_sample % n]
        closest, remaining, contained = func_contact_section(i_face, center, radius, surface_info)
        offset = closest - center
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
                # Positive fans partition each intersecting face. Clip each ray to the unchanged spherical support;
                # this resolves disks crossing elastic edges/vertices without increasing their physical radius.
                vertices = qd.Matrix.rows(
                    [
                        surface_info.face_origin[i_face],
                        surface_info.face_origin[i_face] + surface_info.face_edges[i_face][0, :],
                        surface_info.face_origin[i_face] + surface_info.face_edges[i_face][1, :],
                    ]
                )
                edge_a = vertices[i_triangle, :] - closest
                edge_b = vertices[(i_triangle + 1) % 3, :] - closest
                direction = (1.0 - v) * edge_a + v * edge_b
                squared = direction.norm_sqr()
                area = edge_a.cross(edge_b).norm()
                if squared > 0.0 and area > 0.0:
                    projection = offset.dot(direction)
                    discriminant = qd.sqrt(projection * projection + squared * remaining)
                    limit = (discriminant - projection) / squared
                    if projection >= 0.0:
                        limit = remaining / (discriminant + projection)
                    limit = qd.min(1.0, limit)
                    radial, derivative = limit * u, limit
                    if contact_state.hertz_profile[None]:
                        radial, derivative = limit * (1.0 - u * u), 2.0 * limit * u
                    position = closest + radial * direction
                    weight = area * radial * derivative
            weight *= surface_info.gauss_weights[i_sample // n] * surface_info.gauss_weights[i_sample % n]
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
        if distance.norm_sqr() < radius * radius:
            _closest, remaining, contained = func_contact_section(i_cell, center, radius, surface_info)
            if remaining > 0.0:
                count = surface_info.shape.shape[0]
                start = surface_info.positions.shape[0] + i_cell * (3 * count)
                # Complete disks assign all weight to fan zero; the other two fans contribute exactly zero.
                end = start + qd.select(contained, count, 3 * count)
    elif i_cell < dimensions[0] * dimensions[1] * dimensions[2]:
        start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
    return start, end


@qd.func
def func_contact_index(i_entry: int, surface_info: StressSurfaceInfo):
    i_q = i_entry
    if i_entry < surface_info.positions.shape[0]:
        i_q = surface_info.grid.samples[i_entry]
    return i_q


@qd.kernel
def kernel_anchor(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
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
def func_anchor(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        contact_state.status[i_c, i_b] = 0
        contact_state.force_error[i_c, i_b] = 0.0
        contact_state.moment_error[i_c, i_b] = 0.0
        contact_state.candidate_count[i_c, i_b] = 0
        contact_state.evaluations[i_c, i_b] = 0
        contact_state.is_refined[i_c, i_b] = False
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
                direction = -force / magnitude
                point = contact_state.position[i_c, i_b]
                best = gs.qd_float(-1e30)
                distance = gs.qd_float(1e30)
                for i_f in range(surface_info.face_origin.shape[0]):
                    denominator = surface_info.face_normal[i_f].dot(direction)
                    if denominator > 1e-12:
                        parameter = (
                            surface_info.face_normal[i_f].dot(surface_info.face_origin[i_f] - point) / denominator
                        )
                        bary = surface_info.face_dual[i_f] @ (
                            point + parameter * direction - surface_info.face_origin[i_f]
                        )
                        # The application point identifies the nearest material sheet along the force line
                        if (
                            bary[0] >= -2e-12
                            and bary[1] >= -2e-12
                            and bary.sum() <= 1.0 + 2e-12
                            and qd.abs(parameter) < distance
                        ):
                            contact_state.anchor_face[i_c, i_b] = i_f
                            best = parameter
                            distance = qd.abs(parameter)
                if best > -1e30:
                    contact_state.center[i_c, i_b] = point + best * direction
                    i_f = contact_state.anchor_face[i_c, i_b]
                    edges = surface_info.face_edges[i_f]
                    diameter = qd.max(edges[0, :].norm(), edges[1, :].norm(), (edges[1, :] - edges[0, :]).norm())
                    # A solvable moment fit does not certify integration accuracy when the support is smaller
                    # than the face rule's spacing. Integrate these patches at their physical scale immediately.
                    contact_state.is_refined[i_c, i_b] = radius * surface_info.gauss.shape[0] < diameter
                else:
                    contact_state.status[i_c, i_b] = 2


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
    func_pressure_warp(contact_state, surface_info, refinement)


@qd.func
def func_pressure_warp(
    contact_state: StressContactState,
    surface_info: StressSurfaceInfo,
    refinement: bool,
    enabled: bool = True,
    processed_refined: qd.Tensor | None = None,
):
    qd.loop_config(block_dim=128)
    for i_thread in range(qd.select(enabled, contact_state.active_count[None] * 32, 0)):
        lane = i_thread % 32
        slot = i_thread // 32
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        processed = False
        if qd.static(processed_refined is not None):
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
    qd.loop_config(block_dim=128)
    for i_thread in range(contact_state.active_count[None] * 32):
        lane = i_thread % 32
        slot = i_thread // 32
        pair = contact_state.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contact_state.status[i_c, i_b] == 5:
            first, second = func_force_frame(contact_state.force[i_c, i_b])
            center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            low, dimensions = func_patch_grid(center, radius, surface_info.grid)
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
                            weight /= weight_sum
                            profile = qd.max(0.0, coordinates.dot(candidate))
                            objective -= 0.5 * weight * profile * profile
                            if not line:
                                gradient += weight * profile * coordinates
                                if profile > 0.0:
                                    hessian += weight * coordinates.outer_product(coordinates)
                objective = qd.simt.subgroup.reduce_all_add(objective) + candidate[0]
                if not line:
                    for a in qd.static(range(3)):
                        gradient[a] = qd.simt.subgroup.reduce_all_add(gradient[a])
                    for a, b in qd.static(qd.ndrange(3, 3)):
                        hessian[a, b] = qd.simt.subgroup.reduce_all_add(hessian[a, b])
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
            if lane == 0:
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
                objective = coefficient[0]
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
                        objective -= 0.5 * weight * profile * profile
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
                        candidate_objective = candidate[0]
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
                                candidate_objective -= 0.5 * weight * profile * profile
                        if candidate_objective >= objective + 1e-4 * fraction * gradient.dot(step) - 1e-15:
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
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
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
                            for i_local in range(6):
                                i_n = stress_info.surface_nodes[i_face, i_local]
                                value = shape[i_local] * sample_force
                                for i_a in qd.static(range(3)):
                                    qd.atomic_add(stress_state.force[i_n, i_b][i_a], value[i_a])
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
                distance = qd.max(low - center, 0.0) + qd.max(center - high, 0.0)
                if distance.dot(distance) < radius * radius:
                    n_q = surface_info.shape.shape[0]
                    start, count = i_f * n_q, n_q
                    if contact_state.is_refined[i_c, i_b]:
                        start, count = surface_info.positions.shape[0] + i_f * 3 * n_q, 3 * n_q
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
                        for i_local in range(6):
                            i_n = stress_info.surface_nodes[i_f, i_local]
                            for a in qd.static(range(3)):
                                qd.atomic_add(stress_state.force[i_n, i_b][a], load[i_local, a])
            for a in qd.static(range(3)):
                total[a] = qd.simt.subgroup.reduce_all_add(total[a])
                moment[a] = qd.simt.subgroup.reduce_all_add(moment[a])
            if lane == 0:
                contact_state.force_error[i_c, i_b] = (total - force).norm()
                contact_state.moment_error[i_c, i_b] = moment.norm()
                if (total - force).norm() > 1e-8 * force.norm() or moment.norm() > 1e-8 * force.norm() * radius:
                    contact_state.status[i_c, i_b] = 4
