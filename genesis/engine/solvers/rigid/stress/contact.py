"""Native finite, nonnegative pressure preserving each point-contact wrench."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V, of_kind

from .data import StressInfo, StressState
from .grid import func_patch_cell, func_patch_grid
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class StressContactState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    position: qd.Tensor
    center: qd.Tensor
    force: qd.Tensor
    normal: qd.Tensor
    friction: qd.Tensor
    radius: qd.Tensor = of_kind(DataKind.INFO)  # noqa: RUF009 - typed array metadata
    valid: qd.Tensor
    coefficient: qd.Tensor
    weight_sum: qd.Tensor
    status: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor
    candidate_count: qd.Tensor
    evaluations: qd.Tensor
    frame_position: qd.Tensor
    frame_quaternion: qd.Tensor


def create_contacts(n_contacts: int, n_envs: int, radius: float) -> StressContactState:
    shape = (n_contacts, n_envs)
    state = StressContactState(
        position=V_VEC(3, dtype=gs.qd_float, shape=shape),
        center=V_VEC(3, dtype=gs.qd_float, shape=shape),
        force=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        radius=V(dtype=gs.qd_float, shape=shape),
        valid=V(dtype=gs.qd_bool, shape=shape),
        coefficient=V_VEC(3, dtype=gs.qd_float, shape=shape),
        weight_sum=V(dtype=gs.qd_float, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
        force_error=V(dtype=gs.qd_float, shape=shape),
        moment_error=V(dtype=gs.qd_float, shape=shape),
        candidate_count=V(dtype=gs.qd_int, shape=shape),
        evaluations=V(dtype=gs.qd_int, shape=shape),
        frame_position=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        frame_quaternion=V_VEC(4, dtype=gs.qd_float, shape=(n_envs,)),
    )
    state.radius.fill(radius)
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


@qd.kernel
def kernel_anchor(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        contact_state.status[i_c, i_b] = 0
        contact_state.force_error[i_c, i_b] = 0.0
        contact_state.moment_error[i_c, i_b] = 0.0
        contact_state.candidate_count[i_c, i_b] = 0
        contact_state.evaluations[i_c, i_b] = 0
        if contact_state.valid[i_c, i_b]:
            force = contact_state.force[i_c, i_b]
            magnitude = force.norm()
            normal = contact_state.normal[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            admitted = radius > 0.0 and normal.norm() > 0.0 and contact_state.friction[i_c, i_b] >= 0.0
            admitted = admitted and not qd.math.isnan(magnitude) and not qd.math.isinf(magnitude)
            admitted = admitted and not qd.math.isinf(radius) and not qd.math.isinf(normal.norm())
            admitted = admitted and not qd.math.isinf(contact_state.friction[i_c, i_b])
            for i_a in qd.static(range(3)):
                value = contact_state.position[i_c, i_b][i_a]
                admitted = admitted and not qd.math.isnan(value) and not qd.math.isinf(value)
            if admitted:
                normal = normal.normalized()
                compression = force.dot(normal)
                tangent = force - compression * normal
                allowance = 16.0 * source_epsilon * magnitude
                admitted = (
                    compression >= -allowance
                    and tangent.norm() <= contact_state.friction[i_c, i_b] * compression + allowance
                )
            if not admitted:
                contact_state.status[i_c, i_b] = 1
            elif magnitude > 1e-30:
                direction = -force / magnitude
                point = contact_state.position[i_c, i_b]
                best = gs.qd_float(-1e30)
                for i_f in range(surface_info.face_origin.shape[0]):
                    denominator = surface_info.face_normal[i_f].dot(direction)
                    if denominator > 1e-12:
                        parameter = (
                            surface_info.face_normal[i_f].dot(surface_info.face_origin[i_f] - point) / denominator
                        )
                        bary = surface_info.face_dual[i_f] @ (
                            point + parameter * direction - surface_info.face_origin[i_f]
                        )
                        if bary[0] >= -2e-12 and bary[1] >= -2e-12 and bary.sum() <= 1.0 + 2e-12:
                            best = qd.max(best, parameter)
                if best > -1e30:
                    contact_state.center[i_c, i_b] = point + best * direction
                else:
                    contact_state.status[i_c, i_b] = 2


@qd.kernel(graph=True)
def kernel_pressure(contact_state: StressContactState, surface_info: StressSurfaceInfo):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and contact_state.status[i_c, i_b] == 0:
            force = contact_state.force[i_c, i_b]
            if force.norm() > 1e-30:
                first, second = func_force_frame(force)
                center = contact_state.center[i_c, i_b]
                radius = contact_state.radius[i_c, i_b]
                low, dimensions = func_patch_grid(center, radius, surface_info.grid)
                n_cells = dimensions[0] * dimensions[1] * dimensions[2]
                gram = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i_cell in range(n_cells):
                    start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
                    for i_entry in range(start, end):
                        i_q = surface_info.grid.samples[i_entry]
                        coordinates, weight = func_patch_sample(i_q, center, first, second, radius, surface_info)
                        gram += weight * coordinates.outer_product(coordinates)
                        if weight > 0.0:
                            contact_state.candidate_count[i_c, i_b] += 1
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
                        and (gram @ coefficient - qd.Vector([1.0, 0.0, 0.0])).norm() <= 2e-12
                    )
                    contact_state.weight_sum[i_c, i_b] = weight_sum
                    coefficient /= weight_sum
                contact_state.coefficient[i_c, i_b] = coefficient
                if not accepted:
                    contact_state.status[i_c, i_b] = 5 if nonsingular else 3
    # Keep the uncommon constrained solve out of the integration kernel's register footprint.
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and contact_state.status[i_c, i_b] == 5:
            first, second = func_force_frame(contact_state.force[i_c, i_b])
            center = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            low, dimensions = func_patch_grid(center, radius, surface_info.grid)
            n_cells = dimensions[0] * dimensions[1] * dimensions[2]
            weight_sum = contact_state.weight_sum[i_c, i_b]
            coefficient = contact_state.coefficient[i_c, i_b] * weight_sum
            accepted = False
            for iteration in range(80):
                if not accepted:
                    gradient = qd.Vector([-1.0, 0.0, 0.0])
                    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                    objective = coefficient[0]
                    for i_cell in range(n_cells):
                        start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
                        for i_entry in range(start, end):
                            i_q = surface_info.grid.samples[i_entry]
                            coordinates, weight = func_patch_sample(i_q, center, first, second, radius, surface_info)
                            weight /= weight_sum
                            profile = qd.max(0.0, coordinates.dot(coefficient))
                            gradient += weight * profile * coordinates
                            objective -= 0.5 * weight * profile * profile
                            if profile > 0.0:
                                hessian += weight * coordinates.outer_product(coordinates)
                    contact_state.evaluations[i_c, i_b] = iteration + 1
                    accepted = gradient.norm() <= 2e-12
                    if not accepted and hessian.determinant() > 1e-14 * hessian.trace() ** 3:
                        step = hessian.inverse() @ gradient
                        fraction = gs.qd_float(1.0)
                        line_accepted = False
                        for _ in range(40):
                            if not line_accepted:
                                candidate = coefficient - fraction * step
                                candidate_objective = candidate[0]
                                for i_cell in range(n_cells):
                                    start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
                                    for i_entry in range(start, end):
                                        i_q = surface_info.grid.samples[i_entry]
                                        coordinates, weight = func_patch_sample(
                                            i_q, center, first, second, radius, surface_info
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
def kernel_scatter(
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
                n_cells = dimensions[0] * dimensions[1] * dimensions[2]
                total = qd.Vector.zero(gs.qd_float, 3)
                moment = qd.Vector.zero(gs.qd_float, 3)
                for i_cell in range(n_cells):
                    start, end = func_patch_cell(i_cell, low, dimensions, surface_info.grid)
                    for i_entry in range(start, end):
                        i_q = surface_info.grid.samples[i_entry]
                        coordinates, weight = func_patch_sample(
                            i_q,
                            contact_state.center[i_c, i_b],
                            first,
                            second,
                            contact_state.radius[i_c, i_b],
                            surface_info,
                        )
                        fraction = weight * qd.max(0.0, coordinates.dot(contact_state.coefficient[i_c, i_b]))
                        sample_force = fraction * force
                        total += sample_force
                        moment += (surface_info.positions[i_q] - contact_state.position[i_c, i_b]).cross(sample_force)
                        if weight > 0.0:
                            i_face = i_q // surface_info.shape.shape[0]
                            i_shape = i_q % surface_info.shape.shape[0]
                            for i_local in range(6):
                                i_n = stress_info.surface_nodes[i_face, i_local]
                                value = surface_info.shape[i_shape][i_local] * sample_force
                                for i_a in qd.static(range(3)):
                                    qd.atomic_add(stress_state.force[i_n, i_b][i_a], value[i_a])
                contact_state.force_error[i_c, i_b] = (total - force).norm()
                contact_state.moment_error[i_c, i_b] = moment.norm()
                if (total - force).norm() > 1e-8 * magnitude or moment.norm() > 1e-8 * magnitude * contact_state.radius[
                    i_c, i_b
                ]:
                    contact_state.status[i_c, i_b] = 4
