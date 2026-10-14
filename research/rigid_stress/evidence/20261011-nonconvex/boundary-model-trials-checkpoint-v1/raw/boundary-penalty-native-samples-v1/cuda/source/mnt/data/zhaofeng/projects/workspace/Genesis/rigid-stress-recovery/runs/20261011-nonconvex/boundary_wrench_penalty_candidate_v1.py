"""Generate continuous-domain certificate witnesses on demand, without storing a diagnostic force/geometry cloud."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    func_coulomb_projection,
    func_traction_source_friction,
    func_traction_support_range,
)


@qd.func
def func_boundary_witness(
    i_f: int,
    local: int,
    center: qd.types.vector(3),
    radius: float,
    info: StressTractionInfo,
):
    origin, edges = info.face_origin[i_f], info.face_edges[i_f]
    i_normal = i_f * info.gauss.shape[0] ** 2
    normal = info.inward_normals[i_normal]
    allowance = 128.0 * 2.220446049250313e-16 * (origin.norm() + center.norm() + edges[0, :].norm() + edges[1, :].norm() + radius)
    offset = (center - origin).dot(normal)
    circle_center = center - offset * normal
    circle_squared = radius * radius - offset * offset
    point, valid = circle_center, circle_squared >= 0.0
    if valid:
        if local < 3:
            if local == 0:
                point = origin
            elif local == 1:
                point = origin + edges[0, :]
            else:
                point = origin + edges[1, :]
            valid = (point - center).norm() <= radius + allowance
        elif local < 9:
            edge, root = (local - 3) // 2, (local - 3) % 2
            first, second = origin, origin + edges[0, :]
            if edge == 1:
                first, second = origin + edges[0, :], origin + edges[1, :]
            elif edge == 2:
                first, second = origin + edges[1, :], origin
            delta, relative = second - first, first - center
            qa, qb, qc = delta.dot(delta), 2.0 * relative.dot(delta), relative.dot(relative) - radius * radius
            discriminant = qb * qb - 4.0 * qa * qc
            valid = qa > 0.0 and discriminant >= 0.0
            if valid:
                fraction = (-qb + (2.0 * root - 1.0) * qd.sqrt(discriminant)) / (2.0 * qa)
                valid = 0.0 <= fraction <= 1.0
                point = first + fraction * delta
        elif local >= 10:
            first = edges[0, :] / edges[0, :].norm()
            second = normal.cross(first)
            angle = 2.0 * qd.math.pi * (local - 10) / 64.0
            point = circle_center + qd.sqrt(circle_squared) * (qd.cos(angle) * first + qd.sin(angle) * second)
        if valid:
            uv = info.face_dual[i_f] @ (point - origin)
            bary_allowance = 256.0 * 2.220446049250313e-16 * (1.0 + uv.norm())
            valid = uv[0] >= -bary_allowance and uv[1] >= -bary_allowance
            valid = valid and uv[0] + uv[1] <= 1.0 + bary_allowance
            valid = valid and (point - center).norm() <= radius + allowance
    return point, normal, valid


@qd.func
def func_boundary_wrench_penalty(
    i_c: int,
    i_b: int,
    ray: qd.types.vector(6),
    target: qd.types.vector(6),
    penalty: float,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    n_samples: int,
):
    delta = ray - target
    projected_energy = gs.qd_float(0.0)
    projected_gradient = qd.Vector.zero(gs.qd_float, 6)
    projected_hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    witnesses = 0
    first, last = func_traction_support_range(i_c, i_b, traction_state)
    for i_s in range(first, last):
        center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
        if qd.static(len(traction_state.support.center.shape) == 2):
            center, radius = traction_state.support.center[i_s, i_b], traction_state.support.radius[i_s, i_b]
        friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
        for i_f in range(traction_info.face_origin.shape[0]):
            for local in range(74):
                position, normal, valid = func_boundary_witness(i_f, local, center, radius, traction_info)
                if valid:
                    witnesses += 1
                    relative = (position - contact_state.center[i_c, i_b]) / contact_state.radius[i_c, i_b]
                    value = ray[0:3] + ray[3:6].cross(relative)
                    projected, jacobian = func_coulomb_projection(value, normal, friction)
                    operator = qd.Matrix.zero(gs.qd_float, 6, 3)
                    for a in qd.static(range(3)):
                        operator[a, a] = 1.0
                    operator[3, 1], operator[3, 2] = -relative[2], relative[1]
                    operator[4, 0], operator[4, 2] = relative[2], -relative[0]
                    operator[5, 0], operator[5, 1] = -relative[1], relative[0]
                    projected_energy += 0.5 * projected.dot(projected)
                    projected_gradient += operator @ projected
                    projected_hessian += operator @ jacobian @ operator.transpose()
    fraction = penalty / qd.max(witnesses, 1)
    energy = 0.5 * delta.dot(delta) + fraction * projected_energy
    gradient = delta + fraction * projected_gradient
    hessian = qd.Matrix.identity(gs.qd_float, 6) + fraction * projected_hessian
    return energy, gradient, hessian
