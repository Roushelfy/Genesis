"""Build native boundary witnesses for certificate search on a conservative Euclidean disk superset.

This exploratory cloud affects diagnostic ray generation alone. A ray must still pass the native
continuous-surface upper bound before any physical incompatibility classification. Contact loads,
intrinsic integration, friction and physical radii remain unchanged.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState


@qd.kernel
def kernel_boundary_certificate_support(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    angular_samples: int,
):
    for i_s, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        traction_state.sample_count[i_s, i_b] = traction_state.profile.shape[0]
        traction_state.support.valid[i_s, i_b] = contact_state.valid[i_s, i_b]
        traction_state.support.center[i_s, i_b] = contact_state.center[i_s, i_b]
        traction_state.support.radius[i_s, i_b] = contact_state.radius[i_s, i_b]
        traction_state.support.friction[i_s, i_b] = contact_state.friction[i_s, i_b]
        traction_state.support.start[i_s, i_b] = 0
        traction_state.support.end[i_s, i_b] = contact_state.valid.shape[0]
    for i_q, i_s, i_b in qd.ndrange(
        traction_state.profile.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        i_f, local = i_q // (10 + angular_samples), i_q % (10 + angular_samples)
        center, radius = contact_state.center[i_s, i_b], contact_state.radius[i_s, i_b]
        origin, edges = surface_info.face_origin[i_f], surface_info.face_edges[i_f]
        normal = -surface_info.face_normal[i_f]
        allowance = 128.0 * epsilon * (origin.norm() + center.norm() + edges[0, :].norm() + edges[1, :].norm() + radius)
        offset = (center - origin).dot(normal)
        circle_center = center - offset * normal
        circle_squared = radius * radius - offset * offset
        point, valid = circle_center, contact_state.valid[i_s, i_b] and circle_squared >= 0.0
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
                angle = 2.0 * qd.math.pi * (local - 10) / angular_samples
                point = circle_center + qd.sqrt(circle_squared) * (qd.cos(angle) * first + qd.sin(angle) * second)
            if valid:
                uv = surface_info.face_dual[i_f] @ (point - origin)
                bary_allowance = 256.0 * epsilon * (1.0 + uv.norm())
                valid = uv[0] >= -bary_allowance and uv[1] >= -bary_allowance
                valid = valid and uv[0] + uv[1] <= 1.0 + bary_allowance
                valid = valid and (point - center).norm() <= radius + allowance
        traction_state.profile[i_q, i_s, i_b] = 1.0 if valid else 0.0
        traction_state.sample_positions[i_q, i_s, i_b] = point
        traction_state.sample_normals[i_q, i_s, i_b] = normal
        traction_state.sample_faces[i_q, i_s, i_b] = i_f
