"""Native local disk integration candidate for a footprint wholly inside one material face."""

import quadrants as qd
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState

import genesis as gs


@qd.kernel
def kernel_build_single_face_disks(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        traction_state.sample_count[i_c, i_b] = 0
        i_face = contact_state.anchor_face[i_c, i_b]
        if contact_state.valid[i_c, i_b] and 0 <= i_face < surface_info.face_origin.shape[0]:
            origin, edges = surface_info.face_origin[i_face], surface_info.face_edges[i_face]
            anchor, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            normal = -surface_info.face_normal[i_face]
            roundoff = 128.0 * epsilon * (anchor.norm() + origin.norm() + edges[0, :].norm() + edges[1, :].norm())
            inside = radius > 0.0 and qd.abs((anchor - origin).dot(normal)) <= roundoff
            vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
            for a in range(3):
                edge = vertices[(a + 1) % 3, :] - vertices[a, :]
                direction = normal.cross(edge).normalized()
                if direction.dot(vertices[(a + 2) % 3, :] - vertices[a, :]) < 0.0:
                    direction = -direction
                inside = inside and (anchor - vertices[a, :]).dot(direction) >= radius + roundoff
            if inside:
                traction_state.sample_count[i_c, i_b] = surface_info.gauss.shape[0] ** 2
    for i_q, i_c, i_b in qd.ndrange(
        traction_state.profile.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        traction_state.profile[i_q, i_c, i_b] = 0.0
        if i_q < traction_state.sample_count[i_c, i_b]:
            i_face = contact_state.anchor_face[i_c, i_b]
            edges = surface_info.face_edges[i_face]
            normal = -surface_info.face_normal[i_face]
            tangent = edges[0, :].normalized()
            second = normal.cross(tangent)
            n = surface_info.gauss.shape[0]
            u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
            radius = contact_state.radius[i_c, i_b]
            distance, angle = u * u, 2.0 * qd.math.pi * v
            position = contact_state.center[i_c, i_b] + radius * distance * (
                qd.cos(angle) * tangent + qd.sin(angle) * second
            )
            local = surface_info.face_dual[i_face] @ (position - surface_info.face_origin[i_face])
            bary = qd.Vector([1.0 - local.sum(), local[0], local[1]])
            profile = 4.0 * qd.math.pi * radius * radius * u**3
            profile *= surface_info.gauss_weights[i_q // n] * surface_info.gauss_weights[i_q % n]
            profile *= qd.exp(-0.5 * (distance / 0.45) ** 2) * (1.0 - distance * distance) ** 2
            traction_state.profile[i_q, i_c, i_b] = profile
            traction_state.sample_positions[i_q, i_c, i_b] = position
            traction_state.sample_normals[i_q, i_c, i_b] = normal
            traction_state.sample_faces[i_q, i_c, i_b] = i_face
            traction_state.sample_shape[i_q, i_c, i_b] = qd.Vector(
                [
                    bary[0] * (2.0 * bary[0] - 1.0),
                    bary[1] * (2.0 * bary[1] - 1.0),
                    bary[2] * (2.0 * bary[2] - 1.0),
                    4.0 * bary[0] * bary[1],
                    4.0 * bary[0] * bary[2],
                    4.0 * bary[1] * bary[2],
                ]
            )
