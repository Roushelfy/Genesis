"""Quadrants polar integration panels for verified one-face disks near wrench capacity."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState


@qd.kernel
def kernel_refine_verified_disk(
    contacts: StressContactState,
    sources: StressSourceState,
    tractions: StressTractionState,
    surface: StressSurfaceInfo,
):
    """Partition the same disk into positive-area radial and angular integration panels.

    The caller must first verify that the complete disk lies in one actual material face and set the sample count
    to Q squared. Radius, normal, wrench, local Coulomb coefficient and continuous area/profile law remain fixed.
    """
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if tractions.sample_count[i_c, i_b] == surface.gauss.shape[0] ** 2:
            tractions.sample_count[i_c, i_b] = 9 * surface.gauss.shape[0] ** 2
    for i_q, i_c, i_b in qd.ndrange(tractions.profile.shape[0], contacts.valid.shape[0], contacts.valid.shape[1]):
        tractions.profile[i_q, i_c, i_b] = 0.0
        if i_q < tractions.sample_count[i_c, i_b]:
            i_face = contacts.anchor_face[i_c, i_b]
            normal = -surface.face_normal[i_face]
            edges = surface.face_edges[i_face]
            tangent = edges[0, :].normalized()
            second = normal.cross(tangent)
            radius = contacts.radius[i_c, i_b]
            compression = contacts.force[i_c, i_b].dot(normal)
            centroid = normal.cross(sources.anchor_moment[i_c, i_b]) / (compression * radius)
            fraction = qd.min(1.0 - 1e-12, centroid.norm())
            theta = qd.atan2(centroid.dot(second), centroid.dot(tangent))
            half_width = qd.min(0.5 * qd.math.pi, qd.max(0.1, qd.sqrt(8.0 * (1.0 - fraction))))
            radial_edges = qd.Vector([0.0, qd.max(0.0, fraction - 8.0 * (1.0 - fraction)), 0.5 * (1.0 + fraction), 1.0])
            angular_edges = qd.Vector([theta - qd.math.pi, theta - half_width, theta + half_width, theta + qd.math.pi])
            n = surface.gauss.shape[0]
            panel = i_q // (n * n)
            i_r, i_t = panel // 3, panel % 3
            local = i_q % (n * n)
            u, v = surface.gauss[local // n], surface.gauss[local % n]
            width_r = radial_edges[i_r + 1] - radial_edges[i_r]
            width_t = angular_edges[i_t + 1] - angular_edges[i_t]
            distance = radial_edges[i_r] + width_r * u
            angle = angular_edges[i_t] + width_t * v
            position = contacts.center[i_c, i_b] + radius * distance * (
                qd.cos(angle) * tangent + qd.sin(angle) * second
            )
            local_bary = surface.face_dual[i_face] @ (position - surface.face_origin[i_face])
            bary = qd.Vector([1.0 - local_bary.sum(), local_bary[0], local_bary[1]])
            profile = radius * radius * distance * width_r * width_t
            profile *= surface.gauss_weights[local // n] * surface.gauss_weights[local % n]
            profile *= qd.exp(-0.5 * (distance / 0.45) ** 2) * (1.0 - distance * distance) ** 2
            tractions.profile[i_q, i_c, i_b] = profile
            tractions.sample_positions[i_q, i_c, i_b] = position
            tractions.sample_normals[i_q, i_c, i_b] = normal
            tractions.sample_faces[i_q, i_c, i_b] = i_face
            tractions.sample_shape[i_q, i_c, i_b] = qd.Vector(
                [
                    bary[0] * (2.0 * bary[0] - 1.0),
                    bary[1] * (2.0 * bary[1] - 1.0),
                    bary[2] * (2.0 * bary[2] - 1.0),
                    4.0 * bary[0] * bary[1],
                    4.0 * bary[0] * bary[2],
                    4.0 * bary[1] * bary[2],
                ]
            )
