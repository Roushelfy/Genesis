"""Conservative continuous-surface force incompatibility certificates."""

import quadrants as qd

import genesis as gs

from .contact import StressContactState
from .provenance import func_closest_triangle
from .surface import StressSurfaceInfo
from .traction import (
    StressTractionState,
    StressTractionStatus,
    func_traction_source_friction,
    func_traction_support_range,
)


@qd.kernel
def kernel_certify_continuous_force(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    force_tolerance: float,
):
    """Certify only forces outside every possible local continuous cone sum.

    The Euclidean radius ball contains the declared intrinsic footprint. All
    intersecting triangles therefore form a conservative support superset,
    including branches the intrinsic law cannot reach. A strict common polar
    ray bounds every finite traction distribution on that superset; its force
    margin is a lower bound on the normalized resultant error. This test does
    not establish general six-dimensional feasibility or integration accuracy.
    """
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        status = traction_state.status[i_c, i_b]
        unresolved = status == StressTractionStatus.NUMERICAL_FAILURE
        unresolved = unresolved or status == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
        if contact_state.valid[i_c, i_b] and unresolved:
            i_anchor = contact_state.anchor_face[i_c, i_b]
            force = contact_state.force[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            friction = contact_state.friction[i_c, i_b]
            anchor = contact_state.center[i_c, i_b]
            if 0 <= i_anchor < surface_info.face_normal.shape[0] and force.norm() > 0.0:
                normal = -surface_info.face_normal[i_anchor]
                unit_force = force / force.norm()
                compression = unit_force.dot(normal)
                tangent = unit_force - compression * normal
                if tangent.norm() > friction * compression and tangent.norm() > 0.0:
                    # Construct the polar analytically: subtracting a tiny
                    # force projection would lose relative accuracy near slip.
                    ray = (tangent / tangent.norm() - friction * normal) / qd.sqrt(1.0 + friction**2)
                    ray = (ray - 1024.0 * epsilon * (1.0 + friction) * normal).normalized()
                    roundoff = 64.0 * epsilon * (1.0 + friction)
                    maximum_support = gs.qd_float(-float("inf"))
                    start, end = func_traction_support_range(i_c, i_b, traction_state)
                    for i_s in range(start, end):
                        source_anchor, source_radius = anchor, radius
                        if qd.static(len(traction_state.support.center.shape) == 2):
                            source_anchor = traction_state.support.center[i_s, i_b]
                            source_radius = traction_state.support.radius[i_s, i_b]
                        source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                        geometric_roundoff = 64.0 * epsilon * (source_anchor.norm() + source_radius)
                        for i_f in range(surface_info.face_normal.shape[0]):
                            closest, _ = func_closest_triangle(i_f, source_anchor, surface_info)
                            if (closest - source_anchor).norm() <= source_radius + geometric_roundoff:
                                inward = -surface_info.face_normal[i_f]
                                component = ray.dot(inward)
                                support = component + source_friction * (ray - component * inward).norm()
                                maximum_support = qd.max(maximum_support, support)
                    margin = unit_force.dot(ray)
                    if maximum_support < -roundoff and margin > force_tolerance + roundoff:
                        traction_state.status[i_c, i_b] = StressTractionStatus.CONTINUOUS_FORCE_INFEASIBLE
                        traction_state.certificate[i_c, i_b] = qd.Vector([ray[0], ray[1], ray[2], 0.0, 0.0, 0.0])
                        traction_state.certificate_margin[i_c, i_b] = margin - roundoff
