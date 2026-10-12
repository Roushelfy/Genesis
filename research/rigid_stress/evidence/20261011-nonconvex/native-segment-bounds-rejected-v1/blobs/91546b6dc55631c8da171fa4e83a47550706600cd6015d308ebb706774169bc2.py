"""Byte-exact original Quadrants adjacency walk for the segment-bound ablation."""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo

@qd.func
def func_surface_segment(
    i_anchor: int,
    i_target: int,
    anchor: qd.types.vector(3),
    target: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
):
    """Verify that a straight segment reaches its target through actual coplanar triangles."""
    direction = target - anchor
    scale = qd.max(direction.norm(), 1e-30)
    normal = surface_info.face_normal[i_anchor]
    tolerance = gs.qd_float(2e-12)
    epsilon = gs.qd_float(2.220446049250313e-16)
    if qd.static(gs.qd_float == qd.f32):
        tolerance = 2e-5
        epsilon = 1.1920928955078125e-7
    current, parameter = i_anchor, gs.qd_float(0.0)
    active, accepted = True, False
    for _ in range(stress_info.surface_nodes.shape[0]):
        if active:
            origin = surface_info.face_origin[current]
            # Plane subtraction/dot products round at the coordinate scale, even for a very short segment.
            plane_tolerance = tolerance * scale + 32.0 * epsilon * (anchor.norm() + target.norm() + origin.norm())
            coplanar = (surface_info.face_normal[current] - normal).norm() <= tolerance
            coplanar = coplanar and qd.abs(normal.dot(anchor - origin)) <= plane_tolerance
            coplanar = coplanar and qd.abs(normal.dot(direction)) <= plane_tolerance
            start = surface_info.face_dual[current] @ (anchor - origin)
            delta = surface_info.face_dual[current] @ direction
            bary = qd.Vector([1.0 - start.sum(), start[0], start[1]])
            derivative = qd.Vector([-delta.sum(), delta[0], delta[1]])
            contained = True
            for a in qd.static(range(3)):
                contained = contained and bary[a] + parameter * derivative[a] >= -tolerance
            if not coplanar or not contained:
                active = False
            else:
                end_contained = True
                for a in qd.static(range(3)):
                    end_contained = end_contained and bary[a] + derivative[a] >= -tolerance
                if end_contained:
                    accepted = current == i_target
                    if not accepted:
                        # A segment ending on a shared edge/vertex is also contained in its incident target face.
                        # Match vertex identities, so coincident points on separate sheets never qualify.
                        is_shared_endpoint = True
                        for a in qd.static(range(3)):
                            if bary[a] + derivative[a] > tolerance:
                                i_vertex = stress_info.surface_nodes[current, a]
                                is_shared_vertex = False
                                for b in qd.static(range(3)):
                                    is_shared_vertex = is_shared_vertex or (
                                        i_vertex == stress_info.surface_nodes[i_target, b]
                                    )
                                is_shared_endpoint = is_shared_endpoint and is_shared_vertex
                        accepted = is_shared_endpoint
                    active = False
                else:
                    crossing = gs.qd_float(float("inf"))
                    edge = -1
                    ties = 0
                    for a in qd.static(range(3)):
                        if derivative[a] < -tolerance:
                            candidate = -bary[a] / derivative[a]
                            if candidate < crossing - tolerance:
                                crossing = candidate
                                edge = (a + 1) % 3
                                ties = 1
                            elif qd.abs(candidate - crossing) <= tolerance:
                                ties += 1
                    if edge < 0 or ties != 1 or crossing <= parameter + tolerance or crossing >= 1.0:
                        active = False
                    else:
                        neighbor = stress_info.surface_neighbors[current, edge]
                        if neighbor < 0:
                            active = False
                        else:
                            current = neighbor
                            parameter = crossing
    return accepted

