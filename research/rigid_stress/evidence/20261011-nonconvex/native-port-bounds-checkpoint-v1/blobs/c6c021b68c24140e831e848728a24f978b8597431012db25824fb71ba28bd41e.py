"""Byte-exact native footprint distance for the nonnegative port-bound ablation."""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo, func_surface_segment
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo

@qd.func
def func_footprint_point_distance(
    i_face: int,
    i_c: int,
    i_b: int,
    target: qd.types.vector(3),
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    i_anchor = contact_state.anchor_face[i_c, i_b]
    anchor = contact_state.center[i_c, i_b]
    is_straight = func_surface_segment(i_anchor, i_face, anchor, target, stress_info, surface_info)
    distance = gs.qd_float(float("inf"))
    if is_straight:
        distance = (target - anchor).norm()
    else:
        # A path crossing one actual shared edge has an exact two-triangle
        # unfolding. Both segments stay inside their convex triangles. This
        # removes port-spacing error without connecting separate sheets.
        for i_edge in qd.static(range(3)):
            if stress_info.surface_neighbors[i_anchor, i_edge] == i_face:
                i_first = stress_info.surface_nodes[i_anchor, i_edge]
                i_second = stress_info.surface_nodes[i_anchor, (i_edge + 1) % 3]
                first = stress_info.vertices[i_first]
                edge = stress_info.vertices[i_second] - first
                length = edge.norm()
                direction = edge / length
                anchor_coordinate = (anchor - first).dot(direction)
                target_coordinate = (target - first).dot(direction)
                anchor_height = (anchor - first - anchor_coordinate * direction).norm()
                target_height = (target - first - target_coordinate * direction).norm()
                height = anchor_height + target_height
                crossing = gs.qd_float(0.0)
                if height > 0.0:
                    crossing = (anchor_height * target_coordinate + target_height * anchor_coordinate) / height
                crossing = qd.min(length, qd.max(0.0, crossing))
                point = first + crossing * direction
                distance = qd.min(distance, (anchor - point).norm() + (target - point).norm())
        for i_t in range(path_info.face_ports.shape[1]):
            i_port = path_info.face_ports[i_face, i_t]
            candidate = footprint_state.port_distance[i_port, i_c, i_b]
            candidate += (target - path_info.positions[i_port]).norm()
            distance = qd.min(distance, candidate)
    return distance, is_straight

