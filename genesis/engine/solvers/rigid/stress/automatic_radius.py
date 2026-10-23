"""Witness-based circular Hertz sizes for the original DPR load transfer."""

from enum import IntEnum

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom

from .contact import StressContactState
from .curvature import CurvatureStatus, StressCurvatureInfo, func_closest_surface, func_surface_curvature
from .pair_geometry import PairGeometryStatus, func_opponent_candidate


class StressRadiusStatus(IntEnum):
    MANUAL = 0
    SUCCESS = 1
    UNLOADED = 2
    UNSUPPORTED = 3
    INVALID = 4
    BOX_EDGE_APPROXIMATION = 5


class StressRadiusInfo:
    """Immutable build-time geometry and material references for optional kernel specialization."""

    def __init__(
        self,
        geometry: array_class.DynInfo,
        curvature: StressCurvatureInfo,
        modulus: float,
        unloaded_radius: float,
        epsilon: float,
        box_edges_as_planes: bool = False,
    ):
        self.geometry = geometry
        self.curvature = curvature
        self.modulus = modulus
        self.unloaded_radius = unloaded_radius
        self.epsilon = epsilon
        self.box_edges_as_planes = box_edges_as_planes


@qd.func
def func_witness_curvature(
    i_g: int,
    i_v: int,
    i_b: int,
    point: qd.types.vector(3),
    dyn: array_class.DynState,
    info: qd.template(),
):
    """Reuse an identified mesh vertex, otherwise query only the witnessed geom."""
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
    closest, face, query_surface = point, -1, False
    if info.geometry.geoms.type[i_g] == gs.GEOM_TYPE.MESH:
        if i_v >= 0 and i_v < info.geometry.verts.geom_idx.shape[0]:
            if info.geometry.verts.geom_idx[i_v] == i_g and info.curvature.vertices_weight[i_v] > 0.0:
                rotation = geom.qd_quat_to_R(dyn.geoms.quat[i_g, i_b], info.epsilon)
                tensor = rotation @ info.curvature.vertices_tensor[i_v] @ rotation.transpose()
                status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
        if status == CurvatureStatus.UNSUPPORTED:
            closest, face = func_closest_surface(i_g, i_b, point, dyn, info.geometry, info.curvature, info.epsilon)
            query_surface = True
    else:
        closest, _normal, _gap, _face, boundary_status = func_opponent_candidate(
            i_g, i_b, -1, point, dyn, info.geometry, info.epsilon
        )
        query_surface = boundary_status == PairGeometryStatus.SUCCESS
    if query_surface:
        _normal, tensor, status = func_surface_curvature(
            i_g, face, i_b, closest, dyn, info.geometry, info.curvature, info.epsilon
        )
    return tensor, status


@qd.func
def func_estimate_radius(
    i_c: int,
    i_b: int,
    i_contact: int,
    contacts: StressContactState,
    collider: array_class.ColliderState,
    dyn: array_class.DynState,
    info: qd.template(),
):
    """Estimate each supplied contact's size without changing its force or application point.

    The circular approximation uses the mean positive relative principal curvature. Mesh curvature is a smooth-normal
    estimate at the actual collision witness. Nonpositive curvature remains unsupported. The explicit planar box-edge
    policy ignores only the singular curvature at a witnessed box edge/vertex and labels that approximation.
    """
    contacts.radius[i_c, i_b] = float("nan")
    contacts.radius_status[i_c, i_b] = StressRadiusStatus.INVALID
    normal = contacts.normal[i_c, i_b]
    force = contacts.force[i_c, i_b]
    if normal.norm() > 0.0 and normal.norm() < float("inf") and force.norm() < float("inf"):
        load = force.dot(normal.normalized())
        if force.norm() <= 1e-30:
            contacts.radius[i_c, i_b] = info.unloaded_radius
            contacts.radius_status[i_c, i_b] = StressRadiusStatus.UNLOADED
        elif load > 0.0:
            combined = qd.Matrix.zero(gs.qd_float, 3, 3)
            smooth = True
            approximated_edge = False
            for side in range(2):
                i_g = collider.contact_data.geom_a[i_contact, i_b]
                i_v = collider.contact_data.witness.vertex_a[i_contact, i_b]
                point = collider.contact_data.witness.point_a[i_contact, i_b]
                if side == 1:
                    i_g = collider.contact_data.geom_b[i_contact, i_b]
                    i_v = collider.contact_data.witness.vertex_b[i_contact, i_b]
                    point = collider.contact_data.witness.point_b[i_contact, i_b]
                tensor, status = func_witness_curvature(i_g, i_v, i_b, point, dyn, info)
                combined += tensor
                supported = status < CurvatureStatus.AMBIGUOUS
                if qd.static(info.box_edges_as_planes):
                    if not supported and info.geometry.geoms.type[i_g] == gs.GEOM_TYPE.BOX:
                        local = geom.qd_inv_transform_by_trans_quat(
                            point, dyn.geoms.pos[i_g, i_b], dyn.geoms.quat[i_g, i_b]
                        )
                        half = 0.5 * qd.Vector([info.geometry.geoms.data[i_g][axis] for axis in qd.static(range(3))])
                        tolerance = 128.0 * info.epsilon * (half.norm() + point.norm() + dyn.geoms.pos[i_g, i_b].norm())
                        boundary, inside = 0, True
                        for axis in qd.static(range(3)):
                            boundary += gs.qd_int(qd.abs(qd.abs(local[axis]) - half[axis]) <= tolerance)
                            inside = inside and qd.abs(local[axis]) <= half[axis] + tolerance
                        supported = inside and boundary >= 2
                        approximated_edge = approximated_edge or supported
                smooth = smooth and supported
            contacts.radius_status[i_c, i_b] = StressRadiusStatus.UNSUPPORTED
            if smooth:
                direction = collider.contact_data.normal[i_contact, i_b].normalized()
                projection = qd.Matrix.identity(gs.qd_float, 3) - direction.outer_product(direction)
                curvature = projection @ combined @ projection
                mean = 0.5 * curvature.trace()
                span = qd.sqrt(qd.max(0.0, 0.5 * curvature.norm_sqr() - mean * mean))
                if mean > span and mean < float("inf"):
                    contacts.radius[i_c, i_b] = (0.75 * load / (info.modulus * mean)) ** (1.0 / 3.0)
                    contacts.radius_status[i_c, i_b] = StressRadiusStatus.SUCCESS
                    if approximated_edge:
                        contacts.radius_status[i_c, i_b] = StressRadiusStatus.BOX_EDGE_APPROXIMATION
