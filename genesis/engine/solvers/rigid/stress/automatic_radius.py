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


class StressRadiusInfo:
    """Immutable build-time geometry and material references for optional kernel specialization."""

    def __init__(
        self,
        geometry: array_class.DynInfo,
        curvature: StressCurvatureInfo,
        modulus: float,
        unloaded_radius: float,
        epsilon: float,
    ):
        self.geometry = geometry
        self.curvature = curvature
        self.modulus = modulus
        self.unloaded_radius = unloaded_radius
        self.epsilon = epsilon


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
    normal = qd.Vector.zero(gs.qd_float, 3)
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
    if info.geometry.geoms.type[i_g] == gs.GEOM_TYPE.MESH:
        if i_v >= 0 and i_v < info.geometry.verts.geom_idx.shape[0]:
            if info.geometry.verts.geom_idx[i_v] == i_g and info.curvature.vertices_weight[i_v] > 0.0:
                rotation = geom.qd_quat_to_R(dyn.geoms.quat[i_g, i_b], info.epsilon)
                normal = rotation @ info.geometry.verts.init_normal[i_v].normalized()
                tensor = rotation @ info.curvature.vertices_tensor[i_v] @ rotation.transpose()
                status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
        if status == CurvatureStatus.UNSUPPORTED:
            closest, face = func_closest_surface(i_g, i_b, point, dyn, info.geometry, info.curvature, info.epsilon)
            normal, tensor, status = func_surface_curvature(
                i_g, face, i_b, closest, dyn, info.geometry, info.curvature, info.epsilon
            )
    else:
        closest, _normal, _gap, _face, boundary_status = func_opponent_candidate(
            i_g, i_b, -1, point, dyn, info.geometry, info.epsilon
        )
        if boundary_status == PairGeometryStatus.SUCCESS:
            normal, tensor, status = func_surface_curvature(
                i_g, -1, i_b, closest, dyn, info.geometry, info.curvature, info.epsilon
            )
    return normal, tensor, status


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
    estimate at the actual collision witness. Nonpositive curvature and primitive edges remain unsupported.
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
            normal_a, tensor_a, status_a = func_witness_curvature(
                collider.contact_data.geom_a[i_contact, i_b],
                collider.contact_data.witness.vertex_a[i_contact, i_b],
                i_b,
                collider.contact_data.witness.point_a[i_contact, i_b],
                dyn,
                info,
            )
            normal_b, tensor_b, status_b = func_witness_curvature(
                collider.contact_data.geom_b[i_contact, i_b],
                collider.contact_data.witness.vertex_b[i_contact, i_b],
                i_b,
                collider.contact_data.witness.point_b[i_contact, i_b],
                dyn,
                info,
            )
            contacts.radius_status[i_c, i_b] = StressRadiusStatus.UNSUPPORTED
            if status_a < CurvatureStatus.AMBIGUOUS and status_b < CurvatureStatus.AMBIGUOUS:
                direction = collider.contact_data.normal[i_contact, i_b].normalized()
                projection = qd.Matrix.identity(gs.qd_float, 3) - direction.outer_product(direction)
                curvature = projection @ (tensor_a + tensor_b) @ projection
                mean = 0.5 * curvature.trace()
                span = qd.sqrt(qd.max(0.0, 0.5 * curvature.norm_sqr() - mean * mean))
                if mean > span and mean < float("inf"):
                    contacts.radius[i_c, i_b] = (0.75 * load / (info.modulus * mean)) ** (1.0 / 3.0)
                    contacts.radius_status[i_c, i_b] = StressRadiusStatus.SUCCESS
