"""Retain narrowphase witnesses independently of the rigid application point."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom

from .constants import PORTAL_STATUS


@qd.func
def func_get_witness(
    i_ga: int,
    i_gb: int,
    i_scratch: int,
    i_w: int,
    normal: qd.types.vector(3),
    position: qd.types.vector(3),
    penetration: float,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    used_gjk: bool,
):
    """Return producer points, or an explicitly labelled normal-offset estimate."""
    point_a = position - 0.5 * penetration * normal
    point_b = position + 0.5 * penetration * normal
    method = array_class.ContactWitnessKind.NORMAL_OFFSET
    analytic = (
        dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE
        or (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE)
        or (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE)
        or (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX)
        or (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE)
    )
    if not analytic:
        if used_gjk:
            point_a = gjk_state.witness.point_obj1[i_scratch, i_w]
            point_b = gjk_state.witness.point_obj2[i_scratch, i_w]
            method = array_class.ContactWitnessKind.GJK
        else:
            point_a = mpr_state.witness_a[i_scratch]
            point_b = mpr_state.witness_b[i_scratch]
            method = array_class.ContactWitnessKind.MPR_APPROXIMATE
            if mpr_state.portal_status[i_scratch] == PORTAL_STATUS.EXACT:
                method = array_class.ContactWitnessKind.MPR_EXACT
    return point_a, point_b, method


@qd.func
def func_unrotate_witness(
    point_a: qd.types.vector(3),
    point_b: qd.types.vector(3),
    pivot: qd.types.vector(3),
    qrot: qd.types.vector(4),
):
    point_a = geom.qd_inv_transform_by_quat(point_a - pivot, qrot) + pivot
    point_b = geom.qd_transform_by_quat(point_b - pivot, qrot) + pivot
    return point_a, point_b


@qd.func
def func_set_witness(
    i_c: int,
    i_b: int,
    method: int,
    point_a: qd.types.vector(3),
    point_b: qd.types.vector(3),
    collider_state: array_class.ColliderState,
):
    if qd.static(len(collider_state.contact_data.witness.point_a.shape) > 0):  # noqa: SIM102
        if 0 <= i_c < collider_state.contact_data.geom_a.shape[0]:
            collider_state.contact_data.witness.point_a[i_c, i_b] = point_a
            collider_state.contact_data.witness.point_b[i_c, i_b] = point_b
            collider_state.contact_data.witness.method[i_c, i_b] = method
