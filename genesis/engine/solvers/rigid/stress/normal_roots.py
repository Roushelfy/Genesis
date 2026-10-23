"""Invert the fixed piecewise-linear normal field for planar contact candidates."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom

from .curvature import CurvatureStatus, StressCurvatureInfo


@qd.func
def func_normal_root(
    i_g: int,
    i_b: int,
    direction: qd.types.vector(3),
    seed: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    """Choose the normal-field root nearest the source and return its curvature.

    A mesh root satisfies the normalized barycentric vertex-normal field exactly.
    Its position remains on the collision triangle. Multiple roots retain the
    source-local choice. Visibility and finite opponent support require separate
    checks before pressure admission.
    """
    rotation = geom.qd_quat_to_R(dyn_state.geoms.quat[i_g, i_b], epsilon)
    local_direction = rotation.transpose() @ direction
    local_seed = rotation.transpose() @ (seed - dyn_state.geoms.pos[i_g, i_b])
    center = qd.Vector.zero(gs.qd_float, 3)
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
    i_selected = -1
    weights = qd.Vector.zero(gs.qd_float, 3)
    residual = gs.qd_float(0.0)
    if dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.SPHERE:
        radius = dyn_info.geoms.data[i_g][0]
        center = radius * local_direction
        tensor = (qd.Matrix.identity(gs.qd_float, 3) - local_direction.outer_product(local_direction)) / radius
        status = CurvatureStatus.ANALYTIC
    elif dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.MESH:
        distance = gs.qd_float(float("inf"))
        i_start, i_end = dyn_info.geoms.face_start[i_g], dyn_info.geoms.face_end[i_g]
        for i_block in range(i_start // 64, (i_end + 63) // 64):
            if (local_direction >= curvature_info.search_normal_min[i_block]).all() and (
                local_direction <= curvature_info.search_normal_max[i_block]
            ).all():
                for i_f in range(qd.max(i_start, 64 * i_block), qd.min(i_end, 64 * (i_block + 1))):
                    if curvature_info.normal_inverse_valid[i_f]:
                        barycentric = curvature_info.normal_inverse[i_f] @ local_direction
                        total = barycentric.sum()
                        if total > 128.0 * epsilon * barycentric.norm():
                            barycentric /= total
                            if barycentric.min() >= -128.0 * epsilon:
                                barycentric = qd.max(barycentric, 0.0)
                                barycentric /= barycentric.sum()
                                indices = dyn_info.faces.verts_idx[i_f]
                                candidate = qd.Vector.zero(gs.qd_float, 3)
                                for i in qd.static(range(3)):
                                    i_v = indices[i]
                                    candidate += barycentric[i] * dyn_info.verts.init_pos[i_v]
                                candidate_distance = (candidate - local_seed).norm_sqr()
                                if candidate_distance < distance:
                                    distance, center, i_selected, weights = (
                                        candidate_distance,
                                        candidate,
                                        i_f,
                                        barycentric,
                                    )
        if i_selected >= 0:
            indices = dyn_info.faces.verts_idx[i_selected]
            is_valid = True
            recovered_normal = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(3)):
                i_v = indices[i]
                tensor += weights[i] * curvature_info.vertices_tensor[i_v]
                recovered_normal += weights[i] * dyn_info.verts.init_normal[i_v]
                is_valid = is_valid and curvature_info.vertices_weight[i_v] > 0.0
            residual = (recovered_normal.normalized() - local_direction).norm()
            projection = qd.Matrix.identity(gs.qd_float, 3) - local_direction.outer_product(local_direction)
            tensor = projection @ tensor @ projection
            if is_valid:
                status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
    return (
        dyn_state.geoms.pos[i_g, i_b] + rotation @ center,
        rotation @ tensor @ rotation.transpose(),
        i_selected,
        residual,
        status,
    )
