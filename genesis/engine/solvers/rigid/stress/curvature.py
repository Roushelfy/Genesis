"""Current primitive curvature and cached smooth-normal mesh estimates."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom
from genesis.utils.array_class import DataKind, V, V_MAT, V_VEC


class CurvatureStatus(IntEnum):
    ANALYTIC = 0
    SMOOTH_NORMAL_ESTIMATE = 1
    AMBIGUOUS = 2
    UNSUPPORTED = 3


@dataclass(frozen=True)
class StressCurvatureInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    normal: qd.Tensor
    tensor: qd.Tensor
    normal_spread: qd.Tensor
    antisymmetric_relative: qd.Tensor
    status: qd.Tensor
    vertices_tensor: qd.Tensor
    vertices_weight: qd.Tensor
    vertices_quadric_valid: qd.Tensor
    normal_inverse: qd.Tensor
    normal_inverse_valid: qd.Tensor
    face_origin: qd.Tensor
    face_edges: qd.Tensor
    face_dual: qd.Tensor
    face_valid: qd.Tensor
    face_min: qd.Tensor
    face_max: qd.Tensor
    face_scale: qd.Tensor
    search_min: qd.Tensor
    search_max: qd.Tensor
    search_scale: qd.Tensor
    search_normal_min: qd.Tensor
    search_normal_max: qd.Tensor


@dataclass(frozen=True)
class StressCurvatureFit:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    scale: qd.Tensor
    count: qd.Tensor
    gram: qd.Tensor
    rhs: qd.Tensor


class StressCurvature:
    """Cache signed shape operators in each geom's immutable local frame.

    Primitive curvature is analytical away from edges/seams. Mesh curvature fits
    a local implicit quadric to adjacent vertices and their supplied normals.
    Rank-deficient fits retain the face-normal derivative estimate. It explicitly
    denotes a smooth-surface estimate: interpolation normals do not prove that
    a faceted collision mesh represents the same physical smooth surface.
    Positive principal values describe an outward convex material boundary;
    correctly oriented cavity normals retain negative curvature.
    """

    def __init__(self, dyn_info: array_class.DynInfo, epsilon: float):
        n_faces = dyn_info.faces.verts_idx.shape[0]
        self.info = StressCurvatureInfo(
            normal=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_faces,)),
            normal_spread=V(dtype=gs.qd_float, shape=(n_faces,)),
            antisymmetric_relative=V(dtype=gs.qd_float, shape=(n_faces,)),
            status=V(dtype=gs.qd_int, shape=(n_faces,)),
            vertices_tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=dyn_info.verts.geom_idx.shape),
            vertices_weight=V(dtype=gs.qd_float, shape=dyn_info.verts.geom_idx.shape),
            vertices_quadric_valid=V(dtype=gs.qd_bool, shape=dyn_info.verts.geom_idx.shape),
            normal_inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_faces,)),
            normal_inverse_valid=V(dtype=gs.qd_bool, shape=(n_faces,)),
            face_origin=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_edges=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_faces,)),
            face_dual=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_faces,)),
            face_valid=V(dtype=gs.qd_bool, shape=(n_faces,)),
            face_min=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_max=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_scale=V(dtype=gs.qd_float, shape=(n_faces,)),
            search_min=V_VEC(3, dtype=gs.qd_float, shape=((n_faces + 63) // 64,)),
            search_max=V_VEC(3, dtype=gs.qd_float, shape=((n_faces + 63) // 64,)),
            search_scale=V(dtype=gs.qd_float, shape=((n_faces + 63) // 64,)),
            search_normal_min=V_VEC(3, dtype=gs.qd_float, shape=((n_faces + 63) // 64,)),
            search_normal_max=V_VEC(3, dtype=gs.qd_float, shape=((n_faces + 63) // 64,)),
        )
        kernel_mesh_curvature(dyn_info, self.info, epsilon)
        kernel_mesh_search_bounds(dyn_info, self.info, epsilon)
        kernel_vertex_curvature(dyn_info, self.info)
        fit = StressCurvatureFit(
            scale=V(dtype=gs.qd_float, shape=dyn_info.verts.geom_idx.shape),
            count=V(dtype=gs.qd_int, shape=dyn_info.verts.geom_idx.shape),
            gram=V_MAT(6, 6, dtype=gs.qd_float, shape=dyn_info.verts.geom_idx.shape),
            rhs=V_VEC(6, dtype=gs.qd_float, shape=dyn_info.verts.geom_idx.shape),
        )
        kernel_quadric_scale(dyn_info, self.info, fit)
        kernel_quadric_fit(dyn_info, self.info, fit)
        kernel_quadric_curvature(dyn_info, self.info, fit, epsilon)


@qd.kernel
def kernel_quadric_scale(dyn_info: array_class.DynInfo, curvature: StressCurvatureInfo, fit: StressCurvatureFit):
    for i_v in range(fit.scale.shape[0]):
        fit.scale[i_v], fit.count[i_v] = 0.0, 0
        fit.gram[i_v] = qd.Matrix.zero(gs.qd_float, 6, 6)
        fit.rhs[i_v] = qd.Vector.zero(gs.qd_float, 6)
    for i_f in range(curvature.status.shape[0]):
        if curvature.status[i_f] == CurvatureStatus.SMOOTH_NORMAL_ESTIMATE:
            indices = dyn_info.faces.verts_idx[i_f]
            for i, j in qd.static(qd.ndrange(3, 3)):
                if qd.static(i != j):
                    edge = dyn_info.verts.init_pos[indices[j]] - dyn_info.verts.init_pos[indices[i]]
                    qd.atomic_add(fit.scale[indices[i]], edge.norm())
                    qd.atomic_add(fit.count[indices[i]], 1)


@qd.kernel
def kernel_quadric_fit(dyn_info: array_class.DynInfo, curvature: StressCurvatureInfo, fit: StressCurvatureFit):
    for i_f, i, j in qd.ndrange(curvature.status.shape[0], 3, 3):
        if i != j and curvature.status[i_f] == CurvatureStatus.SMOOTH_NORMAL_ESTIMATE:
            indices = dyn_info.faces.verts_idx[i_f]
            i_v, j_v = indices[i], indices[j]
            scale = fit.scale[i_v] / fit.count[i_v]
            edge = (dyn_info.verts.init_pos[j_v] - dyn_info.verts.init_pos[i_v]) / scale
            normal, other = dyn_info.verts.init_normal[i_v].normalized(), dyn_info.verts.init_normal[j_v].normalized()
            projection = qd.Matrix.identity(gs.qd_float, 3) - other.outer_product(other)
            derivative = qd.Matrix(
                [
                    [edge[0], 0.0, 0.0, edge[1], edge[2], 0.0],
                    [0.0, edge[1], 0.0, edge[0], 0.0, edge[2]],
                    [0.0, 0.0, edge[2], 0.0, edge[0], edge[1]],
                ]
            )
            value = derivative.transpose() @ edge / (2 * edge.norm())
            normal_rows = projection @ derivative
            # A zero implicit level and parallel gradients constrain the same local quadric
            gram = value.outer_product(value) + normal_rows.transpose() @ normal_rows
            rhs = -value * normal.dot(edge) / edge.norm() - normal_rows.transpose() @ (projection @ normal)
            for a, b in qd.static(qd.ndrange(6, 6)):
                qd.atomic_add(fit.gram[i_v][a, b], gram[a, b])
            for a in qd.static(range(6)):
                qd.atomic_add(fit.rhs[i_v][a], rhs[a])


@qd.kernel
def kernel_quadric_curvature(
    dyn_info: array_class.DynInfo, curvature: StressCurvatureInfo, fit: StressCurvatureFit, epsilon: float
):
    for i_v in range(fit.scale.shape[0]):
        valid = fit.count[i_v] > 0 and fit.scale[i_v] > 0.0
        lower, gram = qd.Matrix.zero(gs.qd_float, 6, 6), fit.gram[i_v]
        diagonal = qd.Vector.zero(gs.qd_float, 6)
        for i in qd.static(range(6)):
            valid = valid and gram[i, i] > 0.0
            diagonal[i] = qd.sqrt(qd.max(gram[i, i], epsilon))
        if valid:
            for i in range(6):
                for j in range(i + 1):
                    value = gram[i, j] / (diagonal[i] * diagonal[j])
                    for k in range(j):
                        value -= lower[i, k] * lower[j, k]
                    if i == j:
                        valid = valid and value > 128.0 * epsilon
                        lower[i, j] = qd.sqrt(qd.max(value, epsilon))
                    else:
                        lower[i, j] = value / lower[j, j]
        if valid:
            coefficients, rhs = qd.Vector.zero(gs.qd_float, 6), fit.rhs[i_v]
            for i in range(6):
                value = rhs[i] / diagonal[i]
                for j in range(i):
                    value -= lower[i, j] * coefficients[j]
                coefficients[i] = value / lower[i, i]
            for k in range(6):
                i = 5 - k
                value = coefficients[i]
                for j in range(i + 1, 6):
                    value -= lower[j, i] * coefficients[j]
                coefficients[i] = value / lower[i, i]
            coefficients /= diagonal * (fit.scale[i_v] / fit.count[i_v])
            tensor = qd.Matrix(
                [
                    [coefficients[0], coefficients[3], coefficients[4]],
                    [coefficients[3], coefficients[1], coefficients[5]],
                    [coefficients[4], coefficients[5], coefficients[2]],
                ]
            )
            normal = dyn_info.verts.init_normal[i_v].normalized()
            projection = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
            tensor = projection @ tensor @ projection
            valid = tensor.norm() < float("inf")
            if valid:
                curvature.vertices_tensor[i_v] = tensor
        curvature.vertices_quadric_valid[i_v] = valid


@qd.kernel
def kernel_mesh_curvature(dyn_info: array_class.DynInfo, curvature_info: StressCurvatureInfo, epsilon: float):
    for i_f in range(curvature_info.status.shape[0]):
        indices = dyn_info.faces.verts_idx[i_f]
        i_a, i_b, i_c = indices[0], indices[1], indices[2]
        first = dyn_info.verts.init_pos[i_a]
        edges = qd.Matrix.cols([dyn_info.verts.init_pos[i_b] - first, dyn_info.verts.init_pos[i_c] - first])
        geometric_gram = edges.transpose() @ edges
        curvature_info.face_min[i_f] = qd.min(first, qd.min(dyn_info.verts.init_pos[i_b], dyn_info.verts.init_pos[i_c]))
        curvature_info.face_max[i_f] = qd.max(first, qd.max(dyn_info.verts.init_pos[i_b], dyn_info.verts.init_pos[i_c]))
        curvature_info.face_scale[i_f] = (
            curvature_info.face_min[i_f].norm_sqr() + curvature_info.face_max[i_f].norm_sqr()
        )
        curvature_info.face_origin[i_f] = first
        curvature_info.face_edges[i_f] = edges.transpose()
        curvature_info.face_dual[i_f] = qd.Matrix.zero(gs.qd_float, 2, 3)
        curvature_info.face_valid[i_f] = edges[:, 0].cross(edges[:, 1]).norm() > 0.0
        if curvature_info.face_valid[i_f]:
            curvature_info.face_dual[i_f] = geometric_gram.inverse() @ edges.transpose()
        normals = qd.Matrix.rows(
            [dyn_info.verts.init_normal[i_a], dyn_info.verts.init_normal[i_b], dyn_info.verts.init_normal[i_c]]
        )
        is_invertible = qd.abs(normals.determinant()) > 128.0 * epsilon * normals.norm() ** 3
        curvature_info.normal_inverse_valid[i_f] = is_invertible
        curvature_info.normal_inverse[i_f] = qd.Matrix.zero(gs.qd_float, 3, 3)
        if is_invertible:
            curvature_info.normal_inverse[i_f] = normals.transpose().inverse()
        is_valid = True
        for i in qd.static(range(3)):
            length = normals[i, :].norm()
            is_valid = is_valid and length > 0.0 and length < float("inf")
            if length > 0.0:
                normals[i, :] /= length
        mean_normal = (normals[0, :] + normals[1, :] + normals[2, :]) / 3.0
        is_valid = is_valid and mean_normal.norm() > 128.0 * epsilon
        normal = mean_normal.normalized()
        projection = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
        tangent_edges = projection @ edges
        gram = tangent_edges.transpose() @ tangent_edges
        is_valid = is_valid and gram.determinant() > 128.0 * epsilon * gram.trace() ** 2
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        spread, asymmetry = gs.qd_float(0.0), gs.qd_float(0.0)
        status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
        if is_valid:
            normal_change = qd.Matrix.cols([normals[1, :] - normals[0, :], normals[2, :] - normals[0, :]])
            derivative = projection @ normal_change @ gram.inverse() @ tangent_edges.transpose()
            tensor = 0.5 * (derivative + derivative.transpose())
            scale = derivative.norm()
            if scale > 0.0:
                asymmetry = (derivative - derivative.transpose()).norm() / scale
            for i in qd.static(range(3)):
                spread = qd.max(spread, (normals[i, :] - normal).norm())
            status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
        curvature_info.normal[i_f] = normal
        curvature_info.tensor[i_f] = tensor
        curvature_info.normal_spread[i_f] = spread
        curvature_info.antisymmetric_relative[i_f] = asymmetry
        curvature_info.status[i_f] = status


@qd.kernel
def kernel_mesh_search_bounds(dyn_info: array_class.DynInfo, curvature_info: StressCurvatureInfo, epsilon: float):
    """Bound consecutive face blocks without changing the original ordered surface selection.

    A positive combination of vertex normals stays in their spherical cap when its cosine is positive. The cap's
    Euclidean radius bounds every normalized interpolation, including all roots of the piecewise normal field.
    Conditioning-scaled padding covers the existing barycentric tolerance and inverse/rotation roundoff. Wide or
    ill-conditioned caps conservatively retain the whole sphere. Position bounds also preserve disconnected components.
    """
    for i_block in range(curvature_info.search_scale.shape[0]):
        lower = qd.Vector([float("inf")] * 3)
        upper = qd.Vector([-float("inf")] * 3)
        normal_lower = qd.Vector([float("inf")] * 3)
        normal_upper = qd.Vector([-float("inf")] * 3)
        scale = gs.qd_float(0.0)
        for i_f in range(64 * i_block, qd.min(64 * (i_block + 1), curvature_info.face_scale.shape[0])):
            lower = qd.min(lower, curvature_info.face_min[i_f])
            upper = qd.max(upper, curvature_info.face_max[i_f])
            scale = qd.max(scale, curvature_info.face_scale[i_f])
            if curvature_info.normal_inverse_valid[i_f]:
                indices = dyn_info.faces.verts_idx[i_f]
                normals = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i in qd.static(range(3)):
                    i_v = indices[i]
                    normals[i, :] = dyn_info.verts.init_normal[i_v]
                condition = curvature_info.normal_inverse[i_f].norm() * normals.norm()
                padding = 4096.0 * epsilon * qd.max(1.0, condition)
                center = qd.Vector.zero(gs.qd_float, 3)
                for i in qd.static(range(3)):
                    normals[i, :] = normals[i, :].normalized()
                    center += normals[i, :]
                cosine = gs.qd_float(-1.0)
                if center.norm() > 128.0 * epsilon:
                    center = center.normalized()
                    cosine = qd.min(
                        center.dot(normals[0, :]), qd.min(center.dot(normals[1, :]), center.dot(normals[2, :]))
                    )
                cap_lower = qd.Vector([-1.0 - padding] * 3)
                cap_upper = qd.Vector([1.0 + padding] * 3)
                if cosine > 0.0:
                    radius = qd.sqrt(qd.max(0.0, 2.0 * (1.0 - cosine))) + padding
                    cap_lower = center - radius
                    cap_upper = center + radius
                normal_lower = qd.min(normal_lower, cap_lower)
                normal_upper = qd.max(normal_upper, cap_upper)
        curvature_info.search_min[i_block], curvature_info.search_max[i_block] = lower, upper
        curvature_info.search_scale[i_block] = scale
        curvature_info.search_normal_min[i_block], curvature_info.search_normal_max[i_block] = (
            normal_lower,
            normal_upper,
        )


@qd.kernel
def kernel_vertex_curvature(dyn_info: array_class.DynInfo, curvature_info: StressCurvatureInfo):
    for i_v in range(curvature_info.vertices_weight.shape[0]):
        curvature_info.vertices_tensor[i_v] = qd.Matrix.zero(gs.qd_float, 3, 3)
        curvature_info.vertices_weight[i_v] = 0.0
    for i_f in range(curvature_info.status.shape[0]):
        if curvature_info.status[i_f] == CurvatureStatus.SMOOTH_NORMAL_ESTIMATE:
            indices = dyn_info.faces.verts_idx[i_f]
            i_a, i_b, i_c = indices[0], indices[1], indices[2]
            first = dyn_info.verts.init_pos[i_a]
            weight = (dyn_info.verts.init_pos[i_b] - first).cross(dyn_info.verts.init_pos[i_c] - first).norm()
            for i in qd.static(range(3)):
                i_v = indices[i]
                qd.atomic_add(curvature_info.vertices_weight[i_v], weight)
                for j, k in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(curvature_info.vertices_tensor[i_v][j, k], weight * curvature_info.tensor[i_f][j, k])
    for i_v in range(curvature_info.vertices_weight.shape[0]):
        weight = curvature_info.vertices_weight[i_v]
        if weight > 0.0:
            normal = dyn_info.verts.init_normal[i_v].normalized()
            projection = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
            curvature_info.vertices_tensor[i_v] = projection @ curvature_info.vertices_tensor[i_v] @ projection / weight


@qd.func
def func_surface_curvature(
    i_g: int,
    i_f: int,
    i_b: int,
    position: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    """Evaluate at an actual boundary point; retain analytical/estimated provenance."""
    local = geom.qd_inv_transform_by_trans_quat(position, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])
    kind = dyn_info.geoms.type[i_g]
    normal = qd.Vector.zero(gs.qd_float, 3)
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
    if kind == gs.GEOM_TYPE.MESH:
        if i_f >= dyn_info.geoms.face_start[i_g] and i_f < dyn_info.geoms.face_end[i_g]:
            indices = dyn_info.faces.verts_idx[i_f]
            i_a, i_c, i_d = indices[0], indices[1], indices[2]
            first = dyn_info.verts.init_pos[i_a]
            edges = qd.Matrix.rows([dyn_info.verts.init_pos[i_c] - first, dyn_info.verts.init_pos[i_d] - first])
            local_coordinates = (edges @ edges.transpose()).inverse() @ (edges @ (local - first))
            barycentric = qd.Vector([1.0 - local_coordinates.sum(), local_coordinates[0], local_coordinates[1]])
            is_valid = True
            for i in qd.static(range(3)):
                i_v = indices[i]
                normal += barycentric[i] * dyn_info.verts.init_normal[i_v]
                tensor += barycentric[i] * curvature_info.vertices_tensor[i_v]
                is_valid = is_valid and curvature_info.vertices_weight[i_v] > 0.0
            if is_valid and normal.norm() > 128.0 * epsilon:
                normal = normal.normalized()
                projection = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
                tensor = projection @ tensor @ projection
                status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
    elif kind == gs.GEOM_TYPE.PLANE:
        normal = qd.Vector([dyn_info.geoms.data[i_g][i] for i in qd.static(range(3))]).normalized()
        status = CurvatureStatus.ANALYTIC
    elif kind == gs.GEOM_TYPE.BOX:
        half = 0.5 * qd.Vector([dyn_info.geoms.data[i_g][i] for i in qd.static(range(3))])
        tolerance = 128.0 * epsilon * half.norm()
        count = 0
        for i in qd.static(range(3)):
            if qd.abs(qd.abs(local[i]) - half[i]) <= tolerance:
                normal[i] = qd.select(local[i] >= 0.0, 1.0, -1.0)
                count += 1
        status = CurvatureStatus.AMBIGUOUS
        if count == 1:
            status = CurvatureStatus.ANALYTIC
    elif kind == gs.GEOM_TYPE.SPHERE or kind == gs.GEOM_TYPE.CAPSULE or kind == gs.GEOM_TYPE.CYLINDER:
        radius = dyn_info.geoms.data[i_g][0]
        half = 0.5 * dyn_info.geoms.data[i_g][1]
        tolerance = 128.0 * epsilon * qd.max(radius, half)
        center = qd.Vector.zero(gs.qd_float, 3)
        if kind == gs.GEOM_TYPE.CAPSULE:
            center[2] = qd.min(half, qd.max(-half, local[2]))
        radial = local - center
        status = CurvatureStatus.AMBIGUOUS
        is_spherical = kind == gs.GEOM_TYPE.SPHERE or (
            kind == gs.GEOM_TYPE.CAPSULE and qd.abs(local[2]) > half + tolerance
        )
        if is_spherical and radial.norm() > tolerance:
            normal = radial.normalized()
            tensor = (qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)) / radius
            status = CurvatureStatus.ANALYTIC
        elif kind != gs.GEOM_TYPE.SPHERE and qd.abs(local[2]) < half - tolerance:
            radial[2] = 0.0
            if radial.norm() > tolerance:
                normal = radial.normalized()
                tangent = qd.Vector([-normal[1], normal[0], 0.0])
                tensor = tangent.outer_product(tangent) / radius
                status = CurvatureStatus.ANALYTIC
        elif kind == gs.GEOM_TYPE.CYLINDER:
            radial[2] = 0.0
            if radial.norm() < radius - tolerance:
                normal[2] = qd.select(local[2] >= 0.0, 1.0, -1.0)
                status = CurvatureStatus.ANALYTIC
    rotation = geom.qd_quat_to_R(dyn_state.geoms.quat[i_g, i_b], epsilon)
    return rotation @ normal, rotation @ tensor @ rotation.transpose(), status


@qd.func
def func_closest_surface(
    i_g: int,
    i_b: int,
    position: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    """Find the nearest mesh face using shared rest-shape barycentric maps.

    Face order and the nearest-distance tolerance retain the existing surface query's choice. Conservative box bounds
    skip faces that cannot improve the ordered selection, including on disconnected and nonconvex components.
    """
    local = geom.qd_inv_transform_by_trans_quat(position, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])
    tolerance = 128.0 * epsilon * (dyn_state.geoms.aabb_max[i_g, i_b] - dyn_state.geoms.aabb_min[i_g, i_b]).norm()
    minimum = gs.qd_float(float("inf"))
    selected = -1
    closest = qd.Vector.zero(gs.qd_float, 3)
    local_scale = local.norm_sqr()
    i_start, i_end = dyn_info.geoms.face_start[i_g], dyn_info.geoms.face_end[i_g]
    for i_block in range(i_start // 64, (i_end + 63) // 64):
        block_lower = qd.max(
            curvature_info.search_min[i_block] - local, qd.max(local - curvature_info.search_max[i_block], 0.0)
        )
        block_slack = 128.0 * epsilon * (local_scale + curvature_info.search_scale[i_block] + minimum * minimum)
        if block_lower.norm_sqr() <= minimum * minimum + block_slack:
            for i_f in range(qd.max(i_start, 64 * i_block), qd.min(i_end, 64 * (i_block + 1))):
                lower = qd.max(curvature_info.face_min[i_f] - local, qd.max(local - curvature_info.face_max[i_f], 0.0))
                squared_minimum = minimum * minimum
                slack = 128.0 * epsilon * (local_scale + curvature_info.face_scale[i_f] + squared_minimum)
                if curvature_info.face_valid[i_f] and lower.norm_sqr() <= squared_minimum + slack:
                    origin = curvature_info.face_origin[i_f]
                    edges = curvature_info.face_edges[i_f]
                    coordinates = curvature_info.face_dual[i_f] @ (local - origin)
                    candidate = origin + coordinates @ edges
                    if coordinates.min() < 0.0 or coordinates.sum() > 1.0:
                        squared = gs.qd_float(float("inf"))
                        vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
                        for i in qd.static(range(3)):
                            j = (i + 1) % 3
                            edge = vertices[j, :] - vertices[i, :]
                            fraction = qd.min(1.0, qd.max(0.0, (local - vertices[i, :]).dot(edge) / edge.norm_sqr()))
                            projection = vertices[i, :] + fraction * edge
                            distance = (local - projection).norm_sqr()
                            if distance < squared:
                                squared, candidate = distance, projection
                    distance = (local - candidate).norm()
                    if distance < minimum - tolerance:
                        minimum, closest, selected = distance, candidate, i_f
    return (
        geom.qd_transform_by_trans_quat(closest, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]),
        selected,
    )
