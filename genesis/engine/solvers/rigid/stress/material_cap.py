"""Native material-direction cap adapter for the frozen CPU shell construction.

This load law is deliberately separate from automatic-radius DPR. Every source
has one body-frame force, integrated on its original material-sphere cap. All
floating preparation, ray intersections and physical nodal couples run in QD.
Scalar CPU coordinates map to vec3 blocks as ``translation=2*node`` and
``rotation=2*node+1``.
"""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .quadrilateral import func_quad_shape


@dataclass(frozen=True)
class MaterialCapGeometry:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    macro_coefficients: qd.Tensor
    face_tables: qd.Tensor
    directors: qd.Tensor
    vertices: qd.Tensor


@dataclass(frozen=True)
class MaterialCapSources:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    direction: qd.Tensor
    radius: qd.Tensor
    force: qd.Tensor
    normal_force: qd.Tensor
    friction: qd.Tensor
    tangent: qd.Tensor
    cap_tangent: qd.Tensor
    side: qd.Tensor
    environment: qd.Tensor
    explicit_force: qd.Tensor
    has_tangent: qd.Tensor
    has_cap_tangent: qd.Tensor


@dataclass(frozen=True)
class MaterialCapRule:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    nodes: qd.Tensor
    radial: qd.Tensor
    weights: qd.Tensor
    valid: qd.Tensor


@dataclass(frozen=True)
class MaterialCapState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    raw_force: qd.Tensor
    center: qd.Tensor
    first: qd.Tensor
    second: qd.Tensor
    force: qd.Tensor
    valid: qd.Tensor
    failed_samples: qd.Tensor
    scanned_samples: qd.Tensor
    resultant_force: qd.Tensor
    resultant_moment: qd.Tensor
    expected_moment: qd.Tensor
    sample_directions: qd.Tensor
    sample_positions: qd.Tensor
    sample_nodes: qd.Tensor
    sample_shape: qd.Tensor
    sample_macro: qd.Tensor
    sample_element: qd.Tensor
    sample_local: qd.Tensor
    sample_valid: qd.Tensor


def allocate_material_cap_sources(n_sources: int) -> MaterialCapSources:
    """Allocate physical inputs; callers must populate every source field."""
    if n_sources < 1:
        raise ValueError("Allocate at least one source slot; use source validity for empty batches.")
    shape = (n_sources,)
    return MaterialCapSources(
        direction=V_VEC(3, dtype=gs.qd_float, shape=shape),
        radius=V(dtype=gs.qd_float, shape=shape),
        force=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal_force=V(dtype=gs.qd_float, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        tangent=V_VEC(3, dtype=gs.qd_float, shape=shape),
        cap_tangent=V_VEC(3, dtype=gs.qd_float, shape=shape),
        side=V(dtype=gs.qd_int, shape=shape),
        environment=V(dtype=gs.qd_int, shape=shape),
        explicit_force=V(dtype=gs.qd_int, shape=shape),
        has_tangent=V(dtype=gs.qd_int, shape=shape),
        has_cap_tangent=V(dtype=gs.qd_int, shape=shape),
    )


def build_material_cap_geometry(
    vertices: qd.Tensor,
    face_tables: qd.Tensor,
    directors: qd.Tensor,
    *,
    ncoarse: int = 3,
    subdiv: int = 4,
    vertex_stride: int = 1,
) -> MaterialCapGeometry:
    """Prepare bilinear coefficients in QD from physical nodes and integer charts.

    ``vertex_stride=2`` accepts the shell model's paired translation/rotation
    vertices. Directors and chart node IDs always index physical shell nodes.
    """
    if ncoarse < 1 or subdiv < 1 or vertex_stride not in (1, 2):
        raise ValueError("Positive chart sizes and vertex_stride 1 or 2 are required.")
    if tuple(face_tables.shape) != (6, ncoarse * subdiv + 1, ncoarse * subdiv + 1):
        raise ValueError("face_tables must contain the six frozen cube charts.")
    geometry = MaterialCapGeometry(
        macro_coefficients=V_MAT(4, 3, dtype=gs.qd_float, shape=(6 * ncoarse * ncoarse,)),
        face_tables=face_tables,
        directors=directors,
        vertices=vertices,
    )
    kernel_material_cap_coefficients(geometry, ncoarse, subdiv, vertex_stride)
    return geometry


class MaterialCap:
    """Reusable current-load preparation, with optional per-sample oracle output."""

    def __init__(
        self,
        order: int,
        n_sources: int,
        n_nodes: int,
        *,
        ncoarse: int = 3,
        subdiv: int = 4,
        thickness: float = 0.0005,
        vertex_stride: int = 1,
        audit: bool = False,
    ):
        if min(order, n_sources, n_nodes, ncoarse, subdiv) < 1:
            raise ValueError("Cap order, source/node counts and chart sizes must be positive.")
        if vertex_stride not in (1, 2) or thickness <= 0.0:
            raise ValueError("A positive thickness and vertex_stride 1 or 2 are required.")
        self.order = order
        self.ncoarse = ncoarse
        self.subdiv = subdiv
        self.thickness = thickness
        self.vertex_stride = vertex_stride
        self.audit = audit
        self.rule = MaterialCapRule(
            nodes=V(dtype=gs.qd_float, shape=(order,)),
            radial=V(dtype=gs.qd_float, shape=(order,)),
            weights=V(dtype=gs.qd_float, shape=(order,)),
            valid=V(dtype=gs.qd_bool, shape=()),
        )
        sample_shape = (n_sources, 4 * order * order) if audit else ()
        source_shape = (n_sources,)
        self.state = MaterialCapState(
            raw_force=V_VEC(3, dtype=gs.qd_float, shape=(2 * n_nodes, n_sources)),
            center=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            first=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            second=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            force=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            valid=V(dtype=gs.qd_bool, shape=source_shape),
            failed_samples=V(dtype=gs.qd_int, shape=source_shape),
            scanned_samples=V(dtype=gs.qd_int, shape=source_shape),
            resultant_force=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            resultant_moment=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            expected_moment=V_VEC(3, dtype=gs.qd_float, shape=source_shape),
            sample_directions=V_VEC(3, dtype=gs.qd_float, shape=sample_shape),
            sample_positions=V_VEC(3, dtype=gs.qd_float, shape=sample_shape),
            sample_nodes=V_VEC(4, dtype=gs.qd_int, shape=sample_shape),
            sample_shape=V_VEC(4, dtype=gs.qd_float, shape=sample_shape),
            sample_macro=V(dtype=gs.qd_int, shape=sample_shape),
            sample_element=V(dtype=gs.qd_int, shape=sample_shape),
            sample_local=V_VEC(2, dtype=gs.qd_float, shape=sample_shape),
            sample_valid=V(dtype=gs.qd_bool, shape=sample_shape),
        )
        kernel_material_cap_rule(self.rule)
        kernel_material_cap_rule_normalize(self.rule)

    def assemble(self, geometry: MaterialCapGeometry, sources: MaterialCapSources) -> MaterialCapState:
        if sources.radius.shape[0] != self.state.valid.shape[0]:
            raise ValueError("Source allocation does not match this material cap workspace.")
        self.state.raw_force.fill(0.0)
        kernel_material_cap_sources(geometry, sources, self.rule, self.state, self.ncoarse, self.subdiv)
        kernel_material_cap_scatter(
            geometry, sources, self.rule, self.state, self.ncoarse, self.subdiv, self.thickness, self.audit
        )
        kernel_material_cap_resultants(geometry, self.state, self.vertex_stride)
        return self.state


@qd.func
def func_material_cap_jacobi(n: int, alpha: float, beta: float, x: float):
    previous = gs.qd_float(1.0)
    value = gs.qd_float(1.0)
    if n >= 1:
        value = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x)
        for j in range(2, n + 1):
            a = 2.0 * j + alpha + beta
            first = (a - 1.0) * (a * (a - 2.0) * x + alpha * alpha - beta * beta)
            second = 2.0 * (j + alpha - 1.0) * (j + beta - 1.0) * a
            denominator = 2.0 * j * (j + alpha + beta) * (a - 2.0)
            previous, value = value, (first * value - second * previous) / denominator
    return value


@qd.kernel
def kernel_material_cap_rule(rule: MaterialCapRule):
    """Gauss-Jacobi(.5,0), generated without host floating numerical work."""
    for i in range(rule.nodes.shape[0]):
        n = rule.nodes.shape[0]
        # k runs from the largest root to the smallest; storage follows SciPy's
        # ascending root order, and therefore the original radial sample order.
        k = n - i
        x = qd.cos(qd.math.pi * (4.0 * k) / (4.0 * n + 3.0))
        for _ in range(24):
            value = func_material_cap_jacobi(n, 0.5, 0.0, x)
            derivative = 0.5 * (n + 1.5) * func_material_cap_jacobi(n - 1, 1.5, 1.0, x)
            x -= value / derivative
        derivative = 0.5 * (n + 1.5) * func_material_cap_jacobi(n - 1, 1.5, 1.0, x)
        rule.nodes[i] = x
        rule.radial[i] = qd.sqrt(0.5 * (x + 1.0))
        rule.weights[i] = 1.0 / ((1.0 - x * x) * derivative * derivative)


@qd.kernel
def kernel_material_cap_rule_normalize(rule: MaterialCapRule):
    for _ in range(1):
        total = gs.qd_float(0.0)
        valid = True
        for i in range(rule.nodes.shape[0]):
            total += rule.weights[i]
            valid = valid and rule.nodes[i] > -1.0 and rule.nodes[i] < 1.0 and rule.weights[i] > 0.0
            if i > 0:
                valid = valid and rule.nodes[i] > rule.nodes[i - 1]
        valid = valid and total > 0.0 and total < qd.math.inf
        rule.valid[None] = valid
        for i in range(rule.nodes.shape[0]):
            rule.weights[i] /= total


@qd.kernel
def kernel_material_cap_coefficients(geometry: MaterialCapGeometry, ncoarse: int, subdiv: int, vertex_stride: int):
    for macro in range(geometry.macro_coefficients.shape[0]):
        face = macro // (ncoarse * ncoarse)
        cell = macro % (ncoarse * ncoarse)
        i, j = (cell % ncoarse) * subdiv, (cell // ncoarse) * subdiv
        a = geometry.vertices[vertex_stride * geometry.face_tables[face, j, i]]
        b = geometry.vertices[vertex_stride * geometry.face_tables[face, j, i + subdiv]]
        c = geometry.vertices[vertex_stride * geometry.face_tables[face, j + subdiv, i + subdiv]]
        d = geometry.vertices[vertex_stride * geometry.face_tables[face, j + subdiv, i]]
        geometry.macro_coefficients[macro] = qd.Matrix.rows(
            [(a + b + c + d) * 0.25, (-a + b + c - d) * 0.25, (-a - b + c + d) * 0.25, (a - b + c - d) * 0.25]
        )


@qd.func
def func_material_cap_unit(vector: qd.types.vector(3)):
    length = vector.norm()
    return vector / qd.max(length, 1e-30), length > 1e-30 and length < qd.math.inf


@qd.func
def func_material_cap_tangent(direction: qd.types.vector(3)):
    axis = 0
    if qd.abs(direction[1]) < qd.abs(direction[axis]):
        axis = 1
    if qd.abs(direction[2]) < qd.abs(direction[axis]):
        axis = 2
    coordinate = qd.Vector.zero(gs.qd_float, 3)
    coordinate[axis] = 1.0
    first, _ = func_material_cap_unit(direction.cross(coordinate))
    return first, direction.cross(first)


@qd.func
def func_material_cap_ray_inverse(coefficients: qd.types.matrix(4, 3), ray: qd.types.vector(3)):
    dominant = 0
    if qd.abs(ray[1]) > qd.abs(ray[dominant]):
        dominant = 1
    if qd.abs(ray[2]) > qd.abs(ray[dominant]):
        dominant = 2
    first, second = 0, 1
    if dominant == 0:
        first, second = 1, 2
    elif dominant == 1:
        first, second = 0, 2
    ray_denominator = ray[dominant]
    if qd.abs(ray_denominator) <= 1e-30:
        ray_denominator = 1e-30
    ratios = qd.Vector([ray[first], ray[second]]) / ray_denominator
    projected = qd.Matrix.zero(gs.qd_float, 4, 2)
    for i in qd.static(range(4)):
        projected[i, 0] = coefficients[i, first] - coefficients[i, dominant] * ratios[0]
        projected[i, 1] = coefficients[i, second] - coefficients[i, dominant] * ratios[1]
    a, b, c, d = projected[0, :], projected[1, :], projected[2, :], projected[3, :]
    qa = b[0] * d[1] - b[1] * d[0]
    qb = a[0] * d[1] + b[0] * c[1] - a[1] * d[0] - b[1] * c[0]
    qc = a[0] * c[1] - a[1] * c[0]
    scale = qd.max(qd.abs(qa), qd.abs(qb), qd.abs(qc))
    linear = qd.abs(qa) <= 1e-13 * qd.max(scale, 1e-30)
    roots = qd.Vector.zero(gs.qd_float, 2)
    root_valid = qd.Vector.zero(gs.qd_int, 2)
    if linear:
        if qd.abs(qb) > 1e-15 * qd.max(scale, 1e-30):
            roots[0], root_valid[0] = -qc / qb, 1
    else:
        discriminant = qb * qb - 4.0 * qa * qc
        if discriminant >= -1e-13 * qd.max(scale * scale, 1e-60):
            radical = qd.sqrt(qd.max(discriminant, 0.0))
            stable = -0.5 * (qb + qd.select(qb >= 0.0, radical, -radical))
            roots[0], root_valid[0] = stable / qa, 1
            if qd.abs(stable) > 1e-30:
                roots[1], root_valid[1] = qc / stable, 1
    local = qd.Vector.zero(gs.qd_float, 2)
    valid = False
    for i_root in qd.static(range(2)):
        xi = roots[i_root]
        denominator = c + d * xi
        component = 0
        if qd.abs(denominator[1]) > qd.abs(denominator[0]):
            component = 1
        if root_valid[i_root] and not valid and qd.abs(denominator[component]) > 1e-30:
            eta = -(a + b * xi)[component] / denominator[component]
            residual = a + b * xi + c * eta + d * (xi * eta)
            physical = (
                coefficients[0, :] + coefficients[1, :] * xi + coefficients[2, :] * eta + coefficients[3, :] * xi * eta
            )
            tx = coefficients[1, :] + coefficients[3, :] * eta
            ty = coefficients[2, :] + coefficients[3, :] * xi
            accept = qd.abs(xi) <= 1.0 + 2e-10 and qd.abs(eta) <= 1.0 + 2e-10
            accept = accept and residual.norm() < 1e-11 and physical.dot(ray) > 0.0 and tx.cross(ty).dot(ray) > 0.0
            if accept:
                local, valid = qd.Vector([xi, eta]), True
    return qd.max(-1.0, qd.min(1.0, local)), valid


@qd.func
def func_material_cap_locate(direction: qd.types.vector(3), geometry: MaterialCapGeometry, ncoarse: int, subdiv: int):
    direction, direction_valid = func_material_cap_unit(direction)
    ray = qd.Vector(
        [
            0.022 * (1.0 - 0.18 * direction[2]) * direction[0],
            0.022 * (1.0 - 0.18 * direction[2]) * direction[1],
            0.030 * direction[2],
        ]
    )
    dominant = 0
    if qd.abs(direction[1]) > qd.abs(direction[dominant]):
        dominant = 1
    if qd.abs(direction[2]) > qd.abs(direction[dominant]):
        dominant = 2
    face = 2 * dominant + qd.cast(direction[dominant] < 0.0, gs.qd_int)
    u, v = gs.qd_float(0.0), gs.qd_float(0.0)
    denominator = qd.max(qd.abs(direction[dominant]), 1e-30)
    if face == 0:
        u, v = direction[1] / denominator, direction[2] / denominator
    elif face == 1:
        u, v = -direction[1] / denominator, direction[2] / denominator
    elif face == 2:
        u, v = -direction[0] / denominator, direction[2] / denominator
    elif face == 3:
        u, v = direction[0] / denominator, direction[2] / denominator
    elif face == 4:
        u, v = direction[0] / denominator, direction[1] / denominator
    else:
        u, v = -direction[0] / denominator, direction[1] / denominator
    ix = qd.max(0, qd.min(ncoarse - 1, qd.cast(qd.floor((u + 1.0) * ncoarse / 2.0), gs.qd_int)))
    iy = qd.max(0, qd.min(ncoarse - 1, qd.cast(qd.floor((v + 1.0) * ncoarse / 2.0), gs.qd_int)))
    macro = face * ncoarse * ncoarse + iy * ncoarse + ix
    local, valid = func_material_cap_ray_inverse(geometry.macro_coefficients[macro], ray)
    seam = qd.abs(qd.abs(u) - 1.0) < 1e-11 or qd.abs(qd.abs(v) - 1.0) < 1e-11
    scanned = not valid or seam
    if scanned:
        valid = False
        for candidate in range(geometry.macro_coefficients.shape[0]):
            candidate_local, accepted = func_material_cap_ray_inverse(geometry.macro_coefficients[candidate], ray)
            if accepted and not valid:
                macro, local, valid = candidate, candidate_local, True
    valid = valid and direction_valid
    fine_x = qd.max(0, qd.min(subdiv - 1, qd.cast(qd.floor((local[0] + 1.0) * subdiv / 2.0), gs.qd_int)))
    fine_y = qd.max(0, qd.min(subdiv - 1, qd.cast(qd.floor((local[1] + 1.0) * subdiv / 2.0), gs.qd_int)))
    face = macro // (ncoarse * ncoarse)
    cell = macro % (ncoarse * ncoarse)
    ix, iy = (cell % ncoarse) * subdiv + fine_x, (cell // ncoarse) * subdiv + fine_y
    n = ncoarse * subdiv
    element = face * n * n + iy * n + ix
    nodes = qd.Vector(
        [
            geometry.face_tables[face, iy, ix],
            geometry.face_tables[face, iy, ix + 1],
            geometry.face_tables[face, iy + 1, ix + 1],
            geometry.face_tables[face, iy + 1, ix],
        ]
    )
    sublocal = (local + 1.0) * subdiv - 2.0 * qd.Vector([fine_x, fine_y]) - 1.0
    shape, _shape_x, _shape_y = func_quad_shape(sublocal[0], sublocal[1])
    coefficient = geometry.macro_coefficients[macro]
    position = (
        coefficient[0, :]
        + coefficient[1, :] * local[0]
        + coefficient[2, :] * local[1]
        + coefficient[3, :] * local[0] * local[1]
    )
    normal, normal_valid = func_material_cap_unit(
        (coefficient[1, :] + coefficient[3, :] * local[1]).cross(coefficient[2, :] + coefficient[3, :] * local[0])
    )
    return nodes, shape, position, normal, macro, element, local, valid and normal_valid, scanned


@qd.kernel
def kernel_material_cap_sources(
    geometry: MaterialCapGeometry,
    sources: MaterialCapSources,
    rule: MaterialCapRule,
    state: MaterialCapState,
    ncoarse: int,
    subdiv: int,
):
    for source in range(sources.radius.shape[0]):
        center, valid = func_material_cap_unit(sources.direction[source])
        first, second = func_material_cap_tangent(center)
        if sources.has_cap_tangent[source]:
            vector = sources.cap_tangent[source]
            first, tangent_valid = func_material_cap_unit(vector - vector.dot(center) * center)
            second = center.cross(first)
            valid = valid and tangent_valid
        force = sources.force[source]
        if not sources.explicit_force[source]:
            _nodes, _shape, _position, normal, _macro, _element, _local, located, _scanned = func_material_cap_locate(
                center, geometry, ncoarse, subdiv
            )
            normal *= sources.side[source]
            tangent, _normal_second = func_material_cap_tangent(normal)
            if sources.has_tangent[source]:
                vector = sources.tangent[source]
                tangent, tangent_valid = func_material_cap_unit(vector - vector.dot(normal) * normal)
                valid = valid and tangent_valid
            force = sources.normal_force[source] * (-normal + sources.friction[source] * tangent)
            valid = valid and located
        valid = valid and rule.valid[None] and sources.radius[source] >= 0.0 and sources.radius[source] < qd.math.inf
        valid = valid and (sources.side[source] == 1 or sources.side[source] == -1)
        valid = valid and sources.normal_force[source] >= 0.0 and sources.friction[source] >= 0.0
        valid = valid and qd.abs(force).max() < qd.math.inf
        state.center[source], state.first[source], state.second[source] = center, first, second
        state.force[source], state.valid[source] = force, valid
        state.failed_samples[source], state.scanned_samples[source] = 0, 0
        state.expected_moment[source] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_material_cap_scatter(
    geometry: MaterialCapGeometry,
    sources: MaterialCapSources,
    rule: MaterialCapRule,
    state: MaterialCapState,
    ncoarse: int,
    subdiv: int,
    thickness: float,
    audit: qd.template(),
):
    for source, sample in qd.ndrange(sources.radius.shape[0], 4 * rule.nodes.shape[0] * rule.nodes.shape[0]):
        n = rule.nodes.shape[0]
        if qd.static(audit):
            state.sample_valid[source, sample] = False
        if state.valid[source]:
            radial, angular = sample // (4 * n), sample % (4 * n)
            theta = sources.radius[source] / 0.022 * rule.radial[radial]
            phi = 2.0 * qd.math.pi * (angular + 0.5) / (4 * n)
            direction = qd.cos(theta) * state.center[source] + qd.sin(theta) * (
                qd.cos(phi) * state.first[source] + qd.sin(phi) * state.second[source]
            )
            nodes, shape, position, _, macro, element, local, valid, scanned = func_material_cap_locate(
                direction, geometry, ncoarse, subdiv
            )
            state.scanned_samples[source] += qd.cast(scanned, gs.qd_int)
            if valid:
                weight = rule.weights[radial] / (4 * n)
                interpolated_director = qd.Vector.zero(gs.qd_float, 3)
                for corner in qd.static(range(4)):
                    node = nodes[corner]
                    force = weight * shape[corner] * state.force[source]
                    moment = sources.side[source] * 0.5 * thickness * geometry.directors[node].cross(force)
                    state.raw_force[2 * node, source] += force
                    state.raw_force[2 * node + 1, source] += moment
                    interpolated_director += shape[corner] * geometry.directors[node]
                physical = position + sources.side[source] * 0.5 * thickness * interpolated_director
                state.expected_moment[source] += weight * physical.cross(state.force[source])
            else:
                state.failed_samples[source] += 1
            if qd.static(audit):
                state.sample_directions[source, sample] = direction
                state.sample_positions[source, sample] = position
                state.sample_nodes[source, sample], state.sample_shape[source, sample] = nodes, shape
                state.sample_macro[source, sample], state.sample_element[source, sample] = macro, element
                state.sample_local[source, sample], state.sample_valid[source, sample] = local, valid


@qd.kernel
def kernel_material_cap_resultants(geometry: MaterialCapGeometry, state: MaterialCapState, vertex_stride: int):
    for source in range(state.valid.shape[0]):
        force, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        for node in range(geometry.directors.shape[0]):
            nodal = state.raw_force[2 * node, source]
            force += nodal
            moment += geometry.vertices[vertex_stride * node].cross(nodal) + state.raw_force[2 * node + 1, source]
        state.resultant_force[source], state.resultant_moment[source] = force, moment
        state.valid[source] = state.valid[source] and state.failed_samples[source] == 0
