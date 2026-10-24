"""Bilinear physical-surface geometry shared by shell quadrature and anchoring."""

import quadrants as qd

import genesis as gs


@qd.func
def func_quad_shape(xi: float, eta: float):
    shape = (
        qd.Vector(
            [(1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta), (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)]
        )
        * 0.25
    )
    dxi = qd.Vector([eta - 1.0, 1.0 - eta, 1.0 + eta, -1.0 - eta]) * 0.25
    deta = qd.Vector([xi - 1.0, -1.0 - xi, 1.0 + xi, 1.0 - xi]) * 0.25
    return shape, dxi, deta


@qd.func
def func_quad_inverse(
    vertices: qd.types.matrix(4, 3),
    origin: qd.types.vector(3),
    basis: qd.types.matrix(2, 3),
    target: qd.types.vector(2),
):
    """Invert an orthogonal projection; retain the true surface area and position."""
    constant = 0.25 * (vertices[0, :] + vertices[1, :] + vertices[2, :] + vertices[3, :])
    first = 0.25 * (-vertices[0, :] + vertices[1, :] + vertices[2, :] - vertices[3, :])
    second = 0.25 * (-vertices[0, :] - vertices[1, :] + vertices[2, :] + vertices[3, :])
    mixed = 0.25 * (vertices[0, :] - vertices[1, :] + vertices[2, :] - vertices[3, :])
    local = qd.Vector.zero(gs.qd_float, 2)
    regular = True
    scale = qd.max(first.norm(), second.norm(), 1e-30)
    for _ in range(16):
        residual = (
            basis @ (constant + local[0] * first + local[1] * second + local[0] * local[1] * mixed - origin) - target
        )
        jacobian = qd.Matrix.cols([basis @ (first + local[1] * mixed), basis @ (second + local[0] * mixed)])
        if qd.abs(jacobian.determinant()) > 1e-14 * scale * scale:
            local -= jacobian.inverse() @ residual
        else:
            regular = False
    shape, dxi, deta = func_quad_shape(local[0], local[1])
    position = shape @ vertices
    cross = (dxi @ vertices).cross(deta @ vertices)
    projected_first, projected_second = basis @ (dxi @ vertices), basis @ (deta @ vertices)
    projected_area = qd.abs(projected_first[0] * projected_second[1] - projected_first[1] * projected_second[0])
    residual = basis @ (position - origin) - target
    regular = regular and projected_area > 1e-14 * scale * scale and residual.norm() <= 2e-12 * scale
    regular = regular and qd.abs(local[0]) <= 1.0 + 2e-12 and qd.abs(local[1]) <= 1.0 + 2e-12
    ratio = cross.norm() / qd.max(projected_area, 1e-30)
    return shape, position, cross.normalized(), ratio, regular


@qd.func
def func_quad_section(
    vertices: qd.types.matrix(4, 3),
    origin: qd.types.vector(3),
    normal: qd.types.vector(3),
    center: qd.types.vector(3),
    radius: float,
):
    """Bound a true support ball in a regular face projection without dropping surface area."""
    projected = qd.Matrix.zero(gs.qd_float, 4, 3)
    minimum, maximum = gs.qd_float(float("inf")), gs.qd_float(-float("inf"))
    for i in qd.static(range(4)):
        projected[i, :] = vertices[i, :] - (vertices[i, :] - origin).dot(normal) * normal
        height = (vertices[i, :] - center).dot(normal)
        minimum, maximum = qd.min(minimum, height), qd.max(maximum, height)
    # A bilinear face is a convex combination of its corners. This height bound is exact for planar faces.
    height = qd.max(minimum, 0.0) + qd.min(maximum, 0.0)
    squared_radius = radius * radius - height * height
    projection = center - (center - origin).dot(normal) * normal
    closest = projection
    nearest = gs.qd_float(float("inf"))
    inside, contained = True, squared_radius > 0.0
    for i in range(4):
        a, b = projected[i, :], projected[(i + 1) % 4, :]
        edge = b - a
        signed = edge.cross(projection - a).dot(normal) / edge.norm()
        inside = inside and signed >= 0.0
        contained = contained and signed > qd.sqrt(qd.max(0.0, squared_radius))
        fraction = qd.min(1.0, qd.max(0.0, edge.dot(projection - a) / edge.norm_sqr()))
        candidate = a + fraction * edge
        distance = (candidate - projection).norm_sqr()
        if distance < nearest:
            nearest, closest = distance, candidate
    if inside:
        closest, nearest = projection, 0.0
    return closest, squared_radius - nearest, contained
