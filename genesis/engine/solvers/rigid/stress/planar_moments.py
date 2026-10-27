"""Integrate spherical Hertz pressure through planar triangle displacement maps.

Gauss nodes and weights use the unit interval. Triangle parent shapes map its
three vertices into the four original quadrilateral displacement coordinates.
"""

import quadrants as qd

import genesis as gs

from .contact import func_force_frame


@qd.func
def func_planar_section(
    origin: qd.types.vector(3),
    edges: qd.types.matrix(2, 3),
    dual: qd.types.matrix(2, 3),
    normal: qd.types.vector(3),
    center: qd.types.vector(3),
    radius: float,
):
    """Return the plane section, full-disk containment and triangle intersection."""
    height = normal.dot(center - origin)
    section_center = center - height * normal
    remaining = radius * radius - height * height
    local = dual @ (section_center - origin)
    barycentric = qd.Vector([1.0 - local.sum(), local[0], local[1]])
    gradients = qd.Matrix.rows([-dual[0, :] - dual[1, :], dual[0, :], dual[1, :]])
    is_inside = True
    is_contained = remaining > 0.0
    for i in qd.static(range(3)):
        is_inside = is_inside and barycentric[i] >= 0.0
        is_contained = is_contained and barycentric[i] >= (
            qd.sqrt(qd.max(remaining, 0.0)) * gradients[i, :].norm() * (1.0 + 2e-12)
        )
    vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
    distance_squared = gs.qd_float(float("inf"))
    for i in qd.static(range(3)):
        j = (i + 1) % 3
        edge = vertices[j, :] - vertices[i, :]
        if edge.norm_sqr() > 0.0:
            parameter = qd.max(0.0, qd.min(1.0, (section_center - vertices[i, :]).dot(edge) / edge.norm_sqr()))
            displacement = vertices[i, :] + parameter * edge - section_center
            distance_squared = qd.min(distance_squared, displacement.norm_sqr())
    is_intersecting = remaining > 0.0 and (is_inside or distance_squared < remaining)
    return section_center, remaining, is_contained, is_intersecting


@qd.func
def func_planar_hertz_moments(
    origin: qd.types.vector(3),
    edges: qd.types.matrix(2, 3),
    dual: qd.types.matrix(2, 3),
    normal: qd.types.vector(3),
    parent_shapes: qd.types.matrix(3, 4),
    center: qd.types.vector(3),
    original_position: qd.types.vector(3),
    radius: float,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
):
    """Return exact Gram and parent-load moments on a contained spherical section."""
    section_center, remaining, is_contained, _is_intersecting = func_planar_section(
        origin, edges, dual, normal, center, radius
    )
    gram = qd.Matrix.zero(gs.qd_float, 3, 3)
    load = qd.Matrix.zero(gs.qd_float, 4, 3)
    mass = gs.qd_float(0.0)
    if is_contained and radius > 0.0:
        covariance = remaining / 5.0 * (qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal))
        mass = 2.0 * qd.math.pi * remaining * qd.sqrt(remaining) / (3.0 * radius)
        offset = (section_center - original_position) / radius
        mean = qd.Vector([1.0, offset.dot(first), offset.dot(second)])
        directions = qd.Matrix.rows([qd.Vector.zero(gs.qd_float, 3), first / radius, second / radius])
        gram = mass * (mean.outer_product(mean) + directions @ covariance @ directions.transpose())
        local = dual @ (section_center - origin)
        barycentric = qd.Vector([1.0 - local.sum(), local[0], local[1]])
        gradients = qd.Matrix.rows([-dual[0, :] - dual[1, :], dual[0, :], dual[1, :]])
        parent_mean = parent_shapes.transpose() @ barycentric
        parent_gradients = parent_shapes.transpose() @ gradients
        load = mass * (parent_mean.outer_product(mean) + parent_gradients @ covariance @ directions.transpose())
    return gram, load, mass, is_contained


@qd.func
def func_planar_affine_minimum(
    section_center: qd.types.vector(3),
    normal: qd.types.vector(3),
    original_position: qd.types.vector(3),
    remaining: float,
    radius: float,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    coefficient: qd.types.vector(3),
):
    """Return the affine pressure minimum over the complete section disk."""
    minimum = gs.qd_float(-float("inf"))
    if remaining > 0.0 and radius > 0.0:
        offset = (section_center - original_position) / radius
        mean = qd.Vector([1.0, offset.dot(first), offset.dot(second)])
        direction = first * coefficient[1] + second * coefficient[2]
        tangent = direction - normal * normal.dot(direction)
        minimum = mean.dot(coefficient) - qd.sqrt(remaining) / radius * tangent.norm()
    return minimum


@qd.func
def _func_planar_angle(point: qd.types.vector(2)):
    angle = qd.atan2(point[1], point[0])
    if angle < 0.0:
        angle += 2.0 * qd.math.pi
    return qd.round(angle * 1e14) * 1e-14


@qd.func
def func_planar_clip_breakpoints(polygon: qd.types.matrix(3, 2), section_radius: float):
    """Split polar angles at triangle vertices and circle-edge intersections.

    A triangle supplies three vertex angles, six circle crossings and the two
    interval endpoints. The returned count selects the sorted distinct prefix.
    """
    angles = qd.Vector.zero(gs.qd_float, 11)
    angles[1] = qd.round(2.0 * qd.math.pi * 1e14) * 1e-14
    count = 2
    for i in qd.static(range(3)):
        point = polygon[i, :]
        if point.norm() > 1e-18:
            angles[count] = _func_planar_angle(point)
            count += 1
        j = (i + 1) % 3
        edge = polygon[j, :] - point
        squared = edge.norm_sqr()
        if squared > 0.0:
            linear = 2.0 * point.dot(edge)
            constant = point.norm_sqr() - section_radius * section_radius
            discriminant = linear * linear - 4.0 * squared * constant
            if discriminant > 0.0:
                root = qd.sqrt(discriminant)
                for i_root in qd.static(range(2)):
                    sign = qd.select(i_root == 0, -1.0, 1.0)
                    parameter = (-linear + sign * root) / (2.0 * squared)
                    if parameter > 0.0 and parameter < 1.0:
                        intersection = point + parameter * edge
                        angles[count] = _func_planar_angle(intersection)
                        count += 1
    for i in range(1, count):
        angle = angles[i]
        is_moving = True
        for j_ in range(i):
            j = i - j_ - 1
            if is_moving and angle < angles[j]:
                angles[j + 1] = angles[j]
                angles[j] = angle
            else:
                is_moving = False
    unique_count = 0
    previous = gs.qd_float(-float("inf"))
    for i in range(count):
        angle = angles[i]
        if qd.abs(angle - previous) > 0.5e-14:
            angles[unique_count] = angle
            unique_count += 1
            previous = angle
    return angles, unique_count


@qd.func
def func_planar_clip_geometry(
    origin: qd.types.vector(3),
    edges: qd.types.matrix(2, 3),
    normal: qd.types.vector(3),
    center: qd.types.vector(3),
    radius: float,
):
    """Prepare a plane polygon and its polar integration intervals at a fixed ball center."""
    first, second = func_force_frame(normal)
    plane = qd.Matrix.rows([first, second])
    height = normal.dot(center - origin)
    section_center = center - height * normal
    remaining = radius * radius - height * height
    vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
    polygon = qd.Matrix.zero(gs.qd_float, 3, 2)
    for i in qd.static(range(3)):
        polygon[i, :] = plane @ (vertices[i, :] - section_center)
    orientation = edges[0, :].cross(edges[1, :]).dot(normal)
    if orientation < 0.0:
        temporary = polygon[1, :]
        polygon[1, :] = polygon[2, :]
        polygon[2, :] = temporary
    is_contained = remaining > 0.0
    for i in qd.static(range(3)):
        j = (i + 1) % 3
        edge = polygon[j, :] - polygon[i, :]
        distance = (edge[0] * (-polygon[i, 1]) - edge[1] * (-polygon[i, 0])) / edge.norm()
        is_contained = is_contained and distance >= qd.sqrt(qd.max(remaining, 0.0)) * (1.0 + 2e-12)
    angles, count = func_planar_clip_breakpoints(polygon, qd.sqrt(qd.max(remaining, 0.0)))
    return section_center, plane, polygon, remaining, angles, count, is_contained


@qd.func
def func_planar_sample_count(n_angles: int, n_angular: int, n_radial: int, is_contained: bool):
    """Return the flattened sample count for the contained or clipped polar rule."""
    count = qd.max(0, n_angles - 1) * n_angular * n_radial
    if is_contained:
        count = qd.max(16, 2 * n_angular) * n_radial
    return count


@qd.func
def _func_planar_radial_interval(polygon: qd.types.matrix(3, 2), angle: float, section_radius: float):
    direction = qd.Vector([qd.cos(angle), qd.sin(angle)])
    low, high = gs.qd_float(0.0), section_radius
    for i in qd.static(range(3)):
        j = (i + 1) % 3
        point = polygon[i, :]
        edge = polygon[j, :] - point
        slope = edge[0] * direction[1] - edge[1] * direction[0]
        intercept = edge[0] * point[1] - edge[1] * point[0]
        if slope > 1e-18:
            low = qd.max(low, intercept / slope)
        elif slope < -1e-18:
            high = qd.min(high, intercept / slope)
        elif intercept > 1e-16:
            high = -1.0
    return qd.max(low, 0.0), qd.min(high, section_radius)


@qd.func
def func_planar_hertz_sample(
    i_sample: int,
    origin: qd.types.vector(3),
    dual: qd.types.matrix(2, 3),
    parent_shapes: qd.types.matrix(3, 4),
    section_center: qd.types.vector(3),
    plane: qd.types.matrix(2, 3),
    polygon: qd.types.matrix(3, 2),
    remaining: float,
    angles: qd.types.vector(11),
    n_angles: int,
    original_position: qd.types.vector(3),
    radius: float,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    gauss_angular: qd.Tensor,
    weights_angular: qd.Tensor,
    gauss_radial: qd.Tensor,
    weights_radial: qd.Tensor,
    is_contained: bool,
):
    """Return position, parent shape, physical Hertz weight and moment coordinates.

    The angular and radial Gauss tensors hold nodes and weights on [0, 1].
    Samples include the square-root Hertz density and surface-area measure.
    """
    position = section_center
    parent_shape = qd.Vector.zero(gs.qd_float, 4)
    weight = gs.qd_float(0.0)
    coordinates = qd.Vector([1.0, 0.0, 0.0])
    n_angular, n_radial = gauss_angular.shape[0], gauss_radial.shape[0]
    count = func_planar_sample_count(n_angles, n_angular, n_radial, is_contained)
    if i_sample >= 0 and i_sample < count and remaining > 0.0 and radius > 0.0:
        section_radius = qd.sqrt(remaining)
        n_circle = qd.max(16, 2 * n_angular)
        i_radial = i_sample % n_radial
        if is_contained:
            i_radial = i_sample // n_circle
        radial_node = gauss_radial[i_radial]
        radial_weight = weights_radial[i_radial]
        angle, angular_weight = gs.qd_float(0.0), gs.qd_float(0.0)
        low, high = gs.qd_float(0.0), section_radius
        if is_contained:
            i_angular = i_sample % n_circle
            angle = 2.0 * qd.math.pi * (i_angular + 0.5) / n_circle
            angular_weight = 2.0 * qd.math.pi / n_circle
        else:
            i_angular = (i_sample // n_radial) % n_angular
            i_interval = i_sample // (n_radial * n_angular)
            left, right = angles[i_interval], angles[i_interval + 1]
            angular_node = gauss_angular[i_angular]
            angle = left + (right - left) * qd.sin(0.5 * qd.math.pi * angular_node) ** 2
            angular_weight = (
                (right - left) * 0.5 * qd.math.pi * qd.sin(qd.math.pi * angular_node) * weights_angular[i_angular]
            )
            if right - left < 1e-14:
                angular_weight = 0.0
        if angular_weight > 0.0:
            if not is_contained:
                low, high = _func_planar_radial_interval(polygon, angle, section_radius)
            if high > low + section_radius * 1e-14:
                phi_low = qd.asin(qd.max(0.0, qd.min(1.0, low / section_radius)))
                phi_high = qd.asin(qd.max(0.0, qd.min(1.0, high / section_radius)))
                phi = phi_low + (phi_high - phi_low) * radial_node
                distance = section_radius * qd.sin(phi)
                local_plane = distance * qd.Vector([qd.cos(angle), qd.sin(angle)])
                position = section_center + local_plane @ plane
                local = dual @ (position - origin)
                barycentric = qd.Vector([1.0 - local.sum(), local[0], local[1]])
                parent_shape = parent_shapes.transpose() @ barycentric
                weight = (
                    remaining
                    * section_radius
                    / radius
                    * qd.sin(phi)
                    * qd.cos(phi) ** 2
                    * (phi_high - phi_low)
                    * radial_weight
                    * angular_weight
                )
                relative = (position - original_position) / radius
                coordinates = qd.Vector([1.0, relative.dot(first), relative.dot(second)])
    return position, parent_shape, weight, coordinates
