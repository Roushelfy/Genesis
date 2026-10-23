"""Independent Boussinesq integration of continuous bilinear normal pressure.

Each angular ray crosses piecewise quadratic pressure segments. Radial stress
integrals have elementary antiderivatives, including the exact surface principal
value. Only the periodic angular rule remains numerical. Far queries use
polynomial source interpolation with independently integrated pressure moments.
This CPU reference does not assume Hertz support or belong to the runtime law.
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator, RegularGridInterpolator


def point_tensor(position, poisson=0.3):
    """Unit inward normal point load; coordinates have z into the material."""
    position = np.asarray(position)
    radius = np.linalg.norm(position, axis=-1)
    z = position[..., 2]
    assert np.all(z >= 0) and np.all(radius > 0)
    tensor = (-3 * z / (2 * np.pi * radius**5))[..., None, None] * (position[..., :, None] * position[..., None, :])
    common = (1 - 2 * poisson) / (2 * np.pi)
    tensor[..., :2, :2] += (
        (common * (2 * radius + z) / (radius**3 * (radius + z) ** 2))[..., None, None]
        * position[..., :2, None]
        * position[..., None, :2]
    )
    diagonal = common * (z / radius**3 - 1 / (radius * (radius + z)))
    tensor[..., 0, 0] += diagonal
    tensor[..., 1, 1] += diagonal
    return tensor


def radial_primitives(radius, depth, poisson):
    """Four angular tensor coefficients, integrated against 1, r and r^2.

    The returned coefficients already include their powers of depth, avoiding
    separate singular 1/z^3 primitives close to the material surface.
    """
    distance = np.hypot(radius, depth)
    t = radius / distance
    ratio = depth / distance
    inverse_hyperbolic = np.arcsinh(radius / depth)
    logarithm = np.log(distance + depth)
    a_first = np.stack(
        (
            3 * ratio - ratio**3,
            depth * (t**3 + 3 * t - 3 * inverse_hyperbolic),
            -3 * depth * distance - 6 * depth**2 * ratio + depth**2 * ratio**3,
        ),
        axis=-1,
    )
    a_second = np.stack(
        (
            2 * logarithm + ratio,
            2 * radius - 3 * depth * inverse_hyperbolic + depth * t,
            radius**2 - 3 * depth * distance - depth**2 * ratio,
        ),
        axis=-1,
    )
    b = (1 - 2 * poisson) * np.stack(
        (
            -ratio - logarithm,
            depth * (2 * inverse_hyperbolic - t) - radius,
            2 * depth * distance + depth**2 * ratio - radius**2 / 2,
        ),
        axis=-1,
    )
    c = np.stack(
        (
            -(t**3),
            depth * (3 * ratio - ratio**3),
            depth**2 * (t**3 + 3 * t - 3 * inverse_hyperbolic),
        ),
        axis=-1,
    )
    d = np.stack((ratio**3, -depth * t**3, depth**2 * (3 * ratio - ratio**3)), axis=-1)
    return a_first + (1 - 2 * poisson) * a_second, b, c, d


class BilinearNormalStress:
    def __init__(self, axes, pressure, length_scale, pressure_scale, poisson=0.3):
        self.axes = [np.asarray(axis) / length_scale for axis in axes]
        self.pressure = np.asarray(pressure) / pressure_scale
        self.length_scale, self.pressure_scale, self.poisson = length_scale, pressure_scale, poisson
        self.widths = np.array([axis[1] - axis[0] for axis in self.axes])
        assert self.pressure.shape == tuple(len(axis) for axis in self.axes)
        for axis, width in zip(self.axes, self.widths):
            np.testing.assert_allclose(np.diff(axis), width, rtol=1e-12, atol=0)
        self.interpolation = RegularGridInterpolator(self.axes, self.pressure, bounds_error=False, fill_value=0)

    def ray_segments(self, position, directions):
        crossings = [
            (value - axis[None]) / directions[:, i, None] for i, (axis, value) in enumerate(zip(self.axes, position))
        ]
        lower = np.maximum(
            0,
            np.maximum(
                np.minimum(crossings[0][:, 0], crossings[0][:, -1]), np.minimum(crossings[1][:, 0], crossings[1][:, -1])
            ),
        )
        upper = np.minimum(
            np.maximum(crossings[0][:, 0], crossings[0][:, -1]), np.maximum(crossings[1][:, 0], crossings[1][:, -1])
        )
        valid = upper > lower
        lower, upper = np.where(valid, lower, 0), np.where(valid, upper, 0)
        radii = np.concatenate([lower[:, None], upper[:, None], *crossings], axis=1)
        radii = np.sort(np.clip(radii, lower[:, None], upper[:, None]), axis=1)
        first, last = radii[:, :-1], radii[:, 1:]
        middle = (first + last) / 2
        source = position - middle[..., None] * directions[:, None]
        indices = [
            np.clip(np.searchsorted(axis, source[..., i], side="right") - 1, 0, len(axis) - 2)
            for i, axis in enumerate(self.axes)
        ]
        i, j = indices
        p = self.pressure
        a = p[i, j]
        dx, dy = self.widths
        bx, by = (p[i + 1, j] - a) / dx, (p[i, j + 1] - a) / dy
        mixed = (p[i + 1, j + 1] - p[i + 1, j] - p[i, j + 1] + a) / (dx * dy)
        offset_x, offset_y = position[0] - self.axes[0][i], position[1] - self.axes[1][j]
        ux, uy = directions[:, 0, None], directions[:, 1, None]
        constant = a + bx * offset_x + by * offset_y + mixed * offset_x * offset_y
        linear = -bx * ux - by * uy - mixed * (offset_x * uy + offset_y * ux)
        quadratic = mixed * ux * uy
        return first, last, np.stack((constant, linear, quadratic), axis=-1)

    def near(self, points, angular=256):
        points = np.asarray(points) / self.length_scale
        assert np.all(points[:, 2] >= 0)
        angles = 2 * np.pi * (np.arange(angular) + 0.5) / angular
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        ux, uy = directions.T
        result = np.empty((len(points), 3, 3))
        for index, point in enumerate(points):
            first, last, coefficients = self.ray_segments(point[:2], directions)
            tensor = np.zeros((3, 3))
            if point[2] == 0:
                logarithm = np.log(np.where(last > 0, last, 1) / np.where(first > 0, first, 1))
                radial = np.stack((logarithm, last - first, (last**2 - first**2) / 2), axis=-1)
                integrated = np.einsum("aqk,aqk->a", coefficients, radial)
                value = float(self.interpolation(point[None, :2])[0])
                common = (1 - 2 * self.poisson) / angular
                tensor[0, 0] = common * np.dot(2 * ux * ux - 1, integrated) - (0.5 + self.poisson) * value
                tensor[1, 1] = common * np.dot(2 * uy * uy - 1, integrated) - (0.5 + self.poisson) * value
                tensor[0, 1] = tensor[1, 0] = common * np.dot(2 * ux * uy, integrated)
                tensor[2, 2] = -value
            else:
                primitive_first = radial_primitives(first, point[2], self.poisson)
                primitive_last = radial_primitives(last, point[2], self.poisson)
                a, b, c, d = [
                    np.einsum("aqk,aqk->a", coefficients, high - low)
                    for high, low in zip(primitive_last, primitive_first)
                ]
                tensor[0, 0] = np.sum(a * ux * ux + b) / angular
                tensor[1, 1] = np.sum(a * uy * uy + b) / angular
                tensor[0, 1] = tensor[1, 0] = np.sum(a * ux * uy) / angular
                tensor[0, 2] = tensor[2, 0] = np.sum(c * ux) / angular
                tensor[1, 2] = tensor[2, 1] = np.sum(c * uy) / angular
                tensor[2, 2] = np.sum(d) / angular
            result[index] = tensor * self.pressure_scale
        return result

    def far_sources(self, count):
        nodes = [
            np.cos(np.linspace(np.pi, 0, count)) * (axis[-1] - axis[0]) / 2 + (axis[-1] + axis[0]) / 2
            for axis in self.axes
        ]
        quadrature, weights = np.polynomial.legendre.leggauss((count + 2) // 2)
        positions = [
            (axis[:-1, None] + (quadrature + 1) * width / 2).reshape(-1) for axis, width in zip(self.axes, self.widths)
        ]
        factors = [BarycentricInterpolator(axis, np.eye(count))(position) for axis, position in zip(nodes, positions)]
        x, y = np.meshgrid(*positions, indexing="ij")
        pressure = self.interpolation(np.column_stack((x.reshape(-1), y.reshape(-1)))).reshape(x.shape)
        mass_weights = [np.tile(weights * width / 2, len(axis) - 1) for width, axis in zip(self.widths, self.axes)]
        forces = factors[0].T @ (pressure * mass_weights[0][:, None] * mass_weights[1][None]) @ factors[1]
        source = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, 2)
        return source * self.length_scale, forces.reshape(-1) * self.pressure_scale * self.length_scale**2

    def far(self, points, count=9):
        sources, forces = self.far_sources(count)
        result = np.empty((len(points), 3, 3))
        for start in range(0, len(points), 1024):
            position = (
                np.asarray(points[start : start + 1024])[:, None]
                - np.column_stack((sources, np.zeros(len(sources))))[None]
            )
            result[start : start + len(position)] = np.einsum(
                "nsij,s->nij", point_tensor(position, self.poisson), forces
            )
        return result
