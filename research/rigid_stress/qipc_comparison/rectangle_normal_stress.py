"""Exact positive-depth Boussinesq stress for constant rectangular pressures.

Let Phi=integral(1/R)dA and Psi=integral(log(R+z))dA. Integrating the
existing point tensor gives 2*pi*sxx=-z*Phi_xx+2*nu*Phi_z-(1-2*nu)*Psi_xx,
2*pi*sxy=-z*Phi_xy-(1-2*nu)*Psi_xy, 2*pi*sxz=-z*Phi_xz and
2*pi*szz=Phi_z-z*Phi_zz. The derivatives below are corner differences.
Coordinates have z into material, with tensile stress positive. This is the
uniform-rectangle half-space problem (Love, 1929, doi:10.1098/rsta.1929.0009),
not a contact law or finite-shell model. Independent complex-step quadrature
and boundary-limit controls verify the implementation before its use.
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator

from .bilinear_normal_stress import point_tensor


def rectangle_tensor(points, centers, widths, force, poisson=0.3):
    """Integrate each uniform cell exactly, using bounded target blocks."""
    points, centers, force = np.asarray(points), np.asarray(centers), np.asarray(force)
    assert np.all(points[:, 2] > 0)
    widths = np.broadcast_to(np.asarray(widths), centers.shape)
    assert np.all(widths > 0)
    selected = force != 0
    centers, widths, force = centers[selected], widths[selected], force[selected]
    pressure = force / widths.prod(axis=1)
    result = np.zeros((len(points), 3, 3))
    for start in range(0, len(points), 128):
        current = points[start : start + 128]
        delta = current[:, None, :2] - centers[None]
        z = current[:, None, 2]
        shape = (len(current), len(centers))
        pz, pxx, pyy, pxy, pxz, pyz, qxx, qyy, qxy = [np.zeros(shape) for _ in range(9)]
        for sx, sy in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
            x = delta[..., 0] + sx * widths[None, :, 0] / 2
            y = delta[..., 1] + sy * widths[None, :, 1] / 2
            radius = np.sqrt(x * x + y * y + z * z)
            sign = sx * sy
            pz -= sign * np.arctan2(x * y, z * radius)
            pxx -= sign * x * y / ((x * x + z * z) * radius)
            pyy -= sign * x * y / ((y * y + z * z) * radius)
            pxy += sign / radius
            pxz -= sign * z * y / ((x * x + z * z) * radius)
            pyz -= sign * z * x / ((y * y + z * z) * radius)
            qxx += sign * np.arctan2(x * y, x * x + z * z + z * radius)
            qyy += sign * np.arctan2(x * y, y * y + z * z + z * radius)
            qxy += sign * np.log(radius + z)
        tensor = np.empty((*shape, 3, 3))
        tensor[..., 0, 0] = -z * pxx + 2 * poisson * pz - (1 - 2 * poisson) * qxx
        tensor[..., 1, 1] = -z * pyy + 2 * poisson * pz - (1 - 2 * poisson) * qyy
        tensor[..., 0, 1] = tensor[..., 1, 0] = -z * pxy - (1 - 2 * poisson) * qxy
        tensor[..., 0, 2] = tensor[..., 2, 0] = -z * pxz
        tensor[..., 1, 2] = tensor[..., 2, 1] = -z * pyz
        tensor[..., 2, 2] = pz + z * (pxx + pyy)
        result[start : start + len(current)] = np.einsum("nsij,s->nij", tensor, pressure) / (2 * np.pi)
    return result


class RectangleNormalStress:
    """Exact near-cell integration and controlled far-source interpolation."""

    def __init__(self, centers, widths, force, poisson=0.3):
        self.centers, self.widths, self.force = np.asarray(centers), np.asarray(widths), np.asarray(force)
        self.poisson = poisson
        self.axes = [np.unique(self.centers[:, axis]) for axis in (0, 1)]
        expected = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1).reshape(-1, 2)
        np.testing.assert_array_equal(expected, self.centers)
        self.rules = {}

    def near(self, points):
        return rectangle_tensor(points, self.centers, self.widths, self.force, self.poisson)

    def far_sources(self, count):
        if count not in self.rules:
            nodes, averages = [], []
            knots, weights = np.polynomial.legendre.leggauss((count + 1) // 2)
            for axis, width in zip(self.axes, self.widths):
                middle, extent = (axis[-1] + axis[0]) / 2, (axis[-1] - axis[0] + width) / 2
                values = middle + extent * np.cos(np.linspace(np.pi, 0, count))
                query = axis[:, None] + width * knots / 2
                interpolation = BarycentricInterpolator(values, np.eye(count))(query.reshape(-1))
                averages.append(
                    np.einsum("iqg,q->ig", interpolation.reshape(len(axis), len(knots), count), weights / 2)
                )
                nodes.append(values)
            forces = averages[0].T @ self.force.reshape(*map(len, self.axes)) @ averages[1]
            sources = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, 2)
            np.testing.assert_allclose(forces.sum(), self.force.sum(), rtol=1e-12, atol=0)
            np.testing.assert_allclose(forces.reshape(-1) @ sources, self.force @ self.centers, rtol=1e-9, atol=1e-20)
            self.rules[count] = sources, forces.reshape(-1)
        return self.rules[count]

    def far(self, points, count):
        sources, forces = self.far_sources(count)
        return point_sources_tensor(points, sources, forces, self.poisson)


def point_sources_tensor(points, sources, forces, poisson=0.3):
    """Bounded analytic point-tensor contraction, also valid on unloaded plane."""
    result = np.empty((len(points), 3, 3))
    for start in range(0, len(points), 512):
        current = np.asarray(points[start : start + 512])
        relative = current[:, None] - np.column_stack((sources, np.zeros(len(sources))))[None]
        result[start : start + len(current)] = np.einsum("nsij,s->nij", point_tensor(relative, poisson), forces)
    return result
