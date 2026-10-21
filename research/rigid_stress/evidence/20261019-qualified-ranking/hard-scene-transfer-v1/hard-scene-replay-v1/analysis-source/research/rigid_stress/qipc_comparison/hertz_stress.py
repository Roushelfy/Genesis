"""Normal elliptic Hertz stress from a one-dimensional potential integral.

The Newton potential of the elliptical sqrt-pressure disk is the zero-thickness
limit of a uniformly filled ellipsoid. Differentiating that potential gives the
Boussinesq normal and shear terms. Integrating its planar Hessian from z to
infinity gives the remaining in-plane terms, with the integration order swapped.
This avoids a singular two-dimensional pressure quadrature at the surface.

Coordinates use the ellipse principal frame with z into the material. Axes and
positions use the same length unit; output has the units of peak pressure.
This half-space solution does not include shell or tangential-load corrections.
"""

import numpy as np
from scipy.special import elliprd


def confocal_coordinate(points, axes):
    """Positive ellipsoidal coordinate, including its exact surface limit."""
    squared = points * points
    aa, bb = axes * axes
    coefficient = aa + bb - squared[:, 0] - squared[:, 1]
    constant = aa * bb - bb * squared[:, 0] - aa * squared[:, 1]
    discriminant = np.sqrt(np.maximum(coefficient * coefficient - 4 * constant, 0))
    outside = constant < 0
    surface = np.zeros(len(points))
    surface[outside] = 2 * (-constant[outside]) / (coefficient[outside] + discriminant[outside])
    low = np.maximum(surface, squared[:, 2])
    high = squared.sum(axis=1)
    value = np.maximum(low, high / 2)
    active = points[:, 2] > 0
    for _ in range(56):
        denominator = value[active, None] + np.array([aa, bb, 0])
        equation = (squared[active] / denominator).sum(axis=1) - 1
        derivative = -(squared[active] / denominator**2).sum(axis=1)
        positive = equation > 0
        indices = np.flatnonzero(active)
        low[indices[positive]] = value[indices[positive]]
        high[indices[~positive]] = value[indices[~positive]]
        proposal = value[active] - equation / derivative
        value[active] = np.where(
            (proposal > low[active]) & (proposal < high[active]), proposal, (low[active] + high[active]) / 2
        )
    return np.where(active, value, surface)


def elliptic_stress(points, axes, peak=1.0, poisson=0.3, order=48):
    """Complete normal Hertz tensor, with a finite surface/axis/edge limit.

    Carlson RD evaluates the three differentiated Newton integrals. The remaining
    elementary quadratic-root integrals use one smooth angular Gauss rule.
    The order controls evaluation accuracy only; the physical footprint is fixed.
    """
    assert np.all(points[:, 2] >= 0) and np.all(axes > 0) and -1 < poisson < 0.5
    scale = axes.max()
    position, semiaxes = points / scale, axes / scale
    aa, bb = semiaxes * semiaxes
    x, y, z = position.T
    coordinate = confocal_coordinate(position, semiaxes)
    argument = coordinate[:, None] + np.array([aa, bb, 0])
    interior = z > 0
    integrals = np.zeros_like(position)
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        integrals[interior, i] = 2 * elliprd(argument[interior, j], argument[interior, k], argument[interior, i]) / 3
    # Scale by lambda+b^2 to resolve the far field as well as the edge endpoint.
    gauss, weight = np.polynomial.legendre.leggauss(order)
    theta, weight = (gauss + 1) * np.pi / 4, weight * np.pi / 4
    tangent = np.tan(theta)
    increment = (bb + coordinate[:, None]) * tangent * tangent
    s = coordinate[:, None] + increment
    jacobian = 2 * (bb + coordinate[:, None]) * tangent * (1 + tangent * tangent)
    ax, by = aa + s, bb + s
    # Q(lambda)=a_lambda*b_lambda*z^2/lambda; this representation
    # avoids cancellation exactly on and immediately outside the contact edge.
    q0 = np.zeros(len(points))
    q0[interior] = (aa + coordinate[interior]) * (bb + coordinate[interior]) * z[interior] ** 2 / coordinate[interior]
    inside_surface = (~interior) & (coordinate == 0)
    q0[inside_surface] = np.maximum(aa * bb - bb * x[inside_surface] ** 2 - aa * y[inside_surface] ** 2, 0)
    root = np.sqrt(
        q0[:, None] + increment * (2 * coordinate[:, None] + aa + bb - x[:, None] ** 2 - y[:, None] ** 2 + increment)
    )
    weighted = jacobian * weight
    ell = np.column_stack(
        ((root / (ax * ax * by) * weighted).sum(axis=1), (root / (ax * by * by) * weighted).sum(axis=1))
    )
    moment = np.empty((len(points), 2, 2))
    moment[:, 0, 0] = (weighted / (ax * ax * root)).sum(axis=1)
    moment[:, 1, 1] = (weighted / (by * by * root)).sum(axis=1)
    moment[:, 0, 1] = moment[:, 1, 0] = (weighted / (ax * by * root)).sum(axis=1)
    factor = semiaxes.prod() * peak / 2
    stress = np.zeros((len(points), 3, 3))
    stress[:, :2, :2] = factor * (1 - 2 * poisson) * position[:, :2, None] * position[:, None, :2] * moment
    for i in (0, 1):
        stress[:, i, i] -= factor * (1 - 2 * poisson) * ell[:, i]
    # Surface traction is exact, including points at the contact edge.
    pressure = peak * np.sqrt(np.maximum(1 - x * x / aa - y * y / bb, 0))
    stress[~interior, 0, 0] -= 2 * poisson * pressure[~interior]
    stress[~interior, 1, 1] -= 2 * poisson * pressure[~interior]
    stress[~interior, 2, 2] = -pressure[~interior]
    # Phi_ij = -pi*a*b*p0*(delta_ij*I_i - 2*x_i*x_j*C_ij).
    p, arg, zi = position[interior], argument[interior], z[interior]
    denominator = np.sqrt(arg.prod(axis=1)) * (p * p / arg**2).sum(axis=1)
    derivative = p[:, :, None] * p[:, None, :] / (arg[:, :, None] * arg[:, None, :] * denominator[:, None, None])
    stress[interior, :2, :2] -= 2 * factor * zi[:, None, None] * derivative[:, :2, :2]
    for i in (0, 1):
        stress[interior, i, i] += (
            2 * factor * zi * ((1 - poisson) * integrals[interior, i] - poisson * integrals[interior, 2])
        )
        stress[interior, i, 2] = stress[interior, 2, i] = -2 * factor * zi * derivative[:, i, 2]
    stress[interior, 2, 2] = -2 * factor * zi * derivative[:, 2, 2]
    return stress
