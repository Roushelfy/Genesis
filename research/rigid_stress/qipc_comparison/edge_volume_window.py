"""Smooth local volume quadrature aligned with the ideal sharp-edge singularity."""

import numpy as np


def taper(value, inner, outer):
    t = np.clip((value - inner) / (outer - inner), 0, 1)
    return 1 - 10 * t**3 + 15 * t**4 - 6 * t**5


def window(points, width, along):
    return (
        taper(np.abs(points[:, 0]) / width, 2, 4)
        * taper(np.abs(points[:, 1]) / along, 1.5, 2)
        * taper(points[:, 2] / width, 2, 8)
    )


def quadrature(
    width, along, y_boundaries, angle_order, radial_order, along_order, x_boundaries=None, alignment_depth=None
):
    angles = np.arctan(np.array([0.5, 1, 2, 4]))
    cuts = np.sort(np.r_[0, angles, np.pi / 2, np.pi - angles, np.pi])
    angle_node, angle_weight = np.polynomial.legendre.leggauss(angle_order)
    theta = ((cuts[:-1] + cuts[1:])[:, None] / 2 + np.diff(cuts)[:, None] * angle_node / 2).reshape(-1)
    measure_theta = (np.diff(cuts)[:, None] * angle_weight / 2).reshape(-1)
    rnode, rweight = np.polynomial.legendre.leggauss(radial_order)
    cross_points, cross_weights = [], []
    for angle, weight in zip(theta, measure_theta):
        cosine, sine = np.cos(angle), np.sin(angle)
        maximum = min(4 / abs(cosine), 8 / sine)
        breaks = np.unique(np.r_[0, np.clip([2 / abs(cosine), 2 / sine], 0, maximum), maximum])
        if x_boundaries is not None and cosine > 0:
            # Resolve individual cell edges only near the plane, where their
            # stress transition is sharper than the transverse cell scale.
            radius_edges = x_boundaries / (width * cosine)
            selected_edges = radius_edges[
                (radius_edges > 0) & (radius_edges < maximum) & (radius_edges * width * sine <= alignment_depth)
            ]
            breaks = np.unique(np.r_[breaks, selected_edges])
        radius = ((breaks[:-1] + breaks[1:])[:, None] / 2 + np.diff(breaks)[:, None] * rnode / 2).reshape(-1)
        measure = (np.diff(breaks)[:, None] * rweight / 2).reshape(-1) * radius * weight * width**2
        cross_points.append(np.column_stack((radius * cosine, radius * sine)) * width)
        cross_weights.append(measure)
    cross_points, cross_weights = np.concatenate(cross_points), np.concatenate(cross_weights)
    cuts_y = np.unique(np.r_[np.clip(y_boundaries / along, -2, 2), [-2, -1.5, -1, 0, 1, 1.5, 2]]) * along
    ynode, yweight = np.polynomial.legendre.leggauss(along_order)
    y = ((cuts_y[:-1] + cuts_y[1:])[:, None] / 2 + np.diff(cuts_y)[:, None] * ynode / 2).reshape(-1)
    measure_y = (np.diff(cuts_y)[:, None] * yweight / 2).reshape(-1)
    points = np.column_stack(
        (np.repeat(cross_points[:, 0], len(y)), np.tile(y, len(cross_points)), np.repeat(cross_points[:, 1], len(y)))
    )
    weights = (cross_weights[:, None] * measure_y).reshape(-1) * window(points, width, along)
    selected = weights > 0
    points, weights = points[selected], weights[selected]
    expected = np.array([105 * width**2 * along, 276 * width**3 * along, 330 * width**4 * along])
    observed = np.array([weights.sum(), weights @ points[:, 2], weights @ points[:, 0] ** 2])
    errors = np.abs(observed / expected - 1)
    assert errors.max() < 1e-8, errors
    np.testing.assert_allclose(weights @ points[:, :2], 0, rtol=0, atol=weights.sum() * along * 1e-12)
    return (
        points,
        weights,
        {
            "points": len(points),
            "angle_order": angle_order,
            "radial_order": radial_order,
            "along_order": along_order,
            "analytic_moment_relative_errors": errors.tolist(),
            "minimum_depth_m": float(points[:, 2].min()),
            "weighted_volume_m3": float(weights.sum()),
            "pressure_cell_alignment_depth_m": alignment_depth,
        },
    )
