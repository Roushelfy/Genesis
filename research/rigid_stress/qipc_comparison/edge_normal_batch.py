"""Batched CPU evaluation of the existing normal composite sharp-edge field."""

import numpy as np

from .bilinear_normal_stress import point_tensor
from .edge_fields import flamant_tensor, strip_tensor


def composite_normal(points, model, load, order=32, poisson=0.3):
    """Keep the scalar composite's strip, line correction and depth floor."""
    assert np.all(points[:, 2] > 0)
    knots, weights = np.polynomial.legendre.leggauss(order)
    result = np.empty((len(points), 3, 3))
    for start in range(0, len(points), 512):
        current = np.asarray(points[start : start + 512])
        profile = np.maximum(1 - (current[:, 1] / model.along) ** 2, 0)
        width = model.width * profile
        strip = strip_tensor(current, width, model.exponent, model.amplitude, np.zeros(2), poisson)
        regular = current.copy()
        regular[:, 2] = np.maximum(regular[:, 2], 0.001 * model.along)
        outside = np.maximum.reduce([-regular[:, 0], regular[:, 0] - model.width, np.zeros(len(regular))])
        scale = np.hypot(outside, regular[:, 2])
        middle = np.clip(regular[:, 1], -model.along, model.along)
        limits = np.column_stack((np.full(len(regular), -model.along), middle, np.full(len(regular), model.along)))
        transformed = np.arcsinh((limits - regular[:, 1, None]) / scale[:, None])
        t = (
            (transformed[:, :-1] + transformed[:, 1:])[:, :, None]
            + (transformed[:, 1:] - transformed[:, :-1])[:, :, None] * knots
        ) / 2
        y = regular[:, 1, None, None] + scale[:, None, None] * np.sinh(t)
        measure = (
            weights * (transformed[:, 1:] - transformed[:, :-1])[:, :, None] / 2 * scale[:, None, None] * np.cosh(t)
        )
        shape = np.maximum(1 - (y / model.along) ** 2, 0)
        source_x = model.width * (1 - model.exponent) * shape / 2
        force = 3 * load / (4 * model.along) * shape * measure
        relative = np.stack(
            (
                regular[:, 0, None, None] - source_x,
                regular[:, 1, None, None] - y,
                np.broadcast_to(regular[:, 2, None, None], y.shape),
            ),
            axis=-1,
        ).reshape(len(regular), -1, 3)
        finite = np.einsum("nqij,nq->nij", point_tensor(relative, poisson), force.reshape(len(regular), -1))
        coordinate = regular[:, 0] - width * (1 - model.exponent) / 2 + 1j * regular[:, 2]
        density = 3 * load / (4 * model.along) * profile
        infinite = flamant_tensor(density / coordinate, -density / coordinate**2, regular[:, 2], np.zeros(2), poisson)
        result[start : start + len(current)] = strip + finite - infinite
    return result
