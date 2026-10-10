"""Independent CPU FP64 affine-P2 tensor/von-Mises observation oracle."""

import numpy as np

from .mechanics import EDGES


def stress_field(gradients, elements, displacement, young, poisson):
    barycentric = np.eye(4, dtype=np.float64)
    shape_gradients = np.empty((len(elements), 4, 10, 3), dtype=np.float64)
    shape_gradients[:, :, :4] = (4 * barycentric - 1)[None, :, :, None] * gradients[:, None]
    for edge, (a, b) in enumerate(EDGES):
        shape_gradients[:, :, 4 + edge] = 4 * (
            barycentric[None, :, a, None] * gradients[:, None, b]
            + barycentric[None, :, b, None] * gradients[:, None, a]
        )
    nodal = np.asarray(displacement, dtype=np.float64).reshape((-1, 3))[elements]
    derivative = np.einsum("eni,ecnj->ecij", nodal, shape_gradients)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    sigma = mu * (derivative + derivative.swapaxes(-1, -2))
    sigma += lam * np.trace(derivative, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
    tensor = sigma[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    xx, yy, zz, xy, xz, yz = np.moveaxis(tensor, -1, 0)
    von_mises = np.sqrt(((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) / 2 + 3 * (xy**2 + xz**2 + yz**2))
    return tensor, von_mises
