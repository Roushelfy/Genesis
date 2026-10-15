"""Evaluate independent affine quadratic corner stresses using extended-precision polynomial differentiation."""

from functools import lru_cache

import numpy as np

from .verify_corner_polynomial_decimal import interpolation_derivatives


@lru_cache(maxsize=1)
def native_reference_nodes_and_weights():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    nodes = np.vstack([vertices, vertices[edges].mean(axis=1)])
    return nodes, np.asarray(interpolation_derivatives(), dtype=np.longdouble)


def extended_corner_stress(reference, displacement):
    """Evaluate every corner with exact polynomial weights and extended cofactor geometry."""
    nodes, weights = native_reference_nodes_and_weights()
    locations = reference.scalar_basis.elem.doflocs
    matches = (locations[:, None] == nodes[None]).all(axis=-1)
    assert np.all(matches.sum(axis=1) == 1)
    weights = weights[:, :, matches.argmax(axis=1)]
    points = np.asarray(reference.mesh.p.T[reference.mesh.t.T], dtype=np.longdouble)
    first, second, third = [points[:, i_corner] - points[:, 0] for i_corner in [1, 2, 3]]
    determinant = np.einsum("ei,ei->e", first, np.cross(second, third))
    inverse = np.stack([np.cross(second, third), np.cross(third, first), np.cross(first, second)], axis=1)
    inverse /= determinant[:, None, None]
    local = np.asarray(displacement[reference.basis.element_dofs.T].reshape(-1, 10, 3), dtype=np.longdouble)
    reference_derivative = np.einsum("eli,cjl->ecij", local, weights, optimize=False)
    derivative = reference_derivative @ inverse[:, None]
    young, poisson = np.longdouble(reference.young), np.longdouble(reference.poisson)
    mu = young / (2 * (1 + poisson))
    lame_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    tensor = mu * (derivative + derivative.swapaxes(-1, -2))
    for i_axis in range(3):
        tensor[..., i_axis, i_axis] += lame_lambda * np.trace(derivative, axis1=-2, axis2=-1)
    deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return tensor, np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
