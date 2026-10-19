"""Objective total-Lagrangian P2 elasticity with the declared Lame parameters."""

import numpy as np
from skfem import LinearForm, asm
from skfem.helpers import ddot


def cauchy_from_gradient(gradient, young, poisson):
    strain = (gradient + gradient.swapaxes(-2, -1)) / 2
    strain += 0.5 * gradient.swapaxes(-2, -1) @ gradient
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = young / (2 * (1 + poisson))
    second = 2 * shear * strain + lame * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
    deformation = gradient + np.eye(3)
    determinant = np.linalg.det(deformation)
    assert determinant.min() > 0, "Inverted P2 solid at a sampled point"
    return deformation @ second @ deformation.swapaxes(-2, -1) / determinant[..., None, None]


@LinearForm
def piola_force(test, data):
    return ddot(data.piola, test.grad)


class P2StVK:
    def __init__(self, reference):
        self.reference = reference

    def tensors(self, gradient):
        # G is displacement gradient; this form avoids subtracting I from F^T F.
        strain = (gradient + gradient.swapaxes(0, 1)) / 2
        strain += 0.5 * np.einsum("kieq,kjeq->ijeq", gradient, gradient)
        lame, shear = self.reference.lame
        identity = np.eye(3)[:, :, None, None]
        stress = 2 * shear * strain + lame * np.einsum("iieq->eq", strain)[None, None] * identity
        piola = stress + np.einsum("ikeq,kjeq->ijeq", gradient, stress)
        return strain, stress, piola

    def force_energy(self, displacement):
        gradient = self.reference.basis.interpolate(displacement).grad
        strain, stress, piola = self.tensors(gradient)
        force = asm(piola_force, self.reference.basis, piola=piola)
        energy = 0.5 * np.sum(np.einsum("ijeq,ijeq->eq", strain, stress) * self.reference.basis.dx)
        return force, float(energy)

    def corner_stress(self, displacement):
        gradient = self.reference.corner_basis.interpolate(displacement).grad
        _, _, piola = self.tensors(gradient)
        deformation = gradient + np.eye(3)[:, :, None, None]
        determinant = np.linalg.det(deformation.transpose(2, 3, 0, 1))
        assert determinant.min() > 0, "Inverted P2 solid at a corner"
        stress = np.einsum("ikeq,jkeq->ijeq", piola, deformation).transpose(2, 3, 0, 1)
        return stress / determinant[:, :, None, None]
