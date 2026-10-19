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
    def __init__(self, reference, cache_evaluations=False):
        self.reference = reference
        self.cache_evaluations = cache_evaluations
        self._cached_displacement = None
        self._cached_gradient = None
        self._cached_correction = None

    def _gradient(self, displacement):
        # The reference geometry/material is fixed for this model's lifetime.
        # Own the key: an in-place update by the integrator must invalidate it.
        if not self.cache_evaluations:
            return self.reference.basis.interpolate(displacement).grad
        if self._cached_displacement is None or not np.array_equal(displacement, self._cached_displacement):
            self._cached_displacement = displacement.copy()
            self._cached_gradient = self.reference.basis.interpolate(displacement).grad
            self._cached_correction = None
        return self._cached_gradient

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
        gradient = self._gradient(displacement)
        strain, stress, piola = self.tensors(gradient)
        force = asm(piola_force, self.reference.basis, piola=piola)
        energy = 0.5 * np.sum(np.einsum("ijeq,ijeq->eq", strain, stress) * self.reference.basis.dx)
        return force, float(energy)

    def force_correction(self, displacement):
        """Assemble K u - f(u) without subtracting two complete nodal forces.

        With E=sym(G)+Q, Q=G.T G/2, P=C:sym(G)+C:Q+G(C:E).
        The first term integrates to K u; assemble the negative remainder.
        """
        gradient = self._gradient(displacement)
        if self.cache_evaluations and self._cached_correction is not None:
            return self._cached_correction.copy()
        quadratic = 0.5 * np.einsum("kieq,kjeq->ijeq", gradient, gradient)
        linear = (gradient + gradient.swapaxes(0, 1)) / 2
        lame, shear = self.reference.lame
        identity = np.eye(3)[:, :, None, None]

        def material(strain):
            return 2 * shear * strain + lame * np.einsum("iieq->eq", strain)[None, None] * identity

        quadratic_stress = material(quadratic)
        nonlinear_piola = quadratic_stress + np.einsum("ikeq,kjeq->ijeq", gradient, material(linear) + quadratic_stress)
        correction = -asm(piola_force, self.reference.basis, piola=nonlinear_piola)
        if self.cache_evaluations:
            self._cached_correction = correction.copy()
        return correction

    def corner_stress(self, displacement):
        gradient = self.reference.corner_basis.interpolate(displacement).grad
        _, _, piola = self.tensors(gradient)
        deformation = gradient + np.eye(3)[:, :, None, None]
        determinant = np.linalg.det(deformation.transpose(2, 3, 0, 1))
        assert determinant.min() > 0, "Inverted P2 solid at a corner"
        stress = np.einsum("ikeq,jkeq->ijeq", piola, deformation).transpose(2, 3, 0, 1)
        return stress / determinant[:, :, None, None]
