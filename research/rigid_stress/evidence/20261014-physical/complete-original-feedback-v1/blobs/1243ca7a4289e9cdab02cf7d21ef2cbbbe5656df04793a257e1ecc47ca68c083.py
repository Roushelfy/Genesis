"""Cancellation-resistant first Piola stress for the unchanged neo-Hookean law."""

import numpy as np

from .neohookean_reference import NeoHookeanBulk


class StableNeoHookeanBulk(NeoHookeanBulk):
    """Evaluate P=(mu*(B-I)+lambda*log(J)*I)*F**(-T) from H=grad(u).

    B-I=H+H.T+H@H.T avoids subtracting nearly equal identity tensors.
    det(I+H)-1=tr(H)+((tr(H))**2-tr(H@H))/2+det(H) avoids the
    determinant's subsequent subtraction from one. The constitutive law,
    quadrature and analytic tangent are unchanged.
    """

    def evaluate(self, displacement, hessian=True):
        energy, _gradient, matrix = super().evaluate(displacement, hessian)
        h = np.einsum("aie,ajeq->ijeq", displacement[self.dofs], self.gradients, optimize=True)
        h = h.transpose(2, 3, 0, 1)
        tr = np.trace(h, axis1=-2, axis2=-1)
        delta_j = tr + 0.5 * (tr**2 - np.einsum("...ij,...ji->...", h, h)) + np.linalg.det(h)
        if np.any(delta_j <= -1):
            raise ValueError("Neo-Hookean bulk requires positive deformation determinant")
        log_j = np.log1p(delta_j)
        b_minus_i = h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2)
        kirchhoff = self.mu * b_minus_i + self.lam * log_j[..., None, None] * np.eye(3)
        piola = kirchhoff @ np.linalg.inv(np.eye(3) + h).swapaxes(-1, -2)
        local = np.einsum(
            "ijeq,ajeq,eq->aie", piola.transpose(2, 3, 0, 1), self.gradients, self.basis.dx, optimize=True
        )
        gradient = np.zeros(self.reference.basis.N)
        np.add.at(gradient, self.dofs.ravel(), local.ravel())
        return energy, gradient, matrix
