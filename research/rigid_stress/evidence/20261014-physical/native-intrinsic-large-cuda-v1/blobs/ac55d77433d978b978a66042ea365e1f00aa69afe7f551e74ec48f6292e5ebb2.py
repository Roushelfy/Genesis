"""Extended-precision force contraction for the unchanged neo-Hookean reference."""

import numpy as np

from .stable_neohookean_reference import StableNeoHookeanBulk


class ExtendedPiolaBulk(StableNeoHookeanBulk):
    """Retain the original law/quadrature/tangent and accumulate Piola forces in long double."""

    def evaluate(self, displacement, hessian=True):
        energy, _, matrix = super().evaluate(displacement, hessian)
        assert np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
        h = np.einsum(
            "aie,ajeq->ijeq",
            np.asarray(displacement[self.dofs], dtype=np.longdouble),
            np.asarray(self.gradients, dtype=np.longdouble),
            optimize=True,
        ).transpose(2, 3, 0, 1)
        tr = np.trace(h, axis1=-2, axis2=-1)
        det_h = np.sum(h[..., 0, :] * np.cross(h[..., 1, :], h[..., 2, :]), axis=-1)
        delta = tr + (tr**2 - np.einsum("...ij,...ji->...", h, h)) / 2 + det_h
        if np.any(delta <= -1):
            raise ValueError("Neo-Hookean bulk requires positive deformation determinant")
        identity = np.eye(3, dtype=np.longdouble)
        f = identity + h
        cofactor = np.stack(
            [
                np.cross(f[..., :, 1], f[..., :, 2]),
                np.cross(f[..., :, 2], f[..., :, 0]),
                np.cross(f[..., :, 0], f[..., :, 1]),
            ],
            axis=-1,
        )
        kirchhoff = self.mu * (h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2))
        kirchhoff += self.lam * np.log1p(delta)[..., None, None] * identity
        piola = (kirchhoff @ cofactor) / (1 + delta)[..., None, None]
        local = np.einsum(
            "ijeq,ajeq,eq->aie",
            piola.transpose(2, 3, 0, 1),
            np.asarray(self.gradients, dtype=np.longdouble),
            np.asarray(self.basis.dx, dtype=np.longdouble),
            optimize=True,
        )
        gradient = np.zeros(self.reference.basis.N, dtype=np.longdouble)
        np.add.at(gradient, self.dofs.ravel(), local.ravel())
        return energy, np.asarray(gradient, dtype=float), matrix
