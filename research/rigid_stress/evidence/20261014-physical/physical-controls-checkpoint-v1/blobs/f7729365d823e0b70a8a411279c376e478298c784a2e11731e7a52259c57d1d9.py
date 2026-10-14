"""Temporary-tensor all-corner peak baseline for measured comparison with fused reduction."""

import cupy as cp
import numpy as np

from .mechanics import EDGES


class P2PeakTensorGPU:
    def __init__(self, gradients: np.ndarray, elements: np.ndarray, shear_pa: float) -> None:
        barycentric = np.eye(4)
        derivative = np.zeros((4, 10, 4))
        for vertex in range(4):
            derivative[:, vertex, vertex] = 4 * barycentric[:, vertex] - 1
        for edge, (first, second) in enumerate(EDGES):
            derivative[:, 4 + edge, first] = 4 * barycentric[:, second]
            derivative[:, 4 + edge, second] = 4 * barycentric[:, first]
        self.gradients = cp.einsum("cni,eij->ecnj", cp.asarray(derivative), cp.asarray(gradients))
        self.elements, self.shear_pa = cp.asarray(elements), shear_pa

    def __call__(self, displacement: cp.ndarray) -> cp.ndarray:
        nodal = displacement.reshape(-1, 3, displacement.shape[1])[self.elements]
        nodal = nodal - nodal[:, :1]
        strain = cp.einsum("enab,ecnd->becad", nodal, self.gradients)
        xx_yy, yy_zz, zz_xx = (
            strain[..., 0, 0] - strain[..., 1, 1],
            strain[..., 1, 1] - strain[..., 2, 2],
            (strain[..., 2, 2] - strain[..., 0, 0]),
        )
        xy, yz, xz = (
            strain[..., 0, 1] + strain[..., 1, 0],
            strain[..., 1, 2] + strain[..., 2, 1],
            (strain[..., 0, 2] + strain[..., 2, 0]),
        )
        squared = 2 * (xx_yy**2 + yy_zz**2 + zz_xx**2) + 3 * (xy**2 + yz**2 + xz**2)
        return self.shear_pa * cp.sqrt(squared.max(axis=(1, 2)))
