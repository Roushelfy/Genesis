"""Optional CUDA-capable Torch prototype, NOT the production sparse solver.

Small-mesh dense Cholesky validates device-side batching and peak reduction.
It deliberately refuses large dense systems. The agent must replace this
factor backend with a shared sparse factor implementation before scaling.
No CUDA timing or accuracy result is claimed for this file.
"""
from dataclasses import dataclass

import torch


@dataclass
class DenseDebugFactor:
    lower: torch.Tensor

    @classmethod
    def build(cls, stiffness, device="cuda", dtype=torch.float64, max_dofs=3000):
        if stiffness.shape[0] > max_dofs:
            raise ValueError("Dense debug factor exceeds max_dofs; implement shared sparse solves")
        matrix = torch.as_tensor(stiffness.toarray(), dtype=dtype, device=device)
        return cls(torch.linalg.cholesky(matrix))

    def solve(self, rhs):
        """One matrix, many RHS [free_dof,B]; one factor shared by all envs."""
        return torch.cholesky_solve(rhs, self.lower)


class P2PeakTorch:
    """Device-side full-domain P2 corner scan; no fixed hotspot or contact basis.

    This unfused implementation is an accuracy oracle/prototype. Full stress
    tensors are temporary per tile, not returned for all elements. A fused
    Triton/CUDA/Quadrants implementation should return only max/argmax.
    """
    def __init__(self, gradients, elements, mu, device="cuda", dtype=torch.float64, tile=256):
        self.gradients = torch.as_tensor(gradients, device=device, dtype=dtype)
        self.elements = torch.as_tensor(elements, device=device, dtype=torch.int64)
        self.mu = mu
        self.tile = tile
        self.midpoints = ((0, 4, 5, 6), (4, 1, 7, 8), (5, 7, 2, 9), (6, 8, 9, 3))

    def __call__(self, displacement):
        # displacement [B,node,3]. No scalar .item(), host copies or per-env loops.
        best = torch.zeros(displacement.shape[0], dtype=displacement.dtype, device=displacement.device)
        for lo in range(0, self.elements.shape[0], self.tile):
            nodes = self.elements[lo:lo + self.tile]
            gradient = self.gradients[lo:lo + self.tile]
            local = displacement[:, nodes]
            local = local - local[:, :, :1]
            base = -torch.einsum("ntia,tic->ntac", local[:, :, :4], gradient)
            for corner in range(4):
                du = base
                for vertex in range(4):
                    local_node = corner if vertex == corner else self.midpoints[corner][vertex]
                    du = du + 4 * local[:, :, local_node, :, None] * gradient[None, :, vertex, None, :]
                xx, yy, zz = du[..., 0, 0], du[..., 1, 1], du[..., 2, 2]
                xy = du[..., 0, 1] + du[..., 1, 0]
                yz = du[..., 1, 2] + du[..., 2, 1]
                xz = du[..., 0, 2] + du[..., 2, 0]
                squared = 2 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2) + 3 * (xy**2 + yz**2 + xz**2)
                best = torch.maximum(best, squared.amax(dim=1))
        return self.mu * torch.sqrt(best)


def compact_correct(factor, residual_free, failed, displacement_free):
    """Prototype gather/solve/scatter. Torch nonzero may synchronize on CUDA.

    The optimized version needs device selection with captured fixed/bucketed
    capacities or an explicitly timed count transfer. Zero failure is valid.
    """
    ids = torch.nonzero(failed, as_tuple=True)[0]
    if ids.numel():
        correction = factor.solve(residual_free[:, ids].contiguous())
        displacement_free[:, ids] += correction
    return displacement_free
