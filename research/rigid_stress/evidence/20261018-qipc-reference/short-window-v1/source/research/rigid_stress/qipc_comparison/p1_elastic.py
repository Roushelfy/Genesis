"""Independent CPU P1 Lamé operator and Cauchy stress in the small-strain limit."""

import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse.linalg import splu


def independent_operator(vertices, tetrahedra, young, poisson, density):
    inverse = np.linalg.inv((vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]).transpose(0, 2, 1))
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    volume = 1 / np.abs(np.linalg.det(inverse)) / 6
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    local = (
        lame * np.einsum("tai,tbj->taibj", gradients, gradients)
        + mu * np.einsum("tak,tbk,ij->taibj", gradients, gradients, np.eye(3))
        + mu * np.einsum("taj,tbi->taibj", gradients, gradients)
    ) * volume[:, None, None, None, None]
    dofs = (tetrahedra[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
    rows = np.broadcast_to(dofs[:, :, None], local.reshape(-1, 12, 12).shape)
    columns = np.broadcast_to(dofs[:, None, :], rows.shape)
    matrix = sparse.coo_matrix(
        (local.reshape(-1), (rows.reshape(-1), columns.reshape(-1))), shape=(len(vertices) * 3, len(vertices) * 3)
    ).tocsr()
    masses = np.bincount(tetrahedra.reshape(-1), weights=np.repeat(volume * density / 4, 4), minlength=len(vertices))
    return matrix, masses, gradients, volume


def linear_stress(displacement, tetrahedra, gradients, young, poisson):
    gradient = np.einsum("tai,taj->tij", displacement[tetrahedra], gradients)
    strain = (gradient + gradient.transpose(0, 2, 1)) / 2
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return 2 * mu * strain + lame * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(3)


class FreeElasticP1:
    def __init__(self, vertices, tetrahedra, young, poisson, density):
        self.vertices = vertices
        self.tetrahedra = tetrahedra
        self.young, self.poisson = young, poisson
        self.stiffness, self.masses, self.gradients, self.volumes = independent_operator(
            vertices, tetrahedra, young, poisson, density
        )
        center = self.masses @ vertices / self.masses.sum()
        self.rigid_modes = np.zeros((len(vertices), 3, 6))
        self.rigid_modes[:, :, :3] = np.eye(3)
        for axis in range(3):
            self.rigid_modes[:, :, axis + 3] = np.cross(np.eye(3)[axis], vertices - center)
        self.rigid_modes = self.rigid_modes.reshape(-1, 6)
        self.mass_modes = self.rigid_modes * np.repeat(self.masses, 3)[:, None]
        self.gram = self.rigid_modes.T @ self.mass_modes
        scaled = self.rigid_modes / np.linalg.norm(self.rigid_modes, axis=0)
        _, triangular, pivot = qr(scaled.T, mode="economic", pivoting=True)
        assert np.abs(triangular.diagonal()).min() > 1e-12
        self.free = np.setdiff1d(np.arange(len(vertices) * 3), pivot[:6])
        self.factor = splu((self.stiffness[self.free][:, self.free] / young).tocsc())

    def solve(self, load):
        force = load.reshape(-1)
        correction = self.mass_modes @ np.linalg.solve(self.gram, self.rigid_modes.T @ force)
        rhs = force - correction
        displacement = np.zeros_like(force)
        displacement[self.free] = self.factor.solve(rhs[self.free] / self.young)
        displacement -= self.rigid_modes @ np.linalg.solve(self.gram, self.mass_modes.T @ displacement)
        residual = self.stiffness @ displacement - rhs
        stress = linear_stress(displacement.reshape(-1, 3), self.tetrahedra, self.gradients, self.young, self.poisson)
        return stress, displacement.reshape(-1, 3), correction.reshape(-1, 3), residual.reshape(-1, 3)
