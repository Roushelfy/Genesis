"""Independent affine-P2 compressible neo-Hookean bulk and pad-contact reference."""

from itertools import product

import numpy as np
from scipy import sparse
from skfem import Basis, ElementTetP2

from .contact_reference import IndependentContactSystem

FORMULATION_SOURCE = "https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html"


class NeoHookeanBulk:
    """Integrate W=mu/2*(tr(F.T F)-3)-mu*log(J)+lambda/2*log(J)^2.

    scikit-fem supplies the independent P2 basis, topology, reference gradients
    and volume quadrature. NumPy contracts local forces/tangents and SciPy
    assembles them. E, nu, geometry and mass are inherited without retuning.
    """

    def __init__(self, reference, quadrature_order=6):
        self.reference = reference
        self.lam, self.mu = reference.lame
        self.basis = Basis(reference.mesh, ElementTetP2(), intorder=quadrature_order)
        self.gradients = np.stack([entry[0].grad for entry in self.basis.basis])
        self.dofs = reference.component_dofs[self.basis.element_dofs].transpose(0, 2, 1)
        self.scalar_stiffness = np.einsum(
            "ajeq,bjeq,eq->abe", self.gradients, self.gradients, self.basis.dx, optimize=True
        )
        self.quadrature_order = quadrature_order

    def deformation_gradients(self, displacement, gradients=None):
        gradients = self.gradients if gradients is None else gradients
        local = displacement[self.dofs]
        return np.eye(3)[:, :, None, None] + np.einsum("aie,ajeq->ijeq", local, gradients, optimize=True)

    def evaluate(self, displacement, hessian=True):
        f = self.deformation_gradients(displacement)
        matrices = f.transpose(2, 3, 0, 1)
        sign, logj = np.linalg.slogdet(matrices)
        if np.any(sign <= 0):
            raise ValueError("Neo-Hookean bulk requires positive deformation determinant")
        inverse_transpose = np.linalg.inv(matrices).transpose(3, 2, 0, 1)
        # Singular stretches avoid cancellation of the rigid-rotation energy.
        stretches = np.linalg.svd(matrices, compute_uv=False)
        delta = stretches - 1
        energy_density = self.mu / 2 * np.sum(delta**2 + 2 * (delta - np.log1p(delta)), axis=-1)
        energy_density += self.lam / 2 * logj**2
        piola = self.mu * (f - inverse_transpose) + self.lam * logj[None, None] * inverse_transpose
        local_force = np.einsum("ijeq,ajeq,eq->aie", piola, self.gradients, self.basis.dx, optimize=True)
        gradient = np.zeros(self.reference.basis.N)
        np.add.at(gradient, self.dofs.ravel(), local_force.ravel())
        matrix = None
        if hessian:
            transformed = np.einsum("ijeq,ajeq->aieq", inverse_transpose, self.gradients, optimize=True)
            local_matrix = self.mu * np.einsum("abe,ik->aibke", self.scalar_stiffness, np.eye(3))
            if self.lam:
                local_matrix += self.lam * np.einsum(
                    "aieq,bkeq,eq->aibke", transformed, transformed, self.basis.dx, optimize=True
                )
            local_matrix += np.einsum(
                "akeq,bieq,eq->aibke",
                transformed,
                transformed,
                self.basis.dx * (self.mu - self.lam * logj),
                optimize=True,
            )
            local_matrix = local_matrix.reshape(30, 30, -1)
            local_dofs = self.dofs.reshape(30, -1)
            rows = np.broadcast_to(local_dofs[:, None], local_matrix.shape)
            columns = np.broadcast_to(local_dofs[None], local_matrix.shape)
            matrix = sparse.coo_matrix(
                (local_matrix.ravel(), (rows.ravel(), columns.ravel())),
                shape=(self.reference.basis.N, self.reference.basis.N),
            ).tocsc()
        return float(np.sum(energy_density * self.basis.dx)), gradient, matrix

    def determinant_lower_bound(self, displacement):
        """Bound J everywhere by cubic barycentric Bernstein coefficients.

        P2 displacement gives affine F inside each affine tetrahedron. Expanding
        its three columns by the four vertex values gives the complete cubic
        determinant polynomial. Positive coefficients prove positive J across
        every element, rather than only at selected quadrature/corner points.
        """
        corners = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        basis = Basis(self.reference.mesh, ElementTetP2(), quadrature=(corners, np.ones(4)))
        gradients = np.stack([entry[0].grad for entry in basis.basis])
        f = self.deformation_gradients(displacement, gradients).transpose(2, 3, 0, 1)
        grouped = {}
        for first, second, third in product(range(4), repeat=3):
            key = tuple(sorted((first, second, third)))
            columns = np.stack((f[:, first, :, 0], f[:, second, :, 1], f[:, third, :, 2]), axis=-1)
            grouped.setdefault(key, []).append(np.linalg.det(columns))
        coefficients = np.stack([np.mean(values, axis=0) for values in grouped.values()])
        return float(coefficients.min())


class NeoHookeanPadContactSystem(IndependentContactSystem):
    """Replace only the shoe bulk; keep the body/contact/control equations fixed."""

    def __init__(self, references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor=False):
        super().__init__(references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor)
        self.pad_bulk = [(part, NeoHookeanBulk(part.reference)) for part in self.parts if part.name != "body"]

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact_gradient = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        corrections = [sparse.csc_matrix(self.parts[0].reference.stiffness.shape)] if hessian else []
        for part, bulk in self.pad_bulk:
            count = part.reference.basis.N
            section = slice(part.first_node * 3, part.first_node * 3 + count)
            vector = np.empty(count)
            vector[part.permutation] = displacement[section]
            linear = part.reference.align_rigid_gauge(vector)
            linear_gradient = part.reference.stiffness @ linear
            try:
                new_energy, new_gradient, new_matrix = bulk.evaluate(vector, hessian)
            except ValueError:
                if hessian:
                    raise
                return float("inf"), np.full_like(gradient, np.inf), None, normal, contact_gradient
            energy += new_energy - 0.5 * linear @ linear_gradient
            gradient[section] += (new_gradient - linear_gradient)[part.permutation]
            if hessian:
                difference = new_matrix - part.reference.stiffness
                corrections.append(difference[part.permutation][:, part.permutation])
        if hessian:
            matrix += sparse.block_diag(corrections, format="csc")
        return energy, gradient, matrix, normal, contact_gradient
