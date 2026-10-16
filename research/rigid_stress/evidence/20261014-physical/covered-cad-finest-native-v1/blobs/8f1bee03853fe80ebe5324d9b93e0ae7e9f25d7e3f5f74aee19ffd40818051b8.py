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

    def corner_deformation_gradients(self, displacement):
        corners = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        basis = Basis(self.reference.mesh, ElementTetP2(), quadrature=(corners, np.ones(4)))
        gradients = np.stack([entry[0].grad for entry in basis.basis])
        return self.deformation_gradients(displacement, gradients).transpose(2, 3, 0, 1)

    @staticmethod
    def determinant_coefficients(f):
        grouped = {}
        for first, second, third in product(range(4), repeat=3):
            key = tuple(sorted((first, second, third)))
            columns = np.stack((f[:, first, :, 0], f[:, second, :, 1], f[:, third, :, 2]), axis=-1)
            grouped.setdefault(key, []).append(np.linalg.det(columns))
        return np.stack([np.mean(values, axis=0) for values in grouped.values()])

    def determinant_lower_bound(self, displacement):
        """Bound J everywhere by cubic barycentric Bernstein coefficients.

        P2 displacement gives affine F inside each affine tetrahedron. Expanding
        its three columns by the four vertex values gives the complete cubic
        determinant polynomial. Positive coefficients prove positive J across
        every element, rather than only at selected quadrature/corner points.
        """
        coefficients = self.determinant_coefficients(self.corner_deformation_gradients(displacement))
        return float(coefficients.min())

    def stress_bounds(self, displacement):
        """Bound the complete nonlinear global VM maximum without smoothing.

        The deviatoric Kirchhoff stress is mu*dev(F F.T), quadratic in barycentric
        coordinates. Its Bernstein coefficient norms bound its norm everywhere.
        Dividing by a positive cubic determinant lower bound gives a conservative
        per-element Cauchy VM upper bound; actual corner samples give a lower
        bound. The interval must be reported, not silently called a corner maximum.
        """
        f = self.corner_deformation_gradients(displacement)
        j = np.linalg.det(f)
        b = f @ f.swapaxes(-1, -2)
        tensor = (self.mu * (b - np.eye(3)) + self.lam * np.log(j)[..., None, None] * np.eye(3)) / j[..., None, None]
        deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
        lower_j = self.determinant_coefficients(f).min(axis=0)
        if np.any(lower_j <= 0):
            raise ValueError("No positive whole-element determinant certificate")
        coefficients = []
        for first in range(4):
            for second in range(first, 4):
                product_tensor = 0.5 * (
                    f[:, first] @ f[:, second].swapaxes(-1, -2) + f[:, second] @ f[:, first].swapaxes(-1, -2)
                )
                product_tensor -= np.trace(product_tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
                coefficients.append(self.mu * np.sqrt(1.5 * np.sum(product_tensor**2, axis=(-2, -1))))
        upper = np.max(coefficients, axis=0) / lower_j
        upper += 128 * np.finfo(float).eps * self.mu * np.max(np.sum(f**2, axis=(-2, -1)), axis=1) / lower_j
        return tensor, vm, upper


class NeoHookeanPadContactSystem(IndependentContactSystem):
    """Replace only the shoe bulk; keep the body/contact/control equations fixed."""

    def __init__(self, references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor=False):
        super().__init__(references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor)
        self.pad_bulk = [(part, NeoHookeanBulk(part.reference)) for part in self.parts if part.name != "body"]
        self.body_bulk = None
        self.nonlinear_bulk = self.pad_bulk

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact_gradient = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        corrections = []
        if hessian and self.body_bulk is None:
            corrections.append(sparse.csc_matrix(self.parts[0].reference.stiffness.shape))
        for part, bulk in self.nonlinear_bulk:
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


class NeoHookeanAllContactSystem(NeoHookeanPadContactSystem):
    """Objective finite-deformation body and shoes with the same E, nu and mass."""

    def __init__(self, references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor=False):
        super().__init__(references, dhat, barrier_stiffness, friction, friction_mollifier, reuse_empty_factor)
        self.body_bulk = NeoHookeanBulk(self.parts[0].reference)
        self.nonlinear_bulk = [(self.parts[0], self.body_bulk), *self.pad_bulk]
