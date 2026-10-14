"""Independent affine quadratic elasticity and boundary integration using scikit-fem."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP2, ElementVector, FacetBasis, LinearForm, MeshTet, asm
from skfem.helpers import dot, sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress


@BilinearForm
def vector_mass(u, v, w):
    return w.density * dot(u, v)


@LinearForm
def constant_stress_traction(v, w):
    traction = np.einsum("ij,jfq->ifq", w.tensor, w.n)
    return dot(traction, v)


@LinearForm
def bending_traction(v, w):
    return w.scale * w.x[2] * w.n[0] * v[0]


@dataclass(frozen=True)
class IndependentElasticSolution:
    displacement: np.ndarray
    rhs: np.ndarray
    tensor_pa: np.ndarray
    von_mises_pa: np.ndarray
    energy_j: float
    residual_n: float
    residual_budget_fraction: float
    peak_pa: float
    peak_position_m: np.ndarray
    solve_seconds: float


class SkfemElasticReference:
    """Assemble and solve a free-link problem without native or P2Shell operators."""

    def __init__(self, vertices, tetrahedra, young, poisson, density, gauge="pinned"):
        self.mesh = MeshTet(vertices.T, tetrahedra.T, sort_t=False)
        self.element = ElementVector(ElementTetP2())
        self.scalar_basis = Basis(self.mesh, ElementTetP2(), intorder=4)
        self.basis = Basis(self.mesh, self.element, intorder=4)
        self.coordinates = self.scalar_basis.doflocs.T
        self.component_dofs = np.stack(self.basis.split_indices(), axis=-1)
        self.young = young
        self.poisson = poisson
        self.density = density
        self.lame = lame_parameters(young, poisson)
        self.stiffness = asm(linear_elasticity(*self.lame), self.basis).tocsr()
        self.mass = asm(vector_mass, self.basis, density=density).tocsr()
        self.rigid_modes = np.zeros((self.basis.N, 6))
        self.rigid_modes[self.component_dofs, np.arange(3)[None]] = 1.0
        scalar_mass = self.mass[self.component_dofs[:, 0]][:, self.component_dofs[:, 0]]
        weights = np.asarray(scalar_mass.sum(axis=1)).ravel()
        self.total_mass = weights.sum()
        self.com = weights @ self.coordinates / self.total_mass
        relative = self.coordinates - self.com
        for axis in range(3):
            self.rigid_modes[self.component_dofs, axis + 3] = np.cross(np.eye(3)[axis], relative)
        self.mass_modes = self.mass @ self.rigid_modes
        self.gram = self.rigid_modes.T @ self.mass_modes
        if gauge not in ("pinned", "mass"):
            raise ValueError(f"Unknown independent rigid gauge: {gauge}")
        self.gauge = gauge
        self.factor = None
        scaled_modes = self.rigid_modes / np.linalg.norm(self.rigid_modes, axis=0)
        _, triangular, pivot = qr(scaled_modes.T, mode="economic", pivoting=True)
        if np.min(np.abs(triangular.diagonal())) <= 1e-12:
            raise ValueError("Mesh does not admit six independent rigid constraints")
        self.gauge_dofs = pivot[:6]
        self.free_dofs = np.setdiff1d(np.arange(self.basis.N), self.gauge_dofs)
        self.factor_seconds = 0.0
        self.factor_nonzeros = 0
        corners = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.corner_basis = Basis(self.mesh, self.element, quadrature=(corners, np.ones(4)))
        self.corner_positions = self.mesh.p[:, self.mesh.t].transpose(2, 1, 0)

    def native_dof_permutation(self, coordinates):
        distance, nodes = cKDTree(self.coordinates).query(coordinates)
        assert np.max(distance) <= 1e-12, np.max(distance)
        assert len(np.unique(nodes)) == len(nodes) == len(self.coordinates)
        return self.component_dofs[nodes].reshape(-1)

    def align_rigid_gauge(self, displacement):
        coefficients = np.linalg.solve(self.gram, self.mass_modes.T @ displacement)
        return displacement - self.rigid_modes @ coefficients

    def balanced_rhs(self, force, omega):
        radius = self.coordinates - self.com
        acceleration = -np.cross(omega, np.cross(omega, radius))
        centrifugal = np.empty(self.basis.N)
        centrifugal[self.component_dofs] = acceleration
        load = force + self.mass @ centrifugal
        return load - self.mass_modes @ np.linalg.solve(self.gram, self.rigid_modes.T @ load)

    def corner_stress(self, displacement):
        field = self.corner_basis.interpolate(displacement)
        tensor = linear_stress(*self.lame)(sym_grad(field)).transpose(2, 3, 0, 1)
        deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        von_mises = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
        return tensor, von_mises

    def uniform_stress_load(self, tensor):
        boundary = FacetBasis(self.mesh, self.element, intorder=4)
        return asm(constant_stress_traction, boundary, tensor=tensor)

    def pure_bending_load(self, curvature):
        boundary = FacetBasis(self.mesh, self.element, intorder=4)
        return asm(bending_traction, boundary, scale=self.young * curvature)

    def factorize(self):
        if self.factor is not None:
            return
        started = perf_counter()
        if self.gauge == "pinned":
            operator = self.stiffness[self.free_dofs][:, self.free_dofs] / self.young
        else:
            constraint = self.mass_modes / np.linalg.norm(self.mass_modes, axis=0)
            operator = sparse.bmat(
                [[self.stiffness / self.young, sparse.csr_matrix(constraint)], [sparse.csr_matrix(constraint.T), None]],
                format="csc",
            )
        self.factor = splu(operator.tocsc())
        self.factor_seconds = perf_counter() - started
        self.factor_nonzeros = self.factor.L.nnz + self.factor.U.nnz

    def solve(self, force, omega=None):
        started = perf_counter()
        if omega is None:
            omega = np.zeros(3)
        rhs = self.balanced_rhs(force, omega)
        self.factorize()
        if self.gauge == "pinned":
            displacement = np.zeros(self.basis.N)
            displacement[self.free_dofs] = self.factor.solve(rhs[self.free_dofs] / self.young)
        else:
            displacement = self.factor.solve(np.r_[rhs / self.young, np.zeros(6)])[: self.basis.N]
        displacement = self.align_rigid_gauge(displacement)
        tensor, vm = self.corner_stress(displacement)
        residual = np.linalg.norm(self.stiffness @ displacement - rhs)
        budget = max(1e-11, 1e-8 * np.linalg.norm(rhs))
        element, corner = np.unravel_index(np.argmax(vm), vm.shape)
        return IndependentElasticSolution(
            displacement=displacement,
            rhs=rhs,
            tensor_pa=tensor,
            von_mises_pa=vm,
            energy_j=0.5 * displacement @ (self.stiffness @ displacement),
            residual_n=residual,
            residual_budget_fraction=residual / budget,
            peak_pa=vm[element, corner],
            peak_position_m=self.corner_positions[element, corner],
            solve_seconds=perf_counter() - started,
        )

    def stress_at_points(self, displacement, points):
        elements = self.mesh.element_finder()(*points.T)
        reference = self.basis.mapping.invF(points.T[:, :, None], tind=elements)[:, :, 0]
        barycentric = np.column_stack((1 - reference.sum(axis=0), reference.T))
        corners, _ = self.corner_stress(displacement)
        tensor = np.einsum("nc,ncij->nij", barycentric, corners[elements])
        deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        return tensor, np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
