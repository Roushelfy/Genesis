"""Common-mother linear shell and genuinely reduced local feature spaces."""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from research.rigid_stress.shell_cpu.core import ShellModel

from .geometry import THICKNESS, Geometry, make_geometry, normalize


class SparseFactor:
    """Solve an equilibrated sparse system while retaining its original operator."""

    def __init__(self, matrix: sparse.spmatrix):
        self.K = matrix.tocsr()
        diagonal = self.K.diagonal()
        if np.any(diagonal <= 0):
            raise ValueError("Reduced stiffness has a nonpositive diagonal.")
        self.scale = 1 / np.sqrt(diagonal)
        scaling = sparse.diags(self.scale)
        start = perf_counter()
        self.lu = splu((scaling @ self.K @ scaling).tocsc(), permc_spec="MMD_AT_PLUS_A",
                        diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        self.seconds = perf_counter() - start

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        scale = self.scale if rhs.ndim == 1 else self.scale[:, None]
        return scale * self.lu.solve(np.asfortranarray(scale * rhs))

    @property
    def payload_bytes(self) -> int:
        matrices = (self.lu.L, self.lu.U)
        return sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices) + self.scale.nbytes


class Mother:
    """Construct fixed geometry, stiffness, mass relief and common stress samples."""

    def __init__(self, ncoarse: int = 3, subdiv: int = 4):
        start = perf_counter()
        self.geometry: Geometry = make_geometry(ncoarse, subdiv)
        g = self.geometry
        self.model = ShellModel(g.vertices, g.quads, E=1e10, nu=0.3, h=THICKNESS,
                                density=1000.0, directors=g.directors, gauge="translation")
        self.K = self.model.K
        self.S = self.model.stress_matrix()
        self.P = sparse.kron(g.P_nodes, sparse.eye(6, format="csr"), format="csr")
        self.n_dof = self.model.ndof
        rows = (6 * g.coarse_nodes[:, None] + np.arange(3)).reshape(-1)
        modes = self.model.rigid_modes[rows].copy()
        length = np.sqrt(np.average(np.sum((g.vertices - self.model.center_of_mass)**2, axis=1),
                                    weights=self.model.nodal_mass))
        modes[:, 3:] /= length
        _, triangular, pivots = linalg.qr(modes.T, pivoting=True, mode="economic")
        if np.min(np.abs(np.diag(triangular[:, :6]))) < 1e-12:
            raise ValueError("Coarse translation rows cannot remove all rigid modes.")
        coarse_translation_rows = (6 * np.arange(len(g.coarse_nodes))[:, None] + np.arange(3)).reshape(-1)
        self.coarse_fixed = np.sort(coarse_translation_rows[pivots[:6]])
        self.coarse_free = np.setdiff1d(np.arange(self.P.shape[1]), self.coarse_fixed)
        self.fixed = 6 * g.coarse_nodes[self.coarse_fixed // 6] + self.coarse_fixed % 6
        self.free = np.setdiff1d(np.arange(self.n_dof), self.fixed)
        self.model.gauge_dofs = self.fixed.copy()
        self.factor = SparseFactor(self.K[self.free][:, self.free])
        self.mass_modes = self.model.mass_modes
        self.relief_coefficients = linalg.solve(self.model.mass_gram, self.model.rigid_modes.T, assume_a="pos")
        self.assembly_seconds = perf_counter() - start
        self.spaces: dict[bytes, Space] = {}

    def balance(self, raw_force: np.ndarray) -> np.ndarray:
        return raw_force - self.mass_modes @ (self.relief_coefficients @ raw_force)

    def solve(self, balanced_force: np.ndarray) -> np.ndarray:
        if balanced_force.shape[0] != self.n_dof:
            raise ValueError("Load leading dimension must equal mother DOFs.")
        solution = np.zeros_like(balanced_force)
        solution[self.free] = self.factor.solve(balanced_force[self.free])
        return solution

    def feature(self, center: np.ndarray, core_angle: float | None = None,
                halo: int = 1, name: str = "", *, core: float | None = None) -> Space:
        if core_angle is None:
            core_angle = core
        if core_angle is None or core_angle <= 0 or halo < 0:
            raise ValueError("A positive angular core and nonnegative halo are required.")
        direction = normalize(np.asarray(center))
        angles = np.arccos(np.clip(self.geometry.macro_centers @ direction, -1, 1))
        active = angles <= core_angle + self.geometry.macro_radii + 1e-14
        for _ in range(halo):
            expanded = active.copy()
            for i_macro in np.flatnonzero(active):
                expanded[self.geometry.macro_neighbors[i_macro]] = True
            active = expanded
        return self.space(active, name=name)

    def space(self, active_macros: np.ndarray, name: str = "") -> Space:
        mask = np.asarray(active_macros)
        if mask.dtype != bool:
            indices = mask
            mask = np.zeros(len(self.geometry.macro_quads), dtype=bool)
            mask[indices] = True
        if mask.shape != (len(self.geometry.macro_quads),):
            raise ValueError("Expected one active flag per mother macro element.")
        key = mask.tobytes()
        if key not in self.spaces:
            is_coarse = np.zeros(len(self.geometry.vertices), dtype=bool)
            is_coarse[self.geometry.coarse_nodes] = True
            local_nodes = np.array([i for i, incident in enumerate(self.geometry.node_macros)
                                    if not is_coarse[i] and np.all(mask[incident])], dtype=int)
            local_dofs = (6 * local_nodes[:, None] + np.arange(6)).reshape(-1)
            bubbles = sparse.coo_matrix((np.ones(len(local_dofs)), (local_dofs, np.arange(len(local_dofs)))),
                                        shape=(self.n_dof, len(local_dofs))).tocsr()
            transform = sparse.hstack([self.P[:, self.coarse_free], bubbles], format="csr")
            self.spaces[key] = Space(self, transform, name=name, active_macros=mask.copy(), local_nodes=local_nodes)
        return self.spaces[key]

    def stress(self, solution: np.ndarray) -> np.ndarray:
        return self.S @ solution


class Space:
    """A conforming coarse-plus-local Galerkin space on the fixed mother model."""

    def __init__(self, mother: Mother, transform: sparse.spmatrix, name: str = "",
                 active_macros: np.ndarray | None = None, local_nodes: np.ndarray | None = None):
        self.mother = mother
        self.name = name
        self.T = transform.tocsr()
        self.n_dof = self.T.shape[1]
        self.n_dof_total = self.n_dof + len(mother.coarse_fixed)
        self.n_coarse = len(mother.coarse_free)
        self.active_macros = active_macros
        self.local_nodes = np.array([], dtype=int) if local_nodes is None else local_nodes
        self.K = (self.T.T @ mother.K @ self.T).tocsr()
        self.S = (mother.S @ self.T).tocsr()
        self.factor = SparseFactor(self.K)
        self.relief_projection = np.asarray(self.T.T @ mother.mass_modes)

    @property
    def bubble_nodes(self) -> np.ndarray:
        return self.local_nodes

    @property
    def active_nodes(self) -> np.ndarray:
        return self.local_nodes

    def solve(self, balanced_force: np.ndarray) -> np.ndarray:
        return self.solve_projected(self.project(balanced_force))

    def project(self, balanced_force: np.ndarray) -> np.ndarray:
        return self.T.T @ balanced_force

    def project_raw(self, raw_force: np.ndarray) -> np.ndarray:
        return self.T.T @ raw_force - self.relief_projection @ (self.mother.relief_coefficients @ raw_force)

    def solve_projected(self, reduced_rhs: np.ndarray) -> np.ndarray:
        return self.factor.solve(reduced_rhs)

    def lift(self, reduced_solution: np.ndarray) -> np.ndarray:
        return self.T @ reduced_solution

    def stress(self, reduced_solution: np.ndarray) -> np.ndarray:
        return self.S @ reduced_solution
