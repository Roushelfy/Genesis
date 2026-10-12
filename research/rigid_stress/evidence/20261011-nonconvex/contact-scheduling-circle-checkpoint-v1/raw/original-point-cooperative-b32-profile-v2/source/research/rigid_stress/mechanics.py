"""Full affine P2 shell operators and an offline FP64 sparse direct oracle.

Assembly, consistent mass and rigid-mode relief use the same mathematical model as the preserved research reference.
Surface-column nested dissection changes only the ordering of free unknowns. No symmetry or contact basis is used.
"""

from dataclasses import dataclass
from itertools import combinations
from math import factorial, prod
from time import perf_counter

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import SuperLU, splu

from .cholesky import CholeskyDirectFactor

try:
    import pymetis
except ModuleNotFoundError as import_error:
    if import_error.name != "pymetis":
        raise
    pymetis = None

EDGES = np.array(list(combinations(range(4), 2)))
TET_QUAD = np.full((4, 4), (5 - np.sqrt(5)) / 20)
np.fill_diagonal(TET_QUAD, (5 + 3 * np.sqrt(5)) / 20)


@dataclass(frozen=True)
class ShellMeshMetadata:
    surface_refinement: int
    through_thickness_layers: int
    thickness_m: float
    taper: float
    outer_dimensions_m: np.ndarray
    outer_surface_vertices: int
    outer_surface_triangles: int
    minimum_tet_volume_m3: float


def shell_mesh(
    level: int, layers: int, thickness: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ShellMeshMetadata]:
    if level < 0 or layers < 1 or not 0 < thickness < 0.01:
        raise ValueError("Nonnegative surface refinement, positive layers and a thin positive wall are required")
    golden = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            [-1, golden, 0],
            [1, golden, 0],
            [-1, -golden, 0],
            [1, -golden, 0],
            [0, -1, golden],
            [0, 1, golden],
            [0, -1, -golden],
            [0, 1, -golden],
            [golden, 0, -1],
            [golden, 0, 1],
            [-golden, 0, -1],
            [-golden, 0, 1],
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    # Preserve reference vertex numbering so operator equality can be tested independently of node permutations.
    for _ in range(level):
        points = vertices.tolist()
        edge_cache: dict[tuple[int, int], int] = {}
        refined = []
        for i, j, k in faces:
            mids = []
            for a, b in ((i, j), (j, k), (k, i)):
                key = (int(min(a, b)), int(max(a, b)))
                if key not in edge_cache:
                    point = vertices[a] + vertices[b]
                    point /= np.linalg.norm(point)
                    edge_cache[key] = len(points)
                    points.append(point.tolist())
                mids.append(edge_cache[key])
            ij, jk, ki = mids
            refined.extend([[i, ij, ki], [j, jk, ij], [k, ki, jk], [ij, jk, ki]])
        vertices, faces = np.asarray(points), np.asarray(refined)
    scale = 1 - 0.18 * vertices[:, 2]
    outer = vertices * np.column_stack((0.022 * scale, 0.022 * scale, np.full(len(vertices), 0.030)))
    gradient = np.column_stack(
        (
            2 * vertices[:, 0] / (0.022 * scale),
            2 * vertices[:, 1] / (0.022 * scale),
            2 / 0.030 * (vertices[:, 2] + 0.18 * (vertices[:, 0] ** 2 + vertices[:, 1] ** 2) / scale),
        )
    )
    normal = gradient / np.linalg.norm(gradient, axis=1)[:, None]
    xyz = np.concatenate([outer - depth * normal for depth in np.linspace(0, thickness, layers + 1)])
    n_surface = len(vertices)
    tetrahedra = []
    for layer in range(layers):
        for i, j, k in np.sort(faces, axis=1):
            a, b, c = np.array([i, j, k]) + layer * n_surface
            d, e, f = np.array([i, j, k]) + (layer + 1) * n_surface
            tetrahedra.extend([[a, b, c, f], [a, b, e, f], [a, d, e, f]])
    tets = np.asarray(tetrahedra)
    points = xyz[tets]
    volume = abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
    metadata = ShellMeshMetadata(
        level, layers, thickness, 0.18, np.ptp(outer, axis=0), n_surface, len(faces), float(volume.min())
    )
    if not (volume > 0).all():
        raise ValueError("The shell mesh contains degenerate tetrahedra")
    return xyz, tets, faces, metadata


def p2_mass_template() -> np.ndarray:
    terms = []
    for i in range(4):
        exponent = np.zeros(4, dtype=int)
        exponent[i] = 1
        terms.append([(2.0, 2 * exponent), (-1.0, exponent)])
    for i, j in EDGES:
        exponent = np.zeros(4, dtype=int)
        exponent[i] = exponent[j] = 1
        terms.append([(4.0, exponent)])
    result = np.zeros((10, 10))
    for i, left in enumerate(terms):
        for j, right in enumerate(terms):
            for a, alpha in left:
                for b, beta in right:
                    exponent = alpha + beta
                    result[i, j] += (
                        a * b * 6 * prod(factorial(int(k)) for k in exponent) / factorial(3 + int(exponent.sum()))
                    )
    return result


def shape_gradients(grad_lambda: np.ndarray, bary: np.ndarray) -> np.ndarray:
    vertex = (4 * bary[None, :, :, None] - 1) * grad_lambda[:, None]
    edge = 4 * (
        bary[None, :, EDGES[:, 0], None] * grad_lambda[:, None, EDGES[:, 1]]
        + bary[None, :, EDGES[:, 1], None] * grad_lambda[:, None, EDGES[:, 0]]
    )
    return np.concatenate((vertex, edge), axis=2)


def gauge_rows(rigid_modes: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    if order is None:
        order = np.arange(len(rigid_modes))
    if not np.array_equal(np.sort(order), np.arange(len(rigid_modes))):
        raise ValueError("Gauge candidate order must permute all scalar DOFs")
    _, upper, pivots = linalg.qr(rigid_modes.T[:, order], mode="economic", pivoting=True)
    if abs(upper[5, 5]) <= 1e-12 * abs(upper[0, 0]):
        raise ValueError("Six independent rigid modes are required")
    return order[pivots[:6]]


class SparseAccumulator:
    """Merge assembly chunks in a binary tree to bound memory and avoid quadratic repeated matrix additions."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.parts: list[sparse.csc_matrix | None] = []

    def add(self, matrix: sparse.csc_matrix) -> None:
        level = 0
        while level < len(self.parts) and self.parts[level] is not None:
            matrix = self.parts[level] + matrix
            self.parts[level] = None
            level += 1
        if level == len(self.parts):
            self.parts.append(matrix)
        else:
            self.parts[level] = matrix

    def finish(self) -> sparse.csc_matrix:
        result = sparse.csc_matrix(self.shape)
        for part in self.parts:
            if part is not None:
                result += part
        self.parts.clear()
        return result


class P2Shell:
    def __init__(
        self,
        xyz: np.ndarray,
        tetrahedra: np.ndarray,
        outer: np.ndarray,
        young: float,
        poisson: float,
        density: float,
        layers: int,
        ordering: str = "mmd",
        factor_backend: str = "superlu",
    ) -> None:
        if young <= 0 or density <= 0 or not -1 < poisson < 0.5:
            raise ValueError("A positive material modulus/density and admissible Poisson ratio are required")
        start = perf_counter()
        self.young, self.poisson, self.wall_layers = young, poisson, layers
        self.order, self.quarter, self.nloc = 2, False, 10
        self.base_xyz, self.base_tets, self.outer_faces = xyz, tetrahedra, outer
        self.nbase, self.ne = len(xyz), len(tetrahedra)
        unique, inverse = np.unique(np.sort(tetrahedra[:, EDGES].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
        self.edge_keys = unique[:, 0] * self.nbase + unique[:, 1]
        self.xyz = np.concatenate((xyz, xyz[unique].mean(axis=1)))
        self.elements = np.column_stack((tetrahedra, self.nbase + inverse.reshape(-1, 6)))
        self.ndof = 3 * len(self.xyz)
        self.dofs = (3 * self.elements[:, :, None] + np.arange(3)).reshape(self.ne, 30)
        points = xyz[tetrahedra]
        matrix = np.concatenate((np.ones((self.ne, 4, 1)), points), axis=2)
        self.glambda = np.linalg.inv(matrix)[:, 1:].transpose(0, 2, 1)
        self.volume = abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
        lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        mu = young / (2 * (1 + poisson))
        self.d = np.diag([2 * mu] * 3 + [mu] * 3)
        self.d[:3, :3] += lam
        # Bound transient assembly memory by writing CSR blocks in element chunks.
        stiffness = SparseAccumulator((self.ndof, self.ndof))
        for lo in range(0, self.ne, 1024):
            hi = min(lo + 1024, self.ne)
            gradients = shape_gradients(self.glambda[lo:hi], TET_QUAD)
            values = lam * np.einsum("eqia,eqjb->eiajb", gradients, gradients, optimize=True)
            values += mu * np.einsum("eqib,eqja->eiajb", gradients, gradients, optimize=True)
            dot = np.einsum("eqic,eqjc->eij", gradients, gradients, optimize=True)
            for axis in range(3):
                values[:, :, axis, :, axis] += mu * dot
            values = values.reshape(-1, 30, 30) * self.volume[lo:hi, None, None] / 4
            rows = np.repeat(self.dofs[lo:hi], 30, axis=1).ravel()
            columns = np.tile(self.dofs[lo:hi], (1, 30)).ravel()
            stiffness.add(sparse.coo_matrix((values.ravel(), (rows, columns)), shape=stiffness.shape).tocsc())
        self.k = stiffness.finish()
        self.k.eliminate_zeros()
        scalar_mass = SparseAccumulator((len(self.xyz), len(self.xyz)))
        template = p2_mass_template()
        for lo in range(0, self.ne, 4096):
            hi = min(lo + 4096, self.ne)
            values = (density * self.volume[lo:hi, None, None] * template).ravel()
            rows = np.repeat(self.elements[lo:hi], 10, axis=1).ravel()
            columns = np.tile(self.elements[lo:hi], (1, 10)).ravel()
            scalar_mass.add(sparse.coo_matrix((values, (rows, columns)), shape=scalar_mass.shape).tocsc())
        self.m = sparse.kron(scalar_mass.finish(), sparse.eye(3), format="csc")
        self.m.eliminate_zeros()
        nodal_mass = np.asarray(self.m.sum(axis=1)).ravel()[::3]
        self.mass = float(nodal_mass.sum())
        self.com = nodal_mass @ self.xyz / self.mass
        self.r = np.zeros((self.ndof, 6))
        self.r.reshape(-1, 3, 6)[:, :, :3] = np.eye(3)
        relative = self.xyz - self.com
        self.r[0::3, 4], self.r[0::3, 5] = relative[:, 2], -relative[:, 1]
        self.r[1::3, 3], self.r[1::3, 5] = -relative[:, 2], relative[:, 0]
        self.r[2::3, 3], self.r[2::3, 4] = relative[:, 1], -relative[:, 0]
        self.mr = self.m @ self.r
        self.gram = self.r.T @ self.mr
        self.gfactor = linalg.cho_factor(self.gram, lower=True)
        self.pins = gauge_rows(self.r)
        self.free = np.setdiff1d(np.arange(self.ndof), self.pins)
        self.assembly_s = perf_counter() - start
        start = perf_counter()
        self.ordering = ordering
        if ordering == "column-nd":
            self.free = self.column_order()
            permc_spec = "NATURAL"
        elif ordering == "mmd":
            permc_spec = "MMD_AT_PLUS_A"
        else:
            raise ValueError("Ordering must be mmd or column-nd")
        self.ordering_s = perf_counter() - start
        start = perf_counter()
        self.factor_backend, self.is_unit_lower = factor_backend, factor_backend == "superlu"
        if factor_backend == "superlu":
            self.factor: SuperLU | CholeskyDirectFactor | None = splu(
                self.k[self.free][:, self.free].tocsc(),
                permc_spec=permc_spec,
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        elif factor_backend == "cholmod" and ordering == "column-nd":
            self.factor = CholeskyDirectFactor(self.k[self.free][:, self.free].tocsc())
        elif factor_backend == "none":
            # Device benchmarks can assemble the exact operators without building an unused CPU factor.
            self.factor = None
        else:
            raise ValueError("Use superlu, cholmod with explicit column-nd ordering, or none for operator assembly")
        self.factor_s = perf_counter() - start

    def column_order(self) -> np.ndarray:
        if pymetis is None:
            raise ModuleNotFoundError("Column nested dissection requires the optional pymetis dependency")

        n_surface = self.nbase // (self.wall_layers + 1)
        a, b = (self.edge_keys // self.nbase) % n_surface, (self.edge_keys % self.nbase) % n_surface
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        distinct = lo != hi
        keys, inverse = np.unique(lo[distinct] * n_surface + hi[distinct], return_inverse=True)
        mids = lo.copy()
        mids[distinct] = n_surface + inverse
        columns = np.r_[np.arange(self.nbase) % n_surface, mids]
        face_edges = np.sort(self.outer_faces[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
        edge_columns = n_surface + np.searchsorted(keys, face_edges[:, :, 0] * n_surface + face_edges[:, :, 1])
        face_columns = np.column_stack((self.outer_faces, edge_columns))
        rows = np.repeat(face_columns, 6, axis=1).ravel()
        cols = np.tile(face_columns, (1, 6)).ravel()
        graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_surface + len(keys),) * 2).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        _, inverse_columns = pymetis.nested_dissection(
            adjacency=pymetis.CSRAdjacency(graph.indptr, graph.indices), options=pymetis.Options(seed=17)
        )
        node_order = np.argsort(np.asarray(inverse_columns)[columns], kind="stable")
        dof_order = (3 * node_order[:, None] + np.arange(3)).ravel()
        return dof_order[~np.isin(dof_order, self.pins)]

    def balance(self, raw: np.ndarray) -> np.ndarray:
        return raw - self.mr @ linalg.cho_solve(self.gfactor, self.r.T @ raw)

    def canonical(self, displacement: np.ndarray) -> np.ndarray:
        return displacement - self.r @ linalg.cho_solve(self.gfactor, self.r.T @ (self.m @ displacement))

    def solve(self, raw: np.ndarray, alternate_gauge: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        rhs = self.balance(raw)
        free, factor = self.free, self.factor
        if alternate_gauge:
            pins = gauge_rows(self.r, np.arange(self.ndof)[::-1])
            free = np.setdiff1d(np.arange(self.ndof), pins)
            factor = splu(self.k[free][:, free], permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
        displacement = np.zeros_like(rhs)
        displacement[free] = factor.solve(np.asfortranarray(rhs[free]))
        residual = float(np.linalg.norm(self.k @ displacement - rhs) / max(np.linalg.norm(rhs), 1))
        return displacement, rhs, residual

    def strain(self, displacement: np.ndarray, bary: np.ndarray):
        for lo in range(0, self.ne, 1024):
            hi = min(lo + 1024, self.ne)
            gradients = shape_gradients(self.glambda[lo:hi], bary)
            nodal = displacement[self.dofs[lo:hi]].reshape(-1, 10, 3)
            derivative = np.einsum("eia,eqib->eqab", nodal, gradients, optimize=True)
            strain = np.stack(
                (
                    derivative[:, :, 0, 0],
                    derivative[:, :, 1, 1],
                    derivative[:, :, 2, 2],
                    derivative[:, :, 0, 1] + derivative[:, :, 1, 0],
                    derivative[:, :, 1, 2] + derivative[:, :, 2, 1],
                    derivative[:, :, 0, 2] + derivative[:, :, 2, 0],
                ),
                axis=-1,
            )
            yield lo, hi, strain


class SurfaceGeometry:
    def __init__(self, fem: P2Shell, quadrature: int) -> None:
        if quadrature < 1:
            raise ValueError("Positive quadrature order required")
        self.f, self.quad = fem, quadrature
        x, weights = np.polynomial.legendre.leggauss(quadrature)
        x, weights = (x + 1) / 2, weights / 2
        u, v = np.meshgrid(x, x, indexing="ij")
        wu, wv = np.meshgrid(weights, weights, indexing="ij")
        self.bary = np.column_stack((1 - u.ravel(), (u * (1 - v)).ravel(), (u * v).ravel()))
        self.weights = (2 * u * wu * wv).ravel()
        faces = fem.outer_faces
        vertices = fem.base_xyz[faces]
        edges = np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
        keys = edges[:, :, 0] * fem.nbase + edges[:, :, 1]
        mids = fem.nbase + np.searchsorted(fem.edge_keys, keys)
        self.nodes = np.column_stack((faces, mids))
        bary = self.bary
        self.shape = np.column_stack(
            (
                bary * (2 * bary - 1),
                4 * bary[:, 0] * bary[:, 1],
                4 * bary[:, 0] * bary[:, 2],
                4 * bary[:, 1] * bary[:, 2],
            )
        )
        self.coords = np.einsum("qi,fic->fqc", bary, vertices)
        self.area = (
            np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1) / 2
        )
        self.integration_weights = self.area[:, None] * self.weights[None]
