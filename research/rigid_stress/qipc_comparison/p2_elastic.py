"""Affine P2 elasticity adapter for the independent free-body contact reference."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..skfem_reference import SkfemElasticReference


class FreeElasticP2:
    """Use scikit-fem derivatives and consistent mass, retaining all six rigid modes."""

    def __init__(self, vertices, tetrahedra, young, poisson, density, column_faces=None):
        self.reference = SkfemElasticReference(vertices, tetrahedra, young, poisson, density)
        reference = self.reference
        np.testing.assert_array_equal(reference.mesh.t.T, tetrahedra)
        np.testing.assert_array_equal(reference.component_dofs.reshape(-1), np.arange(reference.basis.N))
        self.vertices = reference.coordinates
        self.tetrahedra = tetrahedra
        self.young, self.poisson = young, poisson
        self.stiffness = reference.stiffness
        self.rigid_modes, self.mass_modes, self.gram = reference.rigid_modes, reference.mass_modes, reference.gram
        self.free = reference.free_dofs
        if column_faces is None:
            self.factor = splu((self.stiffness[self.free][:, self.free] / young).tocsc())
        else:
            import pymetis

            nodes = reference.scalar_basis.element_dofs.T
            rows = np.broadcast_to(nodes[:, :, None], (len(nodes), 10, 10)).reshape(-1)
            columns = np.broadcast_to(nodes[:, None, :], (len(nodes), 10, 10)).reshape(-1)
            graph = sparse.coo_matrix(
                (np.ones(len(rows)), (rows, columns)), shape=(len(self.vertices), len(self.vertices))
            ).tocsr()
            graph.setdiag(0)
            graph.eliminate_zeros()
            order, _ = pymetis.nested_dissection(
                adjacency=pymetis.CSRAdjacency(graph.indptr, graph.indices), options=pymetis.Options(seed=17)
            )
            dofs = (3 * np.asarray(order)[:, None] + np.arange(3)).reshape(-1)
            self.free = dofs[np.isin(dofs, self.free)]
            self.factor = splu(
                (self.stiffness[self.free][:, self.free] / young).tocsc(),
                permc_spec="NATURAL",
                diag_pivot_thresh=0,
                options={"SymmetricMode": True},
            )

    def surface(self, faces):
        """Return the original affine outer boundary's P2 nodes and four subtriangles per face."""
        edges = self.reference.mesh.edges.T
        edge_nodes = self.reference.scalar_basis.edge_dofs[0]
        lookup = {tuple(edge): node for edge, node in zip(edges, edge_nodes)}
        mids = np.array([[lookup[tuple(sorted((a, b)))] for a, b in zip(face, np.roll(face, -1))] for face in faces])
        nodes = np.unique(np.r_[faces.reshape(-1), mids.reshape(-1)])
        a, b, c = faces.T
        ab, bc, ca = mids.T
        triangles = np.concatenate(
            [np.column_stack(part) for part in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
        )
        return nodes, np.searchsorted(nodes, triangles), np.column_stack((faces, mids))

    def displacement_gradient(self, displacement):
        return self.reference.corner_basis.interpolate(displacement.reshape(-1)).grad.transpose(2, 3, 0, 1)

    def solve(self, load):
        force = load.reshape(-1)
        correction = self.mass_modes @ np.linalg.solve(self.gram, self.rigid_modes.T @ force)
        rhs = force - correction
        displacement = np.zeros_like(force)
        displacement[self.free] = self.factor.solve(rhs[self.free] / self.young)
        displacement = self.reference.align_rigid_gauge(displacement)
        residual = self.stiffness @ displacement - rhs
        stress, _ = self.reference.corner_stress(displacement)
        return stress, displacement.reshape(-1, 3), correction.reshape(-1, 3), residual.reshape(-1, 3)


def quadratic_triangle_minimum(values, return_points=False):
    """Exact minimum of a scalar quadratic on each triangle, from its three vertices and edge midpoints."""
    a, b, c, ab, bc, ca = values.T
    linear_x, linear_y = 4 * ab - 3 * a - b, 4 * ca - 3 * a - c
    square_x, square_y = 2 * (a + b - 2 * ab), 2 * (a + c - 2 * ca)
    mixed = 4 * (bc - a - (linear_x + linear_y) / 2 - (square_x + square_y) / 4)
    corner = np.argmin(values[:, :3], axis=1)
    minimum = values[np.arange(len(values)), corner].copy()
    point_x, point_y = (corner == 1).astype(float), (corner == 2).astype(float)
    for edge, (first, second, middle) in enumerate(((a, b, ab), (b, c, bc), (c, a, ca))):
        curvature = 2 * (first + second - 2 * middle)
        slope = 4 * middle - 3 * first - second
        fraction = np.divide(-slope, 2 * curvature, out=np.zeros_like(slope), where=curvature > 0)
        valid = (curvature > 0) & (fraction > 0) & (fraction < 1)
        candidate = np.where(valid, first + slope * fraction + curvature * fraction**2, np.inf)
        improved = candidate < minimum
        edge_x = fraction if edge == 0 else (1 - fraction if edge == 1 else np.zeros_like(fraction))
        edge_y = np.zeros_like(fraction) if edge == 0 else (fraction if edge == 1 else 1 - fraction)
        point_x, point_y = np.where(improved, edge_x, point_x), np.where(improved, edge_y, point_y)
        minimum = np.minimum(minimum, candidate)
    determinant = 4 * square_x * square_y - mixed**2
    valid = (square_x > 0) & (determinant > 0)
    x = np.divide(mixed * linear_y - 2 * square_y * linear_x, determinant, out=np.zeros_like(a), where=valid)
    y = np.divide(mixed * linear_x - 2 * square_x * linear_y, determinant, out=np.zeros_like(a), where=valid)
    valid &= (x > 0) & (y > 0) & (x + y < 1)
    stationary = a + linear_x * x + linear_y * y + square_x * x**2 + mixed * x * y + square_y * y**2
    candidate = np.where(valid, stationary, np.inf)
    improved = candidate < minimum
    point_x, point_y = np.where(improved, x, point_x), np.where(improved, y, point_y)
    minimum = np.minimum(minimum, candidate)
    if return_points:
        return minimum, np.column_stack((1 - point_x - point_y, point_x, point_y))
    return minimum
