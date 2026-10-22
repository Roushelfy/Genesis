"""Build material-boundary ownership and adjacency from vertex identities."""

from dataclasses import dataclass

import numpy as np

import genesis as gs


@dataclass(frozen=True)
class StressBoundaryTopology:
    owners: np.ndarray
    opposite_vertices: np.ndarray
    neighbors: np.ndarray
    element_neighbors: np.ndarray
    triangles: np.ndarray


def build_surface_topology(
    tetrahedra: np.ndarray, surface: np.ndarray, complete: bool = False
) -> StressBoundaryTopology:
    """Associate each surface triangle with its unique material element and edge neighbors.

    Surface winding is unrestricted. Edges connect only matching vertex indices;
    coincident positions on separate surface sheets never create adjacency.
    A selected subset of the material boundary may have open edges.
    Complete coverage appends omitted boundary faces while preserving supplied
    face indices, including disconnected cavity surfaces.
    """
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=gs.np_int)
    faces = np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1)
    selected = np.sort(surface, axis=1)
    if (np.diff(selected, axis=1) == 0).any():
        gs.raise_exception("Stress boundary triangles must have three distinct vertices.")
    unique, inverse = np.unique(np.concatenate((faces, selected)), axis=0, return_inverse=True)
    element_faces, surface_faces = inverse[: len(faces)], inverse[len(faces) :]
    counts = np.bincount(element_faces, minlength=len(unique))
    if (counts > 2).any():
        gs.raise_exception("Stress material faces must have at most two incident tetrahedra.")
    if (counts[surface_faces] != 1).any():
        gs.raise_exception("Each stress boundary triangle must have exactly one owning tetrahedron.")
    if len(np.unique(surface_faces)) != len(surface_faces):
        gs.raise_exception("Stress boundary triangles must not be duplicated, regardless of winding.")

    if complete:
        is_selected = np.zeros(len(unique), dtype=bool)
        is_selected[surface_faces] = True
        missing = np.flatnonzero((counts == 1) & ~is_selected)
        surface = np.concatenate((surface, unique[missing]))
        surface_faces = np.concatenate((surface_faces, missing))

    owners = np.full(len(unique), -1, dtype=gs.np_int)
    opposite_vertices = np.full(len(unique), -1, dtype=gs.np_int)
    owners[element_faces] = np.repeat(np.arange(len(tetrahedra)), 4)
    opposite_vertices[element_faces] = tetrahedra.reshape((-1,))

    element_neighbors = np.full((len(tetrahedra), 4), -1, dtype=gs.np_int)
    face_order = np.argsort(element_faces, kind="stable")
    paired_faces = np.cumsum(counts)[counts == 2] - 2
    first_face, second_face = face_order[paired_faces], face_order[paired_faces + 1]
    element_neighbors[first_face // 4, first_face % 4] = second_face // 4
    element_neighbors[second_face // 4, second_face % 4] = first_face // 4

    edges = np.sort(surface[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1)
    _, edge_inverse, edge_counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if (edge_counts > 2).any():
        gs.raise_exception("Stress boundary edges must have at most two incident triangles.")
    neighbors = np.full((len(surface), 3), -1, dtype=gs.np_int)
    edge_order = np.argsort(edge_inverse, kind="stable")
    paired = np.cumsum(edge_counts)[edge_counts == 2] - 2
    first, second = edge_order[paired], edge_order[paired + 1]
    flat_neighbors = neighbors.reshape((-1,))
    flat_neighbors[first] = second // 3
    flat_neighbors[second] = first // 3
    return StressBoundaryTopology(
        owners[surface_faces], opposite_vertices[surface_faces], neighbors, element_neighbors, surface
    )
