"""Shared structured material meshes for native and independent controls."""

from itertools import permutations

import numpy as np


def voxel_mesh(kind):
    shape = (3, 3, 3) if kind == "cavity" else (3, 3, 1) if kind == "u" else (2, 2, 1)
    cells = np.indices(shape).reshape(3, -1).T
    if kind == "cavity":
        cells = cells[(cells != 1).any(axis=1)]
    elif kind == "u":
        cells = cells[(cells[:, 0] != 1) | (cells[:, 1] == 0)]
    steps = np.eye(3, dtype=int)[np.array(list(permutations(range(3))))]
    offsets = np.concatenate((np.zeros((6, 1, 3), dtype=int), steps.cumsum(axis=1)), axis=1)
    coordinates = cells[:, None, None] + offsets
    grid_shape = tuple(size + 1 for size in shape)
    tets = np.ravel_multi_index(np.moveaxis(coordinates, -1, 0), grid_shape).reshape(-1, 4)
    used, inverse = np.unique(tets, return_inverse=True)
    vertices = np.indices(grid_shape).reshape(3, -1).T[used] * 0.001
    tets = inverse.reshape(-1, 4)
    faces, counts = np.unique(
        np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1),
        axis=0,
        return_counts=True,
    )
    return vertices, tets, faces[counts == 1]

