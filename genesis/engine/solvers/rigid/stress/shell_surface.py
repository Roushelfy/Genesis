"""Triangulated physical shell sides with conservative generalized-load maps."""

import numpy as np
import quadrants as qd

import genesis as gs

from .data import StressInfo, allocate_stress_info
from .shell_operators import ShellInfo


def build_shell_surface(
    quads: np.ndarray, thickness: float, stress_info: StressInfo, shell_info: ShellInfo
) -> StressInfo:
    n_nodes = shell_info.directors.shape[0]
    outer = quads[:, ((0, 1, 2), (0, 2, 3))].reshape((-1, 3))
    inner = n_nodes + outer[:, (0, 2, 1)]
    triangles = np.concatenate((outer, inner))
    edges, inverse = np.unique(
        np.sort(triangles[:, ((0, 1), (0, 2), (1, 2))].reshape((-1, 2)), axis=1), axis=0, return_inverse=True
    )
    surface_nodes = np.column_stack((triangles, 2 * n_nodes + inverse.reshape((-1, 3))))
    n_surface_nodes = 2 * n_nodes + len(edges)
    info = allocate_stress_info(n_surface_nodes, 0, len(triangles), 0, 0, n_load_nodes=n_surface_nodes)
    info.surface_nodes.from_numpy(surface_nodes.astype(gs.np_int, copy=False))
    # The opposite material-side node determines the outward orientation in the existing surface builder.
    opposite = np.concatenate((n_nodes + outer[:, 0], outer[:, 0]))
    info.surface_opposite_vertices.from_numpy(opposite.astype(gs.np_int, copy=False))
    kernel_surface_geometry(edges.astype(gs.np_int, copy=False), thickness, stress_info, shell_info, info)
    return info


@qd.kernel
def kernel_surface_geometry(
    edges: qd.types.ndarray(), thickness: float, stress_info: StressInfo, shell_info: ShellInfo, load_info: StressInfo
):
    for i_n in range(shell_info.directors.shape[0]):
        for i_side in range(2):
            side = qd.select(i_side == 0, 1.0, -1.0)
            i_surface = i_side * shell_info.directors.shape[0] + i_n
            offset = side * 0.5 * thickness * shell_info.directors[i_n]
            load_info.vertices[i_surface] = stress_info.vertices[2 * i_n] + offset
            load_info.load_nodes[i_surface] = qd.Vector([i_n, i_n])
            load_info.load_offsets[i_surface] = qd.Matrix.rows([offset, offset])
    for i_edge in range(edges.shape[0]):
        i_a, i_b = edges[i_edge, 0], edges[i_edge, 1]
        i_surface = 2 * shell_info.directors.shape[0] + i_edge
        load_info.vertices[i_surface] = 0.5 * (load_info.vertices[i_a] + load_info.vertices[i_b])
        load_info.load_nodes[i_surface] = qd.Vector(
            [
                i_a % shell_info.directors.shape[0],
                i_b % shell_info.directors.shape[0],
            ]
        )
        load_info.load_offsets[i_surface] = qd.Matrix.rows(
            [
                load_info.load_offsets[i_a][0, :],
                load_info.load_offsets[i_b][0, :],
            ]
        )
