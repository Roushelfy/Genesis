"""Physical Q4 shell sides and their consistent generalized-load maps."""

import numpy as np
import quadrants as qd

import genesis as gs

from .data import StressInfo, allocate_stress_info
from .shell_operators import ShellInfo


def build_shell_surface(
    quads: np.ndarray, thickness: float, stress_info: StressInfo, shell_info: ShellInfo
) -> StressInfo:
    n_nodes = shell_info.directors.shape[0]
    # Reverse the inner chart to give outward physical normals.
    sides = np.concatenate((quads, n_nodes + quads[:, (0, 3, 2, 1)]))
    # The common scatter interface has six slots; the last two carry zero Q4 shape.
    surface_nodes = np.column_stack((sides, sides[:, (0, 0)]))
    info = allocate_stress_info(2 * n_nodes, 0, len(sides), 0, 0, n_load_nodes=2 * n_nodes)
    info.surface_nodes.from_numpy(surface_nodes.astype(gs.np_int, copy=False))
    kernel_surface_geometry(thickness, stress_info, shell_info, info)
    return info


@qd.kernel
def kernel_surface_geometry(thickness: float, stress_info: StressInfo, shell_info: ShellInfo, load_info: StressInfo):
    for i_n in range(shell_info.directors.shape[0]):
        for i_side in range(2):
            side = qd.select(i_side == 0, 1.0, -1.0)
            i_surface = i_side * shell_info.directors.shape[0] + i_n
            offset = side * 0.5 * thickness * shell_info.directors[i_n]
            load_info.vertices[i_surface] = stress_info.vertices[2 * i_n] + offset
            load_info.load_nodes[i_surface] = qd.Vector([i_n, i_n])
            load_info.load_offsets[i_surface] = qd.Matrix.rows([offset, offset])
