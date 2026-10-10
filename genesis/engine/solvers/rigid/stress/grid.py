"""Exact native uniform-grid candidate search for fixed surface quadrature."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V


@dataclass(frozen=True)
class StressGridInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    bounds: qd.Tensor
    cell_size: qd.Tensor
    starts: qd.Tensor
    samples: qd.Tensor


def allocate_grid(n_samples: int) -> StressGridInfo:
    return StressGridInfo(
        bounds=V_VEC(3, dtype=gs.qd_float, shape=(2,)),
        cell_size=V(dtype=gs.qd_float, shape=()),
        starts=V(dtype=gs.qd_int, shape=(16**3 + 1,)),
        samples=V(dtype=gs.qd_int, shape=(n_samples,)),
    )


def build_grid(positions: qd.Tensor, grid_info: StressGridInfo) -> None:
    cursor = V(dtype=gs.qd_int, shape=(16**3,))
    kernel_bounds(positions, grid_info)
    kernel_counts(positions, grid_info)
    kernel_prefix(cursor, grid_info)
    kernel_samples(positions, cursor, grid_info)


@qd.func
def func_cell(position: qd.types.vector(3), grid_info: StressGridInfo):
    index = qd.cast(qd.floor((position - grid_info.bounds[0]) / grid_info.cell_size[None]), gs.qd_int)
    index = qd.max(0, qd.min(15, index))
    return (index[0] * 16 + index[1]) * 16 + index[2]


@qd.func
def func_patch_grid(center: qd.types.vector(3), radius: float, grid_info: StressGridInfo):
    low = qd.cast(
        qd.max(0.0, qd.min(15.0, qd.floor((center - radius - grid_info.bounds[0]) / grid_info.cell_size[None]))),
        gs.qd_int,
    )
    high = qd.cast(
        qd.max(0.0, qd.min(15.0, qd.floor((center + radius - grid_info.bounds[0]) / grid_info.cell_size[None]))),
        gs.qd_int,
    )
    dimensions = high - low + 1
    return low, dimensions


@qd.func
def func_patch_cell(i: int, low: qd.types.vector(3), dimensions: qd.types.vector(3), grid_info: StressGridInfo):
    z = i % dimensions[2] + low[2]
    y = i // dimensions[2] % dimensions[1] + low[1]
    x = i // (dimensions[1] * dimensions[2]) + low[0]
    cell = (x * 16 + y) * 16 + z
    return grid_info.starts[cell], grid_info.starts[cell + 1]


@qd.kernel(graph=True)
def kernel_bounds(positions: qd.Tensor, grid_info: StressGridInfo):
    for a in range(3):
        grid_info.bounds[0][a] = float("inf")
        grid_info.bounds[1][a] = -float("inf")
    for i in range(positions.shape[0]):
        for a in qd.static(range(3)):
            qd.atomic_min(grid_info.bounds[0][a], positions[i][a])
            qd.atomic_max(grid_info.bounds[1][a], positions[i][a])
    for _ in range(1):
        extent = grid_info.bounds[1] - grid_info.bounds[0]
        grid_info.cell_size[None] = qd.max(extent[0], extent[1], extent[2]) * (1.0 + 1e-12) / 16.0


@qd.kernel(graph=True)
def kernel_counts(positions: qd.Tensor, grid_info: StressGridInfo):
    for i in range(grid_info.starts.shape[0]):
        grid_info.starts[i] = 0
    for i in range(positions.shape[0]):
        cell = func_cell(positions[i], grid_info)
        qd.atomic_add(grid_info.starts[cell + 1], 1)


@qd.kernel
def kernel_prefix(cursor: qd.Tensor, grid_info: StressGridInfo):
    for _ in range(1):
        total = 0
        for i in range(grid_info.starts.shape[0]):
            total += grid_info.starts[i]
            grid_info.starts[i] = total
            if i < cursor.shape[0]:
                cursor[i] = total


@qd.kernel
def kernel_samples(positions: qd.Tensor, cursor: qd.Tensor, grid_info: StressGridInfo):
    for i in range(positions.shape[0]):
        cell = func_cell(positions[i], grid_info)
        entry = qd.atomic_add(cursor[cell], 1)
        grid_info.samples[entry] = i
