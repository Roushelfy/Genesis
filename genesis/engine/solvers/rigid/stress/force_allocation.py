"""Wrench-preserving force allocation over supplied physical contact regions."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, V, V_VEC


@dataclass(frozen=True)
class StressForceAllocationState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    original: qd.Tensor
    center: qd.Tensor
    normal: qd.Tensor
    friction: qd.Tensor
    target: qd.Tensor
    length: qd.Tensor
    force: qd.Tensor
    status: qd.Tensor
    residual: qd.Tensor
    maximum_change: qd.Tensor
    iterations: qd.Tensor


class StressForceAllocation:
    """Minimize squared relative region-force changes under wrench and cone constraints.

    The target stores total force and the moment about the mean supplied center,
    with moment divided by the supplied length. All original free couples belong
    in this target. A six-coordinate dual Newton solve projects each region onto
    its Coulomb cone. Status0 reports numerical convergence,1 is unloaded,2 is
    invalid input,3 is a singular dual derivative and4 is the iteration limit.
    Physical acceptance of the adjusted loads belongs to the caller.
    """

    def __init__(self, n_regions: int, n_envs: int):
        assert n_regions > 0 and n_envs > 0
        self.epsilon = float(np.finfo(gs.np_float).eps)
        shape = (n_regions, n_envs)
        self.state = StressForceAllocationState(
            original=V_VEC(3, dtype=gs.qd_float, shape=shape),
            center=V_VEC(3, dtype=gs.qd_float, shape=shape),
            normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
            friction=V(dtype=gs.qd_float, shape=shape),
            target=V_VEC(6, dtype=gs.qd_float, shape=(n_envs,)),
            length=V(dtype=gs.qd_float, shape=(n_envs,)),
            force=V_VEC(3, dtype=gs.qd_float, shape=shape),
            status=V(dtype=gs.qd_int, shape=(n_envs,)),
            residual=V(dtype=gs.qd_float, shape=(n_envs,)),
            maximum_change=V(dtype=gs.qd_float, shape=(n_envs,)),
            iterations=V(dtype=gs.qd_int, shape=(n_envs,)),
        )

    def update(self):
        kernel_force_allocation(self.state, self.epsilon)


@qd.func
def func_cone_projection(force: qd.types.vector(3), normal: qd.types.vector(3), friction: float):
    """Project a force onto a circular Coulomb cone with its piecewise derivative."""
    axial = force.dot(normal)
    tangent = force - axial * normal
    radial = tangent.norm()
    projected = qd.Vector.zero(gs.qd_float, 3)
    derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
    if friction <= 0 and axial > 0:
        projected, derivative = axial * normal, normal.outer_product(normal)
    elif friction > 0 and radial <= friction * axial and axial >= 0:
        projected, derivative = force, qd.Matrix.identity(gs.qd_float, 3)
    elif axial + friction * radial > 0 and radial > 0:
        direction = tangent / radial
        compression = (axial + friction * radial) / (1 + friction * friction)
        generator = normal + friction * direction
        projected = compression * generator
        derivative = generator.outer_product(generator) / (1 + friction * friction)
        derivative += (
            friction
            * compression
            / radial
            * (qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal) - direction.outer_product(direction))
        )
    return projected, derivative


@qd.func
def func_solve_wrench_system(matrix: qd.types.matrix(6, 6), rhs: qd.types.vector(6), epsilon: float):
    """Solve a positive definite six-coordinate system by Cholesky factorization."""
    lower = qd.Matrix.zero(gs.qd_float, 6, 6)
    value = qd.Vector.zero(gs.qd_float, 6)
    is_valid = matrix.norm() < float("inf") and rhs.norm() < float("inf")
    threshold = 128 * epsilon * matrix.trace()
    for i in range(6):
        for j in range(i + 1):
            if is_valid:
                entry = matrix[i, j]
                for k in range(j):
                    entry -= lower[i, k] * lower[j, k]
                if i == j:
                    is_valid = entry > threshold
                    if is_valid:
                        lower[i, j] = qd.sqrt(entry)
                else:
                    lower[i, j] = entry / lower[j, j]
    if is_valid:
        for i in range(6):
            entry = rhs[i]
            for j in range(i):
                entry -= lower[i, j] * value[j]
            value[i] = entry / lower[i, i]
        for i_ in range(6):
            i = 5 - i_
            entry = value[i]
            for j in range(i + 1, 6):
                entry -= lower[j, i] * value[j]
            value[i] = entry / lower[i, i]
    return value, is_valid


@qd.kernel
def kernel_force_allocation(allocation: StressForceAllocationState, epsilon: float):
    for i_b in range(allocation.status.shape[0]):
        allocation.status[i_b] = 2
        allocation.iterations[i_b], allocation.maximum_change[i_b] = 0, 0
        allocation.residual[i_b] = float("inf")
        scale, count = gs.qd_float(0.0), 0
        origin = qd.Vector.zero(gs.qd_float, 3)
        is_valid = allocation.length[i_b] > 0 and allocation.length[i_b] < float("inf")
        is_valid = is_valid and allocation.target[i_b].norm() < float("inf")
        for i_c in range(allocation.original.shape[0]):
            force = allocation.original[i_c, i_b]
            allocation.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            is_valid = is_valid and force.norm() < float("inf")
            if force.norm() > 0:
                scale += force.norm()
                origin += allocation.center[i_c, i_b]
                count += 1
                is_valid = is_valid and allocation.center[i_c, i_b].norm() < float("inf")
                is_valid = is_valid and qd.abs(allocation.normal[i_c, i_b].norm_sqr() - 1) <= 128 * epsilon
                is_valid = is_valid and allocation.friction[i_c, i_b] >= 0
                is_valid = is_valid and allocation.friction[i_c, i_b] < float("inf")
        if is_valid:
            if count < 1:
                if allocation.target[i_b].norm() <= 0:
                    allocation.status[i_b], allocation.residual[i_b] = 1, 0
            else:
                allocation.status[i_b] = 4
                origin /= count
                dual = qd.Vector.zero(gs.qd_float, 6)
                for i_iteration in range(13):
                    if allocation.status[i_b] == 4:
                        residual = -allocation.target[i_b] / scale
                        jacobian = qd.Matrix.zero(gs.qd_float, 6, 6)
                        maximum_change = gs.qd_float(0.0)
                        for i_c in range(allocation.original.shape[0]):
                            original = allocation.original[i_c, i_b] / scale
                            weight = original.norm_sqr()
                            if weight > 0:
                                position = (allocation.center[i_c, i_b] - origin) / allocation.length[i_b]
                                matrix = qd.Matrix.zero(gs.qd_float, 6, 3)
                                for i in qd.static(range(3)):
                                    unit = qd.Vector.zero(gs.qd_float, 3)
                                    unit[i] = 1.0
                                    moment = position.cross(unit)
                                    matrix[i, i] = 1.0
                                    for j in qd.static(range(3)):
                                        matrix[j + 3, i] = moment[j]
                                force, derivative = func_cone_projection(
                                    original - weight * (matrix.transpose() @ dual),
                                    allocation.normal[i_c, i_b],
                                    allocation.friction[i_c, i_b],
                                )
                                allocation.force[i_c, i_b] = scale * force
                                maximum_change = qd.max(maximum_change, (force - original).norm() / qd.sqrt(weight))
                                residual += matrix @ force
                                jacobian += weight * matrix @ derivative @ matrix.transpose()
                        allocation.residual[i_b] = residual.norm()
                        allocation.maximum_change[i_b] = maximum_change
                        allocation.iterations[i_b] = i_iteration
                        if residual.norm() <= qd.max(1e-12, 128 * epsilon):
                            allocation.status[i_b] = 0
                        elif i_iteration < 12:
                            step, is_regular = func_solve_wrench_system(jacobian, residual, epsilon)
                            if is_regular:
                                dual += step
                            else:
                                allocation.status[i_b] = 3
