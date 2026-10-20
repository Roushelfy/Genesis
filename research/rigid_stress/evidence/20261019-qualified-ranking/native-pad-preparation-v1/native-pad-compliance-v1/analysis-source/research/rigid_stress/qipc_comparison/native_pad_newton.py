"""Batched Quadrants pressure solves for the validated reduced physical pad energy."""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V


@dataclass(frozen=True)
class PadInfo:
    mapping: qd.Tensor
    elastic: qd.Tensor

    @classmethod
    def from_arrays(cls, mapping: np.ndarray, elastic: np.ndarray):
        assert mapping.ndim == 2 and elastic.shape == (mapping.shape[1],)
        assert np.isfinite(mapping).all() and np.isin(elastic, (0, 1)).all()
        info = cls(
            mapping=V(dtype=gs.qd_float, shape=mapping.shape),
            elastic=V(dtype=gs.qd_float, shape=elastic.shape),
        )
        info.mapping.from_numpy(mapping)
        info.elastic.from_numpy(elastic)
        return info


@dataclass(frozen=True)
class PadState:
    gap: qd.Tensor
    weight: qd.Tensor
    elastic_scale: qd.Tensor
    linear: qd.Tensor
    coordinate: qd.Tensor
    force: qd.Tensor
    active_weight: qd.Tensor
    point_product: qd.Tensor
    gradient: qd.Tensor
    preconditioner: qd.Tensor
    increment: qd.Tensor
    residual: qd.Tensor
    direction: qd.Tensor
    product: qd.Tensor
    gradient_squared: qd.Tensor
    residual_squared: qd.Tensor
    residual_preconditioned: qd.Tensor
    next_residual_preconditioned: qd.Tensor
    direction_product: qd.Tensor
    energy: qd.Tensor
    trial_energy: qd.Tensor
    slope: qd.Tensor
    step: qd.Tensor
    status: qd.Tensor
    linear_active: qd.Tensor
    line_active: qd.Tensor
    accepted: qd.Tensor
    iterations: qd.Tensor
    linear_iterations: qd.Tensor


class NativePadNewton:
    """Validate the numerical pressure stage before attaching a geometry/source producer."""

    def __init__(self, info: PadInfo, n_envs: int):
        assert info.elastic.shape == (info.mapping.shape[1],)
        assert n_envs > 0
        n_points, n_dofs = info.mapping.shape
        self.info = info
        self.state = PadState(
            gap=V(dtype=gs.qd_float, shape=(n_points, n_envs)),
            weight=V(dtype=gs.qd_float, shape=(n_points, n_envs)),
            elastic_scale=V(dtype=gs.qd_float, shape=(n_envs,)),
            linear=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            coordinate=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            force=V(dtype=gs.qd_float, shape=(n_points, n_envs)),
            active_weight=V(dtype=gs.qd_float, shape=(n_points, n_envs)),
            point_product=V(dtype=gs.qd_float, shape=(n_points, n_envs)),
            gradient=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            preconditioner=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            increment=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            residual=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            direction=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            product=V(dtype=gs.qd_float, shape=(n_dofs, n_envs)),
            gradient_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            residual_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            residual_preconditioned=V(dtype=gs.qd_float, shape=(n_envs,)),
            next_residual_preconditioned=V(dtype=gs.qd_float, shape=(n_envs,)),
            direction_product=V(dtype=gs.qd_float, shape=(n_envs,)),
            energy=V(dtype=gs.qd_float, shape=(n_envs,)),
            trial_energy=V(dtype=gs.qd_float, shape=(n_envs,)),
            slope=V(dtype=gs.qd_float, shape=(n_envs,)),
            step=V(dtype=gs.qd_float, shape=(n_envs,)),
            status=V(dtype=gs.qd_int, shape=(n_envs,)),
            linear_active=V(dtype=gs.qd_int, shape=(n_envs,)),
            line_active=V(dtype=gs.qd_int, shape=(n_envs,)),
            accepted=V(dtype=gs.qd_bool, shape=(n_envs,)),
            iterations=V(dtype=gs.qd_int, shape=(n_envs,)),
            linear_iterations=V(dtype=gs.qd_int, shape=(n_envs,)),
        )
        self.state.status.fill(1)

    def solve(self, requested: np.ndarray) -> None:
        kernel_begin(requested, self.state)
        for _ in range(40):
            kernel_evaluate(self.state, self.info)
            kernel_linear_begin(self.state, tolerance=1e-10)
            if kernel_count(self.state.status, value=0) == 0:
                break
            for iteration in range(2 * self.info.elastic.shape[0]):
                kernel_linear_step(self.state, self.info, forcing=1e-6)
                if iteration % 8 == 7 and kernel_count(self.state.linear_active, value=1) == 0:
                    break
            kernel_line_begin(self.state)
            for _ in range(40):
                kernel_line_trial(self.state, self.info, epsilon=float(np.finfo(gs.np_float).eps))
                if kernel_count(self.state.line_active, value=1) == 0:
                    break
            kernel_line_end(self.state)
        kernel_evaluate(self.state, self.info)
        kernel_finish(self.state, tolerance=1e-10)


@qd.kernel
def kernel_begin(requested: qd.types.ndarray(), pad_state: PadState):
    for i_b in range(pad_state.status.shape[0]):
        pad_state.linear_active[i_b] = 0
        pad_state.line_active[i_b] = 0
        if requested[i_b]:
            pad_state.status[i_b] = 0
            pad_state.iterations[i_b] = 0
            pad_state.linear_iterations[i_b] = 0
            if not (pad_state.elastic_scale[i_b] > 0.0 and pad_state.elastic_scale[i_b] < qd.math.inf):
                pad_state.status[i_b] = 6
    for i_q, i_b in qd.ndrange(pad_state.gap.shape[0], pad_state.status.shape[0]):
        if requested[i_b]:
            finite = qd.abs(pad_state.gap[i_q, i_b]) < qd.math.inf
            positive = pad_state.weight[i_q, i_b] > 0.0 and pad_state.weight[i_q, i_b] < qd.math.inf
            if not (finite and positive):
                pad_state.status[i_b] = 6


@qd.kernel
def kernel_evaluate(pad_state: PadState, pad_info: PadInfo):
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.status[i_b] == 0:
            pad_state.energy[i_b] = 0.0
            pad_state.gradient_squared[i_b] = 0.0
    for i_q, i_b in qd.ndrange(pad_state.force.shape[0], pad_state.status.shape[0]):
        if pad_state.status[i_b] == 0:
            gap = pad_state.gap[i_q, i_b]
            for i_d in range(pad_info.elastic.shape[0]):
                scale = 1.0 + pad_info.elastic[i_d] * (pad_state.elastic_scale[i_b] - 1.0)
                gap += pad_info.mapping[i_q, i_d] * scale * pad_state.coordinate[i_d, i_b]
            compression = qd.max(-gap, 0.0)
            weight = pad_state.weight[i_q, i_b]
            pad_state.force[i_q, i_b] = weight * compression
            pad_state.active_weight[i_q, i_b] = weight * gs.qd_float(gap < 0.0)
            qd.atomic_add(pad_state.energy[i_b], 0.5 * weight * compression * compression)
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.status[i_b] == 0:
            elastic = pad_info.elastic[i_d]
            coordinate = pad_state.coordinate[i_d, i_b]
            gradient = elastic * coordinate + pad_state.linear[i_d, i_b]
            diagonal = elastic
            for i_q in range(pad_state.force.shape[0]):
                coefficient = pad_info.mapping[i_q, i_d] * (1.0 + elastic * (pad_state.elastic_scale[i_b] - 1.0))
                gradient -= coefficient * pad_state.force[i_q, i_b]
                diagonal += coefficient * coefficient * pad_state.active_weight[i_q, i_b]
            pad_state.gradient[i_d, i_b] = gradient
            pad_state.preconditioner[i_d, i_b] = diagonal
            qd.atomic_add(pad_state.gradient_squared[i_b], gradient * gradient)
            qd.atomic_add(
                pad_state.energy[i_b], 0.5 * elastic * coordinate * coordinate + pad_state.linear[i_d, i_b] * coordinate
            )


@qd.kernel
def kernel_linear_begin(pad_state: PadState, tolerance: float):
    for i_b in range(pad_state.status.shape[0]):
        pad_state.linear_active[i_b] = 0
        if pad_state.status[i_b] == 0:
            if pad_state.gradient_squared[i_b] <= tolerance * tolerance:
                pad_state.status[i_b] = 1
            else:
                pad_state.iterations[i_b] += 1
                pad_state.linear_active[i_b] = 1
                pad_state.residual_preconditioned[i_b] = 0.0
    for i_d, i_b in qd.ndrange(pad_state.gradient.shape[0], pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            diagonal = pad_state.preconditioner[i_d, i_b]
            if not (diagonal > 0.0 and diagonal < qd.math.inf):
                pad_state.status[i_b] = 2
            else:
                residual = -pad_state.gradient[i_d, i_b]
                pad_state.increment[i_d, i_b] = 0.0
                pad_state.residual[i_d, i_b] = residual
                pad_state.direction[i_d, i_b] = residual / diagonal
                qd.atomic_add(pad_state.residual_preconditioned[i_b], residual * residual / diagonal)
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.status[i_b] != 0:
            pad_state.linear_active[i_b] = 0


@qd.kernel
def kernel_linear_step(pad_state: PadState, pad_info: PadInfo, forcing: float):
    for i_b in range(pad_state.status.shape[0]):
        pad_state.direction_product[i_b] = 0.0
        pad_state.next_residual_preconditioned[i_b] = 0.0
        pad_state.residual_squared[i_b] = 0.0
    for i_q, i_b in qd.ndrange(pad_state.force.shape[0], pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            product = gs.qd_float(0.0)
            for i_d in range(pad_info.elastic.shape[0]):
                scale = 1.0 + pad_info.elastic[i_d] * (pad_state.elastic_scale[i_b] - 1.0)
                product += pad_info.mapping[i_q, i_d] * scale * pad_state.direction[i_d, i_b]
            pad_state.point_product[i_q, i_b] = pad_state.active_weight[i_q, i_b] * product
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            product = pad_info.elastic[i_d] * pad_state.direction[i_d, i_b]
            for i_q in range(pad_state.force.shape[0]):
                scale = 1.0 + pad_info.elastic[i_d] * (pad_state.elastic_scale[i_b] - 1.0)
                product += pad_info.mapping[i_q, i_d] * scale * pad_state.point_product[i_q, i_b]
            pad_state.product[i_d, i_b] = product
            qd.atomic_add(pad_state.direction_product[i_b], pad_state.direction[i_d, i_b] * product)
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            pad_state.linear_iterations[i_b] += 1
            if not (pad_state.direction_product[i_b] > 0.0 and pad_state.direction_product[i_b] < qd.math.inf):
                pad_state.status[i_b] = 2
                pad_state.linear_active[i_b] = 0
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            alpha = pad_state.residual_preconditioned[i_b] / pad_state.direction_product[i_b]
            pad_state.increment[i_d, i_b] += alpha * pad_state.direction[i_d, i_b]
            residual = pad_state.residual[i_d, i_b] - alpha * pad_state.product[i_d, i_b]
            pad_state.residual[i_d, i_b] = residual
            qd.atomic_add(pad_state.residual_squared[i_b], residual * residual)
            qd.atomic_add(
                pad_state.next_residual_preconditioned[i_b], residual * residual / pad_state.preconditioner[i_d, i_b]
            )
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            if pad_state.residual_squared[i_b] <= forcing * forcing * pad_state.gradient_squared[i_b]:
                pad_state.linear_active[i_b] = 0
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            beta = pad_state.next_residual_preconditioned[i_b] / pad_state.residual_preconditioned[i_b]
            pad_state.direction[i_d, i_b] = (
                pad_state.residual[i_d, i_b] / pad_state.preconditioner[i_d, i_b] + beta * pad_state.direction[i_d, i_b]
            )
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.linear_active[i_b]:
            pad_state.residual_preconditioned[i_b] = pad_state.next_residual_preconditioned[i_b]


@qd.kernel
def kernel_line_begin(pad_state: PadState):
    for i_b in range(pad_state.status.shape[0]):
        pad_state.line_active[i_b] = 0
        if pad_state.status[i_b] == 0:
            if pad_state.linear_active[i_b]:
                pad_state.status[i_b] = 3
                pad_state.linear_active[i_b] = 0
            else:
                pad_state.line_active[i_b] = 1
                pad_state.step[i_b] = 1.0
                pad_state.slope[i_b] = 0.0
    for i_d, i_b in qd.ndrange(pad_state.gradient.shape[0], pad_state.status.shape[0]):
        if pad_state.line_active[i_b]:
            qd.atomic_add(pad_state.slope[i_b], pad_state.gradient[i_d, i_b] * pad_state.increment[i_d, i_b])


@qd.kernel
def kernel_line_trial(pad_state: PadState, pad_info: PadInfo, epsilon: float):
    for i_b in range(pad_state.status.shape[0]):
        pad_state.accepted[i_b] = False
        pad_state.trial_energy[i_b] = 0.0
    for i_q, i_b in qd.ndrange(pad_state.force.shape[0], pad_state.status.shape[0]):
        if pad_state.line_active[i_b]:
            gap = pad_state.gap[i_q, i_b]
            for i_d in range(pad_info.elastic.shape[0]):
                coordinate = pad_state.coordinate[i_d, i_b] + pad_state.step[i_b] * pad_state.increment[i_d, i_b]
                scale = 1.0 + pad_info.elastic[i_d] * (pad_state.elastic_scale[i_b] - 1.0)
                gap += pad_info.mapping[i_q, i_d] * scale * coordinate
            compression = qd.max(-gap, 0.0)
            qd.atomic_add(pad_state.trial_energy[i_b], 0.5 * pad_state.weight[i_q, i_b] * compression * compression)
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.line_active[i_b]:
            coordinate = pad_state.coordinate[i_d, i_b] + pad_state.step[i_b] * pad_state.increment[i_d, i_b]
            energy = 0.5 * pad_info.elastic[i_d] * coordinate * coordinate + pad_state.linear[i_d, i_b] * coordinate
            qd.atomic_add(pad_state.trial_energy[i_b], energy)
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.line_active[i_b]:
            limit = pad_state.energy[i_b] + 1e-4 * pad_state.step[i_b] * pad_state.slope[i_b]
            limit += 16 * epsilon * qd.abs(pad_state.energy[i_b])
            if pad_state.slope[i_b] < 0.0 and pad_state.trial_energy[i_b] <= limit:
                pad_state.accepted[i_b] = True
                pad_state.line_active[i_b] = 0
            else:
                pad_state.step[i_b] *= 0.5
    for i_d, i_b in qd.ndrange(pad_info.elastic.shape[0], pad_state.status.shape[0]):
        if pad_state.accepted[i_b]:
            pad_state.coordinate[i_d, i_b] += pad_state.step[i_b] * pad_state.increment[i_d, i_b]


@qd.kernel
def kernel_line_end(pad_state: PadState):
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.line_active[i_b]:
            pad_state.status[i_b] = 4
            pad_state.line_active[i_b] = 0


@qd.kernel
def kernel_finish(pad_state: PadState, tolerance: float):
    for i_b in range(pad_state.status.shape[0]):
        if pad_state.status[i_b] == 0:
            pad_state.status[i_b] = 5
            if pad_state.gradient_squared[i_b] <= tolerance * tolerance:
                pad_state.status[i_b] = 1
    for i_q, i_b in qd.ndrange(pad_state.force.shape[0], pad_state.status.shape[0]):
        if pad_state.status[i_b] != 1:
            pad_state.force[i_q, i_b] = float("nan")


@qd.kernel
def kernel_count(values: qd.Tensor, value: int) -> int:
    count = 0
    for i_b in range(values.shape[0]):
        count += gs.qd_int(values[i_b] == value)
    return count
