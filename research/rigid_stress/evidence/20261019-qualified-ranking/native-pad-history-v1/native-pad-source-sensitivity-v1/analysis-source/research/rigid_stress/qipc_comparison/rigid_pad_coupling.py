"""Implicit physical-pad wrenches for an independently audited native rigid step.

This benchmark boundary condition owns its buffers. It does not change the
native collider or stress algorithms. Newton solves seven velocities: a free
rigid link and one translating plate, using frozen world contact Jacobians.
"""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_MAT, V_VEC
from genesis.utils.misc import qd_to_numpy


@dataclass(frozen=True)
class RigidPadState:
    points: qd.Tensor
    area: qd.Tensor
    sign: qd.Tensor
    world: qd.Tensor
    gap: qd.Tensor
    jacobian: qd.Tensor
    previous_normal: qd.Tensor
    force: qd.Tensor
    mass: qd.Tensor
    predicted: qd.Tensor
    velocity: qd.Tensor
    direction: qd.Tensor
    gradient: qd.Tensor
    hessian: qd.Tensor
    energy: qd.Tensor
    slope: qd.Tensor
    residual: qd.Tensor

    @classmethod
    def allocate(cls, points, area):
        state = cls(
            V_VEC(3, dtype=gs.qd_float, shape=(len(points),)),
            V(dtype=gs.qd_float, shape=(len(points),)),
            V(dtype=gs.qd_float, shape=(len(points),)),
            V_VEC(3, dtype=gs.qd_float, shape=(len(points),)),
            V(dtype=gs.qd_float, shape=(len(points),)),
            V_MAT(3, 7, dtype=gs.qd_float, shape=(len(points),)),
            V(dtype=gs.qd_float, shape=(len(points),)),
            V_VEC(3, dtype=gs.qd_float, shape=(len(points),)),
            V_MAT(7, 7, dtype=gs.qd_float, shape=()),
            V_VEC(7, dtype=gs.qd_float, shape=()),
            V_VEC(7, dtype=gs.qd_float, shape=()),
            V_VEC(7, dtype=gs.qd_float, shape=()),
            V_VEC(7, dtype=gs.qd_float, shape=()),
            V_MAT(7, 7, dtype=gs.qd_float, shape=()),
            V(dtype=gs.qd_float, shape=()),
            V(dtype=gs.qd_float, shape=()),
            V(dtype=gs.qd_float, shape=()),
        )
        state.points.from_numpy(points)
        state.area.from_numpy(area)
        state.sign.from_numpy(np.where(points[:, 2] > 0, -1.0, 1.0))
        return state


@qd.kernel
def prepare_geometry(
    position: qd.types.ndarray(),
    rotation: qd.types.ndarray(),
    center: qd.types.ndarray(),
    lower: float,
    upper: float,
    state: RigidPadState,
):
    for i in range(state.points.shape[0]):
        point = qd.Vector([position[0], position[1], position[2]])
        local = state.points[i]
        for j, k in qd.static(qd.ndrange(3, 3)):
            point[j] += rotation[j, k] * local[k]
        arm = (point - qd.Vector([center[0], center[1], center[2]])) / 0.03
        jacobian = qd.Matrix.zero(gs.qd_float, 3, 7)
        for j in qd.static(range(3)):
            jacobian[j, j] = 1.0
        jacobian[0, 4], jacobian[0, 5] = arm[2], -arm[1]
        jacobian[1, 3], jacobian[1, 5] = -arm[2], arm[0]
        jacobian[2, 3], jacobian[2, 4] = arm[1], -arm[0]
        gap = point[2] - lower
        if state.sign[i] < 0:
            gap = upper - point[2]
            jacobian[2, 6] = 1.0
        state.world[i] = point
        state.gap[i] = gap
        state.jacobian[i] = jacobian


@qd.func
def pad_terms(velocity: qd.types.vector(3), gap: float, sign: float, stiffness: float, bound: float, dt: float):
    compression = qd.max(-(gap + dt * sign * velocity[2]), 0.0)
    tangent = qd.Vector([velocity[0], velocity[1], 0.0])
    speed = tangent.norm()
    epsilon = 1e-4
    energy = 0.5 * stiffness * compression**2
    impulse = qd.Vector([0.0, 0.0, dt * sign * stiffness * compression])
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if compression > 0:
        hessian[2, 2] = dt * dt * stiffness
    projector = qd.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    direction = tangent / qd.max(speed, 1e-30)
    if speed < epsilon:
        energy += bound * speed**2 * (1 / epsilon - speed / (3 * epsilon**2))
        impulse -= bound * (2 / epsilon - speed / epsilon**2) * tangent
        hessian += bound * (
            (2 / epsilon - speed / epsilon**2) * projector - speed / epsilon**2 * direction.outer_product(direction)
        )
    else:
        energy += bound * (speed - epsilon / 3)
        impulse -= bound * direction
        hessian += bound / speed * (projector - direction.outer_product(direction))
    return energy, impulse, hessian


@qd.kernel
def evaluate(dt: float, stiffness: float, friction: float, step: float, state: RigidPadState):
    for _ in range(1):
        velocity = state.velocity[None] + step * state.direction[None]
        delta = velocity - state.predicted[None]
        state.energy[None] = 0.5 * delta.dot(state.mass[None] @ delta)
        state.gradient[None] = state.mass[None] @ delta
        state.hessian[None] = state.mass[None]
    for i in range(state.points.shape[0]):
        velocity = state.velocity[None] + step * state.direction[None]
        jacobian = state.jacobian[i]
        energy, impulse, hessian = pad_terms(
            jacobian @ velocity,
            state.gap[i],
            state.sign[i],
            stiffness * state.area[i],
            dt * friction * state.previous_normal[i],
            dt,
        )
        state.energy[None] += energy
        state.gradient[None] -= jacobian.transpose() @ impulse
        state.hessian[None] += jacobian.transpose() @ hessian @ jacobian
        state.force[i] = impulse / dt


@qd.kernel
def newton_direction(state: RigidPadState):
    for _ in range(1):
        factor = qd.Matrix.zero(gs.qd_float, 7, 7)
        intermediate = qd.Vector.zero(gs.qd_float, 7)
        direction = qd.Vector.zero(gs.qd_float, 7)
        for i in range(7):
            for j in range(i + 1):
                value = state.hessian[None][i, j]
                for k in range(j):
                    value -= factor[i, k] * factor[j, k]
                if i == j:
                    factor[i, j] = qd.sqrt(value)
                else:
                    factor[i, j] = value / factor[j, j]
            value = -state.gradient[None][i]
            for j in range(i):
                value -= factor[i, j] * intermediate[j]
            intermediate[i] = value / factor[i, i]
        for reverse in range(7):
            i = 6 - reverse
            value = intermediate[i]
            for j in range(i + 1, 7):
                value -= factor[j, i] * direction[j]
            direction[i] = value / factor[i, i]
        state.direction[None] = direction
        state.slope[None] = state.gradient[None].dot(direction)
        state.residual[None] = state.gradient[None].norm()


@qd.kernel
def accept_step(step: float, state: RigidPadState):
    for _ in range(1):
        state.velocity[None] += step * state.direction[None]
        state.direction[None] = qd.Vector.zero(gs.qd_float, 7)


@qd.kernel
def carry_normal(state: RigidPadState):
    for i in range(state.points.shape[0]):
        state.previous_normal[i] = qd.max(state.sign[i] * state.force[i][2], 0.0)


def solve(state, mass, predicted, dt, stiffness, friction, load_scale):
    """All Newton algebra runs in Quadrants; host reads only scalar stopping data."""
    state.mass.from_numpy(mass)
    state.predicted.from_numpy(predicted)
    state.velocity.from_numpy(predicted)
    state.direction.fill(0)
    for iteration in range(64):
        evaluate(dt, stiffness, friction, 0.0, state)
        newton_direction(state)
        residual = qd_to_numpy(state.residual, transpose=True).item()
        assert np.isfinite(residual), "Nonfinite physical-pad Newton residual"
        if residual <= dt * max(load_scale, 1e-3) * 1e-10:
            return iteration + 1, residual
        energy = qd_to_numpy(state.energy, transpose=True).item()
        slope = qd_to_numpy(state.slope, transpose=True).item()
        assert slope < 0, "Physical-pad Newton direction is not descending"
        step = 1.0
        for _ in range(40):
            evaluate(dt, stiffness, friction, step, state)
            trial = qd_to_numpy(state.energy, transpose=True).item()
            if trial <= energy + 1e-4 * step * slope + 16 * np.finfo(float).eps * abs(energy):
                accept_step(step, state)
                break
            step *= 0.5
        else:
            raise RuntimeError("Physical-pad velocity line search exhausted")
    raise RuntimeError("Physical-pad velocity Newton exhausted")
