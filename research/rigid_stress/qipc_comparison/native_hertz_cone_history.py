"""Quadrants radial-memory projection with force and surface-cone certificates.

An ADMM step uses one shared tridiagonal factor and vector disk projections.
The stored line is reconstructed from feasible slope increments. A failed
projection clears output and latches lost history until release or reset.
Region correspondence and tangent-frame transport remain caller inputs.
"""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_VEC


@dataclass(frozen=True)
class ConeHistoryInput:
    radius: qd.Tensor
    normal: qd.Tensor
    friction: qd.Tensor
    force: qd.Tensor
    angle: qd.Tensor
    reset: qd.Tensor


@dataclass(frozen=True)
class ConeHistoryInfo:
    radius: qd.Tensor
    weight: qd.Tensor
    diagonal_inverse: qd.Tensor
    response: qd.Tensor
    denominator: qd.Tensor
    inverse: qd.Tensor


@dataclass(frozen=True)
class ConeHistoryState:
    line: qd.Tensor
    prior: qd.Tensor
    trial: qd.Tensor
    solution: qd.Tensor
    increment: qd.Tensor
    dual: qd.Tensor
    coefficients: qd.Tensor
    previous_radius: qd.Tensor
    previous_scale: qd.Tensor
    status: qd.Tensor
    lost: qd.Tensor
    residual: qd.Tensor
    dual_residual: qd.Tensor
    line_residual: qd.Tensor
    traction_bound: qd.Tensor
    iterations: qd.Tensor


class NativeConeHistory:
    def __init__(self, count, intervals=16, maximum_iterations=2048, penalty=0.1, cooperative=True):
        assert count > 0 and intervals >= 2 and maximum_iterations > 0
        self.maximum_iterations = maximum_iterations
        self.intervals = intervals
        assert penalty > 0
        self.penalty = penalty
        self.cooperative = cooperative and gs.backend == gs.cuda and intervals <= 32
        self.input = ConeHistoryInput(
            radius=V(dtype=gs.qd_float, shape=(count,)),
            normal=V(dtype=gs.qd_float, shape=(count,)),
            friction=V(dtype=gs.qd_float, shape=(count,)),
            force=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
            angle=V(dtype=gs.qd_float, shape=(count,)),
            reset=V(dtype=gs.qd_bool, shape=(count,)),
        )
        self.info = ConeHistoryInfo(
            radius=V(dtype=gs.qd_float, shape=(intervals + 1,)),
            weight=V(dtype=gs.qd_float, shape=(intervals + 1,)),
            diagonal_inverse=V(dtype=gs.qd_float, shape=(intervals,)),
            response=V(dtype=gs.qd_float, shape=(intervals,)),
            denominator=V(dtype=gs.qd_float, shape=()),
            inverse=V(dtype=gs.qd_float, shape=(intervals, intervals)),
        )
        self.state = ConeHistoryState(
            line=V_VEC(2, dtype=gs.qd_float, shape=(intervals + 1, count)),
            prior=V_VEC(2, dtype=gs.qd_float, shape=(intervals + 1, count)),
            trial=V_VEC(2, dtype=gs.qd_float, shape=(intervals, count)),
            solution=V_VEC(2, dtype=gs.qd_float, shape=(intervals, count)),
            increment=V_VEC(2, dtype=gs.qd_float, shape=(intervals, count)),
            dual=V_VEC(2, dtype=gs.qd_float, shape=(intervals, count)),
            coefficients=V_VEC(2, dtype=gs.qd_float, shape=(intervals, count)),
            previous_radius=V(dtype=gs.qd_float, shape=(count,)),
            previous_scale=V(dtype=gs.qd_float, shape=(count,)),
            status=V(dtype=gs.qd_int, shape=(count,)),
            lost=V(dtype=gs.qd_bool, shape=(count,)),
            residual=V(dtype=gs.qd_float, shape=(count,)),
            dual_residual=V(dtype=gs.qd_float, shape=(count,)),
            line_residual=V(dtype=gs.qd_float, shape=(count,)),
            traction_bound=V(dtype=gs.qd_float, shape=(count,)),
            iterations=V(dtype=gs.qd_int, shape=(count,)),
        )
        kernel_cone_info(self.info, self.penalty)
        kernel_cone_initialize(self.input, self.state)

    def update(self):
        if self.cooperative:
            from .native_hertz_cone_cooperative import kernel_cone_cooperative

            kernel_cone_cooperative(
                self.input, self.state, self.info, self.maximum_iterations, self.penalty, self.intervals
            )
        else:
            kernel_cone_update(self.input, self.state, self.info, self.maximum_iterations, self.penalty)


@qd.kernel
def kernel_cone_info(info: ConeHistoryInfo, penalty: qd.template()):
    qd.loop_config(serialize=True)
    for only in range(1):
        count = info.radius.shape[0] - 1
        for i in range(count + 1):
            radius = i / count
            weight = gs.qd_float(0.0)
            if i > 0:
                lower = (i - 1) / count
                weight += (radius - lower) * (radius + 2 * lower) / (3 * (radius + lower))
            if i < count:
                upper = (i + 1) / count
                weight += (upper - radius) * (2 * upper + radius) / (3 * (upper + radius))
            info.radius[i], info.weight[i] = radius, 1.5 * weight
        for i in range(count):
            diagonal = info.weight[i] + penalty
            if i > 0:
                diagonal += penalty - penalty * penalty * info.diagonal_inverse[i - 1]
            info.diagonal_inverse[i] = 1 / diagonal
            rhs = info.weight[i]
            if i > 0:
                rhs += penalty * info.diagonal_inverse[i - 1] * info.response[i - 1]
            info.response[i] = rhs
        for reverse in range(count):
            i = count - 1 - reverse
            rhs = info.response[i]
            if i < count - 1:
                rhs += penalty * info.response[i + 1]
            info.response[i] = rhs * info.diagonal_inverse[i]
        denominator = gs.qd_float(0.0)
        for i in range(count):
            denominator += info.weight[i] * info.response[i]
        info.denominator[None] = denominator
        for column in range(count):
            for i in range(count):
                rhs = gs.qd_float(0.0)
                if i == column:
                    rhs = 1.0
                if i > 0:
                    rhs += penalty * info.diagonal_inverse[i - 1] * info.inverse[i - 1, column]
                info.inverse[i, column] = rhs
            for reverse in range(count):
                i = count - 1 - reverse
                rhs = info.inverse[i, column]
                if i < count - 1:
                    rhs += penalty * info.inverse[i + 1, column]
                info.inverse[i, column] = rhs * info.diagonal_inverse[i]


@qd.kernel
def kernel_cone_initialize(inputs: ConeHistoryInput, state: ConeHistoryState):
    for env in range(state.status.shape[0]):
        inputs.radius[env], inputs.normal[env], inputs.friction[env], inputs.angle[env] = 0.0, 0.0, 0.0, 0.0
        inputs.force[env], inputs.reset[env] = qd.Vector.zero(gs.qd_float, 2), False
        state.previous_radius[env], state.previous_scale[env] = 0.0, 0.0
        state.status[env], state.lost[env] = 1, False
        state.residual[env], state.dual_residual[env], state.line_residual[env] = 0.0, 0.0, 0.0
        state.traction_bound[env], state.iterations[env] = 0.0, 0
    for i, env in qd.ndrange(state.line.shape[0], state.line.shape[1]):
        state.line[i, env], state.prior[i, env] = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
    for i, env in qd.ndrange(state.increment.shape[0], state.increment.shape[1]):
        state.increment[i, env], state.dual[i, env] = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
        state.coefficients[i, env] = qd.Vector.zero(gs.qd_float, 2)


@qd.func
def func_cone_disk(value: qd.types.vector(2), cap: float):
    result = value
    length = value.norm()
    if length > cap:
        result *= cap / length
    return result


@qd.kernel
def kernel_cone_update(
    inputs: ConeHistoryInput,
    state: ConeHistoryState,
    info: ConeHistoryInfo,
    maximum_iterations: qd.template(),
    penalty: qd.template(),
):
    for env in range(state.status.shape[0]):
        count = info.radius.shape[0] - 1
        normal, radius, friction = inputs.normal[env], inputs.radius[env], inputs.friction[env]
        force, angle = inputs.force[env], inputs.angle[env]
        status = gs.qd_int(0)
        valid = normal >= 0 and normal < float("inf") and radius >= 0 and radius < float("inf")
        valid = (
            valid
            and friction >= 0
            and friction < float("inf")
            and force.norm() < float("inf")
            and qd.abs(angle) < float("inf")
        )
        state.residual[env], state.dual_residual[env], state.line_residual[env] = 0.0, 0.0, 0.0
        state.traction_bound[env], state.iterations[env] = 0.0, 0
        if inputs.reset[env]:
            state.lost[env] = False
        if not valid:
            status = 2
        elif normal == 0 or friction == 0:
            status = 1
            if force.norm() > 0:
                status = 3
        elif radius <= 0 or not friction * normal < float("inf"):
            status = 2
        elif state.lost[env]:
            status = 5
        elif force.norm() > friction * normal * (1 + 1e-12):
            status = 3
        if status == 0:
            scale = 0.75 * friction * normal / radius
            target = force / (friction * normal)
            magnitude = target.norm()
            rotation = qd.Matrix([[qd.cos(angle), -qd.sin(angle)], [qd.sin(angle), qd.cos(angle)]])
            for i in range(count + 1):
                value = qd.Vector.zero(gs.qd_float, 2)
                t = info.radius[i]
                if not inputs.reset[env] and state.previous_radius[env] > 0:
                    old_t = t * radius / state.previous_radius[env]
                    if old_t < 1:
                        j = qd.min(qd.cast(old_t * count, gs.qd_int), count - 1)
                        lower, upper = info.radius[j], info.radius[j + 1]
                        fraction = (old_t * old_t - lower * lower) / (upper * upper - lower * lower)
                        value = (1 - fraction) * state.line[j, env] + fraction * state.line[j + 1, env]
                        value = (state.previous_scale[env] / scale) * (rotation @ value)
                state.prior[i, env] = value
            if magnitude >= 1 - 1e-12:
                for i in range(count):
                    width = info.radius[i + 1] ** 2 - info.radius[i] ** 2
                    state.increment[i, env] = width * target / magnitude
            else:
                for i in range(count):
                    width = info.radius[i + 1] ** 2 - info.radius[i] ** 2
                    state.increment[i, env] = func_cone_disk(state.prior[i, env] - state.prior[i + 1, env], width)
                    state.dual[i, env] = qd.Vector.zero(gs.qd_float, 2)
                converged = False
                iteration = 0
                while iteration < maximum_iterations and not converged:
                    for i in range(count):
                        rhs = info.weight[i] * state.prior[i, env] + penalty * (
                            state.increment[i, env] - state.dual[i, env]
                        )
                        if i > 0:
                            rhs -= penalty * (state.increment[i - 1, env] - state.dual[i - 1, env])
                            rhs += penalty * info.diagonal_inverse[i - 1] * state.trial[i - 1, env]
                        state.trial[i, env] = rhs
                    for reverse in range(count):
                        i = count - 1 - reverse
                        rhs = state.trial[i, env]
                        if i < count - 1:
                            rhs += penalty * state.trial[i + 1, env]
                        state.trial[i, env] = rhs * info.diagonal_inverse[i]
                    current = qd.Vector.zero(gs.qd_float, 2)
                    for i in range(count):
                        current += info.weight[i] * state.trial[i, env]
                    correction = (target - current) / info.denominator[None]
                    for i in range(count):
                        state.solution[i, env] = state.trial[i, env] + info.response[i] * correction
                    for i in range(count):
                        difference = state.solution[i, env]
                        if i < count - 1:
                            difference -= state.solution[i + 1, env]
                        previous = state.increment[i, env]
                        relaxed = 1.6 * difference - 0.6 * previous
                        width = info.radius[i + 1] ** 2 - info.radius[i] ** 2
                        updated = func_cone_disk(relaxed + state.dual[i, env], width)
                        state.dual[i, env] += relaxed - updated
                        state.increment[i, env] = updated
                        state.trial[i, env] = updated - previous
                    dual_error = gs.qd_float(0.0)
                    for i in range(count):
                        difference = state.trial[i, env]
                        if i > 0:
                            difference -= state.trial[i - 1, env]
                        dual_error = qd.max(dual_error, penalty * difference.norm())
                    cumulative, recovered = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
                    line_error = gs.qd_float(0.0)
                    for reverse in range(count):
                        i = count - 1 - reverse
                        cumulative += state.increment[i, env]
                        recovered += info.weight[i] * cumulative
                        line_error = qd.max(line_error, (state.solution[i, env] - cumulative).norm())
                    residual = (recovered - target).norm()
                    state.residual[env], state.dual_residual[env], state.line_residual[env] = (
                        residual,
                        dual_error,
                        line_error,
                    )
                    converged = residual < 1e-9 and dual_error < 1e-10 and line_error < 1e-9
                    iteration += 1
                state.iterations[env] = iteration
                if not converged:
                    status = 4
            if status == 0:
                cumulative, recovered = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
                bound = gs.qd_float(0.0)
                state.line[count, env] = qd.Vector.zero(gs.qd_float, 2)
                for reverse in range(count):
                    i = count - 1 - reverse
                    cumulative += state.increment[i, env]
                    state.line[i, env] = cumulative
                    recovered += info.weight[i] * cumulative
                    width = info.radius[i + 1] ** 2 - info.radius[i] ** 2
                    slope = state.increment[i, env] / width
                    bound = qd.max(bound, slope.norm())
                    coefficient = slope
                    if i < count - 1:
                        next_width = info.radius[i + 2] ** 2 - info.radius[i + 1] ** 2
                        coefficient -= state.increment[i + 1, env] / next_width
                    state.coefficients[i, env] = coefficient
                state.residual[env], state.traction_bound[env] = (recovered - target).norm(), bound
                if state.residual[env] >= 1e-9 or bound > 1 + 1e-10:
                    status = 4
                else:
                    state.previous_radius[env], state.previous_scale[env] = radius, scale
        if status != 0:
            for i in range(count + 1):
                state.line[i, env] = qd.Vector.zero(gs.qd_float, 2)
            for i in range(count):
                state.coefficients[i, env] = qd.Vector.zero(gs.qd_float, 2)
            state.previous_radius[env], state.previous_scale[env] = 0.0, 0.0
            state.lost[env] = status != 1
        state.status[env] = status
