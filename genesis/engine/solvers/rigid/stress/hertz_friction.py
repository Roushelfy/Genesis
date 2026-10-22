"""Fixed16 vector radial memory with exact force and continuous cone checks."""

from dataclasses import dataclass
from typing import ClassVar
import quadrants as qd
import genesis as gs
from genesis.utils.array_class import DataItem, DataKind, V, V_VEC, iter_data, of_kind


@dataclass(frozen=True)
class StressFrictionInput:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    radius: qd.Tensor
    normal: qd.Tensor
    friction: qd.Tensor
    force: qd.Tensor
    angle: qd.Tensor
    previous_index: qd.Tensor
    is_reset: qd.Tensor
    is_valid: qd.Tensor


@dataclass(frozen=True)
class StressFrictionInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT
    radius: qd.Tensor
    weight: qd.Tensor
    diagonal_inverse: qd.Tensor
    response: qd.Tensor
    denominator: qd.Tensor
    inverse: qd.Tensor


@dataclass(frozen=True)
class StressFrictionState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    line: qd.Tensor = of_kind(DataKind.STATE)
    previous_radius: qd.Tensor = of_kind(DataKind.STATE)
    previous_scale: qd.Tensor = of_kind(DataKind.STATE)
    is_lost: qd.Tensor = of_kind(DataKind.STATE)
    prior: qd.Tensor
    prior_is_lost: qd.Tensor
    trial: qd.Tensor
    solution: qd.Tensor
    increment: qd.Tensor
    dual: qd.Tensor
    coefficients: qd.Tensor
    status: qd.Tensor
    residual: qd.Tensor
    dual_residual: qd.Tensor
    line_residual: qd.Tensor
    traction_bound: qd.Tensor
    iterations: qd.Tensor


class StressFrictionHistory:
    """Advance the declared16-interval friction approximation in Quadrants.

    The material contact binding supplies region correspondence and tangent
    transport. A global staging pass reads the prior before any slot is
    overwritten, preserving history when current collider ordering changes.
    Physical line/radius/scale/loss fields travel with solver checkpoints.
    """

    def __init__(self, n_regions: int, maximum_iterations: int = 2048, is_cooperative: bool = True):
        assert n_regions > 0 and maximum_iterations > 0
        self.maximum_iterations = maximum_iterations
        self.penalty = 0.1
        self.is_cooperative = is_cooperative and gs.backend == gs.cuda
        self.input = StressFrictionInput(
            radius=V(dtype=gs.qd_float, shape=(n_regions,)),
            normal=V(dtype=gs.qd_float, shape=(n_regions,)),
            friction=V(dtype=gs.qd_float, shape=(n_regions,)),
            force=V_VEC(2, dtype=gs.qd_float, shape=(n_regions,)),
            angle=V(dtype=gs.qd_float, shape=(n_regions,)),
            previous_index=V(dtype=gs.qd_int, shape=(n_regions,)),
            is_reset=V(dtype=gs.qd_bool, shape=(n_regions,)),
            is_valid=V(dtype=gs.qd_bool, shape=(n_regions,)),
        )
        self.info = StressFrictionInfo(
            radius=V(dtype=gs.qd_float, shape=(17,)),
            weight=V(dtype=gs.qd_float, shape=(17,)),
            diagonal_inverse=V(dtype=gs.qd_float, shape=(16,)),
            response=V(dtype=gs.qd_float, shape=(16,)),
            denominator=V(dtype=gs.qd_float, shape=()),
            inverse=V(dtype=gs.qd_float, shape=(16, 16)),
        )
        self.state = StressFrictionState(
            line=V_VEC(2, dtype=gs.qd_float, shape=(17, n_regions)),
            previous_radius=V(dtype=gs.qd_float, shape=(n_regions,)),
            previous_scale=V(dtype=gs.qd_float, shape=(n_regions,)),
            is_lost=V(dtype=gs.qd_bool, shape=(n_regions,)),
            prior=V_VEC(2, dtype=gs.qd_float, shape=(17, n_regions)),
            prior_is_lost=V(dtype=gs.qd_bool, shape=(n_regions,)),
            trial=V_VEC(2, dtype=gs.qd_float, shape=(16, n_regions)),
            solution=V_VEC(2, dtype=gs.qd_float, shape=(16, n_regions)),
            increment=V_VEC(2, dtype=gs.qd_float, shape=(16, n_regions)),
            dual=V_VEC(2, dtype=gs.qd_float, shape=(16, n_regions)),
            coefficients=V_VEC(2, dtype=gs.qd_float, shape=(16, n_regions)),
            status=V(dtype=gs.qd_int, shape=(n_regions,)),
            residual=V(dtype=gs.qd_float, shape=(n_regions,)),
            dual_residual=V(dtype=gs.qd_float, shape=(n_regions,)),
            line_residual=V(dtype=gs.qd_float, shape=(n_regions,)),
            traction_bound=V(dtype=gs.qd_float, shape=(n_regions,)),
            iterations=V(dtype=gs.qd_int, shape=(n_regions,)),
        )
        kernel_friction_info(self.penalty, self.info)
        kernel_friction_initialize(self.input, self.state)

    def update(self) -> None:
        kernel_friction_prior(self.input, self.state)
        if self.is_cooperative:
            kernel_friction_cooperative(self.maximum_iterations, self.penalty, self.input, self.state, self.info)
        else:
            kernel_friction_serial(self.maximum_iterations, self.penalty, self.input, self.state, self.info)

    def data(self, prefix: str):
        yield DataItem(prefix + ".maximum_iterations", self.maximum_iterations, DataKind.CONFIG)
        yield DataItem(prefix + ".penalty", self.penalty, DataKind.CONFIG)
        yield DataItem(prefix + ".is_cooperative", self.is_cooperative, DataKind.CONFIG)
        yield from iter_data(self.input, prefix + ".input")
        yield from iter_data(self.info, prefix + ".info")
        yield from iter_data(self.state, prefix + ".state")


@qd.func
def func_friction_reset(i_p: int, inputs: StressFrictionInput, state: StressFrictionState):
    (inputs.radius[i_p], inputs.normal[i_p], inputs.friction[i_p], inputs.angle[i_p]) = (0.0, 0.0, 0.0, 0.0)
    (inputs.force[i_p], inputs.is_reset[i_p], inputs.is_valid[i_p]) = (qd.Vector.zero(gs.qd_float, 2), False, True)
    inputs.previous_index[i_p] = i_p
    (state.previous_radius[i_p], state.previous_scale[i_p]) = (0.0, 0.0)
    (state.status[i_p], state.is_lost[i_p], state.prior_is_lost[i_p]) = (1, False, False)
    (state.residual[i_p], state.dual_residual[i_p], state.line_residual[i_p]) = (0.0, 0.0, 0.0)
    (state.traction_bound[i_p], state.iterations[i_p]) = (0.0, 0)
    for i_s in range(17):
        (state.line[i_s, i_p], state.prior[i_s, i_p]) = (qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2))
    for i_s in range(16):
        (state.increment[i_s, i_p], state.dual[i_s, i_p]) = (
            qd.Vector.zero(gs.qd_float, 2),
            qd.Vector.zero(gs.qd_float, 2),
        )
        state.coefficients[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)


@qd.kernel
def kernel_friction_initialize(inputs: StressFrictionInput, state: StressFrictionState):
    for i_p in range(state.status.shape[0]):
        func_friction_reset(i_p, inputs, state)


@qd.kernel
def kernel_friction_prior(inputs: StressFrictionInput, state: StressFrictionState):
    for i_p in range(state.status.shape[0]):
        i_previous = inputs.previous_index[i_p]
        is_previous = not inputs.is_reset[i_p] and i_previous >= 0 and (i_previous < state.status.shape[0])
        state.prior_is_lost[i_p] = False
        if is_previous:
            state.prior_is_lost[i_p] = state.is_lost[i_previous]
    for i_s, i_p in qd.ndrange(17, state.status.shape[0]):
        prior = qd.Vector.zero(gs.qd_float, 2)
        i_previous = inputs.previous_index[i_p]
        (radius, normal, friction) = (inputs.radius[i_p], inputs.normal[i_p], inputs.friction[i_p])
        if not inputs.is_reset[i_p] and i_previous >= 0 and (i_previous < state.status.shape[0]):
            if radius > 0 and normal > 0 and (friction > 0) and (state.previous_radius[i_previous] > 0):
                t = i_s * radius / (16 * state.previous_radius[i_previous])
                if t < 1:
                    i_old = qd.min(qd.cast(t * 16, gs.qd_int), 15)
                    (lower, upper) = (i_old / 16, (i_old + 1) / 16)
                    fraction = (t * t - lower * lower) / (upper * upper - lower * lower)
                    value = (1 - fraction) * state.line[i_old, i_previous] + fraction * state.line[
                        i_old + 1, i_previous
                    ]
                    angle = inputs.angle[i_p]
                    rotation = qd.Matrix([[qd.cos(angle), -qd.sin(angle)], [qd.sin(angle), qd.cos(angle)]])
                    scale = 0.75 * friction * normal / radius
                    prior = state.previous_scale[i_previous] / scale * (rotation @ value)
        state.prior[i_s, i_p] = prior


@qd.kernel
def kernel_friction_info(penalty: float, info: StressFrictionInfo):
    qd.loop_config(serialize=True)
    for i_0 in range(1):
        n_segments = info.radius.shape[0] - 1
        for i_s in range(n_segments + 1):
            radius = i_s / n_segments
            weight = gs.qd_float(0.0)
            if i_s > 0:
                lower = (i_s - 1) / n_segments
                weight += (radius - lower) * (radius + 2 * lower) / (3 * (radius + lower))
            if i_s < n_segments:
                upper = (i_s + 1) / n_segments
                weight += (upper - radius) * (2 * upper + radius) / (3 * (upper + radius))
            (info.radius[i_s], info.weight[i_s]) = (radius, 1.5 * weight)
        for i_s in range(n_segments):
            diagonal = info.weight[i_s] + penalty
            if i_s > 0:
                diagonal += penalty - penalty * penalty * info.diagonal_inverse[i_s - 1]
            info.diagonal_inverse[i_s] = 1 / diagonal
            rhs = info.weight[i_s]
            if i_s > 0:
                rhs += penalty * info.diagonal_inverse[i_s - 1] * info.response[i_s - 1]
            info.response[i_s] = rhs
        for i_r in range(n_segments):
            i_s = n_segments - 1 - i_r
            rhs = info.response[i_s]
            if i_s < n_segments - 1:
                rhs += penalty * info.response[i_s + 1]
            info.response[i_s] = rhs * info.diagonal_inverse[i_s]
        denominator = gs.qd_float(0.0)
        for i_s in range(n_segments):
            denominator += info.weight[i_s] * info.response[i_s]
        info.denominator[None] = denominator
        for j_s in range(n_segments):
            for i_s in range(n_segments):
                rhs = gs.qd_float(0.0)
                if i_s == j_s:
                    rhs = 1.0
                if i_s > 0:
                    rhs += penalty * info.diagonal_inverse[i_s - 1] * info.inverse[i_s - 1, j_s]
                info.inverse[i_s, j_s] = rhs
            for i_r in range(n_segments):
                i_s = n_segments - 1 - i_r
                rhs = info.inverse[i_s, j_s]
                if i_s < n_segments - 1:
                    rhs += penalty * info.inverse[i_s + 1, j_s]
                info.inverse[i_s, j_s] = rhs * info.diagonal_inverse[i_s]


@qd.func
def func_friction_disk(value: qd.types.vector(2), cap: float):
    result = value
    length = value.norm()
    if length > cap:
        result *= cap / length
    return result


@qd.kernel
def kernel_friction_serial(
    maximum_iterations: int,
    penalty: float,
    inputs: StressFrictionInput,
    state: StressFrictionState,
    info: StressFrictionInfo,
):
    for i_p in range(state.status.shape[0]):
        n_segments = info.radius.shape[0] - 1
        (normal, radius, friction) = (inputs.normal[i_p], inputs.radius[i_p], inputs.friction[i_p])
        (force, angle) = (inputs.force[i_p], inputs.angle[i_p])
        status = gs.qd_int(0)
        is_valid = (
            normal >= 0 and normal < float("inf") and (radius >= 0) and (radius < float("inf"))
        ) and inputs.is_valid[i_p]
        is_valid = (
            is_valid
            and friction >= 0
            and (friction < float("inf"))
            and (force.norm() < float("inf"))
            and (qd.abs(angle) < float("inf"))
        )
        (state.residual[i_p], state.dual_residual[i_p], state.line_residual[i_p]) = (0.0, 0.0, 0.0)
        (state.traction_bound[i_p], state.iterations[i_p]) = (0.0, 0)
        if inputs.is_reset[i_p]:
            state.is_lost[i_p] = False
        if not is_valid:
            status = 2
        elif normal == 0 or friction == 0:
            status = 1
            if force.norm() > 0:
                status = 3
        elif radius <= 0 or not friction * normal < float("inf"):
            status = 2
        elif state.prior_is_lost[i_p]:
            status = 5
        elif force.norm() > friction * normal * (1 + 1e-12):
            status = 3
        if status == 0:
            scale = 0.75 * friction * normal / radius
            target = force / (friction * normal)
            magnitude = target.norm()
            rotation = qd.Matrix([[qd.cos(angle), -qd.sin(angle)], [qd.sin(angle), qd.cos(angle)]])
            if magnitude >= 1 - 1e-12:
                for i_s in range(n_segments):
                    width = info.radius[i_s + 1] ** 2 - info.radius[i_s] ** 2
                    state.increment[i_s, i_p] = width * target / magnitude
            else:
                for i_s in range(n_segments):
                    width = info.radius[i_s + 1] ** 2 - info.radius[i_s] ** 2
                    state.increment[i_s, i_p] = func_friction_disk(
                        state.prior[i_s, i_p] - state.prior[i_s + 1, i_p], width
                    )
                    state.dual[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)
                is_converged = False
                iteration = 0
                for i_iteration in range(maximum_iterations):
                    if not is_converged:
                        for i_s in range(n_segments):
                            rhs = info.weight[i_s] * state.prior[i_s, i_p] + penalty * (
                                state.increment[i_s, i_p] - state.dual[i_s, i_p]
                            )
                            if i_s > 0:
                                rhs -= penalty * (state.increment[i_s - 1, i_p] - state.dual[i_s - 1, i_p])
                                rhs += penalty * info.diagonal_inverse[i_s - 1] * state.trial[i_s - 1, i_p]
                            state.trial[i_s, i_p] = rhs
                        for i_r in range(n_segments):
                            i_s = n_segments - 1 - i_r
                            rhs = state.trial[i_s, i_p]
                            if i_s < n_segments - 1:
                                rhs += penalty * state.trial[i_s + 1, i_p]
                            state.trial[i_s, i_p] = rhs * info.diagonal_inverse[i_s]
                        current = qd.Vector.zero(gs.qd_float, 2)
                        for i_s in range(n_segments):
                            current += info.weight[i_s] * state.trial[i_s, i_p]
                        correction = (target - current) / info.denominator[None]
                        for i_s in range(n_segments):
                            state.solution[i_s, i_p] = state.trial[i_s, i_p] + info.response[i_s] * correction
                        for i_s in range(n_segments):
                            difference = state.solution[i_s, i_p]
                            if i_s < n_segments - 1:
                                difference -= state.solution[i_s + 1, i_p]
                            previous = state.increment[i_s, i_p]
                            relaxed = 1.6 * difference - 0.6 * previous
                            width = info.radius[i_s + 1] ** 2 - info.radius[i_s] ** 2
                            updated = func_friction_disk(relaxed + state.dual[i_s, i_p], width)
                            state.dual[i_s, i_p] += relaxed - updated
                            state.increment[i_s, i_p] = updated
                            state.trial[i_s, i_p] = updated - previous
                        dual_error = gs.qd_float(0.0)
                        for i_s in range(n_segments):
                            difference = state.trial[i_s, i_p]
                            if i_s > 0:
                                difference -= state.trial[i_s - 1, i_p]
                            dual_error = qd.max(dual_error, penalty * difference.norm())
                        (cumulative, recovered) = (qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2))
                        line_error = gs.qd_float(0.0)
                        for i_r in range(n_segments):
                            i_s = n_segments - 1 - i_r
                            cumulative += state.increment[i_s, i_p]
                            recovered += info.weight[i_s] * cumulative
                            line_error = qd.max(line_error, (state.solution[i_s, i_p] - cumulative).norm())
                        residual = (recovered - target).norm()
                        (state.residual[i_p], state.dual_residual[i_p], state.line_residual[i_p]) = (
                            residual,
                            dual_error,
                            line_error,
                        )
                        is_converged = residual < 1e-09 and dual_error < 1e-10 and (line_error < 1e-09)
                        iteration += 1
                state.iterations[i_p] = iteration
                if not is_converged:
                    status = 4
            if status == 0:
                (cumulative, recovered) = (qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2))
                bound = gs.qd_float(0.0)
                state.line[n_segments, i_p] = qd.Vector.zero(gs.qd_float, 2)
                for i_r in range(n_segments):
                    i_s = n_segments - 1 - i_r
                    cumulative += state.increment[i_s, i_p]
                    state.line[i_s, i_p] = cumulative
                    recovered += info.weight[i_s] * cumulative
                    width = info.radius[i_s + 1] ** 2 - info.radius[i_s] ** 2
                    slope = state.increment[i_s, i_p] / width
                    bound = qd.max(bound, slope.norm())
                    coefficient = slope
                    if i_s < n_segments - 1:
                        next_width = info.radius[i_s + 2] ** 2 - info.radius[i_s + 1] ** 2
                        coefficient -= state.increment[i_s + 1, i_p] / next_width
                    state.coefficients[i_s, i_p] = coefficient
                (state.residual[i_p], state.traction_bound[i_p]) = ((recovered - target).norm(), bound)
                if state.residual[i_p] >= 1e-09 or bound > 1 + 1e-10:
                    status = 4
                else:
                    (state.previous_radius[i_p], state.previous_scale[i_p]) = (radius, scale)
        if status != 0:
            for i_s in range(n_segments + 1):
                state.line[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)
            for i_s in range(n_segments):
                state.coefficients[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)
            (state.previous_radius[i_p], state.previous_scale[i_p]) = (0.0, 0.0)
            state.is_lost[i_p] = status != 1
        state.status[i_p] = status
        state.is_lost[i_p] = status != 0 and status != 1


@qd.func
def func_friction_next(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i_d in qd.static(range(2)):
        result[i_d] = qd.simt.subgroup.shuffle_down(value[i_d], qd.u32(1))
    return result


@qd.func
def func_friction_previous(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i_d in qd.static(range(2)):
        result[i_d] = qd.simt.subgroup.shuffle_up(value[i_d], qd.u32(1))
    return result


@qd.func
def func_friction_sum(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i_d in qd.static(range(2)):
        result[i_d] = qd.simt.subgroup.reduce_all_add(value[i_d])
    return result


@qd.func
def func_friction_suffix(value: qd.types.vector(2), i_s: int):
    result = value
    for i_o in qd.static((1, 2, 4, 8, 16)):
        shifted = qd.Vector.zero(gs.qd_float, 2)
        for i_d in qd.static(range(2)):
            shifted[i_d] = qd.simt.subgroup.shuffle_down(result[i_d], qd.u32(i_o))
        if i_s + i_o < 32:
            result += shifted
    return result


@qd.kernel
def kernel_friction_cooperative(
    maximum_iterations: int,
    penalty: float,
    inputs: StressFrictionInput,
    state: StressFrictionState,
    info: StressFrictionInfo,
):
    qd.loop_config(block_dim=128)
    for i_t in range(state.status.shape[0] * 32):
        (i_p, i_s) = (i_t // 32, i_t % 32)
        is_active = i_s < 16
        (radius, normal, friction) = (inputs.radius[i_p], inputs.normal[i_p], inputs.friction[i_p])
        (force, angle) = (inputs.force[i_p], inputs.angle[i_p])
        is_lost = state.prior_is_lost[i_p]
        if inputs.is_reset[i_p]:
            is_lost = False
        status = gs.qd_int(0)
        is_valid = (
            normal >= 0 and normal < float("inf") and (radius >= 0) and (radius < float("inf"))
        ) and inputs.is_valid[i_p]
        is_valid = (
            is_valid
            and friction >= 0
            and (friction < float("inf"))
            and (force.norm() < float("inf"))
            and (qd.abs(angle) < float("inf"))
        )
        if not is_valid:
            status = 2
        elif normal == 0 or friction == 0:
            status = 1
            if force.norm() > 0:
                status = 3
        elif radius <= 0 or not friction * normal < float("inf"):
            status = 2
        elif is_lost:
            status = 5
        elif force.norm() > friction * normal * (1 + 1e-12):
            status = 3
        (prior, z, u) = (qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2))
        (weight, width, response) = (gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(0.0))
        inverse_row = qd.Vector.zero(gs.qd_float, 16)
        if is_active:
            weight = info.weight[i_s]
            width = info.radius[i_s + 1] ** 2 - info.radius[i_s] ** 2
            response = info.response[i_s]
            for j_s in qd.static(range(16)):
                inverse_row[j_s] = info.inverse[i_s, j_s]
        iteration = gs.qd_int(0)
        (residual, dual_error, line_error, bound) = (
            gs.qd_float(0.0),
            gs.qd_float(0.0),
            gs.qd_float(0.0),
            gs.qd_float(0.0),
        )
        scale = gs.qd_float(0.0)
        if status == 0:
            scale = 0.75 * friction * normal / radius
            target = force / (friction * normal)
            magnitude = target.norm()
            if is_active:
                prior = state.prior[i_s, i_p]
            neighbor = func_friction_next(prior)
            if i_s == 16 - 1:
                neighbor = qd.Vector.zero(gs.qd_float, 2)
            if is_active:
                z = func_friction_disk(prior - neighbor, width)
            if magnitude >= 1 - 1e-12:
                if is_active:
                    z = width * target / magnitude
            else:
                is_converged = False
                for i_iteration in range(maximum_iterations):
                    if not is_converged:
                        previous = func_friction_previous(z - u)
                        rhs = weight * prior + penalty * (z - u)
                        if i_s > 0:
                            rhs -= penalty * previous
                        value = qd.Vector.zero(gs.qd_float, 2)
                        for j_s in qd.static(range(16)):
                            broadcast = qd.Vector.zero(gs.qd_float, 2)
                            for i_d in qd.static(range(2)):
                                broadcast[i_d] = qd.simt.subgroup.shuffle(rhs[i_d], qd.u32(j_s))
                            value += inverse_row[j_s] * broadcast
                        force_value = func_friction_sum(weight * value)
                        value += response * (target - force_value) / info.denominator[None]
                        neighbor = func_friction_next(value)
                        if i_s == 16 - 1:
                            neighbor = qd.Vector.zero(gs.qd_float, 2)
                        old = z
                        if is_active:
                            relaxed = 1.6 * (value - neighbor) - 0.6 * old
                            z = func_friction_disk(relaxed + u, width)
                            u += relaxed - z
                        change = z - old
                        previous_change = func_friction_previous(change)
                        local_dual = gs.qd_float(0.0)
                        if is_active:
                            if i_s == 0:
                                previous_change = qd.Vector.zero(gs.qd_float, 2)
                            local_dual = penalty * (change - previous_change).norm()
                        dual_error = qd.simt.subgroup.reduce_all_max(local_dual)
                        cumulative = func_friction_suffix(z, i_s)
                        recovered = func_friction_sum(weight * cumulative)
                        local_line = gs.qd_float(0.0)
                        if is_active:
                            local_line = (value - cumulative).norm()
                        line_error = qd.simt.subgroup.reduce_all_max(local_line)
                        residual = (recovered - target).norm()
                        is_converged = residual < 1e-09 and dual_error < 1e-10 and (line_error < 1e-09)
                        iteration += 1
                if not is_converged:
                    status = 4
            cumulative = func_friction_suffix(z, i_s)
            residual = (func_friction_sum(weight * cumulative) - target).norm()
            local_bound = gs.qd_float(0.0)
            if is_active:
                local_bound = (z / width).norm()
            bound = qd.simt.subgroup.reduce_all_max(local_bound)
            if residual >= 1e-09 or bound > 1 + 1e-10:
                status = 4
        accepted = qd.Vector.zero(gs.qd_float, 2)
        if status == 0 and is_active:
            accepted = z / width
        neighbor = func_friction_next(accepted)
        if i_s == 16 - 1:
            neighbor = qd.Vector.zero(gs.qd_float, 2)
        cumulative = func_friction_suffix(z, i_s)
        if is_active:
            state.line[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)
            state.coefficients[i_s, i_p] = qd.Vector.zero(gs.qd_float, 2)
            if status == 0:
                state.line[i_s, i_p] = cumulative
                state.coefficients[i_s, i_p] = accepted - neighbor
            (state.increment[i_s, i_p], state.dual[i_s, i_p]) = (z, u)
        if i_s == 0:
            state.line[16, i_p] = qd.Vector.zero(gs.qd_float, 2)
            (state.previous_radius[i_p], state.previous_scale[i_p]) = (0.0, 0.0)
            if status == 0:
                (state.previous_radius[i_p], state.previous_scale[i_p]) = (radius, scale)
            state.is_lost[i_p] = status != 0 and status != 1
            (state.status[i_p], state.iterations[i_p]) = (status, iteration)
            (state.residual[i_p], state.dual_residual[i_p], state.line_residual[i_p]) = (
                residual,
                dual_error,
                line_error,
            )
            state.traction_bound[i_p] = bound
