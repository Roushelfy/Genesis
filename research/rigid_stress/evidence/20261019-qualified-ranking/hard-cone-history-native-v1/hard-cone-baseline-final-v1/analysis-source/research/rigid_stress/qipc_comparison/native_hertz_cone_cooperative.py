"""Warp-cooperative application of the same small radial-memory projection."""

import quadrants as qd

import genesis as gs

from .native_hertz_cone_history import ConeHistoryInfo, ConeHistoryInput, ConeHistoryState, func_cone_disk


@qd.func
def next_vector(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i in qd.static(range(2)):
        result[i] = qd.simt.subgroup.shuffle_down(value[i], qd.u32(1))
    return result


@qd.func
def previous_vector(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i in qd.static(range(2)):
        result[i] = qd.simt.subgroup.shuffle_up(value[i], qd.u32(1))
    return result


@qd.func
def sum_vector(value: qd.types.vector(2)):
    result = qd.Vector.zero(gs.qd_float, 2)
    for i in qd.static(range(2)):
        result[i] = qd.simt.subgroup.reduce_all_add(value[i])
    return result


@qd.func
def suffix_vector(value: qd.types.vector(2), lane: int):
    result = value
    for offset in qd.static((1, 2, 4, 8, 16)):
        shifted = qd.Vector.zero(gs.qd_float, 2)
        for i in qd.static(range(2)):
            shifted[i] = qd.simt.subgroup.shuffle_down(result[i], qd.u32(offset))
        if lane + offset < 32:
            result += shifted
    return result


@qd.kernel
def kernel_cone_cooperative(
    inputs: ConeHistoryInput,
    state: ConeHistoryState,
    info: ConeHistoryInfo,
    maximum_iterations: qd.template(),
    penalty: qd.template(),
    count: qd.template(),
):
    qd.loop_config(block_dim=128)
    for thread in range(state.status.shape[0] * 32):
        env, lane = thread // 32, thread % 32
        active = lane < count
        radius, normal, friction = inputs.radius[env], inputs.normal[env], inputs.friction[env]
        force, angle = inputs.force[env], inputs.angle[env]
        lost = state.lost[env]
        if inputs.reset[env]:
            lost = False
        status = gs.qd_int(0)
        valid = normal >= 0 and normal < float("inf") and radius >= 0 and radius < float("inf")
        valid = (
            valid
            and friction >= 0
            and friction < float("inf")
            and force.norm() < float("inf")
            and qd.abs(angle) < float("inf")
        )
        if not valid:
            status = 2
        elif normal == 0 or friction == 0:
            status = 1
            if force.norm() > 0:
                status = 3
        elif radius <= 0 or not friction * normal < float("inf"):
            status = 2
        elif lost:
            status = 5
        elif force.norm() > friction * normal * (1 + 1e-12):
            status = 3
        prior, z, u = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
        weight, width, response = gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(0.0)
        inverse_row = qd.Vector.zero(gs.qd_float, count)
        if active:
            weight = info.weight[lane]
            width = info.radius[lane + 1] ** 2 - info.radius[lane] ** 2
            response = info.response[lane]
            for j in qd.static(range(count)):
                inverse_row[j] = info.inverse[lane, j]
        iteration = gs.qd_int(0)
        residual, dual_error, line_error, bound = gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0)
        scale = gs.qd_float(0.0)
        if status == 0:
            scale = 0.75 * friction * normal / radius
            target = force / (friction * normal)
            magnitude = target.norm()
            if active:
                t = info.radius[lane]
                if not inputs.reset[env] and state.previous_radius[env] > 0:
                    old_t = t * radius / state.previous_radius[env]
                    if old_t < 1:
                        j = qd.min(qd.cast(old_t * count, gs.qd_int), count - 1)
                        lower, upper = info.radius[j], info.radius[j + 1]
                        fraction = (old_t * old_t - lower * lower) / (upper * upper - lower * lower)
                        value = (1 - fraction) * state.line[j, env] + fraction * state.line[j + 1, env]
                        rotation = qd.Matrix([[qd.cos(angle), -qd.sin(angle)], [qd.sin(angle), qd.cos(angle)]])
                        prior = (state.previous_scale[env] / scale) * (rotation @ value)
            neighbor = next_vector(prior)
            if lane == count - 1:
                neighbor = qd.Vector.zero(gs.qd_float, 2)
            if active:
                z = func_cone_disk(prior - neighbor, width)
            if magnitude >= 1 - 1e-12:
                if active:
                    z = width * target / magnitude
            else:
                converged = False
                while iteration < maximum_iterations and not converged:
                    previous = previous_vector(z - u)
                    rhs = weight * prior + penalty * (z - u)
                    if lane > 0:
                        rhs -= penalty * previous
                    value = qd.Vector.zero(gs.qd_float, 2)
                    for j in qd.static(range(count)):
                        broadcast = qd.Vector.zero(gs.qd_float, 2)
                        for k in qd.static(range(2)):
                            broadcast[k] = qd.simt.subgroup.shuffle(rhs[k], qd.u32(j))
                        value += inverse_row[j] * broadcast
                    force_value = sum_vector(weight * value)
                    value += response * (target - force_value) / info.denominator[None]
                    neighbor = next_vector(value)
                    if lane == count - 1:
                        neighbor = qd.Vector.zero(gs.qd_float, 2)
                    old = z
                    if active:
                        relaxed = 1.6 * (value - neighbor) - 0.6 * old
                        z = func_cone_disk(relaxed + u, width)
                        u += relaxed - z
                    change = z - old
                    previous_change = previous_vector(change)
                    local_dual = gs.qd_float(0.0)
                    if active:
                        if lane == 0:
                            previous_change = qd.Vector.zero(gs.qd_float, 2)
                        local_dual = penalty * (change - previous_change).norm()
                    dual_error = qd.simt.subgroup.reduce_all_max(local_dual)
                    cumulative = suffix_vector(z, lane)
                    recovered = sum_vector(weight * cumulative)
                    local_line = gs.qd_float(0.0)
                    if active:
                        local_line = (value - cumulative).norm()
                    line_error = qd.simt.subgroup.reduce_all_max(local_line)
                    residual = (recovered - target).norm()
                    converged = residual < 1e-9 and dual_error < 1e-10 and line_error < 1e-9
                    iteration += 1
                if not converged:
                    status = 4
            cumulative = suffix_vector(z, lane)
            residual = (sum_vector(weight * cumulative) - target).norm()
            local_bound = gs.qd_float(0.0)
            if active:
                local_bound = (z / width).norm()
            bound = qd.simt.subgroup.reduce_all_max(local_bound)
            if residual >= 1e-9 or bound > 1 + 1e-10:
                status = 4
        accepted = qd.Vector.zero(gs.qd_float, 2)
        if status == 0 and active:
            accepted = z / width
        neighbor = next_vector(accepted)
        if lane == count - 1:
            neighbor = qd.Vector.zero(gs.qd_float, 2)
        cumulative = suffix_vector(z, lane)
        if active:
            state.line[lane, env] = qd.Vector.zero(gs.qd_float, 2)
            state.coefficients[lane, env] = qd.Vector.zero(gs.qd_float, 2)
            if status == 0:
                state.line[lane, env] = cumulative
                state.coefficients[lane, env] = accepted - neighbor
            state.increment[lane, env], state.dual[lane, env] = z, u
        if lane == 0:
            state.line[count, env] = qd.Vector.zero(gs.qd_float, 2)
            state.previous_radius[env], state.previous_scale[env] = 0.0, 0.0
            if status == 0:
                state.previous_radius[env], state.previous_scale[env] = radius, scale
            state.lost[env] = status != 0 and status != 1
            state.status[env], state.iterations[env] = status, iteration
            state.residual[env], state.dual_residual[env], state.line_residual[env] = residual, dual_error, line_error
            state.traction_bound[env] = bound
