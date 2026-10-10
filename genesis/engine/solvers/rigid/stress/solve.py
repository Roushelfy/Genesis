"""Batched native sparse recovery with complete equilibrium checks."""

import quadrants as qd

import genesis as gs

from .data import StressInfo, StressState
from .operators import func_shape_gradient


@qd.func
def func_precondition(i_n: int, residual: qd.types.vector(3), stress_info: StressInfo, block: qd.template()):
    value = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(block):
        value = stress_info.diagonal_inverse[i_n] @ residual
    else:
        for i_a in qd.static(range(3)):
            value[i_a] = stress_info.diagonal_inverse[i_n][i_a, i_a] * residual[i_a]
    return value


@qd.kernel(graph=True)
def kernel_balance(omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        load = stress_state.force[i_n, i_b] + stress_info.centrifugal[i_n] @ terms
        stress_state.rhs[i_n, i_b] = load
        wrench = stress_info.modes[i_n].transpose() @ load
        for i_a in qd.static(range(6)):
            qd.atomic_add(stress_state.wrench[i_b][i_a], wrench[i_a])
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        acceleration = stress_info.gram_inverse[None] @ stress_state.wrench[i_b]
        stress_state.rhs[i_n, i_b] -= stress_info.mass_modes[i_n] @ acceleration


@qd.kernel(graph=True)
def kernel_direct_init(stress_state: StressState):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.valid[i_b] = True
        stress_state.iterations[i_b] = 1
        stress_state.rhs_norm_squared[i_b] = 0.0
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        qd.atomic_add(stress_state.rhs_norm_squared[i_b], stress_state.rhs[i_n, i_b].dot(stress_state.rhs[i_n, i_b]))


@qd.kernel(graph=True)
def kernel_pcg_init(
    young: float, stress_state: StressState, stress_info: StressInfo, block: qd.template(), warm_start: qd.template()
):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        stress_state.iterations[i_b] = 0
        stress_state.rhs_norm_squared[i_b] = 0.0
        stress_state.residual_norm_squared[i_b] = 0.0
        stress_state.residual_preconditioned[i_b] = 0.0
        stress_state.valid[i_b] = True
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        if qd.static(not warm_start):
            stress_state.displacement[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
        stress_state.displacement[i_n, i_b] *= stress_info.is_free[i_n]
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        product = qd.Vector.zero(gs.qd_float, 3)
        for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
            j_n = stress_info.columns[i_entry]
            product += young * stress_info.stiffness[i_entry] @ stress_state.displacement[j_n, i_b]
        residual = (stress_state.rhs[i_n, i_b] - product) * stress_info.is_free[i_n]
        z = func_precondition(i_n, residual, stress_info, block) / young
        stress_state.residual[i_n, i_b] = residual
        stress_state.preconditioned[i_n, i_b] = z
        stress_state.direction[i_n, i_b] = z
        qd.atomic_add(stress_state.rhs_norm_squared[i_b], stress_state.rhs[i_n, i_b].dot(stress_state.rhs[i_n, i_b]))
        qd.atomic_add(stress_state.residual_norm_squared[i_b], residual.dot(residual))
        qd.atomic_add(stress_state.residual_preconditioned[i_b], residual.dot(z))


@qd.func
def func_pcg_iteration(
    young: float,
    tolerance: float,
    absolute_tolerance: float,
    stress_state: StressState,
    stress_info: StressInfo,
    block: qd.template(),
):
    for i_b in range(stress_state.active.shape[0]):
        budget = qd.max(
            absolute_tolerance * absolute_tolerance, tolerance * tolerance * stress_state.rhs_norm_squared[i_b]
        )
        if stress_state.residual_norm_squared[i_b] <= budget:
            stress_state.active[i_b] = 0
        stress_state.direction_product[i_b] = 0.0
        stress_state.next_residual_preconditioned[i_b] = 0.0
        stress_state.residual_norm_squared[i_b] = 0.0
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        if stress_state.active[i_b]:
            product = qd.Vector.zero(gs.qd_float, 3)
            for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
                j_n = stress_info.columns[i_entry]
                product += young * stress_info.stiffness[i_entry] @ stress_state.direction[j_n, i_b]
            product *= stress_info.is_free[i_n]
            stress_state.product[i_n, i_b] = product
            qd.atomic_add(stress_state.direction_product[i_b], stress_state.direction[i_n, i_b].dot(product))
    for i_b in range(stress_state.active.shape[0]):
        if stress_state.active[i_b]:
            stress_state.iterations[i_b] += 1
            if not (stress_state.direction_product[i_b] > 0.0):
                stress_state.active[i_b] = 0
                stress_state.valid[i_b] = False
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        if stress_state.active[i_b]:
            alpha = stress_state.residual_preconditioned[i_b] / stress_state.direction_product[i_b]
            stress_state.displacement[i_n, i_b] += alpha * stress_state.direction[i_n, i_b]
            residual = stress_state.residual[i_n, i_b] - alpha * stress_state.product[i_n, i_b]
            z = func_precondition(i_n, residual, stress_info, block) / young
            stress_state.residual[i_n, i_b] = residual
            stress_state.preconditioned[i_n, i_b] = z
            qd.atomic_add(stress_state.residual_norm_squared[i_b], residual.dot(residual))
            qd.atomic_add(stress_state.next_residual_preconditioned[i_b], residual.dot(z))
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        if stress_state.active[i_b]:
            beta = stress_state.next_residual_preconditioned[i_b] / stress_state.residual_preconditioned[i_b]
            stress_state.direction[i_n, i_b] = (
                stress_state.preconditioned[i_n, i_b] + beta * stress_state.direction[i_n, i_b]
            )
    for i_b in range(stress_state.active.shape[0]):
        if stress_state.active[i_b]:
            stress_state.residual_preconditioned[i_b] = stress_state.next_residual_preconditioned[i_b]


@qd.kernel(graph=True)
def kernel_pcg_chunk(
    young: float,
    tolerance: float,
    absolute_tolerance: float,
    stress_state: StressState,
    stress_info: StressInfo,
    block: qd.template(),
):
    for _ in qd.static(range(16)):
        func_pcg_iteration(young, tolerance, absolute_tolerance, stress_state, stress_info, block)


@qd.kernel
def kernel_active(stress_state: StressState) -> int:
    count = 0
    for i_b in range(stress_state.active.shape[0]):
        count += stress_state.active[i_b]
    return count


@qd.kernel(graph=True)
def kernel_full_residual(
    young: float, tolerance: float, absolute_tolerance: float, stress_state: StressState, stress_info: StressInfo
):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.residual_norm_squared[i_b] = 0.0
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        product = qd.Vector.zero(gs.qd_float, 3)
        for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
            j_n = stress_info.columns[i_entry]
            product += young * stress_info.stiffness[i_entry] @ stress_state.displacement[j_n, i_b]
        residual = stress_state.rhs[i_n, i_b] - product
        stress_state.residual[i_n, i_b] = residual
        qd.atomic_add(stress_state.residual_norm_squared[i_b], residual.dot(residual))
    for i_b in range(stress_state.active.shape[0]):
        budget = qd.max(
            absolute_tolerance * absolute_tolerance, tolerance * tolerance * stress_state.rhs_norm_squared[i_b]
        )
        stress_state.valid[i_b] = stress_state.valid[i_b] and stress_state.residual_norm_squared[i_b] <= budget


@qd.kernel(graph=True)
def kernel_peak(young: float, poisson: float, stress_state: StressState, stress_info: StressInfo):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.peak[i_b] = 0.0
    for i_e, i_b in qd.ndrange(stress_info.elements.shape[0], stress_state.active.shape[0]):
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        mu = young / (2.0 * (1.0 + poisson))
        peak_squared = gs.qd_float(0.0)
        for i_corner in range(4):
            bary = qd.Vector.zero(gs.qd_float, 4)
            bary[i_corner] = 1.0
            derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_local in range(10):
                i_n = stress_info.elements[i_e, i_local]
                gradient = func_shape_gradient(i_local, bary, stress_info.gradients[i_e], stress_info.edges)
                derivative += stress_state.displacement[i_n, i_b].outer_product(gradient)
            sigma = mu * (derivative + derivative.transpose())
            for i_a in qd.static(range(3)):
                sigma[i_a, i_a] += lam * derivative.trace()
            value = 0.5 * (
                (sigma[0, 0] - sigma[1, 1]) ** 2 + (sigma[1, 1] - sigma[2, 2]) ** 2 + (sigma[2, 2] - sigma[0, 0]) ** 2
            )
            value += 3.0 * (sigma[0, 1] ** 2 + sigma[0, 2] ** 2 + sigma[1, 2] ** 2)
            peak_squared = qd.max(peak_squared, value)
        qd.atomic_max(stress_state.peak[i_b], qd.sqrt(peak_squared))
    for i_b in range(stress_state.active.shape[0]):
        stress_state.step_peak[i_b] = qd.max(stress_state.step_peak[i_b], stress_state.peak[i_b])
