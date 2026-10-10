"""Assemble shared affine-P2 stiffness and consistent mass in Quadrants."""

import quadrants as qd

import genesis as gs

from .data import StressInfo


@qd.func
def func_shape_gradient(i_n: int, bary: qd.types.vector(4), gradient: qd.types.matrix(4, 3), edges: qd.Tensor):
    value = qd.Vector.zero(gs.qd_float, 3)
    if i_n < 4:
        value = (4.0 * bary[i_n] - 1.0) * gradient[i_n, :]
    else:
        i_a = edges[i_n - 4, 0]
        i_c = edges[i_n - 4, 1]
        value = 4.0 * (bary[i_a] * gradient[i_c, :] + bary[i_c] * gradient[i_a, :])
    return value


@qd.func
def func_mass_weight(i_n: int, j_n: int, edges: qd.Tensor):
    value = gs.qd_float(0.0)
    if i_n < 4 and j_n < 4:
        value = 1.0 / 420.0
        if i_n == j_n:
            value = 1.0 / 70.0
    elif i_n >= 4 and j_n >= 4:
        i_a, i_c = edges[i_n - 4, 0], edges[i_n - 4, 1]
        j_a, j_c = edges[j_n - 4, 0], edges[j_n - 4, 1]
        value = 2.0 / 105.0
        if i_n == j_n:
            value = 8.0 / 105.0
        elif i_a == j_a or i_a == j_c or i_c == j_a or i_c == j_c:
            value = 4.0 / 105.0
    else:
        i_v, i_e = i_n, j_n - 4
        if j_n < 4:
            i_v, i_e = j_n, i_n - 4
        i_a, i_c = edges[i_e, 0], edges[i_e, 1]
        value = -1.0 / 70.0
        if i_v == i_a or i_v == i_c:
            value = -1.0 / 105.0
    return value


@qd.kernel
def kernel_geometry(stress_info: StressInfo):
    for i_e in range(stress_info.elements.shape[0]):
        i_0 = stress_info.elements[i_e, 0]
        i_1 = stress_info.elements[i_e, 1]
        i_2 = stress_info.elements[i_e, 2]
        i_3 = stress_info.elements[i_e, 3]
        x_0 = stress_info.vertices[i_0]
        edges = qd.Matrix.cols(
            [stress_info.vertices[i_1] - x_0, stress_info.vertices[i_2] - x_0, stress_info.vertices[i_3] - x_0]
        )
        inverse = edges.inverse()
        gradient = qd.Matrix.zero(gs.qd_float, 4, 3)
        for i_a, i_c in qd.static(qd.ndrange(3, 3)):
            gradient[i_a + 1, i_c] = inverse[i_a, i_c]
            gradient[0, i_c] -= inverse[i_a, i_c]
        stress_info.gradients[i_e] = gradient
        stress_info.volumes[i_e] = qd.abs(edges.determinant()) / 6.0


@qd.kernel
def kernel_assemble(poisson: float, density: float, stress_info: StressInfo):
    for i_e, i_n, j_n in qd.ndrange(stress_info.elements.shape[0], 10, 10):
        gradient = stress_info.gradients[i_e]
        block = qd.Matrix.zero(gs.qd_float, 3, 3)
        lam = poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        mu = 1.0 / (2.0 * (1.0 + poisson))
        for i_q in range(4):
            bary = qd.Vector([0.1381966011250105] * 4)
            bary[i_q] = 0.5854101966249685
            grad_i = func_shape_gradient(i_n, bary, gradient, stress_info.edges)
            grad_j = func_shape_gradient(j_n, bary, gradient, stress_info.edges)
            block += lam * grad_i.outer_product(grad_j) + mu * grad_j.outer_product(grad_i)
            for i_a in qd.static(range(3)):
                block[i_a, i_a] += mu * grad_i.dot(grad_j)
        i_entry = stress_info.element_entries[i_e, i_n, j_n]
        scale = stress_info.volumes[i_e] / 4.0
        for i_a, i_c in qd.static(qd.ndrange(3, 3)):
            qd.atomic_add(stress_info.stiffness[i_entry][i_a, i_c], scale * block[i_a, i_c])
        qd.atomic_add(
            stress_info.mass[i_entry],
            density * stress_info.volumes[i_e] * func_mass_weight(i_n, j_n, stress_info.edges),
        )


@qd.kernel
def kernel_mass_properties(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        mass = gs.qd_float(0.0)
        for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
            mass += stress_info.mass[i_entry]
        qd.atomic_add(stress_info.mass_properties[0], mass)
        for i_a in qd.static(range(3)):
            qd.atomic_add(stress_info.mass_properties[1 + i_a], mass * stress_info.vertices[i_n][i_a])


@qd.kernel
def kernel_modes(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        com = qd.Vector([stress_info.mass_properties[1 + i_a] for i_a in qd.static(range(3))])
        relative = stress_info.vertices[i_n] - com / stress_info.mass_properties[0]
        modes = qd.Matrix.zero(gs.qd_float, 3, 6)
        for i_a in qd.static(range(3)):
            modes[i_a, i_a] = 1.0
        modes[0, 4], modes[0, 5] = relative[2], -relative[1]
        modes[1, 3], modes[1, 5] = -relative[2], relative[0]
        modes[2, 3], modes[2, 4] = relative[1], -relative[0]
        stress_info.modes[i_n] = modes


@qd.kernel
def kernel_mass_modes(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        mass_modes = qd.Matrix.zero(gs.qd_float, 3, 6)
        centrifugal = qd.Matrix.zero(gs.qd_float, 3, 6)
        for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
            j_n = stress_info.columns[i_entry]
            mass = stress_info.mass[i_entry]
            mass_modes += mass * stress_info.modes[j_n]
            com = qd.Vector([stress_info.mass_properties[1 + i_a] for i_a in qd.static(range(3))])
            x = stress_info.vertices[j_n] - com / stress_info.mass_properties[0]
            basis = qd.Matrix(
                [
                    [0.0, x[0], x[0], -x[1], -x[2], 0.0],
                    [x[1], 0.0, x[1], -x[0], 0.0, -x[2]],
                    [x[2], x[2], 0.0, 0.0, -x[0], -x[1]],
                ]
            )
            centrifugal += mass * basis
        stress_info.mass_modes[i_n] = mass_modes
        stress_info.centrifugal[i_n] = centrifugal
        gram = stress_info.modes[i_n].transpose() @ mass_modes
        for i_a, i_c in qd.static(qd.ndrange(6, 6)):
            qd.atomic_add(stress_info.gram[None][i_a, i_c], gram[i_a, i_c])


@qd.kernel
def kernel_gram_inverse(stress_info: StressInfo):
    for _ in range(1):
        lower = qd.Matrix.zero(gs.qd_float, 6, 6)
        gram = stress_info.gram[None]
        for i in range(6):
            for j in range(i + 1):
                value = gram[i, j]
                for k in range(j):
                    value -= lower[i, k] * lower[j, k]
                if i == j:
                    lower[i, j] = qd.sqrt(value)
                else:
                    lower[i, j] = value / lower[j, j]
        inverse = qd.Matrix.zero(gs.qd_float, 6, 6)
        for j in range(6):
            vector = qd.Vector.zero(gs.qd_float, 6)
            for i in range(6):
                value = gs.qd_float(i == j)
                for k in range(i):
                    value -= lower[i, k] * vector[k]
                vector[i] = value / lower[i, i]
            for i_ in range(6):
                i = 5 - i_
                value = vector[i]
                for k in range(i + 1, 6):
                    value -= lower[k, i] * vector[k]
                vector[i] = value / lower[i, i]
            for i in range(6):
                inverse[i, j] = vector[i]
        stress_info.gram_inverse[None] = inverse


@qd.kernel
def kernel_gauge(stress_info: StressInfo):
    for _ in range(1):
        basis = qd.Matrix.zero(gs.qd_float, 6, 6)
        for i_pin in range(6):
            best_norm = gs.qd_float(-1.0)
            best_row = 0
            best_vector = qd.Vector.zero(gs.qd_float, 6)
            for i_n in range(stress_info.vertices.shape[0]):
                for i_a in range(3):
                    vector = stress_info.modes[i_n][i_a, :]
                    # Metres and radians need comparable scales for rank selection.
                    for i_c in range(3, 6):
                        vector[i_c] *= 100.0
                    for j in range(i_pin):
                        vector -= basis[j, :] * basis[j, :].dot(vector)
                    norm = vector.dot(vector)
                    if norm > best_norm:
                        best_norm, best_row, best_vector = norm, 3 * i_n + i_a, vector
            stress_info.pins[i_pin] = best_row
            basis[i_pin, :] = best_vector / qd.sqrt(best_norm)
            stress_info.is_free[best_row // 3][best_row % 3] = 0


@qd.kernel
def kernel_diagonal(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        diagonal = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_entry in range(stress_info.row_start[i_n], stress_info.row_start[i_n + 1]):
            if stress_info.columns[i_entry] == i_n:
                diagonal = stress_info.stiffness[i_entry]
        for i_a, i_c in qd.static(qd.ndrange(3, 3)):
            if stress_info.is_free[i_n][i_a] == 0 or stress_info.is_free[i_n][i_c] == 0:
                diagonal[i_a, i_c] = 0.0
        for i_a in qd.static(range(3)):
            if stress_info.is_free[i_n][i_a] == 0:
                diagonal[i_a, i_a] = 1.0
        stress_info.diagonal_inverse[i_n] = diagonal.inverse()
