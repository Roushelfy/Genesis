"""Fixed-reference quadrilateral shell operators and in-plane stress in Quadrants."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind

from .data import StressInfo, StressState


@dataclass(frozen=True)
class ShellInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    quads: qd.Tensor
    directors: qd.Tensor
    membrane: qd.Tensor
    bending: qd.Tensor
    shear: qd.Tensor
    drilling: qd.Tensor
    axes: qd.Tensor
    inverse: qd.Tensor
    derivative: qd.Tensor
    tangents: qd.Tensor
    interpolated_director: qd.Tensor
    weights: qd.Tensor
    nodal_mass: qd.Tensor
    is_valid: qd.Tensor


@qd.func
def func_shape(xi: float, eta: float):
    shape = qd.Vector.zero(gs.qd_float, 4)
    gradient = qd.Matrix.zero(gs.qd_float, 4, 2)
    for i in qd.static(range(4)):
        sx = qd.static(-1 if i in (0, 3) else 1)
        sy = qd.static(-1 if i < 2 else 1)
        shape[i] = 0.25 * (1.0 + sx * xi) * (1.0 + sy * eta)
        gradient[i, 0] = 0.25 * sx * (1.0 + sy * eta)
        gradient[i, 1] = 0.25 * sy * (1.0 + sx * xi)
    return shape, gradient


@qd.func
def func_geometry(i_e: int, xi: float, eta: float, stress_info: StressInfo, shell_info: ShellInfo):
    shape, gradient = func_shape(xi, eta)
    tangents = qd.Matrix.zero(gs.qd_float, 3, 2)
    director = qd.Vector.zero(gs.qd_float, 3)
    derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(4)):
        i_n = shell_info.quads[i_e, i]
        position = stress_info.vertices[2 * i_n]
        normal = shell_info.directors[i_n]
        director += shape[i] * normal
        for a, b in qd.static(qd.ndrange(3, 2)):
            tangents[a, b] += position[a] * gradient[i, b]
            derivative[a, b] += normal[a] * gradient[i, b]
    return tangents, director, derivative


@qd.kernel
def kernel_directors(stress_info: StressInfo, shell_info: ShellInfo, supplied: qd.template()):
    for i_n in range(shell_info.directors.shape[0]):
        director = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(supplied):
            director = shell_info.directors[i_n]
        else:
            for i_e in range(shell_info.quads.shape[0]):
                is_incident = False
                vertices = qd.Matrix.zero(gs.qd_float, 4, 3)
                for i in qd.static(range(4)):
                    j_n = shell_info.quads[i_e, i]
                    is_incident = is_incident or j_n == i_n
                    vertices[i, :] = stress_info.vertices[2 * j_n]
                if is_incident:
                    director += (vertices[1, :] - vertices[0, :]).cross(vertices[2, :] - vertices[0, :])
                    director += (vertices[2, :] - vertices[0, :]).cross(vertices[3, :] - vertices[0, :])
        if director.norm() > 1e-20:
            shell_info.directors[i_n] = director.normalized()
        else:
            shell_info.is_valid[None] = False


@qd.func
def func_gauss(i_q: int):
    xi = qd.select(i_q == 0 or i_q == 3, -1.0, 1.0) / qd.sqrt(3.0)
    eta = qd.select(i_q < 2, -1.0, 1.0) / qd.sqrt(3.0)
    return xi, eta


@qd.kernel
def kernel_geometry(stress_info: StressInfo, shell_info: ShellInfo):
    for i_e, i_q in qd.ndrange(shell_info.quads.shape[0], 4):
        xi, eta = func_gauss(i_q)
        tangents, director, derivative = func_geometry(i_e, xi, eta, stress_info, shell_info)
        normal = tangents[:, 0].cross(tangents[:, 1])
        weight = normal.norm()
        jacobian = qd.Matrix.cols([tangents[:, 0], tangents[:, 1], director])
        if weight > 1e-20 and jacobian.determinant() > 0.0:
            normal /= weight
            first = tangents[:, 0].normalized()
            shell_info.axes[i_e, i_q] = qd.Matrix.cols([first, normal.cross(first), normal])
            shell_info.inverse[i_e, i_q] = jacobian.inverse()
            shell_info.derivative[i_e, i_q] = derivative
            shell_info.tangents[i_e, i_q] = tangents
            shell_info.interpolated_director[i_e, i_q] = director
            shell_info.weights[i_e, i_q] = weight
        else:
            shell_info.is_valid[None] = False


@qd.func
def func_covariant_column(i_e: int, i_dof: int, xi: float, eta: float, stress_info: StressInfo, shell_info: ShellInfo):
    shape, gradient = func_shape(xi, eta)
    tangents, director, _derivative = func_geometry(i_e, xi, eta, stress_info, shell_info)
    i_local, component = i_dof // 6, i_dof % 6
    value = qd.Vector.zero(gs.qd_float, 2)
    if component < 3:
        value = director[component] * gradient[i_local, :]
    else:
        i_n = shell_info.quads[i_e, i_local]
        unit = qd.Vector.zero(gs.qd_float, 3)
        unit[component - 3] = 1.0
        delta = shape[i_local] * unit.cross(shell_info.directors[i_n])
        value = tangents.transpose() @ delta
    return value


@qd.kernel
def kernel_operators(stress_info: StressInfo, shell_info: ShellInfo):
    for i_e, i_q, i_dof in qd.ndrange(shell_info.quads.shape[0], 4, 24):
        xi, eta = func_gauss(i_q)
        shape, gradient = func_shape(xi, eta)
        i_local, component = i_dof // 6, i_dof % 6
        i_n = shell_info.quads[i_e, i_local]
        d0 = qd.Matrix.zero(gs.qd_float, 3, 3)
        d1 = qd.Matrix.zero(gs.qd_float, 3, 3)
        if component < 3:
            d0[component, 0] = gradient[i_local, 0]
            d0[component, 1] = gradient[i_local, 1]
        else:
            unit = qd.Vector.zero(gs.qd_float, 3)
            unit[component - 3] = 1.0
            delta = unit.cross(shell_info.directors[i_n])
            for a in qd.static(range(3)):
                d0[a, 2] = shape[i_local] * delta[a]
                d1[a, 0] = gradient[i_local, 0] * delta[a]
                d1[a, 1] = gradient[i_local, 1] * delta[a]
        inverse = shell_info.inverse[i_e, i_q]
        axes = shell_info.axes[i_e, i_q]
        g0 = d0 @ inverse
        g1 = (d1 - g0 @ shell_info.derivative[i_e, i_q]) @ inverse
        local0 = axes.transpose() @ g0 @ axes
        local1 = axes.transpose() @ g1 @ axes
        membrane = qd.Vector([local0[0, 0], local0[1, 1], local0[0, 1] + local0[1, 0]])
        bending = qd.Vector([local1[0, 0], local1[1, 1], local1[0, 1] + local1[1, 0]])
        drilling = -0.5 * (local0[1, 0] - local0[0, 1])
        if component >= 3:
            drilling += shape[i_local] * axes[component - 3, 2]
        covariant = qd.Vector.zero(gs.qd_float, 2)
        for i_edge in range(4):
            edge_xi = qd.select(i_edge < 2, 0.0, qd.select(i_edge == 2, -1.0, 1.0))
            edge_eta = qd.select(i_edge >= 2, 0.0, qd.select(i_edge == 0, -1.0, 1.0))
            edge = func_covariant_column(i_e, i_dof, edge_xi, edge_eta, stress_info, shell_info)
            if i_edge < 2:
                covariant[0] += 0.5 * (1.0 + edge_eta * eta) * edge[0]
            else:
                covariant[1] += 0.5 * (1.0 + edge_xi * xi) * edge[1]
        tangents = shell_info.tangents[i_e, i_q]
        director = shell_info.interpolated_director[i_e, i_q]
        local_director = axes.transpose() @ director
        jtan = qd.Matrix.zero(gs.qd_float, 2, 2)
        for a, b in qd.static(qd.ndrange(2, 2)):
            jtan[a, b] = tangents[:, a].dot(axes[:, b])
        eps_dt = qd.Vector(
            [
                membrane[0] * local_director[0] + 0.5 * membrane[2] * local_director[1],
                0.5 * membrane[2] * local_director[0] + membrane[1] * local_director[1],
            ]
        )
        shell_info.membrane[i_e, i_q, i_dof] = membrane
        shell_info.bending[i_e, i_q, i_dof] = bending
        shell_info.shear[i_e, i_q, i_dof] = (jtan.inverse() @ covariant - 2.0 * eps_dt) / local_director[2]
        shell_info.drilling[i_e, i_q, i_dof] = drilling


@qd.func
def func_constitutive(poisson: float):
    return qd.Matrix([[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]) / (
        1.0 - poisson * poisson
    )


@qd.kernel
def kernel_assemble(thickness: float, poisson: float, drilling: float, stress_info: StressInfo, shell_info: ShellInfo):
    for i_entry in range(stress_info.stiffness.shape[0]):
        stiffness = qd.Matrix.zero(gs.qd_float, 3, 3)
        constitutive = func_constitutive(poisson)
        shear_modulus = 0.5 / (1.0 + poisson)
        for i_slot in range(stress_info.assembly_start[i_entry], stress_info.assembly_start[i_entry + 1]):
            i_con = stress_info.assembly_idx[i_slot]
            i_e, i_local, j_local = i_con // 64, (i_con % 64) // 8, i_con % 8
            for i_q in range(4):
                for a, b in qd.static(qd.ndrange(3, 3)):
                    i_dof, j_dof = 3 * i_local + a, 3 * j_local + b
                    value = thickness * shell_info.membrane[i_e, i_q, i_dof].dot(
                        constitutive @ shell_info.membrane[i_e, i_q, j_dof]
                    )
                    value += (
                        thickness**3
                        / 12.0
                        * shell_info.bending[i_e, i_q, i_dof].dot(constitutive @ shell_info.bending[i_e, i_q, j_dof])
                    )
                    value += (
                        5.0
                        / 6.0
                        * shear_modulus
                        * thickness
                        * shell_info.shear[i_e, i_q, i_dof].dot(shell_info.shear[i_e, i_q, j_dof])
                    )
                    value += (
                        drilling
                        * shear_modulus
                        * thickness
                        * shell_info.drilling[i_e, i_q, i_dof]
                        * (shell_info.drilling[i_e, i_q, j_dof])
                    )
                    stiffness[a, b] += shell_info.weights[i_e, i_q] * value
        stress_info.stiffness[i_entry] = stiffness


@qd.kernel
def kernel_mass(thickness: float, density: float, stress_info: StressInfo, shell_info: ShellInfo):
    for i_n in range(shell_info.directors.shape[0]):
        mass = gs.qd_float(0.0)
        for i_e in range(shell_info.quads.shape[0]):
            for i in range(4):
                if shell_info.quads[i_e, i] == i_n:
                    for i_q in range(4):
                        xi, eta = func_gauss(i_q)
                        shape, _gradient = func_shape(xi, eta)
                        mass += density * thickness * shell_info.weights[i_e, i_q] * shape[i]
        shell_info.nodal_mass[i_n] = mass
    for a in range(4):
        total = gs.qd_float(0.0)
        for i_n in range(shell_info.directors.shape[0]):
            mass = shell_info.nodal_mass[i_n]
            if a > 0:
                mass *= stress_info.vertices[2 * i_n][a - 1]
            total += mass
        stress_info.mass_properties[a] = total


@qd.kernel
def kernel_modes(thickness: float, stress_info: StressInfo, shell_info: ShellInfo):
    for i_n in range(shell_info.directors.shape[0]):
        com = qd.Vector([stress_info.mass_properties[a + 1] for a in qd.static(range(3))])
        relative = stress_info.vertices[2 * i_n] - com / stress_info.mass_properties[0]
        modes = qd.Matrix.zero(gs.qd_float, 3, 6)
        rotations = qd.Matrix.zero(gs.qd_float, 3, 6)
        centrifugal = qd.Matrix.zero(gs.qd_float, 3, 6)
        rotary_centrifugal = qd.Matrix.zero(gs.qd_float, 3, 6)
        director = shell_info.directors[i_n]
        mass = shell_info.nodal_mass[i_n]
        rotary_mass = (
            mass * thickness**2 / 12.0 * (qd.Matrix.identity(gs.qd_float, 3) - director.outer_product(director))
        )
        for a in qd.static(range(3)):
            unit = qd.Vector.zero(gs.qd_float, 3)
            unit[a] = 1.0
            modes[a, a] = 1.0
            rotations[a, a + 3] = 1.0
            modes[:, a + 3] = unit.cross(relative)
            centrifugal[:, a] = mass * (relative - relative[a] * unit)
            rotary_centrifugal[:, a] = mass * thickness**2 / 12.0 * director[a] * unit.cross(director)
        for a, b, c in qd.static(((0, 1, 3), (0, 2, 4), (1, 2, 5))):
            centrifugal[a, c] = -mass * relative[b]
            centrifugal[b, c] = -mass * relative[a]
            unit = qd.Vector.zero(gs.qd_float, 3)
            unit[a], unit[b] = director[b], director[a]
            rotary_centrifugal[:, c] = mass * thickness**2 / 12.0 * unit.cross(director)
        stress_info.modes[2 * i_n] = modes
        stress_info.modes[2 * i_n + 1] = rotations
        stress_info.mass_modes[2 * i_n] = mass * modes
        stress_info.mass_modes[2 * i_n + 1] = rotary_mass @ rotations
        stress_info.centrifugal[2 * i_n] = centrifugal
        stress_info.centrifugal[2 * i_n + 1] = rotary_centrifugal
    for a, b in qd.ndrange(6, 6):
        total = gs.qd_float(0.0)
        for i_n in range(stress_info.vertices.shape[0]):
            total += stress_info.modes[i_n][:, a].dot(stress_info.mass_modes[i_n][:, b])
        stress_info.gram[None][a, b] = total


@qd.kernel
def kernel_gauge(stress_info: StressInfo, shell_info: ShellInfo, translation_only: qd.template()):
    for _ in range(1):
        length_squared = gs.qd_float(0.0)
        com = qd.Vector([stress_info.mass_properties[a + 1] for a in qd.static(range(3))])
        com /= stress_info.mass_properties[0]
        for i_n in range(shell_info.directors.shape[0]):
            length_squared += shell_info.nodal_mass[i_n] * (stress_info.vertices[2 * i_n] - com).norm_sqr()
        length = qd.sqrt(length_squared / stress_info.mass_properties[0])
        basis = qd.Matrix.zero(gs.qd_float, 6, 6)
        for i_pin in range(6):
            best_norm = gs.qd_float(-1.0)
            best_row = 0
            best_vector = qd.Vector.zero(gs.qd_float, 6)
            for i_n in range(stress_info.vertices.shape[0]):
                is_eligible = True
                if qd.static(translation_only):
                    is_eligible = i_n % 2 == 0
                if is_eligible:
                    for a in range(3):
                        vector = stress_info.modes[i_n][a, :]
                        if qd.static(translation_only):
                            for b in range(3, 6):
                                vector[b] /= length
                        for j in range(i_pin):
                            vector -= basis[j, :] * basis[j, :].dot(vector)
                        norm = vector.norm_sqr()
                        if norm > best_norm:
                            best_norm, best_row, best_vector = norm, 3 * i_n + a, vector
            if best_norm > 1e-24:
                stress_info.pins[i_pin] = best_row
                basis[i_pin, :] = best_vector / qd.sqrt(best_norm)
                stress_info.is_free[best_row // 3][best_row % 3] = 0
            else:
                shell_info.is_valid[None] = False


@qd.kernel
def kernel_peak(young: float, poisson: float, thickness: float, stress_state: StressState, shell_info: ShellInfo):
    for i_b in range(stress_state.peak.shape[0]):
        stress_state.peak[i_b] = 0.0
    for i_e, i_q, i_side, i_b in qd.ndrange(shell_info.quads.shape[0], 4, 2, stress_state.peak.shape[0]):
        strain = qd.Vector.zero(gs.qd_float, 3)
        z = qd.select(i_side == 0, 0.5, -0.5) * thickness
        for i_local in range(24):
            i_n = shell_info.quads[i_e, i_local // 6]
            i_block = 2 * i_n + (i_local % 6) // 3
            value = stress_state.displacement[i_block, i_b][i_local % 3]
            strain += value * (shell_info.membrane[i_e, i_q, i_local] + z * shell_info.bending[i_e, i_q, i_local])
        local = young * func_constitutive(poisson) @ strain
        vm = qd.sqrt(qd.max(0.0, local[0] ** 2 + local[1] ** 2 - local[0] * local[1] + 3.0 * local[2] ** 2))
        qd.atomic_max(stress_state.peak[i_b], vm)
        if qd.static(stress_state.stress_tensor.shape[0] > 0):
            axes = shell_info.axes[i_e, i_q]
            first, second = axes[:, 0], axes[:, 1]
            tensor = local[0] * first.outer_product(first) + local[1] * second.outer_product(second)
            tensor += local[2] * (first.outer_product(second) + second.outer_product(first))
            stress_state.stress_tensor[2 * i_e + i_side, i_q, i_b] = qd.Vector(
                [
                    tensor[0, 0],
                    tensor[1, 1],
                    tensor[2, 2],
                    tensor[0, 1],
                    tensor[0, 2],
                    tensor[1, 2],
                ]
            )
            stress_state.von_mises[2 * i_e + i_side, i_q, i_b] = qd.Vector([vm])
            if not stress_state.valid[i_b]:
                stress_state.stress_tensor[2 * i_e + i_side, i_q, i_b] = qd.Vector([float("nan")] * 6)
                stress_state.von_mises[2 * i_e + i_side, i_q, i_b] = qd.Vector([float("nan")])
    for i_b in range(stress_state.peak.shape[0]):
        stress_state.step_peak[i_b] = qd.max(stress_state.step_peak[i_b], stress_state.peak[i_b])
