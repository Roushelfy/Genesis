"""Read the PR's unmodified SVK/hinge elastic energy and mechanical state."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class


@qd.kernel
def shell_energy(
    gravity_magnitude: float,
    output: qd.types.ndarray(ndim=2),
    state: array_class.ShellState,
    scratch: array_class.ShellScratch,
    info: array_class.ShellInfo,
):
    for i_b, k in qd.ndrange(output.shape[0], output.shape[1]):
        output[i_b, k] = 0.0
    for i_f, i_b in qd.ndrange(state.faces_thickness.shape[0], state.faces_thickness.shape[1]):
        i_e = info.faces_entity[i_f]
        F, _ = shell.func_face_deformation(i_f, i_b, state, info)
        G = 0.5 * (F.transpose() @ F - qd.Matrix.identity(gs.qd_float, 2))
        resultant = shell.func_membrane_stress(
            F, info.entities_stretching_modulus[i_e] * state.faces_thickness[i_f, i_b], info.entities_nu[i_e]
        )
        output[i_b, 0] += 0.5 * info.faces_rest_area[i_f] * (G * resultant).sum()
    for i_h, i_b in qd.ndrange(scratch.hinges_stiffness.shape[0], state.faces_thickness.shape[1]):
        i_va, i_vb, i_vc, i_vd, intact = shell.func_hinge_verts(i_h, i_b, state, info)
        if intact:
            edge_b, edge_c, edge_d = shell.func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, state)
            delta = (
                shell.func_hinge_angle(edge_b, edge_c, edge_d)
                - info.hinges_rest_angle[i_h]
                - state.hinges_plastic_angle[i_h, i_b]
            )
            stiffness = shell.func_hinge_stiffness(i_h, i_b, state, info)
            output[i_b, 0] += 0.5 * stiffness * delta * delta
    for i_v, i_b in qd.ndrange(state.verts_pos.shape[0], state.verts_pos.shape[1]):
        mass = scratch.verts_mass[i_v, i_b]
        output[i_b, 1] += 0.5 * mass * state.verts_vel[i_v, i_b].dot(state.verts_vel[i_v, i_b])
        output[i_b, 2] += mass * gravity_magnitude * state.verts_pos[i_v, i_b][2]
