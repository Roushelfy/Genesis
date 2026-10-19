"""Record the frozen first contact system from the explicitly truncated Shell diagnostic pin."""

import numpy as np

from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, write_json


def capture(solver, rigid_solver, output):
    state, scratch, info = solver.shell_state, solver._shell_scratch, solver.shell_info
    contact = solver._shell_contact
    corners = qd_to_numpy(state.corners_vert, transpose=True)[0]
    hinge_corner = qd_to_numpy(info.hinges_corner, transpose=True)
    opposite = qd_to_numpy(info.hinges_opposite_corner, transpose=True)
    hinge_vertices = corners[np.column_stack((hinge_corner[:, :2], opposite))]
    assert (corners[hinge_corner[:, :2]] == corners[hinge_corner[:, 2:]]).all()
    assert not qd_to_numpy(state.verts_is_fixed, transpose=True).any()
    assert (qd_to_numpy(state.verts_origin, transpose=True) >= 0).all()
    geom = qd_to_numpy(contact.contacts_geom, transpose=True)[0]
    slots = np.flatnonzero(geom >= 0)
    links = qd_to_numpy(rigid_solver.dyn_info.geoms.link_idx, transpose=True)[geom[slots]]
    trees = qd_to_numpy(rigid_solver.rigid_info.links_tree_idx, transpose=True)[links]
    starts = qd_to_numpy(rigid_solver.rigid_info.trees_dof_start, transpose=True)
    counts = qd_to_numpy(rigid_solver.rigid_info.trees_n_dofs, transpose=True)
    jacobian = qd_to_numpy(contact.contacts_jac, transpose=True)[0, slots]
    inverse = qd_to_numpy(contact.dofs_schur_inv, transpose=True)[0]
    global_jacobian = np.zeros((len(slots), len(inverse), 3))
    for index, tree in enumerate(trees):
        if tree >= 0:
            global_jacobian[index, starts[tree] : starts[tree] + counts[tree]] = jacobian[index, : counts[tree]]
    entity = qd_to_numpy(info.faces_entity, transpose=True)
    path = output / "frozen-linear.npz"
    np.savez_compressed(
        path,
        faces=corners.reshape(-1, 3),
        mass=qd_to_numpy(scratch.verts_mass, transpose=True)[0],
        rhs=qd_to_numpy(scratch.verts_rhs, transpose=True)[0],
        solution=qd_to_numpy(state.verts_dv, transpose=True)[0],
        recursive_residual=qd_to_numpy(scratch.verts_r, transpose=True)[0],
        block_inverse=qd_to_numpy(scratch.verts_prec, transpose=True)[0],
        face_f=qd_to_numpy(scratch.faces_F, transpose=True)[0],
        face_stress=qd_to_numpy(scratch.faces_stress, transpose=True)[0],
        face_y=qd_to_numpy(info.faces_Dm_inv, transpose=True) @ qd_to_numpy(state.faces_plastic, transpose=True)[0],
        face_scale=qd_to_numpy(scratch.faces_stiffness, transpose=True)[0],
        face_modulus=qd_to_numpy(info.entities_stretching_modulus, transpose=True)[entity]
        * qd_to_numpy(state.faces_thickness, transpose=True)[0],
        face_nu=qd_to_numpy(info.entities_nu, transpose=True)[entity],
        hinge_vertices=hinge_vertices,
        hinge_gradient=qd_to_numpy(scratch.hinges_grad, transpose=True)[0],
        hinge_scale=qd_to_numpy(scratch.hinges_stiffness, transpose=True)[0],
        contact_faces=slots // 2,
        contact_bary=qd_to_numpy(contact.contacts_bary, transpose=True)[0, slots],
        contact_hessian=qd_to_numpy(contact.contacts_hessian, transpose=True)[0, slots],
        contact_jacobian=global_jacobian,
        schur_inverse=inverse,
    )
    write_json(
        output / "frozen-linear.json",
        {
            "field": file_id(path),
            "contact_newton_iterations": qd_to_numpy(contact.envs_n_newton_iterations, transpose=True).tolist(),
            "linear_iterations": qd_to_numpy(scratch.envs_n_solve_iterations, transpose=True).tolist(),
            "linear_mode": qd_to_numpy(scratch.envs_pcg_mode, transpose=True).tolist(),
            "status": qd_to_numpy(scratch.envs_solve_status, transpose=True).tolist(),
            "relative_threshold": qd_to_numpy(scratch.envs_residual_threshold, transpose=True).tolist(),
            "velocity_threshold": qd_to_numpy(scratch.envs_vel_error_threshold, transpose=True).tolist(),
            "last_checked_residual": qd_to_numpy(scratch.envs_residual_checked, transpose=True).tolist(),
            "scope": "Diagnostic first Newton linear system. The source pin stops before contact line search; "
            "its returned physical state is not an admitted trajectory. Cached tangents, RHS and PCG iterate are retained.",
        },
    )


def contact_diagnostics(solver, rigid_solver, dt, velocity_tolerance, contact_stiffness):
    """Read contact velocity consistency and express its stopping budget in force units."""
    state, contact = solver.shell_state, solver._shell_contact
    geom = qd_to_numpy(contact.contacts_geom, transpose=True)[0]
    slots = np.flatnonzero(geom >= 0)
    corners = qd_to_numpy(state.corners_vert, transpose=True)[0].reshape(-1, 3)
    bary = qd_to_numpy(contact.contacts_bary, transpose=True)[0, slots]
    shell_change = np.einsum("fa,fai->fi", bary, qd_to_numpy(state.verts_dv, transpose=True)[0, corners[slots // 2]])
    links = qd_to_numpy(rigid_solver.dyn_info.geoms.link_idx, transpose=True)[geom[slots]]
    trees = qd_to_numpy(rigid_solver.rigid_info.links_tree_idx, transpose=True)[links]
    starts = qd_to_numpy(rigid_solver.rigid_info.trees_dof_start, transpose=True)
    counts = qd_to_numpy(rigid_solver.rigid_info.trees_n_dofs, transpose=True)
    jacobian = qd_to_numpy(contact.contacts_jac, transpose=True)[0, slots]
    rigid_change = qd_to_numpy(contact.dofs_dv, transpose=True)[0]
    reconstructed = shell_change.copy()
    for index, tree in enumerate(trees):
        if tree >= 0:
            reconstructed[index] -= (
                rigid_change[starts[tree] : starts[tree] + counts[tree]] @ jacobian[index, : counts[tree]]
            )
    cached = qd_to_numpy(contact.contacts_vel_change, transpose=True)[0, slots]
    stiffness = qd_to_numpy(contact.contacts_stiffness, transpose=True)[0, slots]
    force_budget = velocity_tolerance * stiffness / (contact_stiffness * dt)
    return {
        "contact_slots": len(slots),
        "newton_iterations": qd_to_numpy(contact.envs_n_newton_iterations, transpose=True).tolist(),
        "contact_velocity_change_disagreement_max_m_per_s": float(
            np.linalg.norm(cached - reconstructed, axis=1).max(initial=0)
        ),
        "maximum_per_contact_model_mismatch_budget_n": float(force_budget.max(initial=0)),
        "sum_contact_model_mismatch_budgets_n": float(force_budget.sum()),
        "scope": "Read-only identity J_shell delta_v_shell - J_rigid delta_v_rigid versus cached contact change. "
        "The force budgets convert the existing local nonlinear stopping tolerance, not measured residuals.",
    }
