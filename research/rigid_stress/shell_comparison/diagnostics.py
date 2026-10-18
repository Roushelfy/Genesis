"""Read-only substep instrumentation for a separately labeled PR22 worktree.

Copies and a product into an owned buffer leave original state and solver scratch
unchanged. These intrusive diagnostics are excluded from normal timing runs.
"""

from dataclasses import dataclass

import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class


@qd.kernel
def true_system_product(
    src: qd.Tensor,
    dst: qd.Tensor,
    state: array_class.ShellState,
    scratch: array_class.ShellScratch,
    info: array_class.ShellInfo,
):
    # The original product skips converged environments; this diagnostic evaluates
    # the identical assembled operator on all of them, into a separate buffer.
    for i_v, i_b in qd.ndrange(state.verts_pos.shape[0], state.verts_pos.shape[1]):
        dst[i_v, i_b] = scratch.verts_mass[i_v, i_b] * src[i_v, i_b]
    for i_f, i_b in qd.ndrange(state.faces_thickness.shape[0], state.faces_thickness.shape[1]):
        i_e = info.faces_entity[i_f]
        i_v0 = state.corners_vert[3 * i_f, i_b]
        i_v1 = state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = state.corners_vert[3 * i_f + 2, i_b]
        Y = info.faces_Dm_inv[i_f] @ state.faces_plastic[i_f, i_b]
        k0, k1, k2 = shell.func_membrane_stiffness_product(
            src[i_v0, i_b],
            src[i_v1, i_b],
            src[i_v2, i_b],
            scratch.faces_F[i_f, i_b],
            scratch.faces_stress[i_f, i_b],
            Y,
            info.entities_stretching_modulus[i_e] * state.faces_thickness[i_f, i_b],
            info.entities_nu[i_e],
        )
        stiffness = scratch.faces_stiffness[i_f, i_b]
        dst[i_v0, i_b] += stiffness * k0
        dst[i_v1, i_b] += stiffness * k1
        dst[i_v2, i_b] += stiffness * k2
    for i_h, i_b in qd.ndrange(scratch.hinges_stiffness.shape[0], state.faces_thickness.shape[1]):
        stiffness = scratch.hinges_stiffness[i_h, i_b]
        if stiffness > 0.0:
            i_va, i_vb, i_vc, i_vd, _ = shell.func_hinge_verts(i_h, i_b, state, info)
            gradient = scratch.hinges_grad[i_h, i_b]
            coeff = stiffness * (
                gradient[:, 0].dot(src[i_va, i_b])
                + gradient[:, 1].dot(src[i_vb, i_b])
                + gradient[:, 2].dot(src[i_vc, i_b])
                + gradient[:, 3].dot(src[i_vd, i_b])
            )
            dst[i_va, i_b] += coeff * gradient[:, 0]
            dst[i_vb, i_b] += coeff * gradient[:, 1]
            dst[i_vc, i_b] += coeff * gradient[:, 2]
            dst[i_vd, i_b] += coeff * gradient[:, 3]
    for i_v, i_b in qd.ndrange(state.verts_pos.shape[0], state.verts_pos.shape[1]):
        if not shell.func_is_vert_free(i_v, i_b, state):
            dst[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def begin_step(
    counts: qd.types.ndarray(ndim=1),
    caps: qd.types.ndarray(ndim=1),
    residuals: qd.types.ndarray(ndim=2),
    impulses: qd.types.ndarray(ndim=3),
):
    for i_b in range(counts.shape[0]):
        counts[i_b] = 0
        caps[i_b] = 0
        for k in range(residuals.shape[1]):
            residuals[i_b, k] = 0.0
    for i_v, i_b, k in qd.ndrange(impulses.shape[0], impulses.shape[1], 3):
        impulses[i_v, i_b, k] = 0.0


@qd.kernel
def count_iterations(counts: qd.types.ndarray(ndim=1), scratch: array_class.ShellScratch):
    for i_b in range(counts.shape[0]):
        if scratch.envs_is_solving[i_b]:
            counts[i_b] += 1


@qd.kernel
def residual_sums(
    true_product: qd.Tensor,
    sums: qd.types.ndarray(ndim=2),
    caps: qd.types.ndarray(ndim=1),
    scratch: array_class.ShellScratch,
):
    for i_b, k in qd.ndrange(sums.shape[0], sums.shape[1]):
        sums[i_b, k] = 0.0
    for i_v, i_b in qd.ndrange(scratch.verts_rhs.shape[0], scratch.verts_rhs.shape[1]):
        r = scratch.verts_rhs[i_v, i_b] - true_product[i_v, i_b]
        rhs = scratch.verts_rhs[i_v, i_b]
        sums[i_b, 0] += r.dot(r)
        sums[i_b, 1] += rhs.dot(rhs)
        sums[i_b, 2] += r.dot(scratch.verts_prec[i_v, i_b] @ r)
        sums[i_b, 3] += rhs.dot(scratch.verts_prec[i_v, i_b] @ rhs)
    for i_b in range(caps.shape[0]):
        if scratch.envs_is_solving[i_b]:
            caps[i_b] += 1


@qd.kernel
def summarize_residuals(
    sums: qd.types.ndarray(ndim=2),
    residuals: qd.types.ndarray(ndim=2),
    scratch: array_class.ShellScratch,
):
    for i_b in range(residuals.shape[0]):
        l2 = qd.sqrt(sums[i_b, 0] / qd.max(sums[i_b, 1], 1e-30))
        jacobi = qd.sqrt(sums[i_b, 2] / qd.max(sums[i_b, 3], 1e-30))
        recursive_budget = scratch.envs_rz[i_b] / qd.max(scratch.envs_rz_threshold[i_b], 1e-30)
        qd.atomic_max(residuals[i_b, 0], l2)
        qd.atomic_max(residuals[i_b, 1], jacobi)
        qd.atomic_max(residuals[i_b, 2], recursive_budget)


@qd.kernel
def copy_contact_velocity(
    velocities: qd.types.ndarray(ndim=3),
    state: array_class.ShellState,
):
    for i_v, i_b, k in qd.ndrange(velocities.shape[0], velocities.shape[1], 3):
        velocities[i_v, i_b, k] = state.verts_vel[i_v, i_b][k]


@qd.kernel
def accumulate_contact_impulses(
    velocities: qd.types.ndarray(ndim=3),
    impulses: qd.types.ndarray(ndim=3),
    points: qd.types.ndarray(ndim=3),
    state: array_class.ShellState,
    scratch: array_class.ShellScratch,
):
    for i_v, i_b, k in qd.ndrange(velocities.shape[0], velocities.shape[1], 3):
        impulses[i_v, i_b, k] += scratch.verts_mass[i_v, i_b] * (state.verts_vel[i_v, i_b][k] - velocities[i_v, i_b, k])
        points[i_v, i_b, k] = state.verts_pos[i_v, i_b][k]


@dataclass
class DiagnosticBuffers:
    counts: torch.Tensor
    caps: torch.Tensor
    residuals: torch.Tensor
    sums: torch.Tensor
    velocities: torch.Tensor
    impulses: torch.Tensor
    points: torch.Tensor
    true_product: object

    @classmethod
    def allocate(cls, solver):
        n_verts, n_envs = solver.shell_state.verts_pos.shape
        return cls(
            torch.zeros(n_envs, dtype=gs.tc_int, device=gs.device),
            torch.zeros(n_envs, dtype=gs.tc_int, device=gs.device),
            torch.zeros((n_envs, 3), dtype=gs.tc_float, device=gs.device),
            torch.empty((n_envs, 4), dtype=gs.tc_float, device=gs.device),
            torch.empty((n_verts, n_envs, 3), dtype=gs.tc_float, device=gs.device),
            torch.zeros((n_verts, n_envs, 3), dtype=gs.tc_float, device=gs.device),
            torch.empty((n_verts, n_envs, 3), dtype=gs.tc_float, device=gs.device),
            array_class.V(dtype=gs.qd_vec3, shape=(n_verts, n_envs)),
        )

    def clear(self):
        begin_step(self.counts, self.caps, self.residuals, self.impulses)


ACTIVE: DiagnosticBuffers | None = None


def configure(solver):
    global ACTIVE
    ACTIVE = DiagnosticBuffers.allocate(solver)


def iteration(solver):
    if ACTIVE is not None:
        count_iterations(ACTIVE.counts, solver.shell_scratch)


def after_solve(solver):
    if ACTIVE is not None:
        true_system_product(
            solver.shell_scratch.verts_dv,
            ACTIVE.true_product,
            solver.shell_state,
            solver.shell_scratch,
            solver.shell_info,
        )
        residual_sums(ACTIVE.true_product, ACTIVE.sums, ACTIVE.caps, solver.shell_scratch)
        summarize_residuals(ACTIVE.sums, ACTIVE.residuals, solver.shell_scratch)


def before_contact(solver):
    if ACTIVE is not None:
        copy_contact_velocity(ACTIVE.velocities, solver.shell_state)


def after_contact(solver):
    if ACTIVE is not None:
        accumulate_contact_impulses(
            ACTIVE.velocities, ACTIVE.impulses, ACTIVE.points, solver.shell_state, solver.shell_scratch
        )
