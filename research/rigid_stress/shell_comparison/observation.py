"""Read-only full-field recovery for Kashu PR22, separate from its clipped stiffness/fracture caches.

The membrane resultant is second Piola--Kirchhoff, N/m. Signed hinge curvature is
the PR's face curvature estimator before its absolute-eigenvalue fracture transform.
Linear through-thickness fibers give S(z) = S_mem/h + z C:K (Pa). Their push-forward
is F S(z) F.T / J_area with the PR's unchanged thickness. This is a thin-shell
diagnostic, not transverse stress or a finite-thickness constitutive integration.
"""

from dataclasses import dataclass

import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class


@qd.kernel
def recover_fields(
    tensors: qd.types.ndarray(ndim=4),
    von_mises: qd.types.ndarray(ndim=2),
    membrane: qd.types.ndarray(ndim=4),
    curvature_output: qd.types.ndarray(ndim=4),
    bending: qd.types.ndarray(ndim=4),
    peaks: qd.types.ndarray(ndim=1),
    state: array_class.ShellState,
    info: array_class.ShellInfo,
    full_output: qd.template(),
    reset_peak: qd.template(),
):
    if qd.static(reset_peak):
        for i_b in range(peaks.shape[0]):
            peaks[i_b] = 0.0
    for i_f, i_b in qd.ndrange(state.faces_thickness.shape[0], state.faces_thickness.shape[1]):
        i_e = info.faces_entity[i_f]
        thickness = state.faces_thickness[i_f, i_b]
        F, _ = shell.func_face_deformation(i_f, i_b, state, info)
        stress = shell.func_membrane_stress(F, info.entities_stretching_modulus[i_e] * thickness, info.entities_nu[i_e])
        curvature = qd.Matrix.zero(gs.qd_float, 2, 2)
        Dm = info.faces_Dm[i_f]
        hinges = info.faces_hinge[i_f]
        for k in range(3):
            i_h = hinges[k]
            if i_h >= 0:
                i_va, i_vb, i_vc, i_vd, intact = shell.func_hinge_verts(i_h, i_b, state, info)
                if intact:
                    edge_b, edge_c, edge_d = shell.func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, state)
                    angle = shell.func_hinge_angle(edge_b, edge_c, edge_d)
                    delta = angle - info.hinges_rest_angle[i_h] - state.hinges_plastic_angle[i_h, i_b]
                    edge = qd.Vector([Dm[0, 0], Dm[1, 0]])
                    if k == 1:
                        edge = qd.Vector([Dm[0, 1] - Dm[0, 0], Dm[1, 1] - Dm[1, 0]])
                    elif k == 2:
                        edge = -qd.Vector([Dm[0, 1], Dm[1, 1]])
                    length = qd.max(edge.norm(), shell.NORM_FLOOR)
                    normal = qd.Vector([-edge[1], edge[0]]) / length
                    curvature += delta * length * normal.outer_product(normal)
        curvature /= 2.0 * info.faces_rest_area[i_f]
        nu = info.entities_nu[i_e]
        fiber_gradient = info.entities_stretching_modulus[i_e] * (
            (1.0 - nu) * curvature + nu * curvature.trace() * qd.Matrix.identity(gs.qd_float, 2)
        )
        cross = qd.Vector([F[0, 0], F[1, 0], F[2, 0]]).cross(qd.Vector([F[0, 1], F[1, 1], F[2, 1]]))
        jacobian = qd.max(cross.norm(), shell.NORM_FLOOR)
        if qd.static(full_output):
            for j, k in qd.static(qd.ndrange(2, 2)):
                membrane[i_f, i_b, j, k] = stress[j, k]
                curvature_output[i_f, i_b, j, k] = curvature[j, k]
                bending[i_f, i_b, j, k] = thickness**3 / 12.0 * fiber_gradient[j, k]
        for i_layer in range(3):
            z = 0.5 * (i_layer - 1) * thickness
            sigma = F @ (stress / thickness + z * fiber_gradient) @ F.transpose() / jacobian
            deviator = sigma - sigma.trace() / 3.0 * qd.Matrix.identity(gs.qd_float, 3)
            vm = qd.sqrt(qd.max(1.5 * (deviator * deviator).sum(), 0.0))
            if not (qd.abs(sigma.sum()) < qd.math.inf):
                vm = gs.qd_float(float("inf"))
            i_out = 3 * i_f + i_layer
            if qd.static(full_output):
                for j, k in qd.static(qd.ndrange(3, 3)):
                    tensors[i_out, i_b, j, k] = sigma[j, k]
                von_mises[i_out, i_b] = vm
            qd.atomic_max(peaks[i_b], vm)


@qd.kernel
def locate_peaks(
    von_mises: qd.types.ndarray(ndim=2),
    peaks: qd.types.ndarray(ndim=1),
    locations: qd.types.ndarray(ndim=1),
):
    for i_b in range(peaks.shape[0]):
        locations[i_b] = von_mises.shape[0]
    for i_f, i_b in qd.ndrange(von_mises.shape[0], von_mises.shape[1]):
        if von_mises[i_f, i_b] == peaks[i_b]:
            qd.atomic_min(locations[i_b], i_f)


@dataclass
class FullField:
    tensors: torch.Tensor
    von_mises: torch.Tensor
    membrane: torch.Tensor
    curvature: torch.Tensor
    bending: torch.Tensor
    peaks: torch.Tensor
    locations: torch.Tensor
    output_mode: str

    @classmethod
    def allocate(cls, solver, output_mode="full"):
        assert output_mode in ("full", "max")
        n_faces, n_envs = solver.n_faces, solver.shell_state.faces_thickness.shape[1]
        stored_faces, stored_envs = (n_faces, n_envs) if output_mode == "full" else (1, 1)
        return cls(
            torch.empty((3 * stored_faces, stored_envs, 3, 3), dtype=gs.tc_float, device=gs.device),
            torch.empty((3 * stored_faces, stored_envs), dtype=gs.tc_float, device=gs.device),
            torch.empty((stored_faces, stored_envs, 2, 2), dtype=gs.tc_float, device=gs.device),
            torch.empty((stored_faces, stored_envs, 2, 2), dtype=gs.tc_float, device=gs.device),
            torch.empty((stored_faces, stored_envs, 2, 2), dtype=gs.tc_float, device=gs.device),
            torch.empty(n_envs, dtype=gs.tc_float, device=gs.device),
            torch.empty(n_envs if output_mode == "full" else 0, dtype=gs.tc_int, device=gs.device),
            output_mode,
        )

    def update(self, solver, reset_peak=True):
        recover_fields(
            self.tensors,
            self.von_mises,
            self.membrane,
            self.curvature,
            self.bending,
            self.peaks,
            solver.shell_state,
            solver.shell_info,
            self.output_mode == "full",
            reset_peak,
        )
        if self.output_mode == "full":
            locate_peaks(self.von_mises, self.peaks, self.locations)

    @property
    def nbytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.tensors,
                self.von_mises,
                self.membrane,
                self.curvature,
                self.bending,
                self.peaks,
                self.locations,
            )
        )
