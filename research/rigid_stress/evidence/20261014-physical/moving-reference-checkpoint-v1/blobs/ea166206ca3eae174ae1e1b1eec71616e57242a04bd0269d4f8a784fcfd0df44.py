"""Wide-coordinate force contraction for the unchanged neo-Hookean reference."""

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .stable_neohookean_reference import StableNeoHookeanBulk


class WidePiolaBulk(StableNeoHookeanBulk):
    """Retain the original law/quadrature/tangent and accumulate Piola forces in long double."""

    def evaluate(self, displacement, hessian=True):
        energy, _, matrix = NeoHookeanBulk.evaluate(self, np.asarray(displacement, dtype=float), hessian)
        assert np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
        h = np.einsum(
            "aie,ajeq->ijeq",
            np.asarray(displacement[self.dofs], dtype=np.longdouble),
            np.asarray(self.gradients, dtype=np.longdouble),
            optimize=True,
        ).transpose(2, 3, 0, 1)
        tr = np.trace(h, axis1=-2, axis2=-1)
        det_h = np.sum(h[..., 0, :] * np.cross(h[..., 1, :], h[..., 2, :]), axis=-1)
        delta = tr + (tr**2 - np.einsum("...ij,...ji->...", h, h)) / 2 + det_h
        if np.any(delta <= -1):
            raise ValueError("Neo-Hookean bulk requires positive deformation determinant")
        identity = np.eye(3, dtype=np.longdouble)
        f = identity + h
        cofactor = np.stack(
            [
                np.cross(f[..., :, 1], f[..., :, 2]),
                np.cross(f[..., :, 2], f[..., :, 0]),
                np.cross(f[..., :, 0], f[..., :, 1]),
            ],
            axis=-1,
        )
        kirchhoff = self.mu * (h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2))
        kirchhoff += self.lam * np.log1p(delta)[..., None, None] * identity
        piola = (kirchhoff @ cofactor) / (1 + delta)[..., None, None]
        local = np.einsum(
            "ijeq,ajeq,eq->aie",
            piola.transpose(2, 3, 0, 1),
            np.asarray(self.gradients, dtype=np.longdouble),
            np.asarray(self.basis.dx, dtype=np.longdouble),
            optimize=True,
        )
        gradient = np.zeros(self.reference.basis.N, dtype=np.longdouble)
        np.add.at(gradient, self.dofs.ravel(), local.ravel())
        return energy, np.asarray(gradient, dtype=float), matrix


import json

import ipctk
from scipy.sparse.linalg import splu

from .articulated_franka_contact import ArticulatedContact, BodyContact


class WideBodyContact(BodyContact):
    """Keep body coordinates wide while retaining IPC double-precision geometry."""

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        vector = np.empty(self.n_body_dofs, dtype=np.longdouble)
        permutation = self.parts[0].permutation
        vector[permutation] = displacement[: self.n_body_dofs]
        bulk_gradient = self.bulk.evaluate(vector, False)[1][permutation]
        body_gradient = bulk_gradient + contact[: self.n_body_dofs]
        if external_force is not None:
            body_gradient -= external_force[: self.n_body_dofs]
        if self.dt is not None:
            increment = np.asarray(displacement[: self.n_body_dofs], dtype=np.longdouble) - self.predictor
            body_gradient = body_gradient + self.body_mass @ increment / self.dt**2
        gradient = np.asarray(gradient, dtype=float)
        gradient[: self.n_body_dofs] = body_gradient
        return energy, gradient, matrix, normal, contact


class WideArticulatedContact(ArticulatedContact):
    """Use wide body iterates and a stricter internal Newton target without changing physics."""

    def __init__(self, static, frame, capture_report, mjcf, dt):
        super().__init__(static, frame, capture_report, mjcf, dt)
        self.contact.__class__ = WideBodyContact
        self.contact.n_body_dofs = self.n
        self.contact.bulk = WidePiolaBulk(static.body)

    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
        body_u = np.asarray(body_u, dtype=np.longdouble)
        free = np.setdiff1d(np.arange(self.n), self.held_body) if hold_body else np.arange(self.n)
        joint_free = np.array([7, 8]) if self.dynamic and self.fixed_arm else np.arange(9)
        for iteration in range(max_iterations):
            rb, rj, full, contact, _energy, stencils, blocks = self.evaluate(body_u, variable, previous, True)
            body_error, joint_error = float(np.linalg.norm(rb[free])), float(np.linalg.norm(rj[joint_free]))
            self.failure_state = {
                "body_displacement": body_u.copy(),
                "robot_variable": variable.copy(),
                "full_displacement": full.copy(),
                "body_gradient": rb.copy(),
                "joint_residual": rj.copy(),
                "contact_force_N": -contact.copy(),
            }
            record = {
                "phase": phase,
                "iteration": iteration,
                "body_residual_N": body_error,
                "joint_residual_N_or_Nm": joint_error,
                "normal_stencils": stencils,
            }
            print(json.dumps(record), flush=True)
            if max(body_error, joint_error) <= 1e-10:
                self.history.append(record)
                return body_u, variable, full, contact
            bb, bq, qb, qq = blocks
            factor = splu(bb[free][:, free].tocsc())
            y = factor.solve(-rb[free])
            z = factor.solve(np.asarray(bq[free][:, joint_free]))
            lower = np.asarray(qb[joint_free][:, free])
            direction_q = np.zeros(9)
            direction_q[joint_free] = np.linalg.solve(
                qq[np.ix_(joint_free, joint_free)] - lower @ z, -rj[joint_free] - lower @ y
            )
            direction_b = np.zeros(self.n)
            direction_b[free] = y - z @ direction_q[joint_free]
            merit = np.hypot(body_error, joint_error)
            alpha = 1.0
            current = self.contact.mesh.vertices(self.contact.rest + full.reshape(-1, 3))
            for _ in range(60):
                trial_b, trial_q = body_u + alpha * direction_b, variable + alpha * direction_q
                endpoint_q = self.previous_q + np.longdouble(self.dt) * trial_q if self.dynamic else trial_q
                trial_full = self.pack(trial_b, endpoint_q)[0]
                endpoint = self.contact.mesh.vertices(self.contact.rest + trial_full.reshape(-1, 3))
                safe = ipctk.compute_collision_free_stepsize(self.contact.mesh, current, endpoint)
                if safe >= 1 - 1e-12 and not ipctk.has_intersections(self.contact.mesh, endpoint):
                    try:
                        trial = self.evaluate(trial_b, trial_q, previous, False)
                        norm = np.hypot(np.linalg.norm(trial[0][free]), np.linalg.norm(trial[1][joint_free]))
                        if norm < (1 - 1e-4 * alpha) * merit:
                            break
                    except ValueError:
                        pass
                alpha *= 0.5
            else:
                raise RuntimeError(f"Coupled residual line search failed: {phase} {record}")
            body_u, variable = trial_b, trial_q
        raise RuntimeError(f"Coupled Newton exceeded its budget: {phase}")
