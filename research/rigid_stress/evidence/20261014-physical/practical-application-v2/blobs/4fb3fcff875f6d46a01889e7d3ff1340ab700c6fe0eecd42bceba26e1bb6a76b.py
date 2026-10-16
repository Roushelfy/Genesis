"""Restrict the same contact surface to six rigid-body coordinates for feedback controls."""

import json

import ipctk
import numpy as np
from scipy import sparse

from .articulated_franka_contact import ArticulatedContact, BodyContact
from .contact_reference import IndependentContactSystem


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.longdouble)


def rotation_terms(vector):
    """Cancellation-stable exponential map and its right Jacobian."""
    vector = np.asarray(vector, dtype=np.longdouble)
    square = vector @ vector
    if square < 1e-8:
        a = 1 - square / 6 + square**2 / 120 - square**3 / 5040
        b = 0.5 - square / 24 + square**2 / 720 - square**3 / 40320
        c = 1 / np.longdouble(6) - square / 120 + square**2 / 5040 - square**3 / 362880
    else:
        angle = np.sqrt(square)
        a = np.sin(angle) / angle
        b = 2 * (np.sin(angle / 2) / angle) ** 2
        c = (1 - a) / square
    cross = skew(vector)
    increment = a * cross + b * (cross @ cross)
    rotation = np.eye(3, dtype=np.longdouble) + increment
    right_jacobian = np.eye(3, dtype=np.longdouble) - b * cross + c * (cross @ cross)
    return increment, rotation, right_jacobian


class RigidBodyContact(BodyContact):
    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact = IndependentContactSystem.evaluate(
            self, displacement, previous, tangential, hessian, project_hessian, external_force
        )
        # Rigid internal constraint forces do no work in the six allowed modes.
        # Remove elastic bulk terms, retaining precisely the original contact law.
        elastic = self.elastic_displacement(displacement)
        internal = self.stiffness @ elastic
        energy -= 0.5 * elastic @ internal
        gradient -= internal
        if hessian:
            matrix = matrix - self.stiffness
        if self.dt is not None:
            increment = displacement[: self.body.basis.N] - self.predictor
            inertial = self.body_mass @ increment / self.dt**2
            gradient[: self.body.basis.N] += inertial
            energy += 0.5 * increment @ inertial
            if hessian:
                pad = matrix.shape[0] - self.body.basis.N
                matrix = matrix + sparse.block_diag((self.body_mass / self.dt**2, sparse.csc_matrix((pad, pad))))
        return energy, gradient, matrix, normal, contact


class RigidEggArticulatedContact(ArticulatedContact):
    def __init__(self, static, frame, capture_report, mjcf, dt):
        super().__init__(static, frame, capture_report, mjcf, dt)
        self.contact = RigidBodyContact(static)
        self.body_centre = np.asarray(static.body.com, dtype=np.longdouble)
        self.body_relative = np.asarray(static.rest[: self.n // 3], dtype=np.longdouble) - self.body_centre
        self.radius = float(np.linalg.norm(self.body_relative, axis=1).max())
        self.body_row_scale = np.r_[np.ones(3), np.full(3, 1 / self.radius)]

    def body_kinematics(self, pose):
        increment, rotation, right = rotation_terms(pose[3:])
        displacement = self.body_relative @ increment.T + pose[:3]
        jacobian = np.zeros((self.n // 3, 3, 6))
        jacobian[:, :, :3] = np.eye(3)
        for node, relative in enumerate(self.body_relative):
            jacobian[node, :, 3:] = -rotation @ skew(relative) @ right
        return np.asarray(displacement.reshape(-1), dtype=float), jacobian.reshape(-1, 6)

    def evaluate_rigid(self, pose, variable, previous, hessian):
        body_u, body_jacobian = self.body_kinematics(pose)
        rb, rj, full, contact, energy, stencils, blocks = super().evaluate(body_u, variable, previous, hessian)
        body = (body_jacobian.T @ rb) * self.body_row_scale
        if not hessian:
            return body, rj, full, contact, energy, stencils, None
        bb, bq, qb, qq = blocks
        reduced = body_jacobian.T @ (bb @ body_jacobian)
        for column in range(3, 6):
            direction = np.zeros(6)
            direction[column] = 1e-6
            plus = self.body_kinematics(pose + direction)[1].T @ rb
            minus = self.body_kinematics(pose - direction)[1].T @ rb
            reduced[:, column] += (plus - minus) / 2e-6
        blocks = (
            self.body_row_scale[:, None] * reduced,
            self.body_row_scale[:, None] * (body_jacobian.T @ bq),
            qb @ body_jacobian,
            qq,
        )
        return body, rj, full, contact, energy, stencils, blocks

    def solve_rigid(self, pose, variable, previous, hold_body, phase, max_iterations=100):
        free = np.arange(6, 15) if hold_body else np.arange(15)
        for iteration in range(max_iterations):
            rb, rj, full, contact, _, stencils, blocks = self.evaluate_rigid(pose, variable, previous, True)
            residual = np.r_[rb, rj]
            body_error = 0.0 if hold_body else float(np.linalg.norm(rb))
            joint_error = float(np.linalg.norm(rj))
            self.failure_state = {
                "body_pose": pose.copy(),
                "robot_variable": variable.copy(),
                "full_displacement": full.copy(),
                "body_scaled_residual_N": rb.copy(),
                "joint_residual": rj.copy(),
                "contact_force_N": -contact.copy(),
            }
            row = {
                "phase": phase,
                "iteration": iteration,
                "body_force_equivalent_residual_N": body_error,
                "joint_residual_N_or_Nm": joint_error,
                "normal_stencils": stencils,
            }
            print(json.dumps(row), flush=True)
            if max(body_error, joint_error) <= 1e-9:
                self.history.append(row)
                return pose, variable, full, contact
            bb, bq, qb, qq = blocks
            matrix = np.block([[bb, bq], [qb, qq]])
            direction = np.zeros(15)
            direction[free] = np.linalg.solve(matrix[np.ix_(free, free)], -residual[free])
            current = self.contact.mesh.vertices(self.contact.rest + full.reshape(-1, 3))
            merit = float(np.linalg.norm(residual[free]))
            alpha = 1.0
            for _ in range(60):
                trial_pose = pose + alpha * direction[:6]
                trial_variable = variable + alpha * direction[6:]
                q = self.previous_q + np.longdouble(self.dt) * trial_variable if self.dynamic else trial_variable
                trial_body = self.body_kinematics(trial_pose)[0]
                trial_full = self.pack(trial_body, q)[0]
                endpoint = self.contact.mesh.vertices(self.contact.rest + trial_full.reshape(-1, 3))
                safe = ipctk.compute_collision_free_stepsize(self.contact.mesh, current, endpoint)
                if safe >= 1 - 1e-12 and not ipctk.has_intersections(self.contact.mesh, endpoint):
                    trial = self.evaluate_rigid(trial_pose, trial_variable, previous, False)
                    if np.linalg.norm(np.r_[trial[0], trial[1]][free]) < (1 - 1e-4 * alpha) * merit:
                        break
                alpha *= 0.5
            else:
                raise RuntimeError(f"Rigid-body coupled line search failed: {phase} {row}")
            pose, variable = trial_pose, trial_variable
        raise RuntimeError(f"Rigid-body coupled Newton exceeded its budget: {phase}")
