"""Couple independent P2 contact to exact Panda joint kinematics and dynamics."""

import json

import ipctk
import mujoco
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .contact_reference import IndependentContactSystem
from .franka_contact_reference import RigidFingerContactSystem
from .franka_mujoco_reference import PandaSkeleton
from .precise_panda_kinematics import PrecisePandaKinematics


class BodyContact(RigidFingerContactSystem):
    """Body/contact terms only; all robot inertia and actuation live in joints."""

    def __init__(self, static):
        self.__dict__.update(static.__dict__)
        n = self.body.basis.N
        permutation = self.parts[0].permutation
        self.body_mass = self.body.mass[permutation][:, permutation].tocsc()
        self.dt = None
        self.predictor = np.zeros(n)

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact = IndependentContactSystem.evaluate(
            self, displacement, previous, tangential, hessian, project_hessian, external_force
        )
        n = self.body.basis.N
        permutation = self.parts[0].permutation
        vector = np.empty(n)
        vector[permutation] = displacement[:n]
        linear = self.body.align_rigid_gauge(vector)
        old = self.body.stiffness @ linear
        new_energy, new_gradient, new_matrix = self.bulk.evaluate(vector, hessian)
        energy += new_energy - 0.5 * linear @ old
        gradient[:n] += (new_gradient - old)[permutation]
        delta = None if not hessian else (new_matrix - self.body.stiffness)[permutation][:, permutation]
        if self.dt is not None:
            increment = displacement[:n] - self.predictor
            inertial = self.body_mass @ increment / self.dt**2
            gradient[:n] += inertial
            energy += 0.5 * increment @ inertial
            if hessian:
                delta += self.body_mass / self.dt**2
        if hessian:
            matrix += sparse.block_diag((delta, sparse.csc_matrix((len(displacement) - n,) * 2)), format="csc")
        return energy, gradient, matrix, normal, contact


class ArticulatedContact:
    def __init__(self, static, frame, capture_report, mjcf, dt):
        self.contact = BodyContact(static)
        self.n = static.body.basis.N
        self.robot = PandaSkeleton(mjcf, dt)
        self.robot.model.opt.tolerance = 1e-14
        self.precise = PrecisePandaKinematics(self.robot.model, static.world_coordinate_origin_m)
        self.kinematic_max_discrepancy_m = 0.0
        self.dt = dt
        self.kp = np.array(capture_report["robot_joint_kp"])
        self.kv = np.array(capture_report["robot_joint_kv"])
        self.caps = np.r_[self.robot.model.actuator_forcerange[:7, 1], 4.0, 4.0]
        self.target = np.r_[np.array(capture_report["stationary_actuation"]["arm_target"])[0], 0.04, 0.04]
        self.initial_q = np.array(frame["robot_joint_position"], dtype=np.longdouble)
        self.initial_q[7:] = 0.04
        self.robot.set_state(self.initial_q, np.zeros(9))
        precise_positions, precise_rotations = self.precise.poses(self.initial_q)
        self.finger_local = []
        self.names = ["left_finger", "right_finger"]
        for name, nodes in zip(self.names, static.finger_nodes):
            body = self.robot.body_ids[name]
            origin = precise_positions[body]
            rotation = precise_rotations[body]
            local = (np.asarray(static.rest[nodes], dtype=np.longdouble) - origin) @ rotation
            np.testing.assert_allclose(local[-1], 0, rtol=0, atol=1e-12)
            self.finger_local.append(local)
        self.gravity = static.gravity_force([0, 0, -9.81])
        self.gravity[self.n :] = 0
        held = np.flatnonzero(np.isin(static.parts[0].permutation, static.body.gauge_dofs))
        self.held_body = held
        self.previous_q = self.initial_q.copy()
        self.previous_velocity = np.zeros(9)
        self.dynamic = False
        self.fixed_arm = False
        self.history = []
        _, positions = self.kinematics(self.initial_q)
        np.testing.assert_allclose(positions, static.rest[self.n // 3 :], rtol=0, atol=1e-12)

    def kinematics(self, q):
        self.robot.set_state(q, np.zeros(9))
        precise_positions, precise_rotations = self.precise.poses(q)
        matrices, positions = [], []
        for name, local in zip(self.names, self.finger_local):
            body = self.robot.body_ids[name]
            rotation = precise_rotations[body]
            relative = local @ rotation.T
            centered = relative + precise_positions[body]
            discrepancy = float(
                np.linalg.norm(
                    precise_positions[body] - (self.robot.data.xpos[body] - self.contact.world_coordinate_origin_m)
                )
            )
            self.kinematic_max_discrepancy_m = max(self.kinematic_max_discrepancy_m, discrepancy)
            assert discrepancy < 1e-12
            np.testing.assert_allclose(rotation, self.robot.data.xmat[body].reshape(3, 3), rtol=0, atol=1e-12)
            jacobian = self.robot.jacobian(name)
            point = jacobian[None, :3, :] + np.cross(jacobian[3:].T[None], relative[:, None]).transpose(0, 2, 1)
            positions.append(centered)
            matrices.append(np.asarray(point.reshape(-1, 9), dtype=float))
        return np.vstack(matrices), np.vstack(positions)

    def pack(self, body_u, q):
        jacobian, positions = self.kinematics(q)
        full = np.r_[body_u, np.asarray((positions - self.contact.rest[self.n // 3 :]).reshape(-1), dtype=float)]
        return full, jacobian

    def pd(self, q, velocity):
        raw = self.kp * (self.target - q) - self.kv * velocity
        return np.asarray(np.clip(raw, -self.caps, self.caps), dtype=float), abs(raw) < self.caps

    def static_joint(self, q, finger_gradient, response=False):
        jacobian, _ = self.kinematics(q)
        effort = self.pd(q, np.zeros(9))[0] - jacobian.T @ finger_gradient
        data, model = self.robot.data, self.robot.model
        data.qfrc_applied[:] = effort
        mujoco.mj_forward(model, data)
        mass, acceleration = self.robot.mass(), data.qacc.copy()
        residual = data.qfrc_bias - data.qfrc_applied - data.qfrc_passive - data.qfrc_constraint
        assert np.linalg.norm(residual + mass @ acceleration) < 1e-10
        projection = None
        if response:
            projection = np.empty((9, 9))
            for column in range(9):
                data.qfrc_applied[:] = effort
                data.qfrc_applied[column] += 1
                mujoco.mj_forward(model, data)
                projection[:, column] = mass @ (data.qacc - acceleration)
        return residual, projection

    def evaluate(self, body_u, variable, previous, hessian):
        q = self.previous_q + np.longdouble(self.dt) * variable if self.dynamic else variable
        full, jacobian = self.pack(body_u, q)
        energy, gradient, matrix, normal, contact = self.contact.evaluate_updated_friction(
            full, previous, self.gravity, hessian
        )
        if self.dynamic:
            effort, active = self.pd(q, variable)
            joint = (
                self.inverse_response @ (variable - self.baseline_velocity) - effort + jacobian.T @ contact[self.n :]
            )
            projection, scale = np.eye(9), self.dt
        else:
            joint, projection = self.static_joint(q, contact[self.n :], hessian)
            active, scale = None, 1.0
        if not hessian:
            return gradient[: self.n], joint, full, contact, energy, len(normal), None
        bb = matrix[: self.n, : self.n].tocsc()
        bq = matrix[: self.n, self.n :] @ jacobian * scale
        qb = projection @ (jacobian.T @ matrix[self.n :, : self.n])
        qq = projection @ (jacobian.T @ (matrix[self.n :, self.n :] @ jacobian)) * scale
        for column in range(9):
            step = np.zeros(9)
            step[column] = 1e-6
            if self.dynamic:
                plus = self.kinematics(q + step)[0].T @ contact[self.n :]
                minus = self.kinematics(q - step)[0].T @ contact[self.n :]
                qq[:, column] += self.dt * (plus - minus) / 2e-6
            else:
                plus = self.static_joint(q + step, contact[self.n :])[0]
                minus = self.static_joint(q - step, contact[self.n :])[0]
                qq[:, column] += (plus - minus) / 2e-6
        if self.dynamic:
            qq += self.inverse_response + np.diag((self.dt * self.kp + self.kv) * active)
        return gradient[: self.n], joint, full, contact, energy, len(normal), (bb, bq, qb, qq)

    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
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
            if max(body_error, joint_error) <= 1e-9:
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

    def begin_dynamic_step(self, q, velocity, body_predictor):
        self.dynamic = True
        self.previous_q, self.previous_velocity = np.asarray(q, dtype=np.longdouble).copy(), velocity.copy()
        self.baseline_velocity, response = self.robot.velocity_response(q, velocity)
        self.inverse_response = np.linalg.inv(response)
        self.contact.dt = self.dt
        self.contact.predictor = body_predictor.copy()
        self.contact.friction_potential = ipctk.FrictionPotential(self.dt * 4e-6)

    def verify_robot_step(self, velocity, contact):
        q = self.previous_q + np.longdouble(self.dt) * velocity
        jacobian, _ = self.kinematics(q)
        effort = self.pd(q, velocity)[0] - jacobian.T @ contact[self.n :]
        support = np.zeros(9)
        if self.fixed_arm:
            support[:7] = (self.inverse_response @ (velocity - self.baseline_velocity) - effort)[:7]
        self.last_support_effort = support
        actual = self.robot.step_velocity(self.previous_q, self.previous_velocity, effort + support)
        error = self.inverse_response @ (velocity - actual)
        assert np.linalg.norm(error) <= 1e-9
        return float(np.linalg.norm(error))
