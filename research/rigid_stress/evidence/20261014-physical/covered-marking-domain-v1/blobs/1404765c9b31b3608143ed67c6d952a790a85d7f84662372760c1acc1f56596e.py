"""Assemble the unchanged Panda inertial mass without dense LIL subblock assignment."""

import ipctk
import numpy as np
from scipy import sparse

from .dynamic_franka_reference import InertialRigidFingerContactSystem


class SparseInertialRigidFingerContactSystem(InertialRigidFingerContactSystem):
    def __init__(self, static, dt):
        self.__dict__.update(static.__dict__)
        assert dt > 0
        self.dt = dt
        self.friction_mollifier_velocity_m_s = 4e-6
        self.friction_potential = ipctk.FrictionPotential(dt * self.friction_mollifier_velocity_m_s)
        self.kv, self.passive_damping, self.armature = 10.0, 1.0, 0.1
        count = self.body.basis.N
        permutation = self.parts[0].permutation
        body_mass = self.body.mass[permutation][:, permutation]
        matrix = sparse.block_diag((body_mass, sparse.csc_matrix((self.stiffness.shape[0] - count,) * 2)), format="csc")
        rows, cols, values = [], [], []
        for node, axis, mass in zip(self.finger_com_nodes, self.axes, self.masses, strict=True):
            dofs = np.arange(3 * node, 3 * node + 3)
            local = mass * np.eye(3) + self.armature * np.outer(axis, axis)
            rows.extend(np.repeat(dofs, 3))
            cols.extend(np.tile(dofs, 3))
            values.extend(local.ravel())
        matrix += sparse.coo_matrix((values, (rows, cols)), shape=self.stiffness.shape).tocsc()
        self.inertial_mass = matrix
        self.predicted_displacement = np.zeros(self.stiffness.shape[0])
