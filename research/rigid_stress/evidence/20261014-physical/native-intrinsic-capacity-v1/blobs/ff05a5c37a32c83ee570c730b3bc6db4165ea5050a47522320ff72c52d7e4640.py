"""Assemble rigid contact/inertia directly without adding and cancelling fictitious elastic forces."""

import ipctk
import numpy as np
from scipy import sparse

from .rigid_egg_articulated_contact import RigidBodyContact
from .run_original_rigid_franka import OriginalRigidEggContact


class DirectRigidBodyContact(RigidBodyContact):
    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        normal, current = self.normal_collisions(displacement)
        energy = self.barrier(normal, self.mesh, current)
        contact = self.mesh.to_full_dof(self.barrier.gradient(normal, self.mesh, current))
        relative_motion = self.mesh.vertices((displacement - previous).reshape(-1, 3))
        if self.friction > 0:
            energy += self.friction_potential(tangential, self.mesh, relative_motion)
            contact += self.mesh.to_full_dof(self.friction_potential.gradient(tangential, self.mesh, relative_motion))
        gradient = np.asarray(contact, dtype=np.longdouble).copy()
        if external_force is not None:
            energy -= external_force @ displacement
            gradient -= external_force
        matrix = None
        if hessian:
            projection = ipctk.PSDProjectionMethod.CLAMP if project_hessian else ipctk.PSDProjectionMethod.NONE
            matrix = self.mesh.to_full_dof(self.barrier.hessian(normal, self.mesh, current, projection))
            if self.friction > 0:
                matrix += self.mesh.to_full_dof(self.friction_potential.hessian(tangential, self.mesh, relative_motion))
        if self.dt is not None:
            n = self.body.basis.N
            increment = np.asarray(displacement[:n], dtype=np.longdouble) - self.predictor
            inertial = self.body_mass @ increment / self.dt**2
            gradient[:n] += inertial
            energy += 0.5 * increment @ inertial
            if hessian:
                pad = len(displacement) - n
                matrix += sparse.block_diag((self.body_mass / self.dt**2, sparse.csc_matrix((pad, pad))))
        return energy, gradient, matrix, normal, contact


class DirectOriginalRigidContact(OriginalRigidEggContact):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact.__class__ = DirectRigidBodyContact
