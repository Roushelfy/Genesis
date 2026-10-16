"""Keep original observable precision separate from wide constitutive equilibrium checks."""

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .reference_motion_audit import motion_moments
from .wide_coordinate_franka import WidePiolaBulk


class WideAuditBulk(WidePiolaBulk):
    def stress_bounds(self, displacement):
        return NeoHookeanBulk.stress_bounds(self, np.asarray(displacement, dtype=float))


def observable_frame(body, displacement):
    return finite_frame(body, np.asarray(displacement, dtype=float))


def observable_fields(bulk, displacement):
    return quadrature_fields(bulk, np.asarray(displacement, dtype=float))


def observable_motion(body, permutation, positions, velocity):
    return motion_moments(body, permutation, np.asarray(positions, dtype=float), np.asarray(velocity, dtype=float))
