"""Independently audit wide saved body coordinates at the original external force budgets."""

from . import audit_rigid_egg_feedback, franka_contact_reference
from .wide_audit_observables import WideAuditBulk, observable_fields, observable_frame, observable_motion


def main():
    franka_contact_reference.StableNeoHookeanBulk = WideAuditBulk
    # Stress/rigid-frame observables use the original float64 routines;
    # the separately reconstructed force gradient keeps the saved coordinate dtype.
    audit_rigid_egg_feedback.finite_frame = observable_frame
    audit_rigid_egg_feedback.quadrature_fields = observable_fields
    audit_rigid_egg_feedback.motion_moments = observable_motion
    audit_rigid_egg_feedback.main()


if __name__ == "__main__":
    main()
