"""Audit complete moving time-step histories while retaining wide body coordinates in force checks."""

from . import audit_articulated_contact, audit_articulated_temporal, franka_contact_reference
from .wide_audit_observables import WideAuditBulk, observable_fields, observable_frame, observable_motion


def main():
    franka_contact_reference.StableNeoHookeanBulk = WideAuditBulk
    audit_articulated_contact.finite_frame = observable_frame
    audit_articulated_contact.quadrature_fields = observable_fields
    audit_articulated_contact.motion_moments = observable_motion
    audit_articulated_temporal.main()


if __name__ == "__main__":
    main()
