"""Audit the original motion protocol using extended-precision constitutive force contraction."""

from . import audit_rigid_egg_feedback, franka_contact_reference
from .extended_piola_reference import ExtendedPiolaBulk


def main():
    franka_contact_reference.StableNeoHookeanBulk = ExtendedPiolaBulk
    audit_rigid_egg_feedback.main()


if __name__ == "__main__":
    main()
