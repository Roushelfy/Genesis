"""Repeat the rejected-state tangent diagnostic using direct rigid force assembly."""

from . import diagnose_rigid_continuation as diagnostic
from .direct_rigid_contact import DirectOriginalRigidContact


def main():
    diagnostic.OriginalRigidEggContact = DirectOriginalRigidContact
    diagnostic.main()


if __name__ == "__main__":
    main()
