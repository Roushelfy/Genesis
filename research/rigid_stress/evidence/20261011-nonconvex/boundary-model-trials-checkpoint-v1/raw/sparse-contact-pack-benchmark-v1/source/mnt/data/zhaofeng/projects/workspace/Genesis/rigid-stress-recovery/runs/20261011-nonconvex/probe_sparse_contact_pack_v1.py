"""Patch only raw-slot packing; every active contact remains stably ordered and processed."""

import sys

from sparse_contact_pack_candidate_v1 import kernel_pack_intrinsic_contacts

from genesis.engine.solvers.rigid.stress import intrinsic

intrinsic.kernel_pack_intrinsic_contacts = kernel_pack_intrinsic_contacts
if sys.argv[1] == "tests":
    import pytest

    sys.exit(pytest.main(sys.argv[2:]))
else:
    from examples.speed_benchmark.rigid_stress import main

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    main()
