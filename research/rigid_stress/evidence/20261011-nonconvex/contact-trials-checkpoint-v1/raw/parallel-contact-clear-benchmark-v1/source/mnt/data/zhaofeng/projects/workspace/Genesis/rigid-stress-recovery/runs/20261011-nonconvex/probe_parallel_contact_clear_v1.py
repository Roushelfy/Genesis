"""Patch native packing only; retain the same traction solver, ordering, complete residual and lifecycle checks."""

import sys

from parallel_contact_clear_candidate_v1 import kernel_pack_intrinsic_contacts

from genesis.engine.solvers.rigid.stress import intrinsic

intrinsic.kernel_pack_intrinsic_contacts = kernel_pack_intrinsic_contacts
if sys.argv[1] == "tests":
    import pytest

    sys.exit(pytest.main(sys.argv[2:]))
else:
    from examples.speed_benchmark.rigid_stress import main

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    main()
