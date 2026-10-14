"""Patch only experimental trial evaluation; preserve native FP64 cold retry and all final checks."""

import sys

from traction_gradient_trial_candidate_v1 import kernel_fit_tractions

from genesis.engine.solvers.rigid.stress import cooperative_fit

cooperative_fit.kernel_fit_tractions = kernel_fit_tractions
if sys.argv[1] == "tests":
    import pytest

    sys.exit(pytest.main(sys.argv[2:]))
else:
    from examples.speed_benchmark.rigid_stress import main

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    main()
