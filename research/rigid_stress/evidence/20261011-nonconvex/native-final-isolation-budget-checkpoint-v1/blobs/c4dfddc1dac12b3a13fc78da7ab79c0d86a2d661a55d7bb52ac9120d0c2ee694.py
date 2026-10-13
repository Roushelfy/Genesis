"""Diagnostic lifecycle run with only the approximate Hessian forced to FP64."""
import sys

import pytest

from genesis.engine.solvers.rigid.stress import cooperative_fit

cooperative_fit._HESSIAN_FLOAT32 = cooperative_fit._HESSIAN_FLOAT64
raise SystemExit(pytest.main(sys.argv[1:]))
