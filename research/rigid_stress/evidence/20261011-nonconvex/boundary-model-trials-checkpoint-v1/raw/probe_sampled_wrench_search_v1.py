"""Restore only the frozen sampled diagnostic and its stack depth for a matched current-law cost comparison."""

import importlib.util
import sys
from pathlib import Path

from genesis.engine.solvers.rigid.stress import intrinsic

root = Path(__file__).parent
path = root / "parallel-contact-clear-benchmark-v1/source/genesis/engine/solvers/rigid/stress/compatibility.py"
name = "genesis.engine.solvers.rigid.stress.sampled_wrench_legacy"
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
intrinsic.kernel_search_wrench_certificate = module.kernel_search_wrench_certificate
intrinsic._WRENCH_CERTIFICATE_DEPTH = 8
from examples.speed_benchmark.rigid_stress import main

main()
