"""Measure the unchanged sampled-first diagnosis without the new boundary fallback."""

from genesis.engine.solvers.rigid.stress import intrinsic
from genesis.engine.solvers.rigid.stress.compatibility import (
    kernel_certify_continuous_wrench, kernel_search_wrench_certificate,
)


def diagnose_sampled(contacts, sources, tractions, search, info, surface, epsilon, tolerance):
    kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, tolerance)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface, epsilon, 8, 16384, tolerance)


intrinsic.kernel_diagnose_wrenches = diagnose_sampled
intrinsic._WRENCH_CERTIFICATE_DEPTH = 8
from examples.speed_benchmark.rigid_stress import main

main()
