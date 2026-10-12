"""Validate cooperative scheduling on the unchanged live intrinsic workload and CPU oracle."""

import runpy
from unittest.mock import patch

from traction_original_point_candidate_v10 import kernel_fit_tractions
from traction_sample_cooperative_candidate_v2 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

from genesis.engine.solvers.rigid.stress import intrinsic

with (
    patch.object(intrinsic, "kernel_fit_tractions", kernel_fit_tractions),
    patch.object(intrinsic, "kernel_commit_intrinsic_contacts", kernel_commit_intrinsic_contacts),
    patch.object(intrinsic, "kernel_scatter_tractions", kernel_scatter_tractions),
):
    runpy.run_module("research.rigid_stress.probe_nonconvex_online", run_name="__main__")
