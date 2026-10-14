"""Audit experimental dual initial guesses on actual trajectories with the independent CPU FP64 field oracle."""

import runpy

from traction_workspace_seed_candidate_v1 import kernel_fit_tractions

from genesis.engine.solvers.rigid.stress import cooperative_fit

cooperative_fit.kernel_fit_tractions = kernel_fit_tractions
runpy.run_module("research.rigid_stress.probe_nonconvex_online", run_name="__main__")
