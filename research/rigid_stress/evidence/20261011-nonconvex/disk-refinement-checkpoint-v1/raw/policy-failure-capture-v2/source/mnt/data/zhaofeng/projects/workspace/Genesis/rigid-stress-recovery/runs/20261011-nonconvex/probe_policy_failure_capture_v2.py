"""Capture the first policy load rejection before rigid integration clears its input forces."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from traction_cooperative_fallback_candidate_v1 import kernel_fit_tractions
from traction_sample_cooperative_candidate_v2 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress import intrinsic
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import ErrorCode, iter_data
from genesis.utils.misc import qd_to_numpy

destination = Path(sys.argv[sys.argv.index("--output") + 1]).parent / "failure"
original_recover = RigidStressRecovery.recover
original_step = FrankaEgg.step
current_workload = None
calls = 0
captured = False


def step(workload, residual_action=None):
    global current_workload, calls
    current_workload = workload
    calls += 1
    return original_step(workload, residual_action)


def recover(recovery, i_substep):
    global captured
    solver = recovery.solver
    entry = recovery.links[0]
    before = {
        "applied_ang": qd_to_numpy(solver.dyn_state.links.cfrc_applied_ang, transpose=True, copy=True)[
            :, entry.link.idx
        ],
        "applied_vel": qd_to_numpy(solver.dyn_state.links.cfrc_applied_vel, transpose=True, copy=True)[
            :, entry.link.idx
        ],
        "coupling_ang": qd_to_numpy(solver.dyn_state.links.cfrc_coupling_ang, transpose=True, copy=True)[
            :, entry.link.idx
        ],
        "coupling_vel": qd_to_numpy(solver.dyn_state.links.cfrc_coupling_vel, transpose=True, copy=True)[
            :, entry.link.idx
        ],
        "n_contacts": qd_to_numpy(solver.collider.collider_state.n_contacts, copy=True),
    }
    result = original_recover(recovery, i_substep)
    errno = qd_to_numpy(solver._errno, copy=True)
    if not captured and np.bitwise_and(errno, int(ErrorCode.INVALID_STRESS_LOAD)).any():
        captured = True
        destination.mkdir(parents=True, exist_ok=False)
        values = {f"before.{name}": value for name, value in before.items()}
        values["errno"] = errno
        for name, state in (
            ("contacts", entry.contacts),
            ("state", entry.state),
            ("raw_sources", entry.intrinsic.raw_sources),
            ("fit_contacts", entry.intrinsic.contacts),
            ("fit_sources", entry.intrinsic.sources),
            ("tractions", entry.intrinsic.tractions),
            ("footprints", entry.intrinsic.footprints),
            ("packing", entry.intrinsic.packing),
            ("certificates", entry.intrinsic.certificates),
        ):
            for item in iter_data(state, name):
                values[item.name] = qd_to_numpy(item.value, transpose=True, copy=True)
        np.savez_compressed(destination / "before-rigid-integration.npz", **values)
        unsupported = np.zeros_like(errno, dtype=bool)
        maxima = {}
        for name in ("applied_ang", "applied_vel", "coupling_ang", "coupling_vel"):
            norms = np.linalg.norm(before[name], axis=-1)
            unsupported |= norms > 0
            maxima[name] = norms.tolist()
        capacity = entry.contacts.valid.shape[0]
        unsupported |= before["n_contacts"] > capacity
        metadata = {
            "scope": "Untimed policy failure diagnosis; readbacks alter scheduling cost, no throughput claim.",
            "benchmark_command": sys.argv,
            "step_call": calls,
            "workload_tick_before_step": current_workload.tick,
            "substep": i_substep,
            "errno": errno.tolist(),
            "failed_environments": np.flatnonzero(np.bitwise_and(errno, int(ErrorCode.INVALID_STRESS_LOAD))).tolist(),
            "raw_capacity": capacity,
            "before_n_contacts": before["n_contacts"].tolist(),
            "before_link_force_norms": maxima,
            "unsupported_predicate_environments": np.flatnonzero(unsupported).tolist(),
            "description_before_rigid_integration": recovery.describe_load_failures(),
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    return result


with (
    patch.object(FrankaEgg, "step", step),
    patch.object(RigidStressRecovery, "recover", recover),
    patch.object(intrinsic, "kernel_fit_tractions", kernel_fit_tractions),
    patch.object(intrinsic, "kernel_commit_intrinsic_contacts", kernel_commit_intrinsic_contacts),
    patch.object(intrinsic, "kernel_scatter_tractions", kernel_scatter_tractions),
):
    runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__")
