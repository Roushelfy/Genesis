"""Untimed native policy diagnosis; preserve one failed environment before rigid integration."""

import json
import math
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import ErrorCode, V, V_MAT, V_VEC, iter_data
from genesis.utils.misc import qd_to_numpy

destination = Path(sys.argv[sys.argv.index("--output") + 1]).parent / "failure"
original_recover, original_step = RigidStressRecovery.recover, FrankaEgg.step
current_workload = None
calls, captured = 0, False


@qd.kernel
def kernel_copy_environment(source: qd.template(), target: qd.template(), environment: int):
    for index in qd.grouped(target):
        original = qd.Vector.zero(gs.qd_int, len(source.shape))
        for dimension in qd.static(range(len(source.shape))):
            original[dimension] = index[dimension]
        original[len(source.shape) - 1] = environment
        target[index] = source[original]


def snapshot(value, environment, n_envs):
    if math.prod(value.shape) == 0:
        return None
    if value.shape and value.shape[-1] == n_envs:
        shape = (*value.shape[:-1], 1)
        if isinstance(value, qd.VectorTensor):
            target = V_VEC(*value.element_shape, dtype=value.dtype, shape=shape)
        elif isinstance(value, qd.MatrixTensor):
            target = V_MAT(*value.element_shape, dtype=value.dtype, shape=shape)
        else:
            target = V(dtype=value.dtype, shape=shape)
        kernel_copy_environment(value, target, environment)
        return qd_to_numpy(target, transpose=True, copy=True)
    return qd_to_numpy(value, transpose=True, copy=True)


def step(workload, residual_action=None):
    global current_workload, calls
    current_workload = workload
    calls += 1
    return original_step(workload, residual_action)


def recover(recovery, i_substep):
    global captured
    result = original_recover(recovery, i_substep)
    solver, entry = recovery.solver, recovery.links[0]
    errno = qd_to_numpy(solver._errno, copy=True)
    failed = np.flatnonzero(np.bitwise_and(errno, int(ErrorCode.INVALID_STRESS_LOAD)))
    if not captured and failed.size:
        captured = True
        destination.mkdir(parents=True, exist_ok=False)
        environment, n_envs = int(failed[0]), errno.size
        values = {"errno": errno, "environment": np.array(environment)}
        for name, state in (
            ("contacts", entry.contacts), ("state", entry.state),
            ("raw_sources", entry.intrinsic.raw_sources), ("fit_contacts", entry.intrinsic.contacts),
            ("fit_sources", entry.intrinsic.sources), ("tractions", entry.intrinsic.tractions),
            ("footprints", entry.intrinsic.footprints), ("packing", entry.intrinsic.packing),
            ("certificates", entry.intrinsic.certificates), ("wrench_search", entry.intrinsic.wrench_search),
        ):
            for item in iter_data(state, name):
                array = snapshot(item.value, environment, n_envs)
                if array is not None:
                    values[item.name] = array
        before = {}
        unsupported = np.zeros(n_envs, dtype=bool)
        for name in ("cfrc_applied_ang", "cfrc_applied_vel", "cfrc_coupling_ang", "cfrc_coupling_vel"):
            array = qd_to_numpy(getattr(solver.dyn_state.links, name), transpose=True, copy=True)[:, entry.link.idx]
            norms = np.linalg.norm(array, axis=-1)
            before[name] = norms.tolist()
            unsupported |= norms > 0
            values["before." + name] = array[environment]
        count = qd_to_numpy(solver.collider.collider_state.n_contacts, copy=True)
        unsupported |= count > entry.contacts.valid.shape[0]
        values["before.n_contacts"] = count
        np.savez_compressed(destination / "before-rigid-integration.npz", **values)
        metadata = {"scope": __doc__, "benchmark_command": sys.argv, "step_call": calls,
                    "workload_tick_before_step": current_workload.tick, "substep": i_substep,
                    "selected_environment": environment, "failed_environments": failed.tolist(),
                    "before_link_force_norms": before, "before_n_contacts": count.tolist(),
                    "unsupported_predicate_environments": np.flatnonzero(unsupported).tolist(),
                    "description_before_rigid_integration": recovery.describe_load_failures()}
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    return result


with patch.object(FrankaEgg, "step", step), patch.object(RigidStressRecovery, "recover", recover):
    runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__")
