"""Explicit benchmark response to certified physical load failures; every failed step/reset is counted.

The native feature still raises on invalid loads. This driver checks every completed live/policy step before the
next observation. Only35/36continuous physical certificates terminate/reset the affected environment. Numerical,
sampled-unresolved,geometry,solver and other rigid failures remain fatal. Original forces/radii/friction are untouched.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import runpy

from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_to_numpy, tensor_to_array

original_step, original_restart = FrankaEgg.step, FrankaEgg.restart
output = Path(sys.argv[sys.argv.index("--output") + 1])
scope = sys.argv[sys.argv.index("--scope") + 1]
records, group = [], "warmup"
repeat = 0


def restart(workload):
    global repeat, group
    result = original_restart(workload)
    repeat += 1
    group = f"repetition-{repeat}"
    return result


def step(workload, residual_action=None):
    result = original_step(workload, residual_action)
    if workload.stress:
        solver = workload.scene.rigid_solver
        errors = qd_to_numpy(solver._errno, transpose=True, copy=True)
        ids = np.flatnonzero(errors)
        if len(ids):
            entry = solver.stress_recovery.links[0]
            failure = entry.intrinsic.failure
            status = qd_to_numpy(failure.status, ids, transpose=True, copy=True)
            record = {"group": group, "tick": workload.tick - 1, "environments": ids.tolist(),
                      "errno": errors[ids].tolist(), "status": status.tolist(),
                      "description": solver.stress_recovery.describe_load_failures(),
                      "response": "fatal-unresolved-or-other-error"}
            for name in ("contact", "radius", "friction", "position", "anchor", "force", "free_moment",
                         "group_count", "target_force", "target_moment"):
                record[name] = qd_to_numpy(getattr(failure, name), ids, transpose=True, copy=True).tolist()
            records.append(record)
            if np.any(errors[ids] != int(ErrorCode.INVALID_STRESS_LOAD)) or np.any(~np.isin(status, (35, 36))):
                solver.check_errno()
                raise RuntimeError("Unexpected unresolved error record; benchmark cannot reset it.")
            peers = np.array([i for i in (0, workload.n_envs - 1) if i not in ids], dtype=int)
            if len(peers):
                before_pos = tensor_to_array(workload.egg.get_pos(envs_idx=peers)).copy()
                before_vel = tensor_to_array(workload.egg.get_vel(envs_idx=peers)).copy()
                before_peak = tensor_to_array(workload.link.get_max_stress(envs_idx=peers)).copy()
            workload.reset(ids)
            solver.check_errno()
            if len(peers):
                np.testing.assert_array_equal(tensor_to_array(workload.egg.get_pos(envs_idx=peers)), before_pos)
                np.testing.assert_array_equal(tensor_to_array(workload.egg.get_vel(envs_idx=peers)), before_vel)
                np.testing.assert_array_equal(tensor_to_array(workload.link.get_max_stress(envs_idx=peers)), before_peak)
            assert np.all(tensor_to_array(workload.link.get_max_stress(envs_idx=ids)) == 0)
            record["response"] = "count-invalid-step-and-reset-only-certified-environments"
            record["peer_isolation_verified"] = peers.tolist()
    return result


try:
    # Construction calls restart before warmup,so only the benchmark's explicit repeat restarts advance the labels.
    with patch.object(FrankaEgg, "step", step):
        import examples.speed_benchmark.rigid_stress as benchmark
        original_init = FrankaEgg.__init__

        def initialize(workload, *args, **kwargs):
            result = original_init(workload, *args, **kwargs)
            global repeat, group
            repeat, group = 0, "warmup"
            return result

        with patch.object(FrankaEgg, "restart", restart), patch.object(FrankaEgg, "__init__", initialize):
            benchmark.main()
finally:
    report = {"scope": __doc__, "benchmark_scope": scope,
              "poststep_errno_check": "Every live/policy step uses a bulk device-to-CPU error read inside timing.",
              "failed_environment_steps": sum(len(row["environments"]) for row in records), "events": records,
              "limitations": "These are explicit physically incompatible episode terminations,not successful stress recoveries. All original loads are attempted first. Native final acceptance/error semantics are unchanged. Driver checks/capture/reset/peer-isolation overhead is included in live/policy timing. Numerical or continuum-unresolved failures cannot be reset by this driver. No grasp-success metric or complete RL training is claimed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "certified-failure-responses.json").write_text(json.dumps(report, indent=2) + "\n")
    if output.exists():
        result = json.loads(output.read_text())
        result["certified_failure_rollout"] = report
        for i, row in enumerate(result.get("repeats", []), 1):
            expected = sum(len(event["environments"]) for event in records if event["group"] == f"repetition-{i}")
            row["certified_physical_failed_environment_steps"] = expected
            if scope in ("live", "policy"):
                assert row["invalid_environment_steps"] == expected
        output.write_text(json.dumps(result, indent=2) + "\n")
