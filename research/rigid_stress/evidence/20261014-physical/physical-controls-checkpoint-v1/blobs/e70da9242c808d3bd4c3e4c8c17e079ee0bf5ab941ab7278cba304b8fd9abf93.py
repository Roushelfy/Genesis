"""Matched same-snapshot graph entry experiment; every numerical op is Quadrants."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress import restore_frame
from genesis.engine.solvers.rigid.stress import pipeline, recovery as recovery_module
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    for _ in range(900):
        workload.step()
    qd.sync()
    restore_frame(workload)
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]
    kernels = {
        "func_wrapper": pipeline.kernel_pipeline,
        "direct_body": qd.kernel(graph=True)(pipeline.func_pipeline.fn),
    }
    fields = ("force", "rhs", "displacement", "residual", "step_peak", "step_valid")
    reference = None
    differences = {}
    rows = []
    order = ("func_wrapper", "direct_body") if args.seed == 510000 else ("direct_body", "func_wrapper")
    for route in order:
        recovery_module.kernel_pipeline = kernels[route]
        started = time.perf_counter()
        recovery.recover(0)
        qd.sync()
        preparation = time.perf_counter() - started
        workload.scene.rigid_solver.check_errno()
        observed = {name: qd_to_numpy(getattr(entry.state, name), copy=True) for name in fields}
        if reference is None:
            reference = observed
        else:
            for name in fields:
                if name == "step_valid":
                    np.testing.assert_array_equal(observed[name], reference[name])
                    differences[name] = 0.0
                else:
                    if name == "step_peak":
                        relative, absolute = 1e-4, 1e-3
                    elif name == "displacement":
                        relative, absolute = 1e-8, 1e-18
                    else:
                        relative, absolute = 1e-8, 1e-11
                    np.testing.assert_allclose(observed[name], reference[name], rtol=relative, atol=absolute)
                    differences[name] = float(np.max(np.abs(observed[name] - reference[name])))
        del observed
        repeats = []
        for _ in range(3):
            qd.sync()
            before = qd_to_numpy(entry.state.invalid_steps, copy=True)
            started = time.perf_counter()
            steps = 0
            while steps < 200 or time.perf_counter() - started < 10:
                recovery.recover(0)
                steps += 1
            qd.sync()
            seconds = time.perf_counter() - started
            failed = int((qd_to_numpy(entry.state.invalid_steps) - before).sum())
            workload.scene.rigid_solver.check_errno()
            assert failed == 0
            repeats.append(
                {
                    "steps": steps,
                    "seconds": seconds,
                    "valid_env_steps_per_second": args.envs * steps / seconds,
                    "invalid_environment_steps": failed,
                }
            )
        row = {"route": route, "preparation_seconds_including_JIT": preparation, "repeats": repeats}
        rows.append(row)
        print(route, [repeat["valid_env_steps_per_second"] for repeat in repeats], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "envs": args.envs,
                "seed": args.seed,
                "warmup_steps": 900,
                "rows": rows,
                "maximum_absolute_differences": differences,
                "passed": True,
                "gpu": torch.cuda.get_device_name(),
                "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION"),
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__), Path(pipeline.__file__), Path(recovery_module.__file__)]
                },
                "scope": "Three at-least-ten-second frozen actual pre-integration recovery repeats. Same public recover call, same physical GPU and original complete acceptance; only the Quadrants graph entry declaration differs. Raw body is the exact same Python function used by qd.func. This is not a live throughput measurement.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
