"""Intrusive changing-contact SCR trace with named native dispatch/readback scopes."""

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress import intrinsic
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


ORIGINAL_SCATTER = IntrinsicContacts.scatter
ORIGINAL_GROW = IntrinsicContacts._grow_sample_profiles


def traced_scatter(self, *args, **kwargs):
    with torch.profiler.record_function("SCR.contact_pipeline"):
        return ORIGINAL_SCATTER(self, *args, **kwargs)


def traced_grow(self):
    with torch.profiler.record_function("SCR.capacity_check_and_growth"):
        return ORIGINAL_GROW(self)


def traced_readback(*args, **kwargs):
    with torch.profiler.record_function("SCR.device_scalar_readback"):
        return qd_to_numpy(*args, **kwargs)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--envs", type=int, default=8)
    cli.add_argument("--warmup", type=int, default=900)
    cli.add_argument("--steps", type=int, default=60)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "conditions": file_id(args.conditions),
            "steps": args.steps,
            "envs": args.envs,
        },
    )
    gs.init(backend=gs.gpu, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        level=1,
        seed=623016,
        varied=True,
        conditions=args.conditions,
        contact_model="intrinsic",
        intrinsic_load_partition="body_source_prior",
        history_size=0,
        output_mode="max",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    with torch.inference_mode():
        for _ in range(args.warmup):
            workload.step()
        qd.sync()
        solver.check_errno()
        invalid_before = qd_to_numpy(entry.state.invalid_steps, copy=True)
        with (
            patch.object(IntrinsicContacts, "scatter", traced_scatter),
            patch.object(IntrinsicContacts, "_grow_sample_profiles", traced_grow),
            patch.object(intrinsic, "qd_to_numpy", traced_readback),
        ):
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                record_shapes=False,
                with_stack=False,
            ) as profiler:
                for _ in range(args.steps):
                    with torch.profiler.record_function("native_scene_step"):
                        workload.step()
                qd.sync()
        solver.check_errno()
        invalid = qd_to_numpy(entry.state.invalid_steps) - invalid_before
        assert (invalid == 0).all()
        peak = workload.link.get_max_stress()
        assert torch.isfinite(peak).all() and (peak > 0).any()
    trace = args.output / "trace.json"
    profiler.export_chrome_trace(str(trace))
    # This summary describes inclusive host scopes. GPU work can overlap and
    # none of these observer timings replaces primary production measurements.
    events = profiler.events()
    scopes = []
    for name in (
        "native_scene_step",
        "SCR.contact_pipeline",
        "SCR.capacity_check_and_growth",
        "SCR.device_scalar_readback",
    ):
        selected = [e for e in events if e.name == name]
        scopes.append(
            {
                "name": name,
                "calls": len(selected),
                "inclusive_host_us": sum(e.cpu_time_total for e in selected),
                "host_us_per_scene_step": sum(e.cpu_time_total for e in selected) / args.steps,
            }
        )
    names = Counter(e.name for e in events if "memcpy" in e.name.lower() or "synchron" in e.name.lower())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": True,
            "envs": args.envs,
            "steps": args.steps,
            "warmup": args.warmup,
            "invalid_environment_steps": int(invalid.sum()),
            "gs_use_ndarray": gs.use_ndarray,
            "gs_use_deterministic_algorithms": gs.use_deterministic_algorithms,
            "rigid_config": {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in vars(solver.rigid_config).items()
            },
            "inclusive_host_scopes": scopes,
            "copy_and_sync_event_counts": dict(names),
            "trace": file_id(trace),
            "conditions": file_id(args.conditions),
            "scope": "Intrusive changing-contact trace. Existing method calls are temporarily wrapped in named observer scopes; no engine source or numerical acceptance change. Inclusive host times contain nested readbacks and may overlap GPU work. No primary throughput, physical accuracy or optimization claim.",
        },
    )


if __name__ == "__main__":
    main()
