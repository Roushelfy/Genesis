"""Measure conservative native face reduction for an actual contact snapshot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import (
    kernel_scatter,
    kernel_scatter_serial,
)
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(args.steps):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    kernel_scatter_serial(entry.contacts, entry.state, entry.model.info, entry.surface.info)
    expected = qd_to_numpy(entry.state.force, copy=True)
    kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)
    actual = qd_to_numpy(entry.state.force, copy=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-12)
    assert not qd_to_numpy(entry.contacts.status).any()
    times = {}
    for name, operation in (("serial", kernel_scatter_serial), ("face", kernel_scatter)):
        operation(entry.contacts, entry.state, entry.model.info, entry.surface.info)
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation(entry.contacts, entry.state, entry.model.info, entry.surface.info)
        qd.sync()
        times[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "scatter_ms": times,
        "maximum_nodal_force_difference_N": float(abs(actual - expected).max()),
        "maximum_wrench_force_error_N": float(qd_to_numpy(entry.contacts.force_error).max()),
        "maximum_wrench_moment_error_Nm": float(qd_to_numpy(entry.contacts.moment_error).max()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
