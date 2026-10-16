"""Test shared immutable face bounds on the same complete native scatter."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import kernel_pack_contacts, kernel_scatter_warp
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path("genesis/engine/solvers/rigid/stress/contact.py").read_bytes()).hexdigest()
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    kernel_pack_contacts(entry.contacts)
    reference = lambda: kernel_scatter_warp(entry.contacts, entry.state, entry.model.info, entry.surface.info, False)
    candidate = lambda: kernel_scatter_warp(entry.contacts, entry.state, entry.model.info, entry.surface.info, True)
    reference()
    expected = qd_to_numpy(entry.state.force, copy=True)
    candidate()
    np.testing.assert_allclose(qd_to_numpy(entry.state.force), expected, rtol=2e-10, atol=1e-12)
    assert not qd_to_numpy(entry.contacts.status).any()
    times = {}
    for name, operation in (("computed", reference), ("cached", candidate)):
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        times[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "milliseconds": times,
        "contact_source_sha256": source_hash,
        "maximum_nodal_difference_N": float(abs(qd_to_numpy(entry.state.force) - expected).max()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
