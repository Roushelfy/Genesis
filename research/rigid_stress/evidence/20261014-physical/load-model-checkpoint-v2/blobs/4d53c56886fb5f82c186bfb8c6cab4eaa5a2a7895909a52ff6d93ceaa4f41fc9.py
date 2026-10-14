"""Evaluate one native graph for anchor, pressure, local retry and face scatter."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import kernel_anchor, kernel_pressure, kernel_scatter
from genesis.utils.misc import qd_to_numpy


def generate(directory):
    source = Path("genesis/engine/solvers/rigid/stress/contact.py").read_text()
    names = (
        "kernel_anchor",
        "kernel_pack_contacts",
        "kernel_pressure_warp",
        "kernel_pressure_correct_warp",
        "kernel_refine_contacts",
        "kernel_scatter_warp",
    )
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    blocks = [
        """import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressState, StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.contact import (StressContactState, func_force_frame, func_contact_sample,
    func_contact_cell, func_contact_index, func_pressure_initial, func_pressure_step)
from genesis.engine.solvers.rigid.stress.grid import func_patch_grid
"""
    ]
    for name in names:
        node = functions[name]
        node.name = name.replace("kernel_", "func_")
        node.decorator_list = [ast.Attribute(value=ast.Name(id="qd", ctx=ast.Load()), attr="func", ctx=ast.Load())]
        blocks.append(ast.unparse(node))
    blocks.append("""
@qd.kernel(graph=True)
def kernel_fused(source_epsilon: float, contact_state: StressContactState, stress_state: StressState,
                 stress_info: StressInfo, surface_info: StressSurfaceInfo):
    func_anchor(source_epsilon, contact_state, surface_info)
    func_pack_contacts(contact_state)
    func_pressure_warp(contact_state, surface_info, False)
    func_pressure_correct_warp(contact_state, surface_info)
    func_refine_contacts(contact_state)
    func_pressure_warp(contact_state, surface_info, True)
    func_pressure_correct_warp(contact_state, surface_info)
    func_scatter_warp(contact_state, stress_state, stress_info, surface_info, True)
""")
    destination = directory / "fused_contacts_generated.py"
    destination.write_text("\n\n".join(blocks) + "\n")
    spec = importlib.util.spec_from_file_location("fused_contacts_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_fused, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused, source_hash = generate(args.output.parent)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]

    def reference():
        kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info)
        kernel_pressure(entry.contacts, entry.surface.info)
        kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)

    def candidate():
        fused(recovery.source_epsilon, entry.contacts, entry.state, entry.model.info, entry.surface.info)

    reference()
    expected = qd_to_numpy(entry.state.force, copy=True)
    candidate()
    np.testing.assert_allclose(qd_to_numpy(entry.state.force), expected, rtol=2e-10, atol=1e-12)
    assert not qd_to_numpy(entry.contacts.status).any()
    times = {}
    for name, operation in (("separate", reference), ("combined", candidate)):
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
