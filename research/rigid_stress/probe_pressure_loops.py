"""Compare bounded nested Newton/line loops without changing the native pressure evaluator."""

import argparse
import ast
import copy
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
from genesis.engine.solvers.rigid.stress.contact import (
    kernel_anchor,
    kernel_pack_contacts,
    kernel_pressure_correct_warp,
    kernel_pressure_warp,
    kernel_refine_contacts,
    kernel_scatter,
)
from genesis.utils.misc import qd_to_numpy


def generate(directory):
    source = Path("genesis/engine/solvers/rigid/stress/contact.py").read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_pressure_correct_warp"
    )
    node.name = "kernel_nested"
    original = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.For)
        and isinstance(item.iter, ast.Call)
        and item.iter.args
        and isinstance(item.iter.args[0], ast.BinOp)
        and isinstance(item.iter.args[0].left, ast.Constant)
        and item.iter.args[0].left.value == 80
    )
    evaluator = original.body[0].body
    replacement = ast.parse("""for iteration in range(80):
    if not accepted:
        pass
        for line_pass in range(40):
            if line:
                pass
""").body[0]
    replacement.body[0].body[:1] = copy.deepcopy(evaluator)
    replacement.body[0].body[-1].body[0].body = copy.deepcopy(evaluator)
    # Mutate the existing loop node so its containing block needs no reconstruction.
    original.target, original.iter, original.body = replacement.target, replacement.iter, replacement.body
    header = """import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (StressContactState, func_force_frame,
    func_contact_cell, func_contact_index, func_contact_sample, func_pressure_step)
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.grid import func_patch_grid
"""
    destination = directory / "pressure_loops_generated.py"
    destination.write_text(header + ast.unparse(node) + "\n")
    spec = importlib.util.spec_from_file_location("pressure_loops_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_nested, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nested, source_hash = generate(args.output.parent)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]

    def operation(correct):
        kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info)
        kernel_pack_contacts(entry.contacts)
        kernel_pressure_warp(entry.contacts, entry.surface.info, False)
        correct(entry.contacts, entry.surface.info)
        kernel_refine_contacts(entry.contacts)
        kernel_pressure_warp(entry.contacts, entry.surface.info, True)
        correct(entry.contacts, entry.surface.info)

    operation(kernel_pressure_correct_warp)
    kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)
    expected = qd_to_numpy(entry.state.force, copy=True)
    operation(nested)
    kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)
    actual = qd_to_numpy(entry.state.force, copy=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-12)
    assert not qd_to_numpy(entry.contacts.status).any()
    times = {}
    for name, correct in (("unified", kernel_pressure_correct_warp), ("nested", nested)):
        operation(correct)
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation(correct)
        qd.sync()
        times[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "milliseconds": times,
        "contact_source_sha256": source_hash,
        "maximum_nodal_difference_N": float(abs(actual - expected).max()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
