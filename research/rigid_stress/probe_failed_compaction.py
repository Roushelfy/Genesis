"""Evaluate exact device compaction of failed inverse/residual environments."""

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
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


def generate(directory):
    root = Path("genesis/engine/solvers/rigid/stress")
    blocks = [
        """import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressState, StressInfo
from genesis.engine.solvers.rigid.stress.inverse import StressInverseInfo
"""
    ]
    hashes = {}
    for filename, name, predicate in (
        ("inverse.py", "kernel_apply_inverse", "not stress_state.valid[i_b]"),
        ("solve.py", "kernel_full_residual", "stress_state.active[i_b]"),
    ):
        source = (root / filename).read_text()
        hashes[filename] = hashlib.sha256(source.encode()).hexdigest()
        node = next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name)
        node.name += "_packed"
        node.args.args.extend(
            [
                ast.arg(arg="selected", annotation=ast.parse("qd.Tensor", mode="eval").body),
                ast.arg(arg="count", annotation=ast.parse("qd.Tensor", mode="eval").body),
            ]
        )
        prefix = ast.parse(f"""
for _ in range(1):
    count[None] = 0
for i_b in range(stress_state.active.shape[0]):
    if {predicate}:
        slot = qd.atomic_add(count[None], 1)
        selected[slot] = i_b
""").body
        for loop in node.body:
            if (
                isinstance(loop, ast.For)
                and isinstance(loop.iter, ast.Call)
                and isinstance(loop.iter.func, ast.Attribute)
                and loop.iter.func.attr == "ndrange"
            ):
                loop.target = ast.Name(id="packed_row", ctx=ast.Store())
                loop.iter = ast.parse("range(stress_state.rhs.shape[0] * count[None])", mode="eval").body
                loop.body = (
                    ast.parse("i_n = packed_row // count[None]\ni_b = selected[packed_row % count[None]]").body
                    + loop.body
                )
        node.body = prefix + node.body
        blocks.append(ast.unparse(ast.fix_missing_locations(node)))
    destination = directory / "failed_compaction_generated.py"
    destination.write_text("\n\n".join(blocks) + "\n")
    spec = importlib.util.spec_from_file_location("failed_compaction_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_apply_inverse_packed, module.kernel_full_residual_packed, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=32768)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inverse, residual, hashes = generate(args.output.parent)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    model, state = entry.model, entry.state
    selected = V(dtype=gs.qd_int, shape=(args.envs,))
    count = V(dtype=gs.qd_int, shape=())

    def native():
        model.inverse.apply(model.options.young, state, correction=True)
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active=True,
        )

    def packed():
        inverse(model.options.young, state, model.inverse.info, True, False, selected, count)
        residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            True,
            selected,
            count,
        )

    # Include empty, isolated and dense failed sets. Perturb displacement,
    # independently recompute full residual, then compare both correction paths.
    accepted = qd_to_numpy(state.displacement, copy=True)
    rows = []
    for failed in (0, 1, min(32, args.envs), args.envs):
        perturbed = accepted.copy()
        perturbed[:, :failed] *= 1.01
        state.displacement.from_numpy(perturbed)
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active=False,
        )
        expected_failed = qd_to_numpy(state.valid, copy=True)
        native()
        expected = qd_to_numpy(state.displacement, copy=True)
        state.displacement.from_numpy(perturbed)
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active=False,
        )
        np.testing.assert_array_equal(qd_to_numpy(state.valid), expected_failed)
        packed()
        np.testing.assert_array_equal(qd_to_numpy(state.displacement), expected)
        assert qd_to_numpy(state.valid).all()
        rows.append(
            {
                "perturbed_envs": failed,
                "failed_before_correction": int((~expected_failed).sum()),
                "full_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
            }
        )
    timings = {}
    for name, operation in (("masked", native), ("packed", packed)):
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        timings[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "zero_failure_correction_and_residual_ms": timings,
        "source_sha256": hashes,
        "numerical_cases": rows,
        "note": "Includes device packing; no host failure count read or dropped environment.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
