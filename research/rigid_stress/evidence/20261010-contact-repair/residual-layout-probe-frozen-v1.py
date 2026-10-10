"""Measure complete FP64 residual traversal with contiguous environment tiles."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import pipeline, solve
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle


def generate():
    source = Path("genesis/engine/solvers/rigid/stress/solve.py").read_text()
    function = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "func_full_residual"
    )
    function.name = "func_residual_tiled"
    for index, statement in enumerate(function.body):
        if isinstance(statement, ast.For) and isinstance(statement.target, ast.Tuple):
            loop = ast.parse(
                "for i_thread in range(stress_info.vertices.shape[0] * ((stress_state.active.shape[0] + 31) // 32) * 32):\n"
                "    i_index = i_thread // 32\n"
                "    i_n = i_index % stress_info.vertices.shape[0]\n"
                "    i_b = (i_index // stress_info.vertices.shape[0]) * 32 + i_thread % 32\n"
                "    if i_b < stress_state.active.shape[0]:\n"
                "        pass\n"
            ).body[0]
            loop.body[-1].body = statement.body
            function.body[index : index + 1] = [ast.parse("qd.loop_config(block_dim=256)").body[0], loop]
            break
    generated = (
        "from genesis.engine.solvers.rigid.stress.solve import *\n\n"
        + ast.unparse(ast.fix_missing_locations(function))
        + "\n\n"
        + """@qd.kernel(graph=True)
def kernel_residual_tiled(
    young: float, tolerance: float, absolute_tolerance: float,
    stress_state: StressState, stress_info: StressInfo, only_active: qd.template(),
):
    func_residual_tiled(young, tolerance, absolute_tolerance, stress_state, stress_info, only_active)
"""
    )
    cache = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache" / "residual_layout_trials" / os.environ["SLURM_JOB_ID"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "residual_layout_generated.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("residual_layout_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, generated


def micro(module, remaining):
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(remaining)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    for _ in range(args.warmup):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    info, state, options = entry.model.info, entry.state, entry.link.stress_options
    arguments = options.young, options.tolerance, options.absolute_tolerance, state, info, False
    solve.kernel_full_residual(*arguments)
    expected = qd_to_numpy(state.residual, copy=True)
    expected_valid = qd_to_numpy(state.valid, copy=True)
    rows = []
    for name, operation in (
        ("original", lambda: solve.kernel_full_residual(*arguments)),
        ("env32_node8", lambda: module.kernel_residual_tiled(*arguments)),
    ):
        operation()
        np.testing.assert_array_equal(qd_to_numpy(state.residual), expected)
        np.testing.assert_array_equal(qd_to_numpy(state.valid), expected_valid)
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
            "all_residual_rows_bit_equal": True,
            "all_environments_valid": bool(expected_valid.all()),
            "complete_residual_budget_ratio_max": float(
                (
                    np.sqrt(qd_to_numpy(state.residual_norm_squared))
                    / np.maximum(
                        options.absolute_tolerance,
                        options.tolerance * np.sqrt(qd_to_numpy(state.rhs_norm_squared)),
                    )
                ).max()
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    module, generated = generate()
    if args.micro:
        micro(module, remaining)
    else:
        if args.tiled:
            pipeline.func_full_residual = module.func_residual_tiled
            solve.func_full_residual = module.func_residual_tiled
        sys.argv = ["residual_layout_override", *remaining]
        if args.oracle:
            oracle.main()
        else:
            benchmark.main()
    output.output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "tiled": args.tiled,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "note": "Only node/environment traversal changes. Every matrix row, gauge row, FP64 operation order within a row, acceptance budget, contact and stress sample is retained.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
