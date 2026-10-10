"""Evaluate shared Quadrants inertia-relief projection on the unchanged native workload."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import solve
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import V_MAT
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle


def generate():
    path = Path("genesis/engine/solvers/rigid/stress/solve.py")
    source = path.read_text()
    function = next(
        item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == "func_balance"
    )
    function.name = "func_cached_balance"
    function.args.args.append(ast.arg(arg="relief", annotation=ast.parse("qd.Tensor", mode="eval").body))
    function.body[-1].body = ast.parse("stress_state.rhs[i_n, i_b] -= relief[i_n] @ stress_state.wrench[i_b]").body
    pipeline_kernel = next(
        item
        for item in ast.parse(path.with_name("pipeline.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_pipeline"
    )
    pipeline_kernel.name = "kernel_relief_pipeline"
    pipeline_kernel.args.args.append(ast.arg(arg="relief", annotation=ast.parse("qd.Tensor", mode="eval").body))
    for statement in pipeline_kernel.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "func_balance"
        ):
            statement.value.func.id = "func_cached_balance"
            statement.value.args.append(ast.Name(id="relief", ctx=ast.Load()))
    recovery_class = next(
        item
        for item in ast.parse(path.with_name("recovery.py").read_text()).body
        if isinstance(item, ast.ClassDef) and item.name == "RigidStressRecovery"
    )
    recover = next(item for item in recovery_class.body if isinstance(item, ast.FunctionDef) and item.name == "recover")

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "kernel_pipeline":
                node.func.id = "kernel_relief_pipeline"
                node.args.append(ast.parse("workspaces[id(entry.state)]", mode="eval").body)
            return node

    recover = Calls().visit(recover)
    generated = (
        "from genesis.engine.solvers.rigid.stress.solve import *\n"
        "from genesis.engine.solvers.rigid.stress.pipeline import *\n"
        "from genesis.engine.solvers.rigid.stress.recovery import *\n"
        "shared_relief = None\nworkspaces = {}\n\n"
        + ast.unparse(ast.fix_missing_locations(function))
        + "\n\n"
        + """@qd.kernel
def kernel_build_relief(stress_info: StressInfo, relief: qd.Tensor):
    for i_n in range(stress_info.vertices.shape[0]):
        relief[i_n] = stress_info.mass_modes[i_n] @ stress_info.gram_inverse[None]

@qd.kernel(graph=True)
def kernel_cached_balance(omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo, relief: qd.Tensor):
    func_cached_balance(omega, stress_state, stress_info, relief)
"""
        + ast.unparse(ast.fix_missing_locations(pipeline_kernel))
        + "\n\n"
        + ast.unparse(ast.fix_missing_locations(recover))
        + "\n"
    )
    directory = (
        Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache" / "relief_trials" / os.environ.get("SLURM_JOB_ID", "local")
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "relief_cache_generated.py"
    destination.write_text(generated)
    spec = importlib.util.spec_from_file_location("relief_cache_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source, generated


def create_cache(module, recovery):
    assert len(recovery.links) == 1, "This research override measures one configured stress link."
    info = recovery.links[0].model.info
    module.shared_relief = V_MAT(3, 6, dtype=gs.qd_float, shape=(info.vertices.shape[0],))
    module.kernel_build_relief(info, module.shared_relief)
    module.workspaces[id(recovery.links[0].state)] = module.shared_relief


def micro(module, remaining):
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(remaining)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True, seed=args.seed)
    for _ in range(args.warmup):
        workload.step()
    recovery = workload.scene.rigid_solver.stress_recovery
    create_cache(module, recovery)
    entry = recovery.links[0]
    info, state, options = entry.model.info, entry.state, entry.link.stress_options
    solve.kernel_balance(entry.omega, state, info)
    expected_rhs = qd_to_numpy(state.rhs, copy=True)
    entry.model.solve(options, state, entry.omega, surface_load=True)
    solve.kernel_peak(options.young, options.poisson, state, info, options.cached_peak)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    rows = []
    for name, operation in (
        ("native", lambda: solve.kernel_balance(entry.omega, state, info)),
        ("cached", lambda: module.kernel_cached_balance(entry.omega, state, info, module.shared_relief)),
    ):
        operation()
        actual_rhs = qd_to_numpy(state.rhs, copy=True)
        difference = np.linalg.norm((actual_rhs - expected_rhs).reshape(-1, args.envs, 3), axis=(0, 2))
        budgets = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(expected_rhs, axis=(0, 2)))
        assert (difference <= budgets).all(), (name, difference.max())
        solve.kernel_direct_init(state)
        entry.model.solve(options, state, entry.omega, surface_load=True)
        solve.kernel_full_residual(options.young, options.tolerance, options.absolute_tolerance, state, info, False)
        assert qd_to_numpy(state.valid).all(), name
        solve.kernel_peak(options.young, options.poisson, state, info, options.cached_peak)
        peak = qd_to_numpy(state.peak, copy=True)
        np.testing.assert_allclose(peak, expected_peak, rtol=1e-4, atol=1e-3)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
            "rhs_difference_budget_ratio_max": float((difference / budgets).max()),
            "complete_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
            "global_peak_difference_max_Pa": float(abs(peak - expected_peak).max()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-relief", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--micro", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    module, original, generated = generate()
    if args.micro:
        micro(module, remaining)
    else:
        if args.cached_relief:
            original_init = RigidStressRecovery.__init__

            def initialize(self, *init_args, **kwargs):
                original_init(self, *init_args, **kwargs)
                create_cache(module, self)

            RigidStressRecovery.__init__ = initialize
            RigidStressRecovery.recover = module.recover
            benchmark.kernel_balance = lambda omega, state, info: module.kernel_cached_balance(
                omega, state, info, module.workspaces[id(state)]
            )
        sys.argv = ["native_trajectory_oracle" if args.oracle else "rigid_stress_benchmark", *remaining]
        (oracle.main if args.oracle else benchmark.main)()
    metadata = {
        "command": command,
        "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("genesis/engine/solvers/rigid/stress").glob("*.py"))
        },
        "cached_relief": args.cached_relief or args.micro,
        "oracle": args.oracle,
        "micro": args.micro,
        "original_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "generated_source": generated,
        "shared_projection_bytes": 18 * 8 * module.shared_relief.shape[0] if module.shared_relief is not None else 0,
        "note": "Research shared projection built in Quadrants from the declared mass modes and rigid-mode Gram inverse. Contact law, samples, precision and final acceptance unchanged. Independent CPU oracle and actual rollout retention are separate gates.",
    }
    output.output.with_suffix(".variant.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
