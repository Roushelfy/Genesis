"""Evaluate complete integrated contact-wrench reuse for native inertia relief."""

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
from genesis.engine.solvers.rigid.stress import solve
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import V_MAT
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle


def generate():
    directory = Path("genesis/engine/solvers/rigid/stress")
    node_balance = next(
        item
        for item in ast.parse((directory / "balance.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "func_balance_cooperative"
    )
    node_balance.name = "func_balance_fallback"
    node_balance.args.args.append(ast.arg(arg="enabled", annotation=ast.parse("bool", mode="eval").body))
    for statement in node_balance.body:
        if isinstance(statement, ast.For):
            count = statement.iter.args[-1]
            statement.iter.args[-1] = ast.parse("qd.select(enabled, 0, 0)", mode="eval").body
            statement.iter.args[-1].args[1] = count
    fused = next(
        item
        for item in ast.parse((directory / "pipeline.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_pipeline"
    )
    fused.name = "kernel_reuse_pipeline"
    fused.args.args.append(ast.arg(arg="centrifugal_wrench", annotation=ast.parse("qd.Tensor", mode="eval").body))
    for index, statement in enumerate(fused.body):
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "func_balance"
        ):
            fused.body[index] = ast.parse(
                "if qd.static(face_parallel and is_cooperative_balance):\n"
                "    func_balance_reuse(omega, state, info, contacts, scatter, centrifugal_wrench)\n"
                "else:\n"
                "    func_balance(omega, state, info, is_cooperative_balance)\n"
            ).body[0]
    recovery = next(
        item
        for item in ast.parse((directory / "recovery.py").read_text()).body
        if isinstance(item, ast.ClassDef) and item.name == "RigidStressRecovery"
    )
    recover = next(item for item in recovery.body if isinstance(item, ast.FunctionDef) and item.name == "recover")

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "kernel_pipeline":
                node.func.id = "kernel_reuse_pipeline"
                node.args.append(ast.parse("workspaces[id(entry.state)]", mode="eval").body)
            return node

    recover = Calls().visit(recover)
    generated = (
        "from genesis.engine.solvers.rigid.stress.pipeline import *\n"
        "from genesis.engine.solvers.rigid.stress.recovery import *\n"
        "from genesis.engine.solvers.rigid.stress.balance import *\n"
        "workspaces = {}\n\n"
        + ast.unparse(ast.fix_missing_locations(node_balance))
        + "\n\n"
        + """@qd.kernel
def kernel_centrifugal_wrench(info: StressInfo, centrifugal_wrench: qd.Tensor):
    for i in range(1):
        centrifugal_wrench[None] = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i_n in range(info.vertices.shape[0]):
        value = info.modes[i_n].transpose() @ info.centrifugal[i_n]
        for a, b in qd.static(qd.ndrange(6, 6)):
            qd.atomic_add(centrifugal_wrench[None][a, b], value[a, b])

@qd.func
def func_balance_reuse(
    omega: qd.Tensor, state: StressState, info: StressInfo,
    contacts: StressContactState, scatter: StressScatterWorkspace,
    centrifugal_wrench: qd.Tensor,
):
    admitted = scatter.count[None] <= scatter.tasks.shape[0]
    for i_b in range(qd.select(admitted, state.active.shape[0], 0)):
        w = omega[i_b]
        terms = qd.Vector([w[0]*w[0], w[1]*w[1], w[2]*w[2], w[0]*w[1], w[0]*w[2], w[1]*w[2]])
        state.wrench[i_b] = centrifugal_wrench[None] @ terms
    for i_pair in range(qd.select(admitted, contacts.active_count[None], 0)):
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            accumulated = scatter.wrench[i_pair]
            force = qd.Vector([accumulated[a] for a in qd.static(range(3))])
            moment = qd.Vector([accumulated[a+3] for a in qd.static(range(3))])
            com = qd.Vector([info.mass_properties[a + 1] for a in qd.static(range(3))]) / info.mass_properties[0]
            moment += (contacts.position[i_c, i_b] - com).cross(force)
            for a in qd.static(range(3)):
                qd.atomic_add(state.wrench[i_b][a], force[a])
                qd.atomic_add(state.wrench[i_b][a+3], moment[a])
    for i_n, i_b in qd.ndrange(info.vertices.shape[0], qd.select(admitted, state.active.shape[0], 0)):
        w = omega[i_b]
        terms = qd.Vector([w[0]*w[0], w[1]*w[1], w[2]*w[2], w[0]*w[1], w[0]*w[2], w[1]*w[2]])
        state.rhs[i_n, i_b] = state.force[i_n, i_b] + info.centrifugal[i_n] @ terms - info.relief[i_n] @ state.wrench[i_b]
    func_balance_fallback(omega, state, info, not admitted)

@qd.kernel(graph=True)
def kernel_balance_reuse(
    omega: qd.Tensor, state: StressState, info: StressInfo,
    contacts: StressContactState, scatter: StressScatterWorkspace,
    centrifugal_wrench: qd.Tensor,
):
    func_balance_reuse(omega, state, info, contacts, scatter, centrifugal_wrench)
"""
        + ast.unparse(ast.fix_missing_locations(fused))
        + "\n\n"
        + ast.unparse(ast.fix_missing_locations(recover))
        + "\n"
    )
    cache = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache" / "wrench_reuse_trials" / os.environ["SLURM_JOB_ID"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "wrench_reuse_generated.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("wrench_reuse_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, generated


def create_cache(module, recovery):
    assert len(recovery.links) == 1, "The research comparison configures one stressed link."
    entry = recovery.links[0]
    workspace = V_MAT(6, 6, dtype=gs.qd_float, shape=())
    module.kernel_centrifugal_wrench(entry.model.info, workspace)
    module.workspaces[id(entry.state)] = workspace
    return workspace


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
    workspace = create_cache(module, workload.scene.rigid_solver.stress_recovery)
    solve.kernel_balance(entry.omega, state, info, True)
    expected_rhs = qd_to_numpy(state.rhs, copy=True)
    expected_wrench = qd_to_numpy(state.wrench, copy=True)
    entry.model.solve(options, state, entry.omega, surface_load=True)
    solve.kernel_peak(options.young, options.poisson, state, info, options.cached_peak)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    rows = []
    for name, operation in (
        ("native_node_tiles", lambda: solve.kernel_balance(entry.omega, state, info, True)),
        (
            "integrated_wrench",
            lambda: module.kernel_balance_reuse(entry.omega, state, info, entry.contacts, entry.scatter, workspace),
        ),
    ):
        operation()
        actual_rhs = qd_to_numpy(state.rhs, copy=True)
        difference = np.linalg.norm(actual_rhs - expected_rhs, axis=(0, 2))
        budgets = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(expected_rhs, axis=(0, 2)))
        assert (difference <= budgets).all(), (name, difference.max(), (difference / budgets).max())
        wrench_difference = np.abs(qd_to_numpy(state.wrench) - expected_wrench).max()
        solve.kernel_direct_init(state)
        entry.model.solve(options, state, entry.omega, surface_load=True)
        solve.kernel_full_residual(options.young, options.tolerance, options.absolute_tolerance, state, info, False)
        assert qd_to_numpy(state.valid).all(), name
        solve.kernel_peak(options.young, options.poisson, state, info, options.cached_peak)
        np.testing.assert_allclose(qd_to_numpy(state.peak), expected_peak, rtol=1e-4, atol=1e-3)
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
            "complete_wrench_difference_max": float(wrench_difference),
            "complete_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-wrench", action="store_true")
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
        original = RigidStressRecovery.__init__

        def initialize(recovery, *call_args, **kwargs):
            original(recovery, *call_args, **kwargs)
            create_cache(module, recovery)

        if args.reuse_wrench:
            RigidStressRecovery.__init__ = initialize
            RigidStressRecovery.recover = module.recover
        sys.argv = ["wrench_reuse_override", *remaining]
        if args.oracle:
            oracle.main()
        else:
            benchmark.main()
    output.output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "reuse_wrench": args.reuse_wrench,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "additional_immutable_bytes": 288,
                "note": "Use actual integrated contact force and moment about the authored COM, plus the complete shared centrifugal wrench. Scatter overflow executes the complete original nodal reduction. Final budgets and load samples remain unchanged.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
