"""Reuse current-footprint P2 moments between positive pressure fit and scatter."""

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
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import V_MAT, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle


def function(directory, filename, name):
    return next(
        item
        for item in ast.parse((directory / filename).read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def generate():
    directory = Path("genesis/engine/solvers/rigid/stress")
    scatter = function(directory, "scatter.py", "func_scatter_faces")
    first_loop = next(
        index
        for index, item in enumerate(scatter.body)
        if isinstance(item, ast.Expr)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Attribute)
        and item.value.func.attr == "loop_config"
    )
    prepare = ast.parse(
        "@qd.func\ndef func_prepare_moments(contacts: StressContactState, state: StressState, info: StressInfo, surface: StressSurfaceInfo, workspace: StressScatterWorkspace, cached_bounds: qd.template(), cache: MomentWorkspace):\n    pass\n"
    ).body[0]
    prepare.body = (
        scatter.body[:first_loop]
        + ast.parse("""
for i_pair in range(qd.select(is_admitted, contacts.active_count[None], 0)):
    cache.gram[i_pair] = qd.Matrix.zero(gs.qd_float, 3, 3)
    cache.affine[i_pair] = False
    pair = contacts.active_pairs[i_pair]
    contacts.candidate_count[pair[0], pair[1]] = 0
qd.loop_config(block_dim=128)
for i_thread in range(qd.i32(qd.select(workspace.count[None] <= workspace.tasks.shape[0], workspace.count[None] * 32, 0))):
    lane, i_t = i_thread % 32, i_thread // 32
    i_pair, i_f = workspace.tasks[i_t][0], workspace.tasks[i_t][1]
    pair = contacts.active_pairs[i_pair]
    i_c, i_b = pair[0], pair[1]
    first, second = func_force_frame(contacts.force[i_c, i_b])
    gram = qd.Matrix.zero(gs.qd_float, 3, 3)
    load = qd.Matrix.zero(gs.qd_float, 6, 3)
    candidates = 0
    n_q = surface.shape.shape[0]
    for chunk in range((n_q + 31) // 32):
        i_q = chunk * 32 + lane
        if i_q < n_q:
            coordinates, weight, _position, shape, _face = func_contact_sample(
                i_f * n_q + i_q, i_c, i_b, first, second, contacts, surface)
            gram += weight * coordinates.outer_product(coordinates)
            load += shape.outer_product(weight * coordinates)
            candidates += gs.qd_int(weight > 0.0)
    for a, b in qd.static(qd.ndrange(3, 3)):
        gram[a, b] = qd.simt.subgroup.reduce_all_add(gram[a, b])
    for a, b in qd.static(qd.ndrange(6, 3)):
        load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
    candidates = qd.simt.subgroup.reduce_all_add(candidates)
    if lane == 0:
        cache.loads[i_t] = load
        for a, b in qd.static(qd.ndrange(3, 3)):
            qd.atomic_add(cache.gram[i_pair][a, b], gram[a, b])
        qd.atomic_add(contacts.candidate_count[i_c, i_b], candidates)
for i_pair in range(qd.select(workspace.count[None] <= workspace.tasks.shape[0], contacts.active_count[None], 0)):
    pair = contacts.active_pairs[i_pair]
    i_c, i_b = pair[0], pair[1]
    coefficient, weight_sum, status = func_pressure_initial(cache.gram[i_pair])
    contacts.coefficient[i_c, i_b] = coefficient
    contacts.weight_sum[i_c, i_b] = weight_sum
    contacts.status[i_c, i_b] = status
    cache.affine[i_pair] = status == 0
func_pressure_overflow(contacts, surface, False, workspace.count[None] > workspace.tasks.shape[0])
""").body
    )
    # Keep the prepared task identities; repacking atomics could change their order.
    scatter.name = "func_scatter_moments"
    scatter.args.args.append(ast.arg(arg="cache", annotation=ast.parse("MomentWorkspace", mode="eval").body))
    scatter.body = scatter.body[first_loop:]

    class Reuse(ast.NodeTransformer):
        def visit_For(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == "i_q_block":
                replacement = ast.parse("""
if cache.affine[i_pair] and not contacts.is_refined[i_c, i_b]:
    if i_lane < 6:
        row = qd.Vector([cache.loads[i_t][i_lane, a] for a in qd.static(range(3))])
        nodal_force = row.dot(contacts.coefficient[i_c, i_b]) * force
        load[i_lane, :] = nodal_force
        total = nodal_force
        node = info.surface_nodes[i_f, i_lane]
        moment = (info.vertices[node] - contacts.position[i_c, i_b]).cross(nodal_force)
else:
    pass
""").body[0]
                replacement.orelse = [node]
                return replacement
            return node

    scatter = Reuse().visit(scatter)
    pressure = function(directory, "contact.py", "func_pressure_warp")
    pressure.name = "func_pressure_overflow"
    pressure.args.args.append(ast.arg(arg="enabled", annotation=ast.parse("bool", mode="eval").body))
    loop = next(item for item in pressure.body if isinstance(item, ast.For))
    count = loop.iter.args[0]
    loop.iter.args[0] = ast.Call(
        func=ast.Attribute(value=ast.Name(id="qd", ctx=ast.Load()), attr="select", ctx=ast.Load()),
        args=[ast.Name(id="enabled", ctx=ast.Load()), count, ast.Constant(value=0)],
        keywords=[],
    )
    fused = function(directory, "pipeline.py", "kernel_pipeline")
    fused.name = "kernel_moment_pipeline"
    fused.args.args.append(ast.arg(arg="cache", annotation=ast.parse("MomentWorkspace", mode="eval").body))
    for index, item in enumerate(fused.body):
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name):
            call = item.value
            if (
                call.func.id == "func_pressure_warp"
                and isinstance(call.args[-1], ast.Constant)
                and not call.args[-1].value
            ):
                fused.body[index] = ast.parse(
                    "func_prepare_moments(contacts, state, info, surface, scatter, cached_bounds, cache)"
                ).body[0]
            elif call.func.id == "func_scatter_faces":
                call.func.id = "func_scatter_moments"
                call.args.append(ast.Name(id="cache", ctx=ast.Load()))
        elif isinstance(item, ast.If):
            for expression in item.body:
                if (
                    isinstance(expression, ast.Expr)
                    and isinstance(expression.value, ast.Call)
                    and isinstance(expression.value.func, ast.Name)
                    and expression.value.func.id == "func_scatter_faces"
                ):
                    expression.value.func.id = "func_scatter_moments"
                    expression.value.args.append(ast.Name(id="cache", ctx=ast.Load()))
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
                node.func.id = "kernel_moment_pipeline"
                node.args.append(ast.parse("workspaces[id(entry.state)]", mode="eval").body)
            return node

    recover = Calls().visit(recover)
    generated = (
        "from genesis.engine.solvers.rigid.stress.pipeline import *\n"
        "from genesis.engine.solvers.rigid.stress.contact import *\n"
        "from genesis.engine.solvers.rigid.stress.scatter import *\n"
        "from genesis.engine.solvers.rigid.stress.recovery import *\n"
        "workspaces = {}\n\n@dataclass(frozen=True)\nclass MomentWorkspace:\n"
        "    gram: qd.Tensor\n    loads: qd.Tensor\n    affine: qd.Tensor\n\n"
        + "\n\n".join(
            ast.unparse(ast.fix_missing_locations(item)) for item in (pressure, prepare, scatter, fused, recover)
        )
        + "\n\n@qd.kernel(graph=True)\n"
        "def kernel_contact_moments(epsilon: float, contacts: StressContactState, state: StressState, info: StressInfo, surface: StressSurfaceInfo, scatter: StressScatterWorkspace, cache: MomentWorkspace):\n"
        "    func_anchor(epsilon, contacts, surface)\n    func_pack_contacts(contacts)\n"
        "    func_prepare_moments(contacts, state, info, surface, scatter, True, cache)\n"
        "    func_pressure_correct_warp(contacts, surface)\n    func_refine_contacts(contacts)\n"
        "    func_pressure_warp(contacts, surface, True)\n    func_pressure_correct_warp(contacts, surface)\n"
        "    func_scatter_moments(contacts, state, info, surface, scatter, True, cache)\n"
        "\n@qd.kernel(graph=True)\n"
        "def kernel_contacts_original(epsilon: float, contacts: StressContactState, state: StressState, info: StressInfo, surface: StressSurfaceInfo, scatter: StressScatterWorkspace):\n"
        "    func_anchor(epsilon, contacts, surface)\n    func_pack_contacts(contacts)\n"
        "    func_pressure_warp(contacts, surface, False)\n    func_pressure_correct_warp(contacts, surface)\n"
        "    func_refine_contacts(contacts)\n    func_pressure_warp(contacts, surface, True)\n"
        "    func_pressure_correct_warp(contacts, surface)\n    func_scatter_faces(contacts, state, info, surface, scatter, True)\n"
    )
    cache = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache/contact_moment_trials" / os.environ["SLURM_JOB_ID"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "contact_moments_generated.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("contact_moments_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, generated


def create_cache(module, recovery):
    assert len(recovery.links) == 1, "Research override configures one stressed link."
    entry = recovery.links[0]
    capacity = entry.scatter.tasks.shape[0]
    assert capacity > 0, "Moment reuse requires bounded face scheduling."
    cache = module.MomentWorkspace(
        V_MAT(3, 3, dtype=gs.qd_float, shape=(capacity,)),
        V_MAT(6, 3, dtype=gs.qd_float, shape=(capacity,)),
        V(dtype=gs.qd_bool, shape=(capacity,)),
    )
    module.workspaces[id(entry.state)] = cache
    return cache


def micro(module, remaining):
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--scatter-tasks-per-env", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(remaining)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, scatter_tasks_per_env=args.scatter_tasks_per_env)
    for _ in range(args.warmup):
        workload.step()
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]
    cache = create_cache(module, recovery)
    inputs = (
        float(np.finfo(float).eps),
        entry.contacts,
        entry.state,
        entry.model.info,
        entry.surface.info,
        entry.scatter,
    )
    original = lambda: module.kernel_contacts_original(*inputs)
    trial = lambda: module.kernel_contact_moments(*inputs, cache)
    original()
    expected_force = qd_to_numpy(entry.state.force, copy=True)
    expected_status = qd_to_numpy(entry.contacts.status, copy=True)
    entry.model.recover(entry.omega, entry.state, surface_load=True)
    expected_peak = qd_to_numpy(entry.state.peak, copy=True)
    rows = []
    for name, operation in (("native_fit_scatter", original), ("current_footprint_moments", trial)):
        operation()
        np.testing.assert_array_equal(qd_to_numpy(entry.contacts.status), expected_status)
        actual_force = qd_to_numpy(entry.state.force)
        difference = np.linalg.norm(actual_force - expected_force, axis=(0, 2))
        budget = np.maximum(1e-11, 1e-8 * np.linalg.norm(expected_force, axis=(0, 2)))
        assert (difference <= budget).all(), (name, difference.max(), (difference / budget).max())
        entry.model.recover(entry.omega, entry.state, surface_load=True)
        assert qd_to_numpy(entry.state.valid).all()
        np.testing.assert_allclose(qd_to_numpy(entry.state.peak), expected_peak, rtol=1e-4, atol=1e-3)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
            "force_difference_budget_ratio_max": float((difference / budget).max()),
            "force_error_max_N": float(qd_to_numpy(entry.contacts.force_error).max()),
            "moment_error_max_Nm": float(qd_to_numpy(entry.contacts.moment_error).max()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-moments", action="store_true")
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

        if args.reuse_moments:
            RigidStressRecovery.__init__ = initialize
            RigidStressRecovery.recover = module.recover
        sys.argv = ["contact_moment_override", *remaining]
        if args.oracle:
            oracle.main()
        else:
            benchmark.main()
    output.output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "reuse_moments": args.reuse_moments,
                "micro": args.micro,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "additional_scratch_bytes_per_task_capacity": 217 if args.reuse_moments or args.micro else 0,
                "note": "Current footprint only, refreshed each recovery. Strictly positive affine pressures reuse P2 moments; clipped/refined pressures retain complete sampling. Overflow recomputes all original pressure and loads. Native wrench/full-residual/peak acceptance unchanged.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
