"""Evaluate exact environment chunks of the native packed surface operator."""

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
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual, kernel_peak
from genesis.engine.solvers.rigid.stress.surface_inverse import kernel_surface_pack
from genesis.utils.misc import qd_to_numpy


def generate(directory):
    source = Path("genesis/engine/solvers/rigid/stress/surface_inverse.py").read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_surface_apply_packed"
    )
    node.name = "kernel_chunk"
    for name in ("env_start", "env_count"):
        node.args.args.append(ast.arg(arg=name, annotation=ast.Name(id="int", ctx=ast.Load())))
    for loop in node.body:
        if isinstance(loop, ast.For):
            if isinstance(loop.target, ast.Tuple):
                loop.target.elts[1] = ast.Name(id="local_b", ctx=ast.Store())
                loop.iter.args[1] = ast.Name(id="env_count", ctx=ast.Load())
            else:
                loop.target = ast.Name(id="local_b", ctx=ast.Store())
                loop.iter = ast.parse("range(env_count)", mode="eval").body
            loop.body.insert(0, ast.parse("i_b = env_start + local_b").body[0])
    destination = directory / "surface_chunks_generated.py"
    destination.write_text(
        "import quadrants as qd\nimport genesis as gs\n"
        "from genesis.engine.solvers.rigid.stress.data import StressState\n"
        "from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo\n"
        + ast.unparse(ast.fix_missing_locations(node))
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("surface_chunks_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_chunk, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=32768)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kernel, source_hash = generate(args.output.parent)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    model, state = entry.model, entry.state

    def native():
        model.surface_inverse.apply(model.options.young, entry.omega, state)

    def chunked(size):
        kernel_surface_pack(state, model.surface_inverse.info)
        for start in range(0, args.envs, size):
            kernel(
                model.options.young, entry.omega, state, model.surface_inverse.info, start, min(size, args.envs - start)
            )

    native()
    expected = qd_to_numpy(state.displacement, copy=True)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    operations = [("native", native)]
    for chunk in (1024, 2048, 4096, 8192, 16384, 32768, args.envs):
        operations.append((f"chunk-{chunk}", lambda size=chunk: chunked(size)))
    reports = []
    for name, operation in operations:
        operation()
        np.testing.assert_array_equal(qd_to_numpy(state.displacement), expected)
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active=False,
        )
        assert qd_to_numpy(state.valid).all()
        kernel_peak(model.options.young, model.options.poisson, state, model.info)
        np.testing.assert_array_equal(qd_to_numpy(state.peak), expected_peak)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "ms_including_pack_and_all_chunks": 1e3 * (time.perf_counter() - started) / args.repetitions,
        }
        reports.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps({"envs": args.envs, "reports": reports, "source_sha256": source_hash}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
