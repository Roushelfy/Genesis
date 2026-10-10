"""Measure native packed boundary launch layouts on an actual Panda snapshot."""

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
    path = Path("genesis/engine/solvers/rigid/stress/surface_inverse.py")
    source = path.read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_surface_apply_packed"
    )
    node.name = "kernel_layout"
    node.args.args.append(ast.arg(arg="block_dim", annotation=ast.parse("qd.template()", mode="eval").body))
    node.body.insert(0, ast.parse("qd.loop_config(block_dim=block_dim)").body[0])
    generated = directory / "packed_layout_generated.py"
    generated.write_text(
        "import quadrants as qd\nimport genesis as gs\n"
        "from genesis.engine.solvers.rigid.stress.data import StressState\n"
        "from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo\n"
        + ast.unparse(node)
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("packed_layout_generated", generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_layout, hashlib.sha256(source.encode()).hexdigest()


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
    model.surface_inverse.apply(model.options.young, entry.omega, state, packed=False)
    expected = qd_to_numpy(state.displacement, copy=True)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    expected_peak = qd_to_numpy(state.peak, copy=True)

    def native(packed):
        model.surface_inverse.apply(model.options.young, entry.omega, state, packed=packed)

    def trial(block_dim):
        kernel_surface_pack(state, model.surface_inverse.info)
        kernel(model.options.young, entry.omega, state, model.surface_inverse.info, block_dim)

    operations = [("dense", lambda: native(False)), ("packed-default", lambda: native(True))]
    for block in (64, 128, 256, 512, 1024):
        operations.append((f"packed-block-{block}", lambda b=block: trial(b)))
    rows = []
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
        row = {"name": name, "ms_including_pack": 1e3 * (time.perf_counter() - started) / args.repetitions}
        rows.append(row)
        print(json.dumps(row), flush=True)
    count = qd_to_numpy(state.boundary_count)
    args.output.write_text(
        json.dumps(
            {
                "envs": args.envs,
                "reports": rows,
                "source_sha256": source_hash,
                "nonzero_nodes_min_max_mean": [int(count.min()), int(count.max()), float(count.mean())],
                "full_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
