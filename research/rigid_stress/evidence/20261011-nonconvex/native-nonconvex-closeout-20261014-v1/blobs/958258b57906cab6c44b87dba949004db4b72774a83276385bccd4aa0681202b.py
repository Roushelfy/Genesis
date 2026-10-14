"""Measure node-first packed application with bounded environment tiles."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual, kernel_peak
from genesis.engine.solvers.rigid.stress.surface_inverse import kernel_surface_pack
from genesis.utils.misc import qd_to_numpy


def generate():
    path = Path("genesis/engine/solvers/rigid/stress/surface_inverse.py")
    source = path.read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "func_surface_apply_packed"
    )
    node.name = "func_traversal"
    assert len(node.args.args) == 4, "This prototype records the original four-argument numerical body."
    original_block = copy.deepcopy(node)
    original_block.name = "func_original_block"
    original_block.args.args.append(ast.arg(arg="block_dim", annotation=ast.parse("qd.template()", mode="eval").body))
    original_block.body.insert(0, ast.parse("qd.loop_config(block_dim=block_dim)").body[0])
    for name in ("env_tile", "block_dim"):
        node.args.args.append(ast.arg(arg=name, annotation=ast.parse("qd.template()", mode="eval").body))
    index = next(
        index
        for index, item in enumerate(node.body)
        if isinstance(item, ast.For) and isinstance(item.target, ast.Tuple)
    )
    loop = ast.parse("""
for i_thread in range(stress_state.rhs.shape[0] * ((stress_state.active.shape[0] + env_tile - 1) // env_tile) * env_tile):
    i_index = i_thread // env_tile
    i_n = i_index % stress_state.rhs.shape[0]
    i_b = (i_index // stress_state.rhs.shape[0]) * env_tile + i_thread % env_tile
    if i_b < stress_state.active.shape[0]:
        pass
""").body[0]
    loop.body[-1].body = node.body[index].body
    node.body[index : index + 1] = [ast.parse("qd.loop_config(block_dim=block_dim)").body[0], loop]
    generated = (
        "from genesis.engine.solvers.rigid.stress.surface_inverse import *\n\n"
        + ast.unparse(ast.fix_missing_locations(original_block))
        + "\n\n"
        + ast.unparse(ast.fix_missing_locations(node))
        + "\n\n@qd.kernel(graph=True)\n"
        "def kernel_traversal(young: float, omega: qd.Tensor, stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo, env_tile: qd.template(), block_dim: qd.template()):\n"
        "    func_traversal(young, omega, stress_state, surface_inverse_info, env_tile, block_dim)\n"
        "\n@qd.kernel(graph=True)\n"
        "def kernel_original_block(young: float, omega: qd.Tensor, stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo, block_dim: qd.template()):\n"
        "    func_original_block(young, omega, stress_state, surface_inverse_info, block_dim)\n"
    )
    directory = (
        Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache/packed_traversal_trials" / os.environ["SLURM_JOB_ID"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "packed_traversal_generated.py"
    destination.write_text(generated)
    spec = importlib.util.spec_from_file_location("packed_traversal_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source, generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module, source, generated = generate()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    for _ in range(args.warmup):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    inverse = entry.model.surface_inverse
    inverse.apply(entry.model.options.young, entry.omega, entry.state)
    expected = qd_to_numpy(entry.state.displacement, copy=True)
    kernel_peak(entry.model.options.young, entry.model.options.poisson, entry.state, entry.model.info)
    expected_peak = qd_to_numpy(entry.state.peak, copy=True)

    def trial(tile, block):
        kernel_surface_pack(entry.state, inverse.info)
        module.kernel_traversal(entry.model.options.young, entry.omega, entry.state, inverse.info, tile, block)

    def block_control():
        kernel_surface_pack(entry.state, inverse.info)
        module.kernel_original_block(entry.model.options.young, entry.omega, entry.state, inverse.info, 512)

    operations = [("original", lambda: inverse.apply(entry.model.options.young, entry.omega, entry.state))]
    operations.append(("original-block-512", block_control))
    for tile in (32, 64, 128):
        for block in (256, 512):
            operations.append((f"env-tile-{tile}-block-{block}", lambda t=tile, b=block: trial(t, b)))
    rows = []
    for name, operation in operations:
        operation()
        np.testing.assert_array_equal(qd_to_numpy(entry.state.displacement), expected)
        kernel_full_residual(
            entry.model.options.young,
            entry.model.options.tolerance,
            entry.model.options.absolute_tolerance,
            entry.state,
            entry.model.info,
            False,
        )
        assert qd_to_numpy(entry.state.valid).all()
        kernel_peak(entry.model.options.young, entry.model.options.poisson, entry.state, entry.model.info)
        np.testing.assert_array_equal(qd_to_numpy(entry.state.peak), expected_peak)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {"name": name, "milliseconds_including_pack": 1e3 * (time.perf_counter() - started) / args.repetitions}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "envs": args.envs,
                "seed": args.seed,
                "reports": rows,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
                "note": "All nodes and exact packed nonzero forces, identical displacement and peak, complete residual unchanged. Snapshot diagnostic, not actual rollout throughput.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
