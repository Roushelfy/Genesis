"""Measure coalesced node tiles that reduce rigid-mode wrench atomics in Quadrants."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from research.rigid_stress import probe_relief_cache

NODE_TILE = 8


@qd.func
def func_reduced_balance(omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo, relief: qd.Tensor):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    qd.loop_config(block_dim=32 * NODE_TILE)
    for i_thread in range(
        ((stress_info.vertices.shape[0] + NODE_TILE - 1) // NODE_TILE)
        * ((stress_state.active.shape[0] + 31) // 32)
        * (32 * NODE_TILE)
    ):
        i_t = qd.simt.block.thread_idx()
        i_block = i_thread // (32 * NODE_TILE)
        n_env_tiles = (stress_state.active.shape[0] + 31) // 32
        i_row, i_col = i_t // 32, i_t % 32
        i_n = (i_block // n_env_tiles) * NODE_TILE + i_row
        i_b = (i_block % n_env_tiles) * 32 + i_col
        sh_wrench = qd.simt.block.SharedArray((6, NODE_TILE, 32), gs.qd_float)
        wrench = qd.Vector.zero(gs.qd_float, 6)
        if i_n < stress_info.vertices.shape[0] and i_b < stress_state.active.shape[0]:
            w = omega[i_b]
            terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
            load = stress_state.force[i_n, i_b] + stress_info.centrifugal[i_n] @ terms
            stress_state.rhs[i_n, i_b] = load
            wrench = stress_info.modes[i_n].transpose() @ load
        for a in qd.static(range(6)):
            sh_wrench[a, i_row, i_col] = wrench[a]
        qd.simt.block.sync()
        if i_row == 0 and i_b < stress_state.active.shape[0]:
            total = qd.Vector.zero(gs.qd_float, 6)
            for j_row in qd.static(range(NODE_TILE)):
                for a in qd.static(range(6)):
                    total[a] += sh_wrench[a, j_row, i_col]
            for a in qd.static(range(6)):
                qd.atomic_add(stress_state.wrench[i_b][a], total[a])
        # Grid-stride iterations reuse shared storage after the reducing warp finishes its reads.
        qd.simt.block.sync()
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        stress_state.rhs[i_n, i_b] -= relief[i_n] @ stress_state.wrench[i_b]


@qd.kernel(graph=True)
def kernel_reduced_balance(omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo, relief: qd.Tensor):
    func_reduced_balance(omega, stress_state, stress_info, relief)


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-tile", type=int, choices=(2, 4, 8, 16), default=8)
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    global NODE_TILE
    NODE_TILE = args.node_tile
    original_generate = probe_relief_cache.generate

    def generate():
        module, original, generated = original_generate()
        module.func_cached_balance = func_reduced_balance
        module.kernel_cached_balance = kernel_reduced_balance
        return module, original, generated

    probe_relief_cache.generate = generate
    sys.argv = ["probe_relief_cache", *remaining]
    probe_relief_cache.main()
    variant_path = output.output.with_suffix(".variant.json")
    variant = json.loads(variant_path.read_text())
    variant.update(
        {
            "node_tile": NODE_TILE,
            "reduced_wrench_atomics": variant["cached_relief"],
            "command": command,
            "reduction_source": Path(__file__).read_text(),
            "reduction_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_bytes_per_block": 6 * NODE_TILE * 32 * 8,
            "note": "Research coalesced node tiles reduce complete rigid-mode wrench sums before global atomics. Includes the shared Quadrants relief projection. All nodes, forces, centrifugal terms and final budgets are retained. Actual rollout and CPU oracle are separate retention gates.",
        }
    )
    variant_path.write_text(json.dumps(variant, indent=2) + "\n")
    if variant["micro"]:
        result = json.loads(output.output.read_text())
        result["reports"][1]["name"] = "cached_and_reduced"
        output.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
