"""Native h=4 reorthogonalization cost floor, outside the production feature."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, V


@dataclass(frozen=True)
class HistoryProbe:
    loads: qd.Tensor
    basis: qd.Tensor
    coefficients: qd.Tensor
    norm_squared: qd.Tensor


@qd.kernel(graph=True)
def kernel_qr(state: HistoryProbe):
    for i_h in qd.static(range(4)):
        for i_n, i_b in qd.ndrange(state.loads.shape[0], state.loads.shape[1]):
            state.basis[i_n, i_b, i_h] = state.loads[i_n, i_b, i_h]
        for _ in qd.static(range(2)):
            for i_b in range(state.loads.shape[1]):
                state.coefficients[i_b] = qd.Vector.zero(gs.qd_float, 4)
            for i_n, i_b in qd.ndrange(state.loads.shape[0], state.loads.shape[1]):
                for j in range(i_h):
                    qd.atomic_add(state.coefficients[i_b][j], state.basis[i_n, i_b, i_h].dot(state.basis[i_n, i_b, j]))
            for i_n, i_b in qd.ndrange(state.loads.shape[0], state.loads.shape[1]):
                value = state.basis[i_n, i_b, i_h]
                for j in range(i_h):
                    value -= state.coefficients[i_b][j] * state.basis[i_n, i_b, j]
                state.basis[i_n, i_b, i_h] = value
        for i_b in range(state.loads.shape[1]):
            state.norm_squared[i_b] = 0.0
        for i_n, i_b in qd.ndrange(state.loads.shape[0], state.loads.shape[1]):
            value = state.basis[i_n, i_b, i_h]
            qd.atomic_add(state.norm_squared[i_b], value.dot(value))
        for i_n, i_b in qd.ndrange(state.loads.shape[0], state.loads.shape[1]):
            value = qd.Vector.zero(gs.qd_float, 3)
            if state.norm_squared[i_b] > 1e-30:
                value = state.basis[i_n, i_b, i_h] / qd.sqrt(state.norm_squared[i_b])
            state.basis[i_n, i_b, i_h] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhs", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    source = np.load(args.rhs, mmap_mode="r")
    n_nodes = source.shape[2] // 3
    n_source = source.shape[1]
    shape = (n_nodes, args.envs, 4)
    state = HistoryProbe(
        V_VEC(3, dtype=gs.qd_float, shape=shape),
        V_VEC(3, dtype=gs.qd_float, shape=shape),
        V_VEC(4, dtype=gs.qd_float, shape=(args.envs,)),
        V(dtype=gs.qd_float, shape=(args.envs,)),
    )
    # Replication is asset preparation for a QR microbenchmark, not physical batch throughput.
    data = source[1096:1100, np.arange(args.envs) % n_source].reshape((4, args.envs, n_nodes, 3))
    state.loads.from_numpy(np.ascontiguousarray(data.transpose((2, 1, 0, 3))))
    start = time.perf_counter()
    kernel_qr(state)
    qd.sync()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(100):
        kernel_qr(state)
    qd.sync()
    elapsed = (time.perf_counter() - start) * 10
    result = {
        "scope": "Cost floor for h=4 stable QR only; excludes paired displacement basis, prediction, full residual, selection and append.",
        "envs": args.envs,
        "nodes": n_nodes,
        "history": 4,
        "QR_wall_ms": elapsed,
        "JIT_graph_setup_s": setup,
        "load_basis_bytes": 2 * n_nodes * args.envs * 4 * 3 * 8,
        "source": args.rhs.name,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
