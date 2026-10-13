"""Quadrants sampler counter/reservation cost with irregular accepted samples; no rollout gain claim."""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


@dataclass(frozen=True)
class CounterState:
    candidates: qd.Tensor
    samples: qd.Tensor
    maximum: qd.Tensor
    indices: qd.Tensor


@qd.kernel
def kernel_clear_counters(state: CounterState):
    for i_c in range(state.samples.shape[0]):
        state.candidates[i_c] = 0
        state.samples[i_c] = 0
        state.maximum[i_c] = 0.0


@qd.kernel
def kernel_sampler_counters(n_samples: int, state: CounterState, is_warp: qd.template()):
    qd.loop_config(block_dim=128)
    n_chunks = (n_samples + 31) // 32
    for i_thread in range(state.samples.shape[0] * n_chunks * 32):
        lane = i_thread % 32
        i_c = i_thread // (32 * n_chunks)
        i_q = (i_thread // 32 % n_chunks) * 32 + lane
        is_valid = i_q < n_samples
        is_accepted = is_valid and i_q % 5 != 0
        distance = gs.qd_float(0.0)
        if is_accepted:
            distance = gs.qd_float(i_q % 997 + 1) / 997.0
        if qd.static(is_warp):
            candidates = qd.simt.subgroup.reduce_all_add(gs.qd_int(is_valid))
            samples = qd.simt.subgroup.reduce_all_add(gs.qd_int(is_accepted))
            maximum = qd.simt.subgroup.reduce_all_max(distance)
            prefix = gs.qd_int(is_accepted)
            for offset in qd.static((1, 2, 4, 8, 16)):
                previous = qd.simt.subgroup.shuffle_up(prefix, qd.u32(offset))
                if lane >= offset:
                    prefix += previous
            start = gs.qd_int(0)
            if lane == 0:
                if candidates > 0:
                    qd.atomic_add(state.candidates[i_c], candidates)
                if samples > 0:
                    start = qd.atomic_add(state.samples[i_c], samples)
                    qd.atomic_max(state.maximum[i_c], maximum)
            start = qd.simt.subgroup.shuffle(start, qd.u32(0))
            if is_accepted:
                state.indices[start + prefix - 1, i_c] = i_q
        else:
            if is_valid:
                qd.atomic_add(state.candidates[i_c], 1)
            if is_accepted:
                destination = qd.atomic_add(state.samples[i_c], 1)
                qd.atomic_max(state.maximum[i_c], distance)
                state.indices[destination, i_c] = i_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    rows = []
    for total in (25_477_233, 30_171_852):
        for columns in (32768, 65536, 131072):
            n_samples = (total + columns - 1) // columns
            state = CounterState(
                V(dtype=gs.qd_int, shape=(columns,)),
                V(dtype=gs.qd_int, shape=(columns,)),
                V(dtype=gs.qd_float, shape=(columns,)),
                V(dtype=gs.qd_int, shape=(n_samples, columns), layout=(1, 0)),
            )
            timings = []
            for is_warp in (False, True, False):
                for _ in range(5):
                    kernel_clear_counters(state)
                    kernel_sampler_counters(n_samples, state, is_warp)
                qd.sync()
                seconds = []
                for _ in range(3):
                    start = time.perf_counter()
                    for _ in range(10):
                        kernel_clear_counters(state)
                        kernel_sampler_counters(n_samples, state, is_warp)
                    qd.sync()
                    seconds.append((time.perf_counter() - start) / 10)
                expected = np.arange(n_samples, dtype=gs.np_int)
                accepted = expected[expected % 5 != 0]
                print(json.dumps({"is_warp": is_warp, "n_samples": n_samples, "columns": columns, "candidates": qd_to_numpy(state.candidates, transpose=True)[:16].tolist(), "candidate_range": [int(qd_to_numpy(state.candidates, transpose=True).min()), int(qd_to_numpy(state.candidates, transpose=True).max())], "samples": qd_to_numpy(state.samples, transpose=True)[:16].tolist()}), flush=True)
                assert np.all(qd_to_numpy(state.candidates, transpose=True) == n_samples)
                assert np.all(qd_to_numpy(state.samples, transpose=True) == len(accepted))
                np.testing.assert_allclose(qd_to_numpy(state.maximum, transpose=True), ((accepted % 997 + 1) / 997).max(), rtol=0, atol=1e-15)
                actual = qd_to_numpy(state.indices, transpose=True)
                assert np.all(np.sort(actual[:, :len(accepted)], axis=1) == accepted)
                timings.append({"is_warp": is_warp, "seconds": seconds})
            rows.append({"nominal_candidates": total, "columns": columns, "candidates_per_column": n_samples,
                         "actual_candidates": n_samples * columns, "actual_accepted": len(accepted) * columns,
                         "timings": timings})
            print(json.dumps({"total": total, "columns": columns, "mean_ms": [1000 * np.mean(row["seconds"]) for row in timings]}), flush=True)
    args.output.write_text(json.dumps({"rows": rows, "precision": "64", "scope": __doc__,
        "limitations": "Synthetic uniform source counts bracket captured aggregate candidate diagnostics, which may duplicate body-union work.20percent irregular rejected samples; geometry/profile computation excluded. Kernel clear, candidate/sample reservation counters, FP64 maximum and complete index stores included. Off/on/off repeats are a cost/benefit probe, not a formal production bound, stress qualification or rollout speedup. Every column count/maximum and every accepted index permutation is verified."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
