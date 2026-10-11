"""Check device reset selections against declared initial and untouched states."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from examples.rigid.egg_stress_controller import ResetIndexCache
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array


def state_arrays(solver):
    state = solver.get_state(0)
    return {
        name: tensor_to_array(getattr(state, name)).copy()
        for name in ("qpos", "dofs_vel", "dofs_acc", "links_pos", "links_quat", "friction_ratio")
    }


def observations(workload):
    return [
        tensor_to_array(value).copy()
        for value in (workload.link.get_max_stress(), *workload.link.get_stress_field(), workload.phase)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(16, varied=True, seed=args.seed, output_mode="full", resident_reset_indices=False)
    cache = ResetIndexCache(16, True)
    solver = workload.scene.rigid_solver
    initial = state_arrays(solver)
    for _ in range(900):
        workload.step()
    solver.check_errno()
    live = workload.scene.get_state()
    rows = []
    for ids in (np.array([3]), np.array([0, 15]), np.array([0, 3, 8, 15]), np.arange(16)):
        restored = []
        for route in ("cpu", "resident"):
            workload.scene.reset(state=live)
            workload.tick = 901
            workload.step()
            solver.check_errno()
            before = state_arrays(solver)
            before_observations = observations(workload)
            if route == "cpu":
                workload.reset(ids)
            else:
                selection = cache.select(ids)
                workload.scene.reset(state=workload.reset_state, envs_idx=selection)
                workload.phase[selection] = 0.0
                workload.reset_count += len(ids)
            actual = state_arrays(solver)
            observed = observations(workload)
            untouched = np.setdiff1d(np.arange(16), ids)
            for name, value in actual.items():
                np.testing.assert_array_equal(value[untouched], before[name][untouched])
                np.testing.assert_allclose(value[ids], initial[name][ids], rtol=0, atol=1e-14)
            for index in range(4):
                np.testing.assert_array_equal(observed[index][untouched], before_observations[index][untouched])
                if index in (0, 3):
                    np.testing.assert_array_equal(observed[index][ids], 0)
                else:
                    assert np.isnan(observed[index][ids]).all()
            restored.append(actual)
            rows.append(
                {
                    "route": route,
                    "selected": ids.tolist(),
                    "rigid_initial_state_passed": True,
                    "untouched_state_and_observations_exact": True,
                    "reset_stress_invalidation_passed": True,
                }
            )
        for name in restored[0]:
            np.testing.assert_array_equal(restored[0][name], restored[1][name])
        print("Reset selections equivalent", ids.tolist(), flush=True)
    assert cache.evictions > 0 and cache.elements <= 16
    ids = np.array([0, 15])
    owned = cache.select(ids)
    ids[0] = 4
    np.testing.assert_array_equal(tensor_to_array(owned), [0, 15])
    assert cache.select(np.array([0, 15])).data_ptr() == owned.data_ptr()
    assert cache.elements == sum(len(key) for key in cache.buffers)
    assert cache.describe()["bytes"] <= 16 * 8
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rows": rows,
                "passed": True,
                "cache": cache.describe(),
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        Path("examples/rigid/egg_stress_controller.py"),
                        Path("genesis/engine/solvers/rigid/rigid_solver.py"),
                        Path("genesis/engine/solvers/rigid/stress/recovery.py"),
                        Path("examples/rigid/franka_egg_stress.py"),
                    )
                },
                "note": "Same public Scene.reset initial state with CPU indices versus resident int64 CUDA indices (Python integer for one selected environment). Six rigid fields match exactly between routes; selected rows restore their declared initial state and untouched rows/stress observations stay exactly unchanged. Selected max/phase clear to zero and all full-field values become NaN. No numerical budget changes.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
