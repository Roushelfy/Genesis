"""Check rigid isolation and complete stress agreement across serial post fusion."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workloads = [
        FrankaEgg(args.envs, seed=args.seed, varied=True, output_mode="full", fused_rigid_post=enabled)
        for enabled in (False, True)
    ]
    assert workloads[0].scene.rigid_solver.stress_recovery.post_kernel is None
    assert workloads[1].scene.rigid_solver.stress_recovery.post_kernel is not None
    maxima = {name: 0.0 for name in ("nodal_force_N", "peak_Pa", "tensor_Pa", "von_mises_Pa")}
    checks = 0
    for tick in range(args.steps):
        for workload in workloads:
            workload.step()
            workload.scene.rigid_solver.check_errno()
        if tick % 20 == 19:
            solvers = [workload.scene.rigid_solver for workload in workloads]
            states = [solver.get_state(0) for solver in solvers]
            for name in ("qpos", "dofs_vel", "dofs_acc", "links_pos", "links_quat", "friction_ratio"):
                values = [tensor_to_array(getattr(state, name)) for state in states]
                np.testing.assert_array_equal(*values, err_msg=f"rigid {name} tick {tick}")
            entries = [solver.stress_recovery.links[0] for solver in solvers]
            for name in ("valid", "step_valid", "invalid_steps"):
                np.testing.assert_array_equal(*[qd_to_numpy(getattr(entry.state, name)) for entry in entries])
            np.testing.assert_array_equal(*[qd_to_numpy(entry.contacts.valid) for entry in entries])
            for label, values, relative, absolute in (
                ("nodal_force_N", [qd_to_numpy(entry.state.force) for entry in entries], 1e-8, 1e-12),
                ("peak_Pa", [tensor_to_array(workload.link.get_max_stress()) for workload in workloads], 1e-4, 1e-3),
                (
                    "tensor_Pa",
                    [tensor_to_array(workload.link.get_stress_field()[0]) for workload in workloads],
                    1e-4,
                    1e-3,
                ),
                (
                    "von_mises_Pa",
                    [tensor_to_array(workload.link.get_stress_field()[1]) for workload in workloads],
                    1e-4,
                    1e-3,
                ),
            ):
                np.testing.assert_allclose(*values, rtol=relative, atol=absolute)
                maxima[label] = max(maxima[label], float(np.max(np.abs(values[0] - values[1]))))
            np.testing.assert_array_equal(*[tensor_to_array(workload.phase) for workload in workloads])
            checks += 1
        if tick % 400 == 399:
            print("Equivalent tick", tick, "max differences", maxima, flush=True)
    result = {
        "passed": True,
        "seed": args.seed,
        "envs": args.envs,
        "steps": args.steps,
        "checked_states": checks,
        "maximum_absolute_differences": maxima,
        "resets": [workload.reset_count for workload in workloads],
        "invalid_environment_steps": [
            int(qd_to_numpy(workload.scene.rigid_solver.stress_recovery.links[0].state.invalid_steps).sum())
            for workload in workloads
        ],
        "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                    Path("genesis/engine/solvers/rigid/rigid_solver.py"),
                    Path("genesis/options/rigid_stress.py"),
                    Path("examples/rigid/franka_egg_stress.py"),
                    Path("examples/rigid/egg_stress_controller.py"),
                    Path(__file__),
                ]
            )
        },
        "scope": "Untimed live full-field trajectories; serial post fusion on/off, ordinary public resets. Six rigid fields and phase must agree exactly; complete stress retains the original final budgets. CPU FP64 wrench/full-gauge oracles are separate.",
    }
    assert result["invalid_environment_steps"] == [0, 0]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Passed", checks, "state comparisons", flush=True)


if __name__ == "__main__":
    main()
