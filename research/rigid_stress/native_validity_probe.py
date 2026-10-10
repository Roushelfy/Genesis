"""Untimed every-step validity audit across warmup and a complete reset trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils import array_class
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_audit(
    state: StressState,
    contacts: StressContactState,
    failures: qd.Tensor,
    counters: qd.Tensor,
    errors: qd.Tensor,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    phase: qd.types.ndarray(),
    egg: int,
    left: int,
    right: int,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    for i_b in range(failures.shape[0]):
        if not state.step_valid[i_b]:
            failures[i_b] += 1
        counters[i_b][4] += state.corrections[i_b]
        counters[i_b][5] += state.fallbacks[i_b]
        errors[i_b][0] = qd.max(
            errors[i_b][0],
            qd.sqrt(state.residual_norm_squared[i_b])
            / qd.max(absolute_tolerance, relative_tolerance * qd.sqrt(state.rhs_norm_squared[i_b])),
        )
        errors[i_b][1] = qd.max(errors[i_b][1], state.step_peak[i_b])
        fingers = qd.Vector.zero(qd.i32, 2)
        for i_c in range(contacts.valid.shape[0]):
            magnitude = contacts.force[i_c, i_b].norm()
            if contacts.valid[i_c, i_b] and magnitude > 1e-30:
                counters[i_b][0] += 1
                counters[i_b][1] += gs.qd_int(contacts.is_refined[i_c, i_b])
                counters[i_b][2] += contacts.evaluations[i_c, i_b]
                counters[i_b][3] = qd.max(counters[i_b][3], contacts.evaluations[i_c, i_b])
                errors[i_b][2] = qd.max(errors[i_b][2], contacts.force_error[i_c, i_b] / magnitude)
                errors[i_b][3] = qd.max(
                    errors[i_b][3], contacts.moment_error[i_c, i_b] / (magnitude * contacts.radius[i_c, i_b])
                )
                errors[i_b][4] = qd.max(errors[i_b][4], contacts.friction[i_c, i_b])
                errors[i_b][5] = qd.min(errors[i_b][5], contacts.friction[i_c, i_b])
                source = collider_state.contact_sort_idx[i_c, i_b]
                a, b = collider_state.contact_data.link_a[source, i_b], collider_state.contact_data.link_b[source, i_b]
                fingers[0] += gs.qd_int(a == left or b == left)
                fingers[1] += gs.qd_int(a == right or b == right)
        if phase[i_b] >= 0.4 and phase[i_b] <= 0.65:
            held = dyn_state.links.pos[egg, i_b][2] > 0.09 and fingers[0] > 0 and fingers[1] > 0
            counters[i_b][6] += gs.qd_int(held)
            counters[i_b][7] += gs.qd_int(not held)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("live", "policy"), required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, conditions=args.conditions)
    torch.manual_seed(99173)
    policy = (
        torch.nn.Sequential(
            torch.nn.Linear(26, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 7),
        )
        .to(device=gs.device, dtype=gs.tc_float)
        .eval()
    )
    failures = V(dtype=gs.qd_int, shape=(args.envs,))
    failures.fill(0)
    counters = V_VEC(8, dtype=qd.i64, shape=(args.envs,))
    errors = V_VEC(6, dtype=gs.qd_float, shape=(args.envs,))
    counters.fill(0)
    initial = np.zeros((args.envs, 6))
    initial[:, 5] = np.inf
    errors.from_numpy(initial)
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    left, right = [workload.robot.get_link(name).idx for name in ("left_finger", "right_finger")]
    with torch.inference_mode():
        for count in (args.warmup, args.steps):
            for _ in range(count):
                action = policy(workload.observation()) if args.scope == "policy" else None
                workload.step(action)
                kernel_audit(
                    entry.state,
                    entry.contacts,
                    failures,
                    counters,
                    errors,
                    solver.dyn_state,
                    solver.collider.collider_state,
                    workload.phase,
                    workload.link.idx,
                    left,
                    right,
                    entry.link.stress_options.tolerance,
                    entry.link.stress_options.absolute_tolerance,
                )
            workload.restart()
    counts = qd_to_numpy(failures)
    tail = qd_to_numpy(counters)
    accuracy = qd_to_numpy(errors)
    result = {
        "scope": args.scope,
        "envs": args.envs,
        "warmup_steps": args.warmup,
        "trajectory_steps": args.steps,
        "seed": args.seed,
        "load_model": "finite_pad_adaptive_q10",
        "condition_count": workload.condition_count,
        "conditions": str(args.conditions) if args.conditions is not None else None,
        "every_step_failures": counts.tolist(),
        "failed_environment_steps": int(counts.sum()),
        "counter_columns": [
            "nonzero_contacts",
            "local_retries",
            "fit_evaluations_total",
            "fit_evaluations_max",
            "corrections",
            "fallbacks",
            "bilateral_lift_steps",
            "failed_hold_checks",
        ],
        "per_environment_counters": tail.tolist(),
        "error_columns": [
            "full_residual_budget_ratio_max",
            "global_peak_max_Pa",
            "wrench_force_relative_max",
            "wrench_moment_relative_max",
            "actual_mu_max",
            "actual_mu_min",
        ],
        "per_environment_errors": accuracy.tolist(),
        "environments_with_bilateral_lift": int((tail[:, 6] > 0).sum()),
        "environments_without_bilateral_lift": int((tail[:, 6] == 0).sum()),
        "note": "Separate untimed native check of every observation, surviving per-environment resets; no rate claimed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert not counts.any(), "A reset trajectory contained invalid stress observations."
    print("All", args.envs * (args.warmup + args.steps), "environment observations valid.")


if __name__ == "__main__":
    main()
