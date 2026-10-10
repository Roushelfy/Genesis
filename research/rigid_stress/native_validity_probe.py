"""Untimed every-step validity audit across warmup and a complete reset trajectory."""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
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
    parser.add_argument("--policy-precision", choices=("64", "32"), default="32")
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(
        args.envs, seed=args.seed, varied=True, conditions=args.conditions, output_mode=args.output_mode
    )
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
    policy_dtype = torch.float32 if args.policy_precision == "32" else torch.float64
    reference = copy.deepcopy(policy) if args.policy_precision == "32" else None
    policy = policy.to(dtype=policy_dtype)
    policy_error_rad = 0.0
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
            for tick in range(count):
                action = None
                if args.scope == "policy":
                    observation = workload.observation()
                    action = policy(observation.to(dtype=policy_dtype))
                    if reference is not None and tick % 100 == 0:
                        expected = reference(observation)
                        difference = (action.to(dtype=torch.float64) - expected).abs().max().item()
                        policy_error_rad = max(policy_error_rad, 1e-4 * difference)
                        assert policy_error_rad < 1e-9
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
        "output_mode": args.output_mode,
        "condition_count": workload.condition_count,
        "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                    Path("genesis/options/rigid_stress.py"),
                    Path("examples/rigid/franka_egg_stress.py"),
                    Path(__file__),
                ]
            )
        },
        "command": [sys.executable, *sys.argv],
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
        "quadrants": qd.__version__,
        "torch": torch.__version__,
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
        "hold_check_count": int(tail[:, 6].sum() + tail[:, 7].sum()),
        "bilateral_lift_check_count": int(tail[:, 6].sum()),
        "failed_hold_check_count": int(tail[:, 7].sum()),
        "hold_check_success_fraction": float(tail[:, 6].sum() / max(1, tail[:, 6].sum() + tail[:, 7].sum())),
        "environments_with_failed_hold_checks": int((tail[:, 7] > 0).sum()),
        "hold_check_note": "Height >0.09 m and both fingers in nonzero contact during phase [0.4,0.65]; this is a per-step lift/hold proxy, not an episode-level success certificate.",
        "per_environment_work_quantiles": {
            "quantiles": [0.5, 0.9, 0.99, 1.0],
            "fit_evaluations_total": np.quantile(tail[:, 2], [0.5, 0.9, 0.99, 1.0]).tolist(),
            "local_retries": np.quantile(tail[:, 1], [0.5, 0.9, 0.99, 1.0]).tolist(),
            "note": "Work counts proxy per-environment tails; GPU shared execution is not individually timed.",
        },
        "policy": {
            "layers": [26, 128, 128, 7],
            "dtype": "float32" if args.policy_precision == "32" else "float64",
            "seed": 99173,
            "residual_scale_rad": 1e-4,
            "scope": "inference rollout",
            "reference_max_control_error_rad": policy_error_rad,
            "control_error_budget_rad": 1e-9,
        },
        "note": "Separate untimed native check of every observation, surviving per-environment resets; no rate claimed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert not counts.any(), "A reset trajectory contained invalid stress observations."
    print("All", args.envs * (args.warmup + args.steps), "environment observations valid.")


if __name__ == "__main__":
    main()
