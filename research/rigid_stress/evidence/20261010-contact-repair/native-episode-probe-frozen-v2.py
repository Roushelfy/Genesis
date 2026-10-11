"""Untimed, every-environment completed-episode grasp and numerical audit."""

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
from genesis.engine.solvers.rigid.stress.scatter import StressScatterWorkspace
from genesis.utils import array_class
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.native_validity_probe import kernel_audit


@qd.kernel
def kernel_episode(
    tick: int,
    delays: qd.types.ndarray(),
    initial_position: qd.types.ndarray(),
    state: StressState,
    contacts: StressContactState,
    dyn: array_class.DynState,
    collider: array_class.ColliderState,
    egg: int,
    left: int,
    right: int,
    flags: qd.Tensor,
    geometry: qd.Tensor,
    outcomes: qd.Tensor,
):
    for i_b in range(flags.shape[0]):
        if tick >= delays[i_b]:
            local_tick = (tick - delays[i_b]) % 600
            phase = gs.qd_float(local_tick) / 599.0
            if local_tick == 0:
                flags[i_b] = qd.Vector.zero(qd.i64, 8)
                flags[i_b][0] = 1
                geometry[i_b] = qd.Vector([initial_position[i_b, 2], 0.0, 0.0])
                outcomes[i_b][0] += 1
            if flags[i_b][0]:
                flags[i_b][4] |= gs.qd_int(not state.step_valid[i_b])
                fingers = qd.Vector.zero(qd.i32, 2)
                for i_c in range(contacts.valid.shape[0]):
                    if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 1e-30:
                        source = collider.contact_sort_idx[i_c, i_b]
                        a, b = collider.contact_data.link_a[source, i_b], collider.contact_data.link_b[source, i_b]
                        fingers[0] += gs.qd_int(a == left or b == left)
                        fingers[1] += gs.qd_int(a == right or b == right)
                height = dyn.links.pos[egg, i_b][2] - geometry[i_b][0]
                bilateral = fingers[0] > 0 and fingers[1] > 0
                if phase >= 0.3 and phase <= 0.5:
                    flags[i_b][1] |= gs.qd_int(height >= 0.08 and bilateral)
                if phase >= 0.5 and phase <= 0.6:
                    flags[i_b][5] += 1
                    flags[i_b][2] |= gs.qd_int(height < 0.08 or not bilateral)
                geometry[i_b][1] = qd.max(geometry[i_b][1], height)
                if phase >= 0.7 and phase < 0.76:
                    flags[i_b][7] += 1
                    geometry[i_b][2] = qd.max(geometry[i_b][2], geometry[i_b][1] - height)
                if phase >= 0.97:
                    flags[i_b][6] += 1
                    flags[i_b][3] |= gs.qd_int(height <= 0.01 and fingers[0] == 0 and fingers[1] == 0)
                if local_tick == 599:
                    failed_lift = not flags[i_b][1]
                    failed_hold = flags[i_b][2] != 0 or flags[i_b][5] == 0
                    failed_release = not flags[i_b][3] or flags[i_b][6] == 0
                    failed_numeric = flags[i_b][4] != 0
                    outcomes[i_b][1] += 1
                    outcomes[i_b][2] += gs.qd_int(not (failed_lift or failed_hold or failed_release or failed_numeric))
                    outcomes[i_b][3] += gs.qd_int(failed_lift)
                    outcomes[i_b][4] += gs.qd_int(failed_hold)
                    outcomes[i_b][5] += gs.qd_int(failed_release)
                    outcomes[i_b][6] += gs.qd_int(failed_numeric)
                    outcomes[i_b][8] += gs.qd_int(flags[i_b][7] > 0)
                    outcomes[i_b][9] += gs.qd_int(geometry[i_b][2] > 0.01)
                    flags[i_b][0] = 0


@qd.kernel
def kernel_abort(flags: qd.Tensor, outcomes: qd.Tensor):
    for i_b in range(flags.shape[0]):
        outcomes[i_b][7] += flags[i_b][0]
        flags[i_b][0] = 0


@qd.kernel
def kernel_tail_work(
    contacts: StressContactState,
    state: StressState,
    scatter: StressScatterWorkspace,
    frame_counts: qd.Tensor,
    totals: qd.Tensor,
    reuse_moments: qd.template(),
    n_q: qd.template(),
):
    for i_b in range(totals.shape[0]):
        frame_counts[i_b] = qd.Vector.zero(qd.i64, 4)
        totals[i_b][0] += state.boundary_count[i_b]
        totals[i_b][1] = qd.max(totals[i_b][1], state.boundary_count[i_b])
    for i_t in range(qd.i32(qd.select(scatter.count[None] <= scatter.tasks.shape[0], scatter.count[None], 0))):
        i_pair, i_f = scatter.tasks[i_t][0], scatter.tasks[i_t][1]
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        qd.atomic_add(frame_counts[i_b][0], 1)
        affine = False
        if qd.static(reuse_moments):
            qd.atomic_add(frame_counts[i_b][2], n_q)
            affine = (
                scatter.affine_pressure[i_pair] and not contacts.is_refined[i_c, i_b] and contacts.status[i_c, i_b] == 0
            )
        if affine:
            qd.atomic_add(frame_counts[i_b][1], 1)
        elif contacts.status[i_c, i_b] == 0:
            samples = 7 * n_q if contacts.is_refined[i_c, i_b] and i_f == contacts.anchor_face[i_c, i_b] else n_q
            qd.atomic_add(frame_counts[i_b][3], samples)
    for i_b in range(totals.shape[0]):
        totals[i_b][2] += frame_counts[i_b][0]
        totals[i_b][3] = qd.max(totals[i_b][3], frame_counts[i_b][0])
        totals[i_b][4] += frame_counts[i_b][1]
        totals[i_b][5] += frame_counts[i_b][2]
        totals[i_b][6] += frame_counts[i_b][3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("live", "policy"), required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--contact-moment-reuse", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--scatter-tasks-per-env", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        seed=args.seed,
        varied=True,
        contact_moment_reuse={"auto": None, "on": True, "off": False}[args.contact_moment_reuse],
        scatter_tasks_per_env=args.scatter_tasks_per_env,
    )
    torch.manual_seed(99173)
    reference = (
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
    policy = copy.deepcopy(reference).to(dtype=torch.float32)
    flags = V_VEC(8, dtype=qd.i64, shape=(args.envs,))
    geometry = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    outcomes = V_VEC(10, dtype=qd.i64, shape=(args.envs,))
    failures = V(dtype=gs.qd_int, shape=(args.envs,))
    counters = V_VEC(9, dtype=qd.i64, shape=(args.envs,))
    errors = V_VEC(6, dtype=gs.qd_float, shape=(args.envs,))
    tail_counts = V_VEC(4, dtype=qd.i64, shape=(args.envs,))
    tail_totals = V_VEC(7, dtype=qd.i64, shape=(args.envs,))
    tail_totals.fill(0)
    for field in (flags, geometry, outcomes, failures, counters):
        field.fill(0)
    initial_errors = np.zeros((args.envs, 6))
    initial_errors[:, 5] = np.inf
    errors.from_numpy(initial_errors)
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    left, right = [workload.robot.get_link(name).idx for name in ("left_finger", "right_finger")]
    stages = []
    policy_error = 0.0
    with torch.inference_mode():
        for stage, length in (("warmup", args.warmup), ("trajectory", args.steps)):
            for step in range(length):
                action = None
                if args.scope == "policy":
                    observation = workload.observation()
                    action = policy(observation.to(dtype=torch.float32))
                    if step % 100 == 0:
                        difference = (action.to(dtype=torch.float64) - reference(observation)).abs().max().item()
                        policy_error = max(policy_error, 1e-4 * difference)
                        assert policy_error < 1e-9
                tick = workload.tick
                workload.step(action)
                kernel_audit(
                    entry.state,
                    entry.contacts,
                    entry.scatter,
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
                    entry.scatter.tasks.shape[0] > 0,
                )
                kernel_episode(
                    tick,
                    workload.device_delays,
                    workload.initial_position,
                    entry.state,
                    entry.contacts,
                    solver.dyn_state,
                    solver.collider.collider_state,
                    workload.link.idx,
                    left,
                    right,
                    flags,
                    geometry,
                    outcomes,
                )
                kernel_tail_work(
                    entry.contacts,
                    entry.state,
                    entry.scatter,
                    tail_counts,
                    tail_totals,
                    entry.reuse_contact_moments,
                    entry.surface.info.shape.shape[0],
                )
            unfinished = qd_to_numpy(flags)[:, 0]
            if stage == "warmup":
                kernel_abort(flags, outcomes)
            rows = qd_to_numpy(outcomes)
            stages.append(
                {
                    "stage": stage,
                    "steps": length,
                    "per_environment_cumulative_outcomes": rows.tolist(),
                    "cumulative_totals": rows.sum(axis=0).tolist(),
                    "unfinished_episodes": unfinished.tolist(),
                }
            )
            if stage == "warmup":
                workload.restart()
    counts, work, accuracy = (qd_to_numpy(field) for field in (failures, counters, errors))
    observed = work[:, 0] > 0
    trajectory_outcomes = np.asarray(stages[1]["per_environment_cumulative_outcomes"]) - np.asarray(
        stages[0]["per_environment_cumulative_outcomes"]
    )
    result = {
        "scope": args.scope,
        "envs": args.envs,
        "seed": args.seed,
        "warmup_steps": args.warmup,
        "trajectory_steps": args.steps,
        "stages": stages,
        "trajectory_only_totals": trajectory_outcomes.sum(axis=0).tolist(),
        "per_environment_trajectory_only_outcomes": trajectory_outcomes.tolist(),
        "outcome_columns": [
            "started",
            "completed",
            "successful",
            "failed_lift",
            "failed_hold",
            "failed_release",
            "failed_numeric",
            "aborted_by_restart",
            "weak_grip_observed",
            "weak_grip_height_drop_gt_1cm",
        ],
        "criteria": {
            "lift": "At least one bilateral nonzero-finger-contact step with root height 0.08 m above authored initial position during phase [0.3,0.5].",
            "hold": "Every sampled step of phase [0.5,0.6] has bilateral nonzero finger contact and root height >= initial+0.08 m.",
            "release": "At least one step of phase [0.97,1] has zero nonzero-force finger contacts and root height <= initial+0.01 m.",
            "numeric": "Every step in the completed episode passes unchanged native acceptance.",
            "weak_grip": "Maximum earlier root height minus current root height during phase [0.7,0.76]; >0.01 m is a drop proxy, not a measured tangential slip distance.",
            "failure_counts": "Reasons overlap. Failed total is completed minus successful. Warmup cumulative counts are subtracted to obtain trajectory-only counts. Incomplete and restart-aborted episodes are separately reported.",
        },
        "failed_environment_steps": int(counts.sum()),
        "every_step_failures": counts.tolist(),
        "per_environment_work": work.tolist(),
        "counter_columns": [
            "nonzero_contacts",
            "local_integration_retries",
            "pressure_newton_evaluations_sum",
            "pressure_newton_evaluations_max",
            "inverse_corrections",
            "factor_fallbacks",
            "hold_proxy_success_steps",
            "hold_proxy_failed_steps",
            "batch_scatter_overflow_affected_steps",
        ],
        "tail_cost_scope": "Per-environment work counters; these are not individual GPU latency measurements. Line-search scans are not separately counted.",
        "contact_moment_reuse_requested": args.contact_moment_reuse,
        "effective_contact_moment_reuse": entry.reuse_contact_moments,
        "scatter_tasks_per_env": args.scatter_tasks_per_env,
        "additional_per_environment_work": qd_to_numpy(tail_totals).tolist(),
        "additional_work_columns": [
            "packed_boundary_columns_sum",
            "packed_boundary_columns_max",
            "admitted_face_tasks_sum",
            "admitted_face_tasks_max",
            "affine_contracted_face_tasks_sum",
            "moment_prepare_sample_visits_sum",
            "face_scatter_sample_visits_sum",
        ],
        "additional_work_scope": "Work counts across warmup and trajectory. Overflow work is separately counted and does not appear in the bounded admitted-task sample totals; nonlinear pressure/grid work is not included in these sample counts.",
        "error_columns": [
            "full_residual_budget_ratio_max",
            "global_peak_max_Pa",
            "wrench_force_relative_max",
            "wrench_moment_relative_max",
            "actual_mu_max",
            "actual_mu_min",
        ],
        "per_environment_error_maxima": accuracy.tolist(),
        "full_residual_budget_ratio_max": float(accuracy[:, 0].max()),
        "actual_mu_range": [float(accuracy[observed, 5].min()), float(accuracy[observed, 4].max())]
        if observed.any()
        else None,
        "scatter_overflow_calls": int(qd_to_numpy(entry.scatter.overflow_calls)),
        "policy": {
            "scope": "FP32 inference rollout",
            "seed": 99173,
            "layers": [26, 128, 128, 7],
            "reference_control_error_rad": policy_error,
        },
        "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                    Path("genesis/options/rigid_stress.py"),
                    Path("examples/rigid/franka_egg_stress.py"),
                    Path("research/rigid_stress/native_validity_probe.py"),
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
        "note": "Intrusive untimed audit of every environment; no throughput inferred. Grasp criteria concern the scripted rigid trajectory, not fracture or mesh convergence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Completed",
        args.scope,
        args.envs,
        "invalid",
        int(counts.sum()),
        "episode totals",
        stages[-1]["cumulative_totals"],
        flush=True,
    )


if __name__ == "__main__":
    main()
