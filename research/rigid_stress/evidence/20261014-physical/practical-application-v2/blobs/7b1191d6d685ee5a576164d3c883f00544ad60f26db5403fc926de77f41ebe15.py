"""Measure raw-stress reward repeatability and paired 1N/4N grasp discrimination.

Each of eight frozen conditions is duplicated with only the commanded strong
grip force limit changed. The original approach/lift/slide/weak/release schedule,
arm targets, radii, friction and resets are retained. Intrusive observations
make this a behavioral experiment, never a throughput benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.egg_stress_controller import kernel_inputs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array


def choice_step(workload, strong_limits):
    ids = np.flatnonzero((workload.tick > workload.delays) & ((workload.tick - workload.delays) % 600 == 0))
    if len(ids):
        before = tensor_to_array(workload.link.get_max_stress(copy=True))
        workload.reset(ids)
        after = tensor_to_array(workload.link.get_max_stress(copy=True))
        np.testing.assert_array_equal(after[ids], 0)
        remaining = np.setdiff1d(np.arange(workload.n_envs), ids)
        np.testing.assert_array_equal(after[remaining], before[remaining])
    workspace = workload.controller_inputs
    kernel_inputs(
        workload.tick,
        workload.device_delays,
        workload.initial_joints,
        workload.pick_joints,
        workload.lift_joints,
        workload.slide_joints,
        workload.base_radius,
        workspace["arm_indices"],
        workspace["zero_action"],
        workload.phase,
        workspace["target"],
        workspace["grip"],
        workspace["limit"],
        workspace["radius"],
        False,
        False,
        True,
    )
    # Only the strong-grip limit differs; the original weak window is preserved.
    weak = (workload.phase >= 0.7) & (workload.phase < 0.76)
    limit = torch.where(weak, workspace["limit"], strong_limits)[:, None].expand(-1, 2)
    workload.robot.control_dofs_position(workspace["target"][:, :7], range(7))
    workload.robot.set_dofs_force_range(-limit, limit, range(7, 9))
    workload.robot.control_dofs_position(workspace["grip"][:, None].expand(-1, 2), range(7, 9))
    workload.link.set_stress_contact_radius(workspace["radius"][:, None])
    workload.scene.step()
    workload.tick += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--model", choices=["pressure", "body_source_prior"], required=True)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.radius_scale > 0
    args.output.mkdir(parents=True, exist_ok=False)
    bank_hash = hashlib.sha256(args.conditions.read_bytes()).hexdigest()
    with np.load(args.conditions) as saved:
        count = len(saved["position"])
    assert count == 8
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        2 * count,
        1,
        True,
        varied=True,
        conditions=args.conditions,
        history_size=4,
        output_mode="max",
        contact_model="pressure" if args.model == "pressure" else "intrinsic",
        intrinsic_load_partition="body_source_prior",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    workload.base_radius.mul_(args.radius_scale)
    strong_limits = torch.tensor([1.0] * count + [4.0] * count, dtype=torch.get_default_dtype(), device=gs.device)
    for _ in range(900):
        choice_step(workload, strong_limits)
    peak = np.empty((3, 1200, 2 * count))
    height, phase = np.empty_like(peak), np.empty_like(peak)
    resets = []
    for repetition in range(3):
        workload.restart()
        np.testing.assert_array_equal(tensor_to_array(workload.link.get_max_stress()), 0)
        for step in range(1200):
            choice_step(workload, strong_limits)
            peak[repetition, step] = tensor_to_array(workload.link.get_max_stress())
            height[repetition, step] = tensor_to_array(workload.egg.get_pos())[:, 2]
            phase[repetition, step] = tensor_to_array(workload.phase)
        assert np.isfinite(peak[repetition]).all()
        resets.append(workload.reset_count)
        print(
            json.dumps(
                {
                    "repetition": repetition,
                    "raw_peak_Pa": peak[repetition].max(axis=0).tolist(),
                    "resets": workload.reset_count,
                }
            ),
            flush=True,
        )
    qd.sync()
    workload.scene.rigid_solver.check_errno()
    window = (phase >= 0.3) & (phase <= 0.6)
    lift_peak = np.where(window, peak, 0).max(axis=1)
    lift_exposure = np.where(window, peak, 0).sum(axis=1) * 0.01
    lift_height = np.where(window, height, -np.inf).max(axis=1)
    span = np.ptp(lift_peak, axis=0)
    relative_span = span / np.maximum(lift_peak.max(axis=0), np.finfo(float).tiny)
    paired = []
    for index in range(count):
        first, second = lift_peak[:, index], lift_peak[:, index + count]
        paired.append(
            {
                "condition": index,
                "strong_limit_N": [1, 4],
                "lift_window_peak_1N_Pa": first.tolist(),
                "lift_window_peak_4N_Pa": second.tolist(),
                "4N_over_1N": (second / first).tolist(),
                "4N_peak_exceeds_1N_in_all_repetitions": bool((second > first).all()),
                "minimum_separation_Pa": float((second - first).min()),
                "maximum_lift_height_1N_m": lift_height[:, index].tolist(),
                "maximum_lift_height_4N_m": lift_height[:, index + count].tolist(),
            }
        )
    np.savez_compressed(
        args.output / "raw_trace.npz",
        maximum_stress_Pa=peak,
        height_m=height,
        phase=phase,
        strong_limit_N=tensor_to_array(strong_limits),
        condition_index=np.arange(2 * count) % count,
        lift_window_raw_peak_Pa=lift_peak,
        lift_window_stress_exposure_Pa_s=lift_exposure,
    )
    assert hashlib.sha256(args.conditions.read_bytes()).hexdigest() == bank_hash
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "model": args.model,
                "radius_scale": args.radius_scale,
                "condition_bank": str(args.conditions),
                "condition_bank_sha256": bank_hash,
                "envs": 2 * count,
                "warmup": 900,
                "steps_per_repetition": 1200,
                "repetitions": 3,
                "observed_environment_steps": int(peak.size),
                "invalid_stress_steps": int((~np.isfinite(peak)).sum()),
                "partial_resets_per_repetition": resets,
                "lift_window_phase": [0.3, 0.6],
                "lift_window_peak_relative_span": relative_span.tolist(),
                "paired_choices": paired,
                "reset_and_unselected_peak_checks_passed": True,
                "goal_completion": False,
                "limitations": "Raw stress and exposure are uncalibrated reward proxies for two explicitly controlled grip choices. This is numerical repeatability and useful discrimination conditional on the chosen480-tet mesh, actual rigid contact and model/radius assumptions. It does not certify a fracture threshold, physical stress accuracy or complete RL training. Different models can change stress rankings without changing rigid motion. All values and any rejected input remain; intrusive observer timing is excluded from throughput.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
