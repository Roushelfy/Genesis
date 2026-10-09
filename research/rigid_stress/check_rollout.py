"""Audit physical grasp stages and identical-mesh numerical errors from recorded rigid trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-grasp-failures", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.rollout.read_text())
    frames = source["records"]
    phase = np.array([item["phase"] for item in frames])
    heights = np.array([item["com_position_m"] for item in frames])[..., 2]
    grasp_contacts = np.array([item["grasp_contact_count"] for item in frames])
    tangent_force = np.array([item["tangent_force_norm_sum_n"] for item in frames])
    tangent_speed = np.array([item["pre_solve_grasp_tangent_speed_max_m_s"] for item in frames])
    peaks = np.array([item["peak_pa"] for item in frames])
    residual_n = np.array([item["full_residual_absolute_n"] for item in frames])
    rhs_norm = np.array([item["rhs_norm_n"] for item in frames])
    peak_error_pa = np.array([item["gpu_peak_error_pa"] for item in frames])
    relative_error = peak_error_pa / np.maximum(peaks, 1)
    allowed = np.maximum(1e-11, 1e-6 * rhs_norm)
    equations_passed = bool(np.all(residual_n <= allowed))
    output_passed = bool(relative_error.max() <= 1e-4)
    environments = []
    for i_env in range(source["envs"]):
        rest = float(np.median(heights[phase < 0.15, i_env]))
        lift = float(heights[(phase >= 0.3) & (phase < 0.5), i_env].max())
        hold = float(heights[(phase >= 0.5) & (phase < 0.6), i_env].min())
        hold_contact = bool(np.all(grasp_contacts[(phase >= 0.5) & (phase < 0.6), i_env] > 0))
        slip = float(tangent_speed[(phase >= 0.7) & (phase < 0.76), i_env].max())
        released_height = float(np.median(heights[-10:, i_env]))
        released_contacts = int(grasp_contacts[-10:, i_env].max())
        passed = lift - rest >= 0.08 and hold - rest >= 0.08 and hold_contact and released_height <= rest + 0.01
        environments.append(
            {
                "environment": i_env,
                "grasp_passed": bool(passed),
                "rest_com_height_m": rest,
                "maximum_lift_com_height_m": lift,
                "minimum_hold_com_height_m": hold,
                "hold_finger_contacts_present": hold_contact,
                "weak_grip_tangent_speed_max_m_s": slip,
                "slip_observed": slip >= 0.005,
                "tangent_force_max_n": float(tangent_force[:, i_env].max()),
                "released_com_height_m": released_height,
                "released_finger_contacts_max": released_contacts,
                "peak_max_pa": float(peaks[:, i_env].max()),
            }
        )
    above_one_pa = peaks > 1
    report = {
        "source": str(args.rollout),
        "scope": "Recorded real rigid grasp with per-frame host CPU FP64 verification",
        "physical_mesh_converged": source["stress_mesh_converged"],
        "passed": equations_passed and output_passed and all(item["grasp_passed"] for item in environments),
        "equations_passed": equations_passed,
        "output_error_passed": output_passed,
        "grasp_success_fraction": sum(item["grasp_passed"] for item in environments) / source["envs"],
        "absolute_residual_floor_n": 1e-11,
        "relative_residual_max_above_1pa": float((residual_n / np.maximum(rhs_norm, 1e-30))[above_one_pa].max()),
        "absolute_peak_error_max_pa": float(peak_error_pa.max()),
        "relative_peak_error_max_above_1pa": float(relative_error[above_one_pa].max()),
        "absolute_peak_error_max_at_or_below_1pa": float(peak_error_pa[~above_one_pa].max(initial=0)),
        "near_zero_frames": int((~above_one_pa).sum()),
        "net_contact_force_difference_max_n": max(max(item["net_contact_force_difference_n"]) for item in frames),
        "newton_balance_force_error_max_n": max(max(item["newton_balance_force_error_n"]) for item in frames),
        "euler_balance_torque_error_max_nm": max(max(item["euler_balance_torque_error_nm"]) for item in frames),
        "finite_patch_moment_error_max_nm": max(
            max(item["finite_patch_point_moment_difference_nm"]) for item in frames
        ),
        "egg_as_a_samples": sum(sum(item["egg_as_a_contact_count"]) for item in frames),
        "egg_as_b_samples": sum(sum(item["egg_as_b_contact_count"]) for item in frames),
        "environments": environments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not equations_passed or not output_passed or (not args.allow_grasp_failures and not report["passed"]):
        raise ArithmeticError("Recorded grasp or same-mesh numerical acceptance failed")


if __name__ == "__main__":
    main()
