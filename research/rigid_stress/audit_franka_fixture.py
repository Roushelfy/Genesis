"""Audit fixed-hand rigid history, temporary support motion and static diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_franka_reference import cross_matrix, rotation
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    report = json.loads((args.run / "results/summary.json").read_text())
    config = json.loads((args.run / "results/configuration.json").read_text())
    capture = Path(report["source_capture"])
    closed_source(capture)
    original = json.loads((capture / "results/summary.json").read_text())
    original_pose = original["captures"][-1]
    metadata = original["body_asset_metadata"]
    mass, local_com = metadata["mass_kg"], np.array(metadata["com_m"])
    data = []
    held_positions, held_angles = [], []
    for frame in report["captures"]:
        orientation = rotation(frame["egg_quaternion_wxyz"])
        centre = np.array(frame["egg_position_m"]) + orientation @ local_com
        contacts = frame["contacts"]
        points = np.array([row["world_point_m"] for row in contacts]).reshape(-1, 3)
        force = np.array([row["body_force_N"] for row in contacts]).reshape(-1, 3)
        resultant = force.sum(axis=0) + [0, 0, -mass * 9.81]
        moment = np.cross(points - centre, force).sum(axis=0)
        if len(points):
            matrix = np.concatenate(
                [np.vstack((np.eye(3), cross_matrix(point - centre) / 0.02)) for point in points], axis=1
            )
            values = np.linalg.svd(matrix, compute_uv=False)
            target = np.array([0, 0, mass * 9.81, 0, 0, 0])
            span_residual = float(np.linalg.norm(matrix @ np.linalg.lstsq(matrix, target, rcond=1e-10)[0] - target))
            rank = int(np.count_nonzero(values > values[0] * 1e-10))
        else:
            values, span_residual, rank = np.array([]), mass * 9.81, 0
        if frame["phase"] == "held_body_pd_closure":
            held_positions.append(
                float(np.linalg.norm(np.array(frame["egg_position_m"]) - original_pose["egg_position_m"]))
            )
            relative = orientation @ rotation(original_pose["egg_quaternion_wxyz"]).T
            held_angles.append(float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
        data.append(
            {
                "phase": frame["phase"],
                "tick": frame["tick"],
                "physical_substep": frame["physical_substep"],
                "physical_time_s": 0.01 * (frame["tick"] + frame["physical_substep"] / config["substeps"]),
                "contacts": len(points),
                "contact_geoms": [row["opposite_geom"] for row in contacts],
                "normal_forces_N": [row["normal_force_N"] for row in contacts],
                "jaw_position_m": frame["jaw_position_m"],
                "jaw_control_force_N": frame["jaw_control_force_N"],
                "body_speed_m_s": float(np.linalg.norm(frame["egg_velocity_m_s"])),
                "body_angular_speed_rad_s": float(np.linalg.norm(frame["egg_angular_velocity_rad_s"])),
                "body_net_force_including_gravity_N": resultant.tolist(),
                "body_net_moment_about_com_Nm": moment.tolist(),
                "complete_static_wrench_residual_scaled_N": float(np.linalg.norm(np.r_[resultant, moment / 0.02])),
                "point_force_span_residual_scaled_N": span_residual,
                "point_force_wrench_rank": rank,
                "point_force_wrench_singular_values": values.tolist(),
            }
        )
    released = [row for row in data if row["phase"] == "released_free_body_fixed_hand"]
    assert len(released) > 4
    assert len(data) == len({(r["phase"], r["tick"], r["physical_substep"]) for r in data})
    last = released[-1]
    tail = released[len(released) // 2 :]
    output = {
        "scope": __doc__,
        "run": str(args.run),
        "captures": len(data),
        "temporary_weld_max_body_position_change_m": max(held_positions),
        "temporary_weld_max_body_orientation_change_rad": max(held_angles),
        "released_tail_sampled_speed_rms_m_s": float(np.sqrt(np.mean([r["body_speed_m_s"] ** 2 for r in tail]))),
        "released_tail_sampled_angular_speed_rms_rad_s": float(
            np.sqrt(np.mean([r["body_angular_speed_rad_s"] ** 2 for r in tail]))
        ),
        "last_released_frame": last,
        "time_history": data,
        "goal_completion": False,
        "limitations": "Static wrench/span diagnostics are not dynamic equilibrium errors or global Coulomb feasibility proofs. Sampled histories do not prove temporal convergence. The held rigid weld is compliant whereas the current independent initializer pins six displacement rows. Full arm PD and soft finger equality are omitted; no rigid/deformable feedback accuracy is inferred until supports/history, corresponding solved contact and temporal/spatial/regularization axes are qualified.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "time_history"}), flush=True)


if __name__ == "__main__":
    main()
