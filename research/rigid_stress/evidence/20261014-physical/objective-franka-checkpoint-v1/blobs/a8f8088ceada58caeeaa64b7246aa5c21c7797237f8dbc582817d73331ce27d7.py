"""Audit captured rigid wrenches, temporal settling and complete finger geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def rotation(quaternion):
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cross_matrix(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def geometry_file(run, row):
    stem = f"{row['mode']}-{row['tick']:04d}"
    current = run / f"results/{stem}-s{row['substep']:02d}_geometry.npz"
    return current if current.exists() else run / f"results/{stem}_geometry.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnose-geometry", action="store_true", help="Retain failed geometry guards as unresolved diagnostics"
    )
    parser.add_argument(
        "--body-metadata", type=Path, default=Path("examples/rigid/assets/hollow_egg/level1/egg_shell.metadata.json")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        report = json.loads((run / "results/summary.json").read_text())
        metadata_path = run / "source/examples/rigid/assets/hollow_egg/level1/egg_shell.metadata.json"
        contemporaneous_asset = metadata_path.exists()
        if not contemporaneous_asset:
            metadata_path = args.body_metadata
        metadata = json.loads(metadata_path.read_text())
        mass, local_com = metadata["mass_kg"], np.array(metadata["com_m"])
        hold = [r for r in report["captures"] if r["mode"] == "stationary_lift_hold"]
        rows = []
        for row in hold:
            transform = rotation(row["egg_quaternion_wxyz"])
            np.testing.assert_allclose(transform.T @ transform, np.eye(3), rtol=0, atol=1e-12)
            com = np.array(row["egg_position_m"]) + transform @ local_com
            if "egg_world_com_m" in row:
                np.testing.assert_allclose(com, row["egg_world_com_m"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(mass, row["egg_mass_kg"], rtol=0, atol=1e-14)
            points = np.array([c["world_point_m"] for c in row["contacts"]])
            forces = np.array([c["body_force_N"] for c in row["contacts"]])
            matrix = np.concatenate([np.vstack((np.eye(3), cross_matrix(p - com) / 0.02)) for p in points], axis=1)
            target = np.array([0, 0, mass * 9.81, 0, 0, 0])
            _, values, _ = np.linalg.svd(matrix)
            projected = matrix @ np.linalg.lstsq(matrix, target, rcond=1e-10)[0]
            observed = np.r_[forces.sum(axis=0), np.cross(points - com, forces).sum(axis=0) / 0.02]
            rows.append(
                {
                    "tick": row["tick"],
                    "substep": row["substep"],
                    "contacts": len(points),
                    "effective_friction": [c["friction"] for c in row["contacts"]],
                    "normal_forces_N": [c["normal_force_N"] for c in row["contacts"]],
                    "body_speed_m_s": float(np.linalg.norm(row["egg_velocity_m_s"])),
                    "angular_speed_rad_s": float(np.linalg.norm(row["egg_angular_velocity_rad_s"])),
                    "complete_observed_static_wrench_residual_scaled_N": float(np.linalg.norm(observed - target)),
                    "unconstrained_fixed_point_force_span_residual_scaled_N": float(np.linalg.norm(projected - target)),
                    "wrench_span_singular_values": values.tolist(),
                    "fixed_point_wrench_rank": int(np.count_nonzero(values > values[0] * 1e-10)),
                    "target_static_world_force_N": target[:3].tolist(),
                    "observed_origin_moment_about_com_Nm": (observed[3:] * 0.02).tolist(),
                }
            )
        first, last = hold[0], hold[-1]
        r_first, r_last = rotation(first["hand_quaternion_wxyz"]), rotation(last["hand_quaternion_wxyz"])
        axes, errors = [], []
        with np.load(geometry_file(run, first)) as a, np.load(geometry_file(run, last)) as b:
            for index, finger in enumerate(last["fingers"]):
                axis = r_last.T @ rotation(finger["world_quaternion_wxyz"]) @ [0, 1, 0]
                axes.append(axis.tolist())
                dq = last["finger_position_m"][index] - first["finger_position_m"][index]
                for geom in report["geometry"]:
                    if geom["link"] != finger["link"]:
                        continue
                    name = f"geom{geom['geom']}_world_vertices_m"
                    old = (a[name] - first["hand_position_m"]) @ r_first
                    new = (b[name] - last["hand_position_m"]) @ r_last
                    error = float(np.linalg.norm(new - old - dq * axis, axis=1).max())
                    delta = new - old
                    diagnostic = {
                        "geom": geom["geom"],
                        "rigid_slider_reconstruction_error_m": error,
                        "unchanged_geometry_guard_passed": error < 1e-10,
                        "mean_translation_error_m": (delta.mean(axis=0) - dq * axis).tolist(),
                        "nontranslation_change_m": float(np.linalg.norm(delta - delta.mean(axis=0), axis=1).max()),
                        "joint_translation_m": dq,
                    }
                    errors.append(diagnostic)
                    if not args.diagnose_geometry:
                        assert diagnostic["unchanged_geometry_guard_passed"], (run, diagnostic)
        summaries.append(
            {
                "run": str(run),
                "substeps": report.get("substeps", 1),
                "mass_kg": mass,
                "body_asset_contemporaneously_snapshotted": contemporaneous_asset,
                "body_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "rows": rows,
                "finger_axes_in_hand_frame": axes,
                "geometry_checks": errors,
                "summary_sha256": hashlib.sha256((run / "results/summary.json").read_bytes()).hexdigest(),
            }
        )
    output = {
        "scope": __doc__,
        "runs": summaries,
        "goal_completion": False,
        "limitations": "Wrench moments are scaled by fixed20mm solely for rank/residual diagnostics. Unconstrained point-force span is an outer support test, not a friction-cone feasibility or global grasp impossibility claim. Contact birth, geometry/pose changes and finite contact moments can change support. No stationary frame is accepted from translational speed alone; angular motion, six-component equilibrium and temporal trends remain explicit. All12 finger geoms must be retained in corresponding references.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
