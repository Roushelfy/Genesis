"""Audit plate equilibria and compare complete solid fields at declared load endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation, tensor_error
from .plates import equilibrium
from .p1_elastic import FreeElasticP1


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--labels", nargs="+", required=True)
    cli.add_argument("--window", type=float, default=0.05)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(args.runs) == len(args.labels)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], []
    baseline = None
    for label, run in zip(args.labels, args.runs):
        receipt_path = run / "result.json"
        start_path = run / "start.json"
        if not receipt_path.is_file() or not start_path.is_file():
            rows.append({"label": label, "run": str(run), "receipt_missing": True, "admitted": False})
            continue
        receipt, start = json.loads(receipt_path.read_text()), json.loads(start_path.read_text())
        history = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        levels = []
        solid_frames = []
        for level, target in enumerate(start["args"]["loads"]):
            end = (level + 1) * (start["args"]["ramp"] + start["args"]["hold"])
            window = [row for row in history if end - args.window - 1e-12 <= row["time_s"] <= end + 1e-12]
            checks = []
            mechanical_checks = []
            for row in window:
                if "egg_kinetic_j" in row or "egg_kinetic_upper_bound_j" in row:
                    mechanical_row = dict(row)
                    mechanical_row["egg_kinetic_j"] = row.get("egg_kinetic_j", row.get("egg_kinetic_upper_bound_j"))
                    mechanical_row["equation_residual_relative"] = 0.0
                    mechanical = equilibrium(mechanical_row, target)
                    del mechanical["equation_residual"]
                    mechanical_checks.append(mechanical)
                if "egg_kinetic_j" in row and "equation_residual_relative" in row:
                    checks.append(equilibrium(row, target))
            frame = run / "frames" / f"level{level}.npz"
            complete = frame.is_file() and any(abs(row["time_s"] - end) < 1e-12 for row in history)
            levels.append(
                {
                    "load_level": level,
                    "command_force_n": target,
                    "endpoint_s": end,
                    "frame": file_id(frame) if frame.is_file() else None,
                    "requested_endpoint_present": complete,
                    "settling_window_observations": len(window),
                    "observations_with_complete_equilibrium_metrics": len(checks),
                    "mechanical_window_passed": bool(
                        complete
                        and len(window) >= 5
                        and len(mechanical_checks) == len(window)
                        and all(all(check.values()) for check in mechanical_checks)
                    ),
                    "endpoint_equation_passed": bool(
                        complete
                        and window
                        and window[-1].get("equation_residual_relative", float("inf")) <= 1e-3
                        and window[-1].get("invalid_stress_steps", 0) == 0
                    ),
                    "pcg_caps_observed_in_window": sum(row.get("pcg_max", 0) >= 4096 for row in window),
                    "entire_window_equilibrium_passed": bool(
                        complete
                        and len(window) >= 5
                        and len(checks) == len(window)
                        and all(all(check.values()) for check in checks)
                    ),
                    "last_observation": window[-1] if window else None,
                }
            )
            if frame.is_file() and "config" in start:
                with np.load(run / "rest.npz") as mesh, np.load(frame) as saved:
                    vertices, tets = mesh["vertices"], mesh["tetrahedra"]
                    volumes = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
                    masses = np.bincount(
                        tets.reshape(-1),
                        weights=np.repeat(volumes * start["config"]["density"] / 4, 4),
                        minlength=len(vertices),
                    )
                    rotation = corotation(vertices, saved["positions_m"], masses)
                    tensor = rotation.T @ saved["cauchy_pa"] @ rotation
                    center = masses @ saved["positions_m"] / masses.sum()
                    relative = saved["positions_m"] - center
                    velocity = saved["velocities_m_per_s"]
                    mean_velocity = masses @ velocity / masses.sum()
                    inertia = np.eye(3) * np.einsum("v,vi,vi->", masses, relative, relative) - np.einsum(
                        "v,vi,vj->ij", masses, relative, relative
                    )
                    angular_momentum = (masses[:, None] * np.cross(relative, velocity - mean_velocity)).sum(axis=0)
                    angular_velocity = np.linalg.solve(inertia, angular_momentum)
                    deformational_velocity = velocity - mean_velocity - np.cross(angular_velocity, relative)
                    elastic = FreeElasticP1(
                        vertices,
                        tets,
                        start["config"]["young"],
                        start["config"]["poisson"],
                        start["config"]["density"],
                    )
                    static, displacement, correction, residual = elastic.solve(saved["contact_forces_n"] @ rotation)
                    footprints = {}
                    for side, sign in (("upper", -1), ("lower", 1)):
                        weights = np.maximum(sign * saved["contact_forces_n"][:, 2], 0)
                        selected = weights > max(target, 1e-3) * 1e-8
                        center_xy = weights @ saved["positions_m"][:, :2] / max(weights.sum(), 1e-30)
                        radius_squared = ((saved["positions_m"][:, :2] - center_xy) ** 2).sum(axis=1)
                        footprints[side] = {
                            "positive_normal_resultant_n": weights.sum().item(),
                            "reported_vertex_force_threshold_n": max(target, 1e-3) * 1e-8,
                            "force_bearing_vertices": int(selected.sum()),
                            "vertex_indices": np.flatnonzero(selected).tolist(),
                            "vertex_forces_n": weights[selected].tolist(),
                            "weighted_center_xy_m": center_xy.tolist(),
                            "weighted_rms_radius_m": np.sqrt(
                                weights @ radius_squared / max(weights.sum(), 1e-30)
                            ).item(),
                            "maximum_force_bearing_radius_m": np.sqrt(radius_squared[selected]).max().item()
                            if selected.any()
                            else None,
                            "scope": "Actual FEM nodal normal loads; no inferred finite pressure area for a point force.",
                        }
                    levels[-1]["normal_traction_footprints"] = footprints
                    levels[-1]["motion_diagnostics"] = {
                        "center_velocity_m_per_s": mean_velocity.tolist(),
                        "angular_velocity_rad_per_s": angular_velocity.tolist(),
                        "deformational_kinetic_j": (0.5 * masses[:, None] * deformational_velocity**2).sum().item(),
                        "translation_kinetic_j": (0.5 * masses.sum() * np.dot(mean_velocity, mean_velocity)).item(),
                        "rotation_kinetic_j": (0.5 * angular_velocity @ inertia @ angular_velocity).item(),
                        "material_rotation_matrix": rotation.tolist(),
                    }
                    levels[-1]["independent_same_traction_static_P1"] = {
                        "complete_tensor_difference": tensor_error(tensor, static, volumes),
                        "gauge_balance_correction_l2_n": np.linalg.norm(correction).item(),
                        "gauge_balance_correction_resultant_n": correction.sum(axis=0).tolist(),
                        "equation_residual_l2_n": np.linalg.norm(residual).item(),
                        "maximum_displacement_m": np.linalg.norm(displacement, axis=1).max().item(),
                        "scope": "Independent CPU linear Lamé solve on the same full P1 domain, using actual measured nodal traction rotated to material coordinates. Six mass-orthogonal rigid modes are gauged; report every balancing correction, not a hidden support reaction. This controls elasticity at the observed footprint, not contact convergence.",
                    }
                    solid_frames.append((tensor, volumes, file_id(frame)))
        observed_caps = sum(row.get("pcg_max", 0) >= 4096 for row in history)
        admitted = bool(
            receipt["completed"]
            and levels
            and all(level["entire_window_equilibrium_passed"] for level in levels)
            and receipt.get("newton_cap_steps", 0) == 0
            and receipt.get("pcg_cap_steps") == 0
            and not any(row.get("edge_triangle_intersection_count", 0) for row in history)
        )
        rows.append(
            {
                "label": label,
                "run": str(run),
                "receipt": file_id(receipt_path),
                "source_input": start.get("input"),
                "completed": receipt["completed"],
                "pcg_cap_steps_recorded": receipt.get("pcg_cap_steps"),
                "pcg_cap_observations_lower_bound": observed_caps,
                "load_levels": levels,
                "admitted_equilibrium_and_solver_exits": admitted,
                "mechanical_equilibrium_with_endpoint_equation": bool(
                    receipt["completed"]
                    and levels
                    and all(level["mechanical_window_passed"] and level["endpoint_equation_passed"] for level in levels)
                ),
                "reference_qualified": False,
            }
        )
        if baseline is None and solid_frames:
            baseline = (label, solid_frames)
        elif baseline is not None and solid_frames:
            assert len(baseline[1]) == len(solid_frames)
            differences = []
            for level, (expected, actual) in enumerate(zip(baseline[1], solid_frames)):
                np.testing.assert_array_equal(expected[1], actual[1])
                differences.append(
                    {
                        "load_level": level,
                        "baseline_field": expected[2],
                        "candidate_field": actual[2],
                        "full3D_solid_volume_weighted_tensor": tensor_error(actual[0], expected[0], actual[1]),
                        "scope": "Same tetrahedral mesh, complete spatial field, corotated material frame; no temporal average or smoothing.",
                    }
                )
            comparisons.append({"baseline": baseline[0], "candidate": label, "levels": differences})
    write_json(
        args.output / "result.json",
        {
            "kind": "parallel-plate-pilot-admission",
            "identity": runtime_identity(),
            "settling_window_s": args.window,
            "runs": rows,
            "same_mesh_solid_field_contrasts": comparisons,
            "scope": "Admission checks and separate inner/outer-tolerance contrasts. No physical reference qualification from numerical contrasts alone; spatial, timestep and contact uncertainty remain independent gates.",
        },
    )


if __name__ == "__main__":
    main()
