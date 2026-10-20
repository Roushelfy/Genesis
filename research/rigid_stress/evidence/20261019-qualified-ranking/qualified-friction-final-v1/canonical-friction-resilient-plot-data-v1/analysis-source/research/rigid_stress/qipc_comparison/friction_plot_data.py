"""Prepare audited actual force histories and unsmoothed pad traction samples."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--reference-run", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, histories, traction, checks = [], [], [], {}
    native_arrays = {}

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    runs = (
        ("dpr", args.data_root / "native-pad-full-source-stability-v1/group2/native"),
        ("scr", args.data_root / "native-pad-full-source-stability-v1/group3/native"),
        ("shell", args.data_root / "compliant-pad-friction250-history-v1/group0/fine"),
        ("reference", args.reference_run),
    )
    for method, run in runs:
        start, result = read(run / "start.json"), read(run / "result.json")
        parameters = start["args"]
        assert parameters["loads"] == [0.1, 0.1]
        assert parameters["dt"] == 0.001 and parameters["friction"] == 0.3 and parameters["gravity_x"] == 1
        assert parameters["record_stride"] == 1
        assert result["completed"]
        if method == "shell":
            audit = read(run.parent / "audit/result.json")["runs"][0]
            assert audit["result"] == file_id(run / "result.json")
            assert audit["completed"] and len(audit["equilibrium_windows"]) == 2
            assert all(w["mechanical_window_passed"] for w in audit["equilibrium_windows"])
            fields = {f["field"]["path"]: f["field"] for f in audit["fields"]}
        else:
            assert result["steps"] == 400 and result["mechanical_history_passed"]
            assert all(result["history_checks"].values())
            assert len(result["windows"]) == 2 and all(w["mechanical_window_passed"] for w in result["windows"])
            fields = None
            if method == "reference":
                assert start["fixed_dynamic_dofs"] == 0 and parameters["elasticity"] == "stvk"
                audit = read(run.parent / "moment/result.json")
                assert audit["passed"] and audit["run"] == file_id(run / "result.json")
                fields = {f["field"]["path"]: f["field"] for f in audit["frames"]}
        journal = run / "steps.jsonl"
        receipts.append(file_id(journal))
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(rows) == 400 and [r["step"] for r in rows] == list(range(1, 401))
        endpoints = [next(r for r in reversed(rows) if r["level"] == level) for level in range(2)]
        for row in rows:
            if method == "shell":
                assert not any(row["solver_failure_flags"]) and row["equation_residual_relative"] <= 1e-3
                assert row["physical_pad"]["normal_law_and_transfer_passed"]
                assert row["physical_pad"]["friction"]["lagged_law_passed"]
                body = row["body_force_resultant_n"]
                tangent = row["physical_pad"]["friction"]["tangential_resultant_n"]
            else:
                body, tangent = row["body_force_n"], row["tangential_resultant_n"]
                assert row["equation_residual_relative"] <= 1e-7
            histories.append(
                {
                    "method": method,
                    "step": row["step"],
                    "time_s": row["time_s"],
                    "level": row["level"],
                    "command_force_n": row["command_force_n"],
                    "upper_reaction_n": row["upper_reaction_n"],
                    "lower_reaction_n": row["lower_reaction_n"],
                    "plate_gap_closure_m": row["compression_coordinate_m"] - 2 * parameters["clearance"],
                    "body_force_x_n": body[0],
                    "tangential_force_x_n": tangent[0],
                    "tangential_force_y_n": tangent[1],
                    "egg_inertia_resultant_norm_n": row["egg_inertia_resultant_norm_n"],
                    "egg_kinetic_j": row["egg_kinetic_j"],
                    "mass_kg": start["mass_kg"],
                }
            )
        for level, row in enumerate(endpoints):
            path = run / "frames" / f"level{level}.npz"
            receipt = file_id(path)
            receipts.append(receipt)
            if fields is not None:
                assert fields[str(path)] == receipt, "Frame changed after its own audit"
            with np.load(path) as saved:
                if method in ("dpr", "scr"):
                    # These are the actual prior-point locations used by the
                    # semi-implicit rigid wrench, not the later predicted points.
                    points = saved["applied_points_world_m"]
                    final_points = saved["points_world_m"]
                    force, area = saved["contact_force_n"], saved["projected_area_m2"]
                    assert row["stress"]["native_valid"]
                elif method == "shell":
                    points = final_points = saved["pad_points_world_m"]
                    force, area = saved["pad_point_force_n"], saved["pad_projected_area_m2"]
                else:
                    points = final_points = saved["source_points_current_m"]
                    force, area = saved["source_forces_n"], saved["projected_area_m2"]
            assert points.shape == force.shape == (len(area), 3)
            assert all(np.isfinite(a).all() for a in (points, final_points, force, area)) and np.all(area >= 0)
            active = np.any(force != 0, axis=1)
            assert active.any() and np.all(area[active] > 0)
            upper = points[:, 2] > 0
            sign = np.where(upper, -1, 1)
            normal = sign * force[:, 2]
            assert np.all(normal >= 0)
            normal_density, tangent_density = np.zeros(len(area)), np.zeros((len(area), 2))
            normal_density[active] = normal[active] / area[active]
            tangent_density[active] = force[active, :2] / area[active, None]
            totals = np.stack([force[~upper].sum(axis=0), force[upper].sum(axis=0)])
            expected = (
                row["physical_pad"]["friction"]["tangential_resultant_n"]
                if method == "shell"
                else row["tangential_resultant_n"]
            )
            np.testing.assert_allclose(totals.sum(axis=0)[:2], expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                [totals[0, 2], -totals[1, 2]],
                [row["lower_reaction_n"], row["upper_reaction_n"]],
                rtol=0,
                atol=1e-12,
            )
            if method in ("dpr", "scr"):
                native_arrays[method, level] = (points, final_points, force, area)
            output = args.output / f"traction-{method}-level{level}.npz"
            np.savez_compressed(
                output,
                applied_points_world_m=points,
                final_points_world_m=final_points,
                force_n=force,
                reference_projected_area_m2=area,
                active=active,
                upper=upper,
                normal_density_pa=normal_density,
                tangent_density_pa=tangent_density,
            )
            traction.append(
                {
                    "method": method,
                    "level": level,
                    "input": receipt,
                    "output": file_id(output),
                    "point_count": len(area),
                    "active_point_count": int(active.sum()),
                    "complete_active_xy_extent_m": float(np.abs(points[active, :2]).max()),
                    "maximum_normal_density_pa": float(normal_density.max()),
                    "maximum_tangent_density_pa": float(np.linalg.norm(tangent_density, axis=1).max()),
                    "maximum_current_cone_excess_n": float((np.linalg.norm(force[:, :2], axis=1) - 0.3 * normal).max()),
                    "lower_upper_resultants_n": totals.tolist(),
                    "lower_upper_world_moments_nm": [
                        np.cross(points[mask], force[mask]).sum(axis=0).tolist() for mask in (~upper, upper)
                    ],
                    "maximum_application_to_final_point_distance_m": float(
                        np.linalg.norm(final_points - points, axis=1).max()
                    ),
                }
            )
        checks[method + "_own_history_and_traction_totals"] = True
    for level in range(2):
        for first, second in zip(native_arrays["dpr", level], native_arrays["scr", level]):
            np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-12)
    checks["native_dpr_scr_actual_contact_data_agree"] = True
    with (args.output / "force-history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(histories)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": all(checks.values()),
            "checks": checks,
            "receipts": receipts,
            "trajectory_rows": len(histories),
            "traction": traction,
            "force_history": file_id(args.output / "force-history.csv"),
            "spatial_reference_qualified": False,
            "whole_goal_complete": False,
            "scope": "Data preparation only, with no rendered figure or accuracy qualification. Actual complete400-step mechanics and all unsmoothed point forces are retained. Native points are the actual prior-point wrench locations, with final points preserved separately; Shell/P2 use their recorded current contact points. Densities divide actual forces by declared projected reference areas, not deformed surface area. All nonzero force samples, including tiny residues, are kept. Requested native equilibrium observations do not admit its failed every-step mode. Final plots require a separate matching spatial qualification and common-field inputs.",
        },
    )


if __name__ == "__main__":
    main()
