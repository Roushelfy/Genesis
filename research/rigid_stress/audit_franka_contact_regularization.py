"""Audit raw rigid penetration and fixed-step contact-regularization sensitivity."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .audit_franka_reference import rotation
from .franka_fixture_geometry import fit_rigid
from .summarize_fine_references import closed_source


def final_geometry(run, report, phase):
    row = [frame for frame in report["captures"] if frame["phase"] == phase][-1]
    assert "geometry_snapshot" in row
    finger_geoms = {g["fixture_geom"]: g["source_geom"] for g in report["geometry"]}
    (body_geom,) = set(report["contact_geom_indices"]) - set(finger_geoms)
    source_run = Path(report["source_capture"])
    source_report = json.loads((source_run / "results/summary.json").read_text())
    with np.load(source_run / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz") as bulk:
        body_reference = bulk["vertices"] @ rotation(row["egg_quaternion_wxyz"]).T + row["egg_position_m"]
    records = []
    with np.load(run / "results" / row["geometry_snapshot"]) as world:
        body = world[f"geom{body_geom}_world_vertices_m"]
        nearest = cKDTree(body_reference).query(body)[0]
        assert nearest.max() < 1e-8, nearest.max()
        for contact in row["contacts"]:
            geom = contact["opposite_geom"]
            source_geom = next(g for g in source_report["geometry"] if g["geom"] == finger_geoms[geom])
            assert source_geom["type"] == "GEOM_TYPE.BOX"
            with np.load(source_run / "results" / source_geom["mesh"]) as local:
                box_source = local["vertices_m"]
            box = world[f"geom{geom}_world_vertices_m"]
            orientation, centre, fit_error = fit_rigid(box_source, box)
            assert fit_error < 1e-10
            normal = np.array(contact["body_inward_normal"])
            local_normal = orientation.T @ normal
            normal_axis = int(np.argmax(np.abs(local_normal)))
            assert abs(abs(local_normal[normal_axis]) - 1) < 1e-10
            projection = body @ normal
            witness = body[np.argmin(projection)]
            coordinates = (witness - centre) @ orientation
            transverse = [axis for axis in range(3) if axis != normal_axis]
            half_sizes = np.array(source_geom["data"][:3]) / 2
            inside_face = bool(np.all(np.abs(coordinates[transverse]) <= half_sizes[transverse] + 1e-10))
            # Only call this a face-overlap check when the body support vertex
            # projects inside the finite box face, not merely its infinite plane.
            overlap = float((box @ normal).max() - projection.min())
            raw = contact["penetration_m"]
            record = {
                "fixture_geom": geom,
                "source_geom": finger_geoms[geom],
                "support_vertex_projects_inside_finite_box_face": inside_face,
                "normal_axis": normal_axis,
                "rigid_fit_max_error_m": fit_error,
                "raw_collider_penetration_m": raw,
                "projected_overlap_along_captured_normal_m": overlap,
                "overlap_minus_collider_penetration_m": overlap - raw,
                "penetration_over_wall_thickness": raw / 0.0005,
                "normal_force_N": contact["normal_force_N"],
                "force_over_positive_penetration_N_per_m": contact["normal_force_N"] / raw if raw > 0 else None,
                "mixed_contact_sol_params": contact["mixed_contact_sol_params"],
            }
            if inside_face:
                assert abs(overlap - raw) < 1e-8, record
            records.append(record)
    return {
        "phase": phase,
        "time_s": 0.01 * (row["tick"] + row["physical_substep"] / 8),
        "body_collision_to_independent_mesh_vertex_max_distance_m": float(nearest.max()),
        "contacts": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, fixed, geometry, before_params, weld = [], None, None, None, None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        frozen = {key: value for key, value in config.items() if key not in ("contact_timeconst", "output")}
        assert config["substeps"] == 8
        report = json.loads((run / "results/summary.json").read_text())
        geom = json.loads((run / "results/fixture-assets/manifest.json").read_text())
        if fixed is None:
            fixed, geometry = frozen, geom
            before_params = report["contact_geom_sol_params_before"]
            weld = report["runtime_temporary_weld_sol_params"]
        assert frozen == fixed and geom == geometry and report["runtime_temporary_weld_sol_params"] == weld
        assert report["contact_geom_sol_params_before"] == before_params
        np.testing.assert_array_equal(
            np.array(report["contact_geom_sol_params_after"])[:, 1:], np.array(before_params)[:, 1:]
        )
        for frame in report["captures"]:
            for contact in frame["contacts"]:
                assert contact["friction"] == 0.6
                assert abs(contact["mixed_contact_sol_params"][0] - config["contact_timeconst"]) < 1e-14
        rows = [
            r
            for r in report["captures"]
            if r["phase"] == "released_free_body_fixed_hand" and r["physical_substep"] == 0
        ]
        assert len(rows) == 80
        cases.append((config["contact_timeconst"], run, rows))
    cases.sort(reverse=True)
    comparisons = []
    for a, b in pairwise(cases):
        assert [r["tick"] for r in a[2]] == [r["tick"] for r in b[2]]
        old = np.array([[c["normal_force_N"] for c in r["contacts"]] for r in a[2]])
        new = np.array([[c["normal_force_N"] for c in r["contacts"]] for r in b[2]])
        assert old.shape == new.shape == (80, 2)
        assert all(
            [c["opposite_geom"] for c in x["contacts"]] == [c["opposite_geom"] for c in y["contacts"]]
            for x, y in zip(a[2], b[2], strict=True)
        )
        comparisons.append(
            {
                "previous_contact_timeconst_s": a[0],
                "current_contact_timeconst_s": b[0],
                "common_physical_samples": 80,
                "normal_force_l2_relative_change": float(np.linalg.norm(new - old) / np.linalg.norm(new)),
                "maximum_sample_normal_force_change_N": float(np.max(np.abs(new - old))),
            }
        )
    records = []
    for tau, run, _rows in cases:
        report = json.loads((run / "results/summary.json").read_text())
        records.append(
            {
                "run": str(run),
                "contact_timeconst_s": tau,
                "source_summary_sha256": hashlib.sha256((run / "results/summary.json").read_bytes()).hexdigest(),
                "final_geometric_checks": [
                    final_geometry(run, report, phase)
                    for phase in ("held_body_pd_closure", "released_free_body_fixed_hand")
                ],
            }
        )
    output = {
        "scope": __doc__,
        "fixed_integration_interval_s": 0.00125,
        "fixed_temporary_weld_sol_params": weld,
        "cases": records,
        "common_time_comparisons": comparisons,
        "goal_completion": False,
        "limitations": "Raw collider penetration is a solver-regularization observable, not material deformation. The independent body mesh agrees with runtime collision vertices within the declared geometry guard. Force/depth ratios are state diagnostics, not a universal contact stiffness; dynamics and solver impedance also contribute. Contact-only time constants change at fixed geometry, material, actuation, integration interval and recorded temporary-weld parameters. Three levels do not by themselves establish a credible zero-regularization tail. The body may retain angular motion; solved deformable/dynamic contact, support/history matching and native load replay remain required before interpreting deformation feedback or physical operating range.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
