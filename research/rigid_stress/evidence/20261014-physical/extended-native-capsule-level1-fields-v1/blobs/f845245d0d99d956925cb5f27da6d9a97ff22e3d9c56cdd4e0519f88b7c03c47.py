"""Locate independently audited contact samples relative to fixed numerical refinement caps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_franka_reference import rotation
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) == len(args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for run, audit_run in zip(args.run, args.audit, strict=True):
        for source in [run, audit_run]:
            closed_source(source)
            assert int((source / "returncode.txt").read_text()) == 0
        config_path, audit_path = run / "results/configuration.json", audit_run / "results/summary.json"
        config = json.loads(config_path.read_text())
        audit = json.loads(audit_path.read_text())
        assert Path(audit["run"]).resolve() == run.resolve() and audit["states_checked"] == config["steps"]
        geometry = Path(config["geometry_run"])
        closed_source(geometry)
        geometry_path = geometry / "results/summary.json"
        metadata = json.loads(geometry_path.read_text())
        anchors = np.array(metadata["fixed_unit_sphere_marking_anchors"])
        angle = metadata["conservative_parameter_cap_angle_rad"]
        g = metadata["declared_geometry"]
        closure = Path(config["closure_run"]) / "results/configuration.json"
        capture = Path(json.loads(closure.read_text())["capture"])
        closed_source(capture)
        capture_path = capture / "results/summary.json"
        frame = [r for r in json.loads(capture_path.read_text())["captures"] if r["mode"] == "stationary_lift_hold"][-1]
        matrix = rotation(frame["egg_quaternion_wxyz"])
        shift = np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
        contact_path = run / "results/contact_geometry.npz"
        with np.load(contact_path) as saved:
            rest, faces = saved["rest_positions"].copy(), saved["faces"].copy()
        edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
        incident = np.zeros(len(rest))
        np.maximum.at(incident, edges.ravel(), np.repeat(lengths, 2))
        for original in audit["rows"]:
            step = original["step"]
            field_path = audit_run / f"results/step{step:04d}.npz"
            with np.load(field_path) as saved:
                active = saved["active_body_contact_nodes"].copy()
                normal = saved["normal_surface_force_N"].copy()
            positions = (rest[active] - shift) @ matrix
            parameter = positions[:, 2] / g["polar_radius_m"]
            direction = np.column_stack(
                (positions[:, :2] / (g["equatorial_radius_m"] * (1 - g["taper"] * parameter[:, None])), parameter)
            )
            norms = np.linalg.norm(direction, axis=1)
            direction /= norms[:, None]
            angles = np.arccos(np.clip(direction @ anchors.T, -1, 1)).min(axis=1)
            outside = angles > angle
            load = np.linalg.norm(normal[active], axis=1)
            row = {
                "run": str(run),
                "step": step,
                "time_s": original["time_s"],
                "tetrahedra": original["body_tetrahedra"],
                "active_body_samples": len(active),
                "active_samples_outside_marking_caps": int(outside.sum()),
                "outside_unsigned_normal_sample_load_fraction": float(load[outside].sum() / load.sum()),
                "minimum_and_maximum_sample_angle_over_cap_angle": [
                    float((angles / angle).min()),
                    float((angles / angle).max()),
                ],
                "active_reference_trace_incident_edge_m_percentiles": np.percentile(
                    incident[active], [0, 50, 100]
                ).tolist(),
                "maximum_sample_radial_parameter_norm_difference_from_one": float(abs(norms - 1).max()),
                "raw_global_peak_lower_Pa": original["raw_global_peak_lower_Pa"],
            }
            rows.append(row)
            hashes[str(field_path)] = hashlib.sha256(field_path.read_bytes()).hexdigest()
        for path in [config_path, audit_path, geometry_path, closure, capture_path, contact_path]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}
    result = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": hashes,
        "physical_accuracy_accepted": False,
        "goal_completion": False,
        "limitations": "Reference contact-sample positions are mapped to the same angular parameter used for numerical mesh marking; this is not a geodesic distance or resolved physical contact boundary. Unsigned sample-force norms avoid cancellation of opposing jaws and are not a traction-area integral. Incident trace edges are measured in the undeformed collision discretization. Samples outside caps reveal coverage gaps; samples inside caps do not exclude missed contact caused by coarse geometry elsewhere. No stress smoothing, force/geometry modification, convergence acceptance or original goal completion follows.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "last_states": [r for r in rows if r["step"] == 5]}), flush=True)


if __name__ == "__main__":
    main()
