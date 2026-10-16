"""Locate rejected rigid-step CCD candidates without accepting or changing physical states."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .original_franka_ground import build_original
from .run_original_rigid_franka import OriginalRigidEggContact
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.recovery]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) != 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    system = OriginalRigidEggContact(static, frame, capture, args.mjcf, config["dt"])
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    with np.load(paths[-1]) as saved:
        previous, q, velocity, body_velocity = [
            saved[key].copy() for key in ["displacement", "joint_position", "joint_velocity", "body_velocity"]
        ]
    substeps = round(protocol["macro_dt_s"] / config["dt"])
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    system.target = np.array(controls[len(paths) // substeps]["target"])
    system.caps[7:] = controls[len(paths) // substeps]["finger_cap_N"]
    system.begin_dynamic_step(q, velocity, previous[: system.n] + config["dt"] * body_velocity)
    mesh = system.contact.mesh
    start = mesh.vertices(system.contact.rest + previous.reshape(-1, 3))
    ids = mesh.to_full_vertex_id()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted((args.recovery / "results/numerical_recovery").glob("attempt*.npz")):
        with np.load(path) as saved:
            unknown = np.asarray(saved["unknown"], dtype=np.longdouble)
        result = system.evaluate_rigid(unknown[:6], unknown[6:], previous, False)
        end = mesh.vertices(system.contact.rest + result[2].reshape(-1, 3))
        candidates = ipctk.Candidates()
        candidates.build(mesh, start, end)
        rejected = []
        for kind in ["vv", "ev", "ee", "fv"]:
            attribute = kind + "_candidates"
            for candidate in getattr(candidates, attribute):
                individual = ipctk.Candidates()
                setattr(individual, attribute, [candidate])
                fraction = individual.compute_collision_free_stepsize(mesh, start, end)
                if fraction >= 1 - 1e-12:
                    continue
                vertex_ids = np.array(candidate.vertex_ids(mesh.edges, mesh.faces))
                vertex_ids = vertex_ids[vertex_ids >= 0]
                tolerances = []
                for tolerance in [1e-8, 1e-10, 1e-12]:
                    tolerances.append(
                        {
                            "ccd_tolerance": tolerance,
                            "safe_fraction": float(
                                individual.compute_collision_free_stepsize(
                                    mesh, start, end, narrow_phase_ccd=ipctk.TightInclusionCCD(tolerance=tolerance)
                                )
                            ),
                        }
                    )
                rejected.append(
                    {
                        "kind": kind,
                        "candidate": str(candidate),
                        "full_vertex_ids": ids[vertex_ids].tolist(),
                        "groups": static.groups[ids[vertex_ids]].tolist(),
                        "start_positions_m": start[vertex_ids].tolist(),
                        "end_positions_m": end[vertex_ids].tolist(),
                        "default_safe_fraction": float(fraction),
                        "tolerance_diagnostics": tolerances,
                    }
                )
        records.append(
            {
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "body_residual_N": float(np.linalg.norm(result[0])),
                "robot_effort_residual": float(np.linalg.norm(result[1])),
                "start_has_intersections": ipctk.has_intersections(mesh, start),
                "end_has_intersections": ipctk.has_intersections(mesh, end),
                "default_safe_fraction": float(ipctk.compute_collision_free_stepsize(mesh, start, end)),
                "swept_candidate_count": len(candidates),
                "rejected_candidates": rejected,
            }
        )
        np.savez_compressed(args.output / path.name, start=start, end=end)
    (args.output / "summary.json").write_text(
        json.dumps({"scope": __doc__, "step": len(paths) + 1, "attempts": records, "accepted_states": 0}, indent=2)
        + "\n"
    )
    print(json.dumps({"attempts": len(records), "rejected_counts": [len(r["rejected_candidates"]) for r in records]}))


if __name__ == "__main__":
    main()
