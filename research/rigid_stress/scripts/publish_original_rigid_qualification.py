"""Publish rejected rigid-contact roots, measured CCD diagnosis and independently qualified original step857."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    meta = p / f"original-rigid-qualification-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-original-rigid-counterpart-pilot-v1": ("717371_0", "FAILED", 1),
        "franka-original-rigid-counterpart-pilot-audit-v1": ("717438_0", "COMPLETED", 0),
        "franka-original-rigid-step857-recovery-v1": ("717450_0", "FAILED", 1),
        "franka-original-rigid-step857-ccd-diagnosis-v1": ("717488_0", "COMPLETED", 0),
        "franka-original-rigid-step857-recovery-v2": ("717491_0", "FAILED", 1),
        "franka-original-rigid-step857-ccd-diagnosis-v2": ("717495_0", "COMPLETED", 0),
        "franka-original-rigid-step857-recovery-v3": ("717500_0", "COMPLETED", 0),
        "franka-original-rigid-step857-recovery-audit-v3": ("717501_0", "COMPLETED", 0),
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(row[0].split("_")[0] for row in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%24000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    records, identities = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, (job, state, code) in jobs.items():
        run = p / name
        assert states[job] == state and int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, name_in_source = line.split("  ", 1)
                actual = run / kind / name_in_source
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected
                identities.append({"run": name, "kind": kind, "path": name_in_source, "sha256": expected})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "state": state, "returncode": code})
    recovered = json.loads((p / "franka-original-rigid-step857-recovery-v3/results/summary.json").read_text())
    audit = json.loads((p / "franka-original-rigid-step857-recovery-audit-v3/results/summary.json").read_text())
    assert recovered["recovered_step"] == audit["states_checked"] == len(audit["rows"]) == 857
    assert recovered["byte_identical_prefix_states"] == 856
    assert not recovered["complete_requested_history"] and not audit["complete_requested_history"]
    assert audit["producer_returncode"] == 0 and audit["requested_states"] == 880
    for row in audit["rows"]:
        assert row["body_force_equivalent_residual_N"] <= 1e-9
        assert row["fresh_actual_robot_effort_residual"] <= 1e-9
        assert row["independently_verified_straight_motion_CCD_intervals"] > 0
    for record in recovered["original_state_identities"]:
        original = Path(record["source"])
        assert hashlib.sha256(original.read_bytes()).hexdigest() == record["sha256"]
        assert (
            original.read_bytes()
            == (p / "franka-original-rigid-step857-recovery-v3/results" / original.name).read_bytes()
        )
    for path, expected in audit["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    accepted = [row for row in recovered["numerical_recovery"]["attempts"] if row["original_physical_gates_passed"]]
    assert len(accepted) == 1 and accepted[0]["method"] == "CCD_initialized_original_Newton"
    assert accepted[0]["original_physical_motion_CCD"]["intervals"] == 4
    for version in ["v1", "v2"]:
        extras.add(p / f"meta/original-rigid-full-history-commands-{version}.json")
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "qualified_recovery": recovered,
        "all_state_independent_audit": audit,
        "observables": {
            "qualified_rigid_physical_states": 857,
            "byte_identical_retained_states": 856,
            "maximum_body_force_equivalent_residual_N": max(
                row["body_force_equivalent_residual_N"] for row in audit["rows"]
            ),
            "maximum_actual_robot_effort_residual": max(
                row["fresh_actual_robot_effort_residual"] for row in audit["rows"]
            ),
            "recovered_state_independent_audit": audit["rows"][-1],
            "complete_original_rigid_history": False,
        },
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"original-rigid-qualification-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "Failed pilot and intersecting roots remain rejected. All857 accepted original rigid states have independent contact force, body net force/moment, actual old-state Panda step and straight-trajectory CCD audits. The recovered step uses a collision-free numerical initializer; four contiguous default CCD queries certify the same original straight vertex paths without subdividing physical time or changing CCD settings. The856 retained physical states are byte-identical. Code0 for the recovered prefix does not establish880 requested or2640 full-original states. Full same-law rigid/deformable feedback, physical convergence, native finite-footprint accuracy and warmed accuracy/cost remain open.",
        p / "publication-staging",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.rigid_stress.scripts.verify_runtime_publication",
            "--manifest",
            str(destination / "manifest.json"),
            "--runtime-root",
            str(args.runtime_root),
            "--staging-root",
            str(p / "publication-staging" / destination.name),
            "--output",
            str(meta / "runtime-verification.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
