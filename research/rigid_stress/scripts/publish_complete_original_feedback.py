"""Publish the complete original rigid counterpart, all-state audit, matched feedback and actual native replays."""

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
    meta = p / f"complete-original-feedback-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-original-rigid-full-history-v4": "718525_0",
        "franka-original-rigid-full-audit-v4": "718526_0",
        "franka-original-rigid-deformable-feedback-v2": "718527_0",
        "franka-original-rigid-resolved-native-v2": "718528_0",
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    identities, records = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            if not before.exists():
                continue
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                digest, relative = line.split("  ", 1)
                actual = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
                identities.append({"run": name, "kind": kind, "file": relative, "sha256": digest})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "scheduler_state": "COMPLETED", "actual_wrapper_code": 0})

    def read(name):
        return json.loads((p / name / "results/summary.json").read_text())

    producer, audit, feedback, native = [read(name) for name in jobs]
    assert producer["complete_original_330_macro_history"] and producer["requested_macro_steps"] == 330
    assert (
        len(list((p / "franka-original-rigid-full-history-v4/results").glob("step[0-9][0-9][0-9][0-9][0-9].npz")))
        == 2640
    )
    assert audit["complete_requested_history"] and audit["states_checked"] == len(audit["rows"]) == 2640
    assert all(
        max(row["body_force_equivalent_residual_N"], row["fresh_actual_robot_effort_residual"]) <= 1e-9
        for row in audit["rows"]
    )
    assert (
        feedback["complete_original_330_macro_history"] and feedback["states_checked"] == len(feedback["rows"]) == 2640
    )
    assert not feedback["audited_prefix_explicitly_allowed"]
    selected = [11, 80, 866, 869, 1440, 2000, 2640]
    assert feedback["saved_snapshot_steps"] == selected
    assert native["complete_original_330_macro_history"] and len(native["rows"]) == 14
    assert {(row["step"], row["mode"]) for row in native["rows"]} == {
        (step, mode) for step in selected for mode in ["full_inertia", "rigid_relief"]
    }
    assert all(row["native_complete_budget_fraction"] <= 1 for row in native["rows"])
    for report in [audit, feedback]:
        for path, digest in report["input_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    original_parent = args.destination_root / "complete-original-history-v1/manifest.json"
    assert original_parent.exists()
    summaries = [row for row in feedback["rows"] if "resolved_load_history_feedback_tensor" in row]
    assert [row["step"] for row in summaries] == selected
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "independently_qualified_original_states": 2640,
        "original_control_cycles": 330,
        "physical_duration_s": 3.3,
        "complete_rigid_audit": audit,
        "feedback_snapshot_rows": summaries,
        "actual_native_replays": native,
        "deformable_parent_archive": {
            "manifest": str(original_parent),
            "sha256": hashlib.sha256(original_parent.read_bytes()).hexdigest(),
            "original_runtime_input_hashes_reverified": True,
        },
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    for name in ["original-rigid-inexact-lagged-commands-v1.json", "original-rigid-resolved-native-commands-v2.json"]:
        extras.add(p / "meta" / name)
    destination = args.destination_root / f"complete-original-feedback-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "The original rigid counterpart and independently solved deformable parent now both cover all330 controls/2640 states/3.3s at identical represented geometry, mass, actuation, timestep and contact law. All2640 rigid states independently pass original body/moment/actual robot/contact/CCD gates. Seven full material-frame resolved-load comparisons separate load/history feedback from deformable linear-versus-nonlinear recovery, and14 actual native nodal replays retain original numerical budgets. The23.17% snapshot869 feedback difference is conditional coarse480-tet evidence, not a converged continuum truth or history-wide tensor maximum. Sources are actual fingers/support, not continuum islands. The previously published deformable full-history parent is referenced with its immutable manifest SHA256; all actual parent input hashes are reverified rather than duplicating those artifacts here. Full physical geometry/contact/time/regularization convergence, finite-footprint load-model/island accuracy and warmed physical accuracy/cost remain open. Production Quadrants and rigid dynamics are unchanged.",
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
