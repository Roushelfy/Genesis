"""Publish the failed original rigid continuation and independently qualified step2129 recovery."""

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
    meta = p / f"rigid-inexact-recovery-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-original-rigid-full-history-v3": ("717843_0", "FAILED", 1),
        "franka-original-rigid-full-prefix-audit-v3": ("718517_0", "COMPLETED", 0),
        "franka-original-rigid-step2129-inexact-lagged-v1": ("718523_0", "COMPLETED", 0),
        "franka-original-rigid-step2129-inexact-audit-v1": ("718524_0", "COMPLETED", 0),
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job, _, _ in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    records, identities = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, (job, state, code) in jobs.items():
        run = p / name
        assert states[job] == state and int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                digest, relative = line.split("  ", 1)
                actual = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
                identities.append({"run": name, "kind": kind, "file": relative, "sha256": digest})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "scheduler_state": state, "actual_wrapper_code": code})

    def read(name):
        return json.loads((p / name / "results/summary.json").read_text())

    original = p / "franka-original-rigid-full-history-v3"
    recovered = p / "franka-original-rigid-step2129-inexact-lagged-v1"
    prefix = read("franka-original-rigid-full-prefix-audit-v3")
    producer = read(recovered.name)
    audit = read("franka-original-rigid-step2129-inexact-audit-v1")
    assert prefix["states_checked"] == len(prefix["rows"]) == producer["byte_identical_prefix_states"] == 2128
    assert audit["states_checked"] == len(audit["rows"]) == producer["recovered_step"] == 2129
    assert not prefix["complete_requested_history"] and not audit["complete_requested_history"]
    assert Path(audit["run"]).resolve() == recovered.resolve()
    for report in [prefix, audit]:
        for row in report["rows"]:
            assert max(row["body_force_equivalent_residual_N"], row["fresh_actual_robot_effort_residual"]) <= 1e-9
        for path, digest in report["input_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for path in sorted((original / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz")):
        assert path.read_bytes() == (recovered / "results" / path.name).read_bytes()
    for identity in producer["original_state_identities"]:
        path = Path(identity["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
        assert path.read_bytes() == (recovered / "results" / path.name).read_bytes()
    numerical = producer["numerical_recovery"]
    assert numerical["recovered_original_equation"] and numerical["final_acceptance_uses_original_current_friction"]
    assert numerical["original_physical_gate_N_or_Nm"] == 1e-9
    assert numerical["frozen_numerical_stagnation_working_bound_N_or_Nm"] == 1e-8
    assert numerical["original_previous_state_controls_dt_predictor_fixed"]
    assert not numerical["numerical_iterations_are_physical_history"]
    assert len(numerical["attempts"]) == 26
    assert sum(row["inexact_frozen_numerical_candidate"] is not None for row in numerical["attempts"]) == 1
    assert numerical["attempts"][-1]["original_physical_gates_passed"]
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "failed_continuation_independently_audited_states": 2128,
        "recovered_independently_audited_states": 2129,
        "original_2128_prefix_files_byte_identical": True,
        "numerical_recovery": numerical,
        "independent_final_state": audit["rows"][-1],
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    extras.update(
        p / "meta" / name
        for name in [
            "original-rigid-full-prefix-audit-commands-v3.json",
            "original-rigid-inexact-lagged-commands-v1.json",
            "original-rigid-resolved-native-commands-v2.json",
        ]
    )
    destination = args.destination_root / f"rigid-inexact-recovery-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "The original full rigid continuation fails at2129 after2128 accepted states; all2128 independently pass the original physical gates. One frozen-friction inner stagnation iterate at1.0108e-9N is only a numerical seed, and26 outer numerical stages are not physical history. Recovered2129 passes the unchanged current-friction1e-9N body/fresh actual robot gates and original straight-path default CCD; all2128 prefix files remain byte-identical. The2129-state prefix is not the full2640-state application or physical convergence. Full continuation/feedback/native replay, contact/geometry/regularization, load-model accuracy and warmed physical accuracy/cost remain open. Production Quadrants and rigid dynamics are unchanged.",
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
