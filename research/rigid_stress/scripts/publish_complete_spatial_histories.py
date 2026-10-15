"""Publish four complete stationary histories and their ten matched full-field comparisons."""

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
    meta = p / f"complete-spatial-history-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-tip18-retained-full-history-v1": "716331_0",
        "franka-tip18-retained-full-audit-v1": "716332_0",
        "franka-four-complete-histories-fields-v1": "717369_0",
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%24000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    comparison = json.loads((p / "franka-four-complete-histories-fields-v1/results/summary.json").read_text())
    audit = json.loads((p / "franka-tip18-retained-full-audit-v1/results/summary.json").read_text())
    assert len(comparison["common_snapshots"]) == 10
    assert [row["step"] for row in comparison["common_snapshots"]] == list(range(8, 81, 8))
    assert audit["states_checked"] == len(audit["rows"]) == 80 and audit["complete_source_requested_history"]
    assert all(
        row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
        for row in audit["rows"]
    )
    sources = [Path(name) for name in comparison["run"]]
    identities = []
    for run in [*sources, p / "franka-tip18-retained-full-audit-v1", p / "franka-four-complete-histories-fields-v1"]:
        assert int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            if not identity.exists():
                continue
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, name = line.split("  ", 1)
                path = Path(name) if Path(name).is_absolute() else run / kind / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
                identities.append({"run": str(run), "kind": kind, "path": name, "sha256": expected})
    for run in sources:
        config = json.loads((run / "results/configuration.json").read_text())
        report = json.loads((run / "results/summary.json").read_text())
        paths = sorted((run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
        assert config["steps"] == len(report["steps"]) == len(paths) == 80
        assert [path.name for path in paths] == [f"step{step:04d}.npz" for step in range(1, 81)]
    for name, job in jobs.items():
        assert states[job] == "COMPLETED"
    for path, expected in comparison["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for snapshot in comparison["common_snapshots"]:
        assert hashlib.sha256(Path(snapshot["summary"]).read_bytes()).hexdigest() == snapshot["summary_sha256"]
        assert len(snapshot["rows"]) == 4
        assert all(
            row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
            for row in snapshot["rows"]
        )
    observables = {
        "complete_histories": 4,
        "states_per_history": 80,
        "horizon_s": 0.1,
        "matched_field_snapshots": 40,
        "finest_independently_audited_states": 80,
        "finest_history_raw_peak_lower_Pa": max(row["raw_global_peak_lower_Pa"] for row in audit["rows"]),
        "finest_history_raw_peak_upper_Pa": max(row["raw_global_peak_upper_Pa"] for row in audit["rows"]),
        "maximum_finest_body_balance_N": max(row["complete_body_dynamic_residual_N"] for row in audit["rows"]),
        "maximum_finest_mapped_balance_N": max(row["complete_mapped_dynamic_residual_N"] for row in audit["rows"]),
        "final_finest_pair": comparison["common_snapshots"][-1]["comparisons"][-1],
        "final_finest_state": comparison["common_snapshots"][-1]["rows"][-1],
    }
    extras = {
        meta / "jobs.txt",
        meta / "job-commands.txt",
        p / "meta/complete-spatial-peak-anchor-distances-v1.json",
        p / "meta/four-complete-histories-commands-v1.json",
    }
    for job in jobs.values():
        extras.update(p.glob(f"*{job}.log"))
    summary = {
        "scope": __doc__,
        "closed_new_jobs": jobs,
        "source_identity_records": identities,
        "observables": observables,
        "comparison": comparison,
        "finest_independent_audit": audit,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"complete-spatial-histories-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [*sources, p / "franka-tip18-retained-full-audit-v1", p / "franka-four-complete-histories-fields-v1"],
        sorted(extras),
        summary,
        "Four complete80-state stationary0.1s histories use fixed physical controls, material, original polyhedron and mass. Only the finest80 states and40 matched full-field states are independently audited here;320 saved states is not320 independently audited states. The finest initializer retains the five previously saved prefix states within declared original budgets; the other75 are new physical states. Ten common times preserve raw maxima, full corner/quadrature tensors, objective displacements, strain and common physical probes. Last-pair tensor15.52%, energy9.15% and raw lower peak45.42% changes and fine0.451 corner Green strain remain unresolved. This is not the original330-control approach/lift protocol, physical convergence, feedback or warmed-cost acceptance.",
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
