"""Publish the closed three-level U fields and six final normal-contact trace audits."""

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
    meta = p / f"u-contact-trace-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "u-nonlinear-rigid-init-spatial-1-v1": "715802_0",
        "u-nonlinear-rigid-init-spatial-2-v1": "715802_1",
        "u-nonlinear-rigid-init-spatial-3-v2": "716308_0",
        "u-nonlinear-fixed-body-pad5-v2": "716309_0",
        "u-nonlinear-fixed-body-pad9-v2": "716310_0",
        "u-nonlinear-fixed-pad5-body5669-v1": "716333_0",
        "u-nonlinear-three-level-fields-v2": "716316_0",
        "u-nonlinear-fixed-body-pad-fields-v2": "716317_0",
        "u-nonlinear-fixed-pad5-body-fields-v1": "716334_0",
        "u-nonlinear-final-contact-traces-v1": "718505_0",
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(sorted({job.split("_")[0] for job in jobs.values()}))]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    identities, records = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt", p / "meta/u-final-contact-traces-commands-v1.json"}
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                digest, relative = line.split("  ", 1)
                actual = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
                identities.append({"run": name, "kind": kind, "file": relative, "sha256": digest})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "scheduler_state": states[job], "actual_wrapper_code": 0})
    reports = {}
    for name in [
        "u-nonlinear-three-level-fields-v2",
        "u-nonlinear-fixed-body-pad-fields-v2",
        "u-nonlinear-fixed-pad5-body-fields-v1",
        "u-nonlinear-final-contact-traces-v1",
    ]:
        report = json.loads((p / name / "results/summary.json").read_text())
        for path, digest in report["input_sha256"].items():
            path = Path(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            extras.add(path)
        reports[name] = report
    coupled = reports["u-nonlinear-three-level-fields-v2"]
    assert [row["body_tetrahedra"] for row in coupled["rows"]] == [845, 2339, 5669]
    assert [row["pad_resolution"] for row in coupled["rows"]] == [5, 7, 9]
    assert coupled["common_material_points"] == 75
    for report in list(reports.values())[:3]:
        assert len(report["rows"]) == 3
        for row in report["rows"]:
            assert (
                max(
                    row["independent_body_residual_N"],
                    row["independent_original_mapped_residual_N"],
                    row["maximum_recorded_original_residual_N"],
                )
                <= 1e-9
            )
    trace = reports["u-nonlinear-final-contact-traces-v1"]
    assert len(trace["rows"]) == 6 and len(trace["comparisons"]) == 3
    assert {row["normal_stencils"] for row in trace["rows"]} == {15, 19, 37, 39, 47, 80}
    assert all(row["normal_gradient_reconstruction_max_difference_N"] <= 1e-12 for row in trace["rows"])
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "reports": reports,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"u-contact-trace-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "Three coupled U levels retain12.21% last tensor change and5.34% raw lower peak change. Fixed-body pad7-to9 tensor change1.20% does not establish contact pressure convergence: left maximum discrete normal density increases25.48%, and left/right active lumped areas decrease12.80%/8.31%. All six final normal contact stencils are independently rebuilt and both physical shoe wrenches retained. Saved total-minus-normal shear is explicitly not an independent friction-history reconstruction. Lumped sample-force densities and source identities are not continuum pressure/area/islands. All original raw fields/maxima, materials, physical load laws and force budgets remain. Full physical convergence, regularization/history, model/feedback and warmed accuracy/cost acceptance remain open.",
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
