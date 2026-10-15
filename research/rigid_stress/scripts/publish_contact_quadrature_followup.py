"""Publish validated contact figures, complete coverage corrections and retained failed rendering evidence."""

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
    meta = p / f"contact-quadrature-followup-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "egg-contact-quadrature-figures-v1": ("717198_0", "FAILED", 1),
        "egg-contact-quadrature-figures-v2": ("717235_0", "COMPLETED", 0),
        "egg-contact-quadrature-figures-v3": ("717276_0", "COMPLETED", 0),
        "egg-body-finger-contact-density-audit-v1": ("717277_0", "COMPLETED", 0),
        "egg-adaptive-contact-cap-coverage-levels34-v1": ("717286_0", "COMPLETED", 0),
        "egg-adaptive-cad-level4-contact-v1": ("716803_0", "COMPLETED", 0),
        "egg-adaptive-cad-level4-contact-v1-audit": ("716804_0", "COMPLETED", 0),
        "egg-covered-cad-level1-contact-v1": ("716883_0", "COMPLETED", 0),
        "egg-covered-cad-level1-contact-v1-audit": ("716884_0", "COMPLETED", 0),
        "egg-covered-cad-level1-coverage-v1": ("716885_0", "COMPLETED", 0),
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(row[0].split("_")[0] for row in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%20000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    identities, records = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, (job, state, code) in jobs.items():
        run = p / name
        assert states[job] == state and int((run / "returncode.txt").read_text()) == code
        if state == "COMPLETED":
            assert (run / "results/summary.json").exists()
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, path = line.split("  ", 1)
                actual = run / kind / path if not Path(path).is_absolute() else Path(path)
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected
                identities.append({"run": name, "identity": kind, "file": path, "sha256": expected})
        records.append({"run": name, "job": job, "state": state, "returncode": code})
        extras.update(p.glob(f"*{job}.log"))
    accepted = json.loads((p / "meta/contact-quadrature-figure-visual-qa-v1.json").read_text())
    assert accepted["visual_qa_passed"] and not accepted["matplotlib_text_qa_issues"]
    for name in [
        "metrics.csv",
        "surface_centroids.csv",
        "stencil_weights.csv",
        "contact_quadrature.png",
        "contact_quadrature_preview.png",
    ]:
        assert (p / "egg-contact-quadrature-figures-v2/results" / name).read_bytes() == (
            p / "egg-contact-quadrature-figures-v3/results" / name
        ).read_bytes()
    coverage = json.loads((p / "egg-covered-cad-level1-coverage-v1/results/summary.json").read_text())
    assert len(coverage["rows"]) == 5 and all(
        row["active_samples_outside_marking_caps"] == 0 for row in coverage["rows"]
    )
    audit = json.loads((p / "egg-covered-cad-level1-contact-v1-audit/results/summary.json").read_text())
    assert audit["states_checked"] == len(audit["rows"]) == 5
    assert all(
        row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
        for row in audit["rows"]
    )
    extras.update(
        p / path
        for path in [
            "meta/contact-quadrature-figure-visual-qa-v1.json",
            "meta/contact-quadrature-figure-commands-v1.json",
            "meta/contact-quadrature-figure-commands-v2.json",
            "meta/contact-quadrature-figure-commands-v3.json",
            "meta/contact-density-audit-commands-v1.json",
            "meta/old-caps-levels34-coverage-commands-v1.json",
            "meta/covered-cad-third-level-commands-v1.json",
            "meta/broad-covered-cad-contact-commands-v1.json",
            "contact-quadrature-publication-v1/runtime-verification.json",
            "finger-quadrature-publication-v1/runtime-verification.json",
        ]
    )
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "figure_QA": accepted,
        "corrected_contact_coverage": coverage,
        "independent_corrected_contact_audit": audit,
        "separate_body_finger_density_audit": json.loads(
            (p / "egg-body-finger-contact-density-audit-v1/results/summary.json").read_text()
        ),
        "old_caps_last_two_levels": json.loads(
            (p / "egg-adaptive-contact-cap-coverage-levels34-v1/results/summary.json").read_text()
        ),
        "production_engine_changed": False,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"egg-contact-quadrature-followup-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "The failed first render, accepted editable SVG source and matching PDF/data/raster exports remain distinct. Discrete density norms and stencil weights do not establish physical pressure or contact area. All four old small-cap levels miss actual contacts; corrected first-level samples are covered but neither this coverage nor its40.4MPa raw peak establishes convergence. Finer/broader contact, thickness, original histories, model/feedback and physically verified warmed-cost gates remain open.",
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
