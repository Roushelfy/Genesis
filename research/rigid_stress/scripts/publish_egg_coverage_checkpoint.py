"""Publish closed egg contact, refinement-coverage and complete ordering comparisons."""

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
    meta = p / f"egg-coverage-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "egg-adaptive-contact-cap-coverage-v1": "716833_0",
        "egg-adaptive-cad-polyhedron-geometry-v4": "716878_0",
        "egg-adaptive-cad-polyhedron-geometry-v5": "716880_0",
        "egg-paired-polyhedron-level4-contact-v1": "716500_0",
        "egg-paired-polyhedron-level4-contact-v1-audit": "716501_0",
        "egg-paired-polyhedron-four-level-fields-v1": "716881_0",
        "egg-paired-cad-level4-generic-contact-v1": "716805_0",
        "egg-paired-cad-level4-generic-contact-v1-audit": "716806_0",
        "egg-paired-cad-level4-ordered-contact-v1": "716824_0",
        "egg-paired-cad-level4-ordered-contact-v1-audit": "716825_0",
        "egg-cad-level4-saved-ordering-qualification-v1": "716882_0",
        "egg-adaptive-cad-level1-contact-v1": "716797_0",
        "egg-adaptive-cad-level1-contact-v1-audit": "716798_0",
        "egg-adaptive-cad-level2-contact-v1": "716799_0",
        "egg-adaptive-cad-level2-contact-v1-audit": "716800_0",
        "egg-adaptive-cad-level3-contact-v1": "716801_0",
        "egg-adaptive-cad-level3-contact-v1-audit": "716802_0",
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
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
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        assert (run / "results/summary.json").exists()
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, path = line.split("  ", 1)
                assert hashlib.sha256((run / kind / path).read_bytes()).hexdigest() == expected
                identities.append({"run": name, "identity": kind, "file": path, "sha256": expected})
        records.append({"run": name, "job": job, "state": states[job], "returncode": 0})
        extras.update(p.glob(f"*{job}.log"))
    comparison = json.loads((p / "egg-cad-level4-saved-ordering-qualification-v1/results/summary.json").read_text())
    assert comparison["all_original_arrays_and_complete_fields_qualified"] and comparison["states_checked"] == 5
    coverage = json.loads((p / "egg-adaptive-contact-cap-coverage-v1/results/summary.json").read_text())
    assert len(coverage["rows"]) == 10
    assert all(row["outside_unsigned_normal_sample_load_fraction"] == 1 for row in coverage["rows"])
    polyhedron = json.loads((p / "egg-paired-polyhedron-four-level-fields-v1/results/summary.json").read_text())
    assert len(polyhedron["rows"]) == 4 and polyhedron["common_physical_probes"] == 2400
    for name in [
        "egg-paired-polyhedron-level4-contact-v1-audit",
        "egg-paired-cad-level4-generic-contact-v1-audit",
        "egg-paired-cad-level4-ordered-contact-v1-audit",
        *(f"egg-adaptive-cad-level{level}-contact-v1-audit" for level in [1, 2, 3]),
    ]:
        audit = json.loads((p / name / "results/summary.json").read_text())
        assert audit["states_checked"] == len(audit["rows"]) == 5
        assert all(
            row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
            for row in audit["rows"]
        )
    extras.update(
        p / path
        for path in [
            "egg-thickness-fields-publication-v1/runtime-verification.json",
            "reference-ordering-publication-v2/verification.json",
            "publication-binary-sharding-reusable-qualification-v1/summary.json",
            "meta/adaptive-contact-coverage-correction-v1.json",
            "meta/adaptive-v5-and-polyhedron-four-level-commands-v1.json",
            "meta/cad4-saved-ordering-qualification-commands-v1.json",
            "meta/covered-cad-contact-commands-v1.json",
            "meta/thickness-five-six-commands-v1.json",
        ]
    )
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "complete_CAD30720_state_and_field_equivalence": comparison,
        "independent_refinement_coverage": coverage,
        "fixed_polyhedron_four_level_fields": polyhedron,
        "corrected_geometry_summaries": [
            json.loads((p / f"egg-adaptive-cad-polyhedron-geometry-{version}/results/summary.json").read_text())
            for version in ["v4", "v5"]
        ],
        "production_engine_changed": False,
        "goal_completion": False,
    }
    limitations = (
        "The first two small-cap adaptive cases place every actual active sample outside their refinement caps; "
        "their near-equal raw peaks do not establish convergence. Corrected geometry adds independently audited "
        "contact positions and force-normal support directions but does not certify future contact coverage. "
        "New actual contact jobs are scheduled only. Four fixed-polyhedron levels retain large stress/energy "
        "changes. All saved30720-tet baseline/candidate arrays and complete corner/quadrature fields agree "
        "within unchanged budgets, with separate all-state physical force audits; this is numerical equivalence, "
        "not physical convergence or controlled performance. Full original physical/contact/geometry/thickness/"
        "history/model/feedback and physically verified warmed-cost requirements remain open."
    )
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"egg-contact-coverage-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        limitations,
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
