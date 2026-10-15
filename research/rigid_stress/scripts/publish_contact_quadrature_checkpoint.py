"""Publish the fixed-bulk egg contact-integration axis and its complete zero-level qualification."""

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
    parser.add_argument("--finger", action="store_true")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    publication_kind = "finger-quadrature" if args.finger else "contact-quadrature"
    meta = p / f"{publication_kind}-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "egg-cad-level2-quadrature0-v1": "717003_0",
        "egg-cad-level2-quadrature0-v1-audit": "717004_0",
        "egg-cad-level2-quadrature-zero-qualification-v1": "717005_0",
        "egg-cad-level2-quadrature1-v1": "717006_0",
        "egg-cad-level2-quadrature1-v1-audit": "717007_0",
        "egg-cad-level2-quadrature2-v1": "717008_0",
        "egg-cad-level2-quadrature2-v1-audit": "717009_0",
        "egg-cad-level2-quadrature-three-level-fields-v1": "717035_0",
        "egg-contact-quadrature-stencil-observations-v1": "717111_0",
    }
    if args.finger:
        jobs = {
            "egg-rigid-finger-trace-construction-qualification-v1": "717133_0",
            "egg-finger-quadrature0-v1": "717150_0",
            "egg-finger-quadrature0-v1-audit": "717151_0",
            "egg-finger-quadrature-zero-qualification-v1": "717152_0",
            "egg-finger-quadrature1-v1": "717153_0",
            "egg-finger-quadrature1-v1-audit": "717154_0",
            "egg-finger-quadrature2-v1": "717155_0",
            "egg-finger-quadrature2-v1-audit": "717156_0",
            "egg-finger-quadrature-three-level-fields-v1": "717157_0",
            "egg-finger-quadrature-stencil-observations-v1": "717240_0",
        }
    zero_name = (
        "egg-finger-quadrature-zero-qualification-v1"
        if args.finger
        else "egg-cad-level2-quadrature-zero-qualification-v1"
    )
    comparison_name = (
        "egg-finger-quadrature-three-level-fields-v1"
        if args.finger
        else "egg-cad-level2-quadrature-three-level-fields-v1"
    )
    observations_name = (
        "egg-finger-quadrature-stencil-observations-v1"
        if args.finger
        else "egg-contact-quadrature-stencil-observations-v1"
    )
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
    zero = json.loads((p / zero_name / "results/summary.json").read_text())
    assert zero["all_original_arrays_and_complete_fields_qualified"] and zero["states_checked"] == 5
    for path, expected in zero["input_sha256"].items():
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert path.stat().st_size < 50_000_000
        extras.add(path)
    comparison = json.loads((p / comparison_name / "results/summary.json").read_text())
    assert len(comparison["rows"]) == 3 and len(comparison["comparisons"]) == 2
    assert {row["body_tetrahedra"] for row in comparison["rows"]} == {1920}
    axis_key = "finger_refinement" if args.finger else "contact_refinement"
    assert [row[axis_key] for row in comparison["rows"]] == [0, 1, 2]
    for level in [0, 1, 2]:
        producer_prefix = "egg-finger-quadrature" if args.finger else "egg-cad-level2-quadrature"
        audit = json.loads((p / f"{producer_prefix}{level}-v1-audit/results/summary.json").read_text())
        assert audit["states_checked"] == len(audit["rows"]) == 5
        assert all(
            row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
            for row in audit["rows"]
        )
    metadata_names = (
        [
            "finger-quadrature-axis-commands-v1.json",
            "finger-quadrature-stencil-commands-v1.json",
            "finger-trace-construction-commands-v1.json",
        ]
        if args.finger
        else [
            "contact-quadrature-axis-commands-v1.json",
            "contact-quadrature-fields-commands-v1.json",
            "contact-quadrature-memory-update-v1.json",
            "contact-quadrature-stencil-commands-v1.json",
        ]
    )
    extras.update(p / "meta" / name for name in metadata_names)
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "zero_level_complete_state_and_field_qualification": zero,
        "fixed_bulk_contact_integration_comparison": comparison,
        "actual_stencil_observations": json.loads((p / observations_name / "results/summary.json").read_text()),
        "production_engine_changed": False,
        "goal_completion": False,
    }
    limitations = (
        "All15 new states and their independent original-force audits are preserved on one identical1920-tet "
        "material geometry and mass. Only body P2 contact trace integration changes; fingers stay fixed. "
        "The zero case preserves all original saved arrays and complete corner/quadrature fields under "
        "existing budgets after independently repeating the same held controls. Its initializer is "
        "fully compared with its actual byte identity recorded. Adjacent fixed-bulk tensor/energy changes "
        "remain conditional on fixed bulk/geometry, finger discretization and contact regularization. "
        "Fixed active counts and changing discrete area weights do not establish a physical pressure/area tail. "
        "Lumped force-density norms are discrete diagnostics; total frictional surface moment is retained "
        "without adding a zero-moment gate to the original law. Full physical/contact/geometry/thickness/"
        "history/model/feedback and physically verified warmed-cost requirements remain open."
    )
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"egg-{publication_kind}-{args.version}"
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
