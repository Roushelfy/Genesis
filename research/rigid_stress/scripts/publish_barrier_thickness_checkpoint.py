"""Publish closed all-stage barrier comparisons or fully reassembled thickness fields with verified archive parents."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded
from research.rigid_stress.scripts.verify_nonconvex_publication import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--kind", choices=["barrier", "thickness"], required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = "fixed-history-barrier" if args.kind == "barrier" else "five-level-thickness"
    meta = p / f"{label}-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "barrier": {
            "u-fixed-history-barrier-1250nm-v2": "718718_0",
            "u-fixed-history-barrier-1250nm-audit-v2": "718719_0",
            "u-fixed-history-barrier-625nm-v2": "718720_0",
            "u-fixed-history-barrier-625nm-audit-v2": "718721_0",
            "u-fixed-history-barrier-all-stage-comparison-v1": "718883_0",
        },
        "thickness": {
            "egg-fixed-surface-layers5-v1": "716835_0",
            "egg-fixed-surface-layers6-v1": "716836_0",
            "egg-fixed-surface-five-level-audit-v1": "716837_0",
            "egg-five-level-thickness-fem-reassembly-v1": "718858_0",
        },
    }[args.kind]
    parent_label, parent_sha, verification_folder = {
        "barrier": (
            "recorded-contact-history-v1",
            "7f9accb7d15815fd1e73614c43e456071668a19cf0fe7d23494a0f66d9674e87",
            "recorded-contact-history-publication-v1",
        ),
        "thickness": (
            "egg-thickness-fields-v1",
            "915665a4ada17beb8ef5d4d21b678e8de4b2e5f5086a44440cc0d68228b3de52",
            "egg-thickness-fields-publication-v1",
        ),
    }[args.kind]
    parent = args.destination_root / parent_label / "manifest.json"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_sha
    parent_archives = verify_manifest(parent, True, {})
    parent_proof_path = p / verification_folder / "runtime-verification.json"
    parent_proof = json.loads(parent_proof_path.read_text())
    assert parent_proof["passed"] and parent_proof["manifest_sha256"] == parent_sha
    parent_sources = parent_proof["actual_original_sha256_and_bytes"]
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    identities, records, parent_references = [], [], {}
    extras = {meta / "jobs.txt", meta / "job-commands.txt", parent_proof_path}
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        scopes = [run, *[run / child for child in ["fem", "probes"] if (run / child / "source.sha256").exists()]]
        for scope in scopes:
            assert int((scope / "returncode.txt").read_text()) == 0
            for kind in ["source", "input-source"]:
                before = scope / f"{kind}.sha256"
                if not before.exists():
                    continue
                assert before.read_bytes() == (scope / f"{kind}-after.sha256").read_bytes()
                for line in before.read_text().splitlines():
                    digest, relative = line.split("  ", 1)
                    actual = Path(relative) if Path(relative).is_absolute() else scope / kind / relative
                    assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
                    identities.append({"scope": str(scope), "kind": kind, "file": relative, "sha256": digest})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "scheduler_state": "COMPLETED", "actual_wrapper_code": 0})

    def read(name):
        value = json.loads((p / name / "results/summary.json").read_text())
        for input_name, digest in value.get("input_sha256", {}).items():
            path = Path(input_name)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            if input_name in parent_sources:
                assert parent_sources[input_name][0] == digest
                parent_references[input_name] = digest
            elif not any(path.is_relative_to(p / run_name) for run_name in jobs):
                extras.add(path)
        return value

    if args.kind == "barrier":
        audits = [
            read(name) for name in ["u-fixed-history-barrier-1250nm-audit-v2", "u-fixed-history-barrier-625nm-audit-v2"]
        ]
        comparison = read("u-fixed-history-barrier-all-stage-comparison-v1")
        assert comparison["stages_compared_per_level"] == len(comparison["rows"]) == 744 and comparison["levels"] == 3
        assert all(len(row["pairs"]) == len(row["levels"]) == 3 for row in comparison["rows"])
        for audit in audits:
            assert audit["stages_checked"] == len(audit["rows"]) == 744
            assert audit["all_retained_stages_independent_current_friction_and_original_force_gate"]
            assert audit["all_retained_stages_full_corner_quadrature_and_raw_bounds"]
            assert all(row["independent_complete_constrained_residual_N"] <= 1e-9 for row in audit["rows"])
            assert (
                max(audit["independent_final_body_residual_N"], audit["independent_final_original_mapped_residual_N"])
                <= 1e-9
            )
            assert not audit["same_solution_equivalence_qualified"]
        details = {
            "independent_new_all_stage_audits": audits,
            "all744_three_level_material_frame_comparison": comparison,
        }
        extras.update(
            p / "meta" / name
            for name in [
                "u-fixed-history-barrier-commands-v2.json",
                "u-fixed-barrier-all-stage-comparison-commands-v1.json",
            ]
        )
        limitations = "All744 corresponding stages at2.5/1.25/0.625um barriers retain independent current normal/friction forces, original1e-9N balances, complete corner/quadrature fields and raw unsmoothed global bounds. Original stage counts, geometry/mass/material, actuation/gravity/friction and tangential mollifier remain fixed. All-stage material-frame tensor, energy, displacement and source-wrench comparisons retain absolute errors near zero. The immutable baseline parent archive is fully reverified and referenced, not duplicated; every referenced actual parent input is rehashed. This coarse regularization family is not spatial/contact-pressure/history convergence, a continuum hard-contact limit, full Franka/model/feedback acceptance or verified warmed cost. Startup failures remain in their preceding evidence packet."
    else:
        comparison = read("egg-fixed-surface-five-level-audit-v1")
        audit = read("egg-five-level-thickness-fem-reassembly-v1")
        assert comparison["fields_verified"] == len(comparison["rows"]) == 25
        assert audit["fields_independently_reassembled"] == len(audit["rows"]) == 25
        assert audit["all_original_complete_force_budgets_passed"]
        assert audit["all_full_corner_and_quadrature_tensors_and_original_probes_checked"]
        assert all(
            row["complete_reassembled_budget_fraction"] <= 1 and row["all_original_physical_probes_checked"] == 36
            for row in audit["rows"]
        )
        assert [row["layers"] for row in comparison["independent_boundary_and_surface_flux_audit"]["rows"]] == [
            2,
            3,
            4,
            5,
            6,
        ]
        assert not comparison["physical_accuracy_accepted"]
        details = {
            "original_five_level_archive_geometry_traction_comparison": comparison,
            "fresh_full_matrix_corner_quadrature_and_probe_audit": audit,
        }
        extras.update(
            p / "meta" / name
            for name in [
                "thickness-five-six-commands-v1.json",
                "thickness-five-level-fem-reassembly-commands-v1.json",
                "thickness-publication-large-extra-correction-v1.json",
                "publish_barrier_thickness_checkpoint_pre_extras_fix_v1.py",
            ]
        )
        limitations = "All25 original2/3/4/5/6-layer FEM fields keep fixed actual boundaries, mass, original tractions and36 probes. Fresh independent stiffness/mass/inertia-relief assembly rechecks every original complete force budget, full corner/quadrature tensors, energy, probes and unsmoothed global maxima. The original archive comparator's producer-recorded residual scope remains explicit rather than being mislabeled as fresh matrix reassembly. The15 older fields' immutable parent archive is fully reverified and referenced, not duplicated; every referenced actual parent input is rehashed. Tensor changes remain unresolved despite stable peak/energy. This is not three-dimensional bulk/CAD/contact/model/feedback or warmed physical accuracy/cost acceptance."
    assert parent_references
    summary = {
        "scope": __doc__,
        "kind": args.kind,
        "closed_runs": records,
        "source_identity_records": identities,
        "parent_archive": {
            "manifest": str(parent),
            "sha256": parent_sha,
            "complete_archive_reverification": parent_archives,
            "referenced_actual_original_sha256_reverified": parent_references,
        },
        **details,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    assert all(path.stat().st_size < 50_000_000 for path in extras), "Large inputs must use bounded run staging"
    destination = args.destination_root / f"{label}-{args.version}"
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
