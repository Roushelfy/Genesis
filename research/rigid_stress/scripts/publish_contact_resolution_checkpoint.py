"""Publish complete covered-contact fields or independently qualified reference methods without accepting unresolved physical convergence."""

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
    parser.add_argument("--kind", choices=["fields", "methods"], required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = "covered-contact-fields" if args.kind == "fields" else "contact-reference-methods"
    meta = p / f"{label}-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    if args.kind == "methods":
        cases = {
            "franka-original-rigid-step866-tangent-diagnosis-v1": ("717697_0", 0),
            "franka-original-rigid-step866-direct-tangent-v1": ("717719_0", 0),
            "franka-original-rigid-direct-prefix-qualification-v1": ("717734_0", 0),
            "franka-original-rigid-step866-scaled-continuation-v1": ("717745_0", 1),
            "franka-original-rigid-step866-deformable-seed-v1": ("717809_0", 1),
            "franka-original-rigid-step866-lagged-v1": ("717812_0", 0),
            "franka-original-rigid-step866-lagged-audit-v1": ("717813_0", 0),
            "franka-original-feedback-prefix-qualification-v1": ("717919_0", 1),
            "franka-original-feedback-prefix-qualification-v2": ("717928_0", 1),
            "franka-original-feedback-prefix-qualification-v3": ("717941_0", 0),
            "reference-metis-dependency-v1": ("717849_0", 0),
            "egg-cad1920-metis-qualification-candidate-v1": ("717897_0", 0),
            "egg-cad1920-metis-qualification-audit-v1": ("717916_0", 0),
            "egg-cad1920-metis-original-fields-qualification-v1": ("717917_0", 0),
            "egg-covered-cad-level3-metis-contact-v1": ("717938_0", 1),
        }
    else:
        cases = {
            "egg-covered-cad-level2-contact-v1": ("716886_0", 0),
            "egg-covered-cad-level2-contact-v1-audit": ("716887_0", 0),
            "egg-covered-cad-level2-coverage-v1": ("716888_0", 0),
            "egg-covered-cad-levels12-fields-v2": ("717744_0", 137),
            "egg-covered-cad-levels12-fields-v3": ("717746_0", 1),
            "egg-covered-cad-levels12-fields-v4": ("717788_0", 0),
            "u-nonlinear-fixed-body-pad9-v2": ("716310_0", 0),
            "u-nonlinear-fixed-body-pad-fields-v2": ("716317_0", 0),
        }
        for version in range(1, 5):
            journal = p / f"meta/covered-contact-axes-figure-commands-v{version}.json"
            command = json.loads(journal.read_text())
            cases[Path(command["run"]).name] = (command["job"] + "_0", 0)
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job, _ in cases.values())]
    if args.kind == "fields":
        query[-1] += ",717706"
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
    for name, (job, code) in cases.items():
        run = p / name
        assert states[job] == ("COMPLETED" if code == 0 else "FAILED")
        assert int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            if not identity.exists():
                continue
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, file = line.split("  ", 1)
                actual = Path(file) if Path(file).is_absolute() else run / kind / file
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected
                identities.append({"run": name, "kind": kind, "file": file, "sha256": expected})
        records.append({"run": name, "job": job, "scheduler_state": states[job], "actual_wrapper_code": code})
        extras.update(p.glob(f"*{job}.log"))

    def read(name):
        return json.loads((p / name / "results/summary.json").read_text())

    if args.kind == "methods":
        direct = read("franka-original-rigid-direct-prefix-qualification-v1")
        assert direct["states_checked"] == 865 and direct["all_original_states_and_forces_preserved"]
        audit = read("franka-original-rigid-step866-lagged-audit-v1")
        producer = read("franka-original-rigid-step866-lagged-v1")
        assert audit["states_checked"] == len(audit["rows"]) == 866 and not audit["complete_requested_history"]
        assert producer["byte_identical_prefix_states"] == 865 and producer["recovered_step"] == 866
        assert len(producer["numerical_recovery"]["attempts"]) == 38
        assert all(
            max(r["body_force_equivalent_residual_N"], r["fresh_actual_robot_effort_residual"]) <= 1e-9
            for r in audit["rows"]
        )
        feedback = read("franka-original-feedback-prefix-qualification-v3")
        assert feedback["states_checked"] == len(feedback["rows"]) == 866
        assert not feedback["complete_original_330_macro_history"] and feedback["audited_prefix_explicitly_allowed"]
        ordering = read("egg-cad1920-metis-original-fields-qualification-v1")
        assert (
            ordering["all_original_arrays_and_complete_fields_qualified"]
            and ordering["states_checked"] == len(ordering["rows"]) == 5
        )
        force_checks = [a for r in ordering["rows"] for a in r["independent_force_audits"]]
        assert len(force_checks) == 10 and all(
            max(r["complete_body_dynamic_residual_N"], r["complete_mapped_dynamic_residual_N"]) <= 1e-9
            for r in force_checks
        )
        for report in [feedback, ordering]:
            for file, expected in report["input_sha256"].items():
                file = Path(file)
                assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
                extras.add(file)
        summary = {
            "direct_assembly_preserves_original_states": direct["states_checked"],
            "independently_qualified_rigid_states": 866,
            "lagged_original_equation_recovery": producer,
            "independent_rigid_audit": audit,
            "feedback_prefix": feedback,
            "metis_original_field_qualification": ordering,
        }
        extras.update(
            p / "meta" / name
            for name in [
                "original-rigid-full-history-commands-v3.json",
                "original-rigid-deformable-feedback-commands-v1.json",
                "original-rigid-resolved-native-commands-v1.json",
                "covered-cad-level3-metis-commands-v2.json",
            ]
        )
        limitations = "Lagged friction is used only for numerical stages; final forces/friction, physical timestep, old state and1e-9 gates are unchanged and independently audited. Every retained865 prefix file remains byte-identical. The866-state feedback prefix is a postprocessor qualification, not the complete2640-state grasp or a physically converged truth. Scaled continuation/intersecting roots and earlier postprocessor failures remain rejected. METIS qualifies only the five saved1920-tet dynamic states, held initializer and full corner/quadrature fields before a separate fine retry; reference timing is not warmed native cost. Production Quadrants/rigid dynamics are unchanged."
    else:
        cad = read("egg-covered-cad-levels12-fields-v4")
        audit = read("egg-covered-cad-level2-contact-v1-audit")
        coverage = read("egg-covered-cad-level2-coverage-v1")
        pad = read("u-nonlinear-fixed-body-pad-fields-v2")
        assert [r["tetrahedra"] for r in cad["rows"]] == [31668, 64716] and cad["common_physical_probes"] == 158340
        assert audit["states_checked"] == len(audit["rows"]) == len(coverage["rows"]) == 5
        assert all(
            max(r["complete_body_dynamic_residual_N"], r["complete_mapped_dynamic_residual_N"]) <= 1e-9
            for r in audit["rows"]
        )
        assert all(r["active_samples_outside_marking_caps"] == 0 for r in coverage["rows"])
        assert len(pad["rows"]) == 3 and {r["body_mesh_sha256"] for r in pad["rows"]} == {
            pad["rows"][0]["body_mesh_sha256"]
        }
        assert all(
            max(
                r["independent_body_residual_N"],
                r["independent_original_mapped_residual_N"],
                r["maximum_recorded_original_residual_N"],
            )
            <= 1e-9
            for r in pad["rows"]
        )
        for report in [cad, pad]:
            for file, expected in report["input_sha256"].items():
                file = Path(file)
                assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
                extras.add(file)
        qa_path = p / "meta/covered-contact-axes-visual-qa-v3.json"
        qa = json.loads(qa_path.read_text())
        assert qa["visual_qa_passed"] and qa["svg_embedded_images"] == 0
        extras.add(qa_path)
        first = p / "egg-covered-cad-levels12-fields-v1"
        assert (
            states["717706_0"] == "FAILED"
            and not (first / "returncode.txt").exists()
            and not (first / "source-after.sha256").exists()
        )
        for file in first.rglob("*"):
            if file.is_file() and not any(part in {"tmp", "__pycache__"} for part in file.relative_to(first).parts):
                extras.add(file)
        extras.update(p.glob("*717706_0.log"))
        summary = {
            "covered_CAD_full_fields": cad,
            "independent_second_CAD_audit": audit,
            "second_contact_coverage": coverage,
            "fixed_body_U_pad_axis": pad,
            "figure_QA": qa,
            "early_setup_failure": {
                "job": "717706_0",
                "phase": "source patching before Python",
                "missing_natural_exit_or_final_source_identity_not_fabricated": True,
            },
        }
        limitations = "All five second-level CAD states pass independent original force gates and contact coverage. The158340-probe comparison retains complete corners/fields: tensor9.68%, energy0.407%, raw peak13.89%; two coupled bulk/geometry/contact levels are not physical convergence. The U pad5/7/9 sequence keeps body geometry/mass fixed and has a conditional1.20% last tensor change; body/geometry and regularization remain separate. OOM, original-tolerance manufactured-field failures and partial source-setup failure are preserved with true status. Figure scalar/vector/raster QA does not establish physical accuracy."
    summary = {
        "scope": __doc__,
        "kind": args.kind,
        "closed_runs": records,
        "source_identity_records": identities,
        **summary,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"{label}-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in cases],
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
