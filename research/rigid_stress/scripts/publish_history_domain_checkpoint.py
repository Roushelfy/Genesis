"""Publish closed contact-history, marking-domain or observable-method evidence with explicit scope."""

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
    parser.add_argument("--kind", choices=["history", "domain", "methods"], required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = {
        "history": "recorded-contact-history",
        "domain": "covered-marking-domain",
        "methods": "contact-observable-methods",
    }[args.kind]
    meta = p / f"{label}-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    cases = {
        "history": {
            "u-nonlinear-recorded-history-qualification-v1": ("718566_0", 0),
            "u-nonlinear-recorded-history-audit-v2": ("718584_0", 0),
        },
        "domain": {
            "egg-broad-covered-cad-level1-contact-v1": ("716992_0", 0),
            "egg-broad-covered-cad-level1-contact-v1-audit": ("716993_0", 0),
            "egg-broad-covered-cad-level1-coverage-v1": ("716994_0", 0),
            "egg-covered-marking-domain-fields-v1": ("718622_0", 0),
        },
        "methods": {
            "nested-bulk-observable-qualification-v1": ("718623_0", 0),
            "u-initialization-constraint-geometry-v1": ("718726_0", 0),
            "u-initialization-constraint-geometry-v2": ("718741_0", 0),
            "u-fixed-history-barrier-1250nm-v1": ("718703_0", 1),
            "u-fixed-history-barrier-625nm-v1": ("718705_0", 1),
        },
    }[args.kind]
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job, _ in cases.values())]
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
        expected_state = "COMPLETED" if code == 0 else "FAILED"
        assert states[job] == expected_state and int((run / "returncode.txt").read_text()) == code
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
        records.append({"run": name, "job": job, "scheduler_state": expected_state, "actual_wrapper_code": code})

    def report(name):
        value = json.loads((p / name / "results/summary.json").read_text())
        for path, digest in value.get("input_sha256", {}).items():
            path = Path(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            extras.add(path)
        return value

    if args.kind == "history":
        result = report("u-nonlinear-recorded-history-audit-v2")
        assert result["stages_checked"] == len(result["rows"]) == 744
        assert result["all_retained_stages_independent_current_friction_and_original_force_gate"]
        assert result["all_retained_stages_full_corner_quadrature_and_raw_bounds"]
        assert all(row["independent_complete_constrained_residual_N"] <= 1e-9 for row in result["rows"])
        assert (
            max(result["independent_final_body_residual_N"], result["independent_final_original_mapped_residual_N"])
            <= 1e-9
        )
        assert (
            result["original_accepted_stage_keys_identical"]
            and not result["original_accepted_residual_records_identical"]
        )
        extras.update(
            p / "meta" / name
            for name in [
                "u-recorded-history-qualification-commands-v1.json",
                "u-recorded-history-audit-commands-v1.json",
                "u-recorded-history-audit-commands-v2.json",
                "u-recorded-history-bit-identity-observation-v1.json",
            ]
        )
        details = {"complete_recorded_stage_audit": result}
        limitations = "All744 original quasi-static initialization/load stages independently pass current normal/friction forces and original1e-9N constrained balances, with full corner/quadrature fields and raw whole-volume bounds. Original final geometry/applied force arrays remain exact; measured displacement/contact/stress/iteration bit differences remain explicit, with final full-quadrature relative difference6.62e-11. No old intermediate full fields exist for an identical-history claim. These stages are not physical timesteps. One coarse845-tet/pad5 case is not contact-pressure/spatial/history/barrier convergence or full Franka/model/warmed-cost acceptance. Live barrier and initialization controls are excluded."
    elif args.kind == "domain":
        result = report("egg-covered-marking-domain-fields-v1")
        audit = report("egg-broad-covered-cad-level1-contact-v1-audit")
        coverage = report("egg-broad-covered-cad-level1-coverage-v1")
        assert result["common_physical_probes"] == 158340 and len(result["rows"]) == 2
        assert [row["tetrahedra"] for row in result["rows"]] == [31668, 90240]
        assert audit["states_checked"] == len(audit["rows"]) == len(coverage["rows"]) == 5
        assert all(
            max(row["complete_body_dynamic_residual_N"], row["complete_mapped_dynamic_residual_N"]) <= 1e-9
            for row in audit["rows"]
        )
        assert all(row["active_samples_outside_marking_caps"] == 0 for row in coverage["rows"])
        factories = []
        for version in [4, 5]:
            root = p / f"egg-adaptive-cad-polyhedron-geometry-v{version}"
            source = {}
            for line in (root / "source.sha256").read_text().splitlines():
                digest, relative = line.split("  ", 1)
                if not relative.endswith(".py"):
                    continue
                path = root / "source" / relative
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                source[relative] = digest
                extras.add(path)
            factories.append(source)
        assert factories[0] == factories[1] and len(factories[0]) == 8
        extras.update(
            p / "meta" / name
            for name in ["broad-covered-cad-closed-readonly-v1.json", "covered-marking-domain-fields-commands-v1.json"]
        )
        details = {
            "all_original_common_point_fields": result,
            "broad_five_state_audit": audit,
            "broad_contact_coverage": coverage,
            "all_eight_geometry_constructor_Python_sources_identical": factories[0],
        }
        limitations = "The150um/500um numerical marking domains share declared CAD inputs, identical seed hashes and all eight constructor Python sources. All five broad states pass original independent force and contact-coverage gates. All158340 original probes/full raw fields remain: tensor changes4.27% despite raw peak0.00555%, energy0.173% and displacement0.173%. Represented geometry/mass, bulk and trace approximation remain coupled; neither coverage nor a matching scalar peak establishes a physical tail. Actual different provenance roots remain retained. Full original application/geometry/contact/regularization/model/feedback and warmed accuracy/cost acceptance stay open."
    else:
        result = report("nested-bulk-observable-qualification-v1")
        anchors = report("u-initialization-constraint-geometry-v2")
        report("u-initialization-constraint-geometry-v1")
        assert (
            result["qualification_passed"]
            and result["analytic_affine_tensor_squared_volume_integral_Pa2_m3"]
            == result["computed_exact_simplex_formula_Pa2_m3"]
        )
        assert anchors["common_candidate_points"] == 23 and len(anchors["common_anchor_components"]) == 6
        assert not all(row["all_six_physical_coordinate_constraints_match_baseline"] for row in anchors["rows"])
        for name in ["u-fixed-history-barrier-1250nm-v1", "u-fixed-history-barrier-625nm-v1"]:
            assert not (p / name / "contact.log").exists()
        extras.update(
            p / "meta" / name
            for name in [
                "nested-bulk-observable-qualification-commands-v1.json",
                "frozen-traction-local-bulk-audit-commands-v1.json",
                "u-initialization-constraint-commands-v1.json",
                "u-initialization-constraint-commands-v2.json",
                "u-fixed-history-barrier-commands-v1.json",
                "u-fixed-history-barrier-commands-v2.json",
                "u-fixed-history-barrier-startup-failures-v1.json",
                "u-common-initialization-control-commands-v1.json",
            ]
        )
        details = {
            "manufactured_observable_qualification": result,
            "original_initialization_coordinates_and_shared_anchor_asset": anchors,
        }
        limitations = "Manufactured6/12-tet P2 transfer and affine-tensor volume integration pass analytic and independent quadrature checks; a nonzero nested energy identity is verified at original numerical force gates. This is a numerical method control, not physical contact convergence. Actual old U initialization coordinate selections differ across meshes; their final physical/history effect is unmeasured. A23-point common set supplies rank6 anchors for a separate controlled study, without changing any original result. Both missing-frozen-import startup failures retain real code1/source closure and no FEM output; their unstarted audits were cancelled. Fixed-history barrier and common-initialization controls remain live/unverified and are excluded as accepted results. Full original physical/model/feedback/warmed-cost objective remains."
    summary = {
        "scope": __doc__,
        "kind": args.kind,
        "closed_runs": records,
        "source_identity_records": identities,
        **details,
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
