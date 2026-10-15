"""Publish the fixed-pad U body axis and closed rigid-continuation/fine-factorization failures."""

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
    meta = p / f"closed-physical-followup-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-original-rigid-full-history-v2": ("717502_0", "FAILED", 1),
        "franka-original-rigid-full-prefix865-audit-v1": ("717589_0", "COMPLETED", 0),
        "franka-original-rigid-step866-recovery-v1": ("717590_0", "FAILED", 1),
        "franka-original-rigid-step866-homotopy-v1": ("717624_0", "FAILED", 1),
        "egg-covered-cad-level3-contact-v1": ("717295_0", "FAILED", 1),
        "u-nonlinear-fixed-pad5-body5669-v1": ("716333_0", "COMPLETED", 0),
        "u-nonlinear-fixed-pad5-body-fields-v1": ("716334_0", "COMPLETED", 0),
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(row[0].split("_")[0] for row in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%24000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    records, identities = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, (job, state, code) in jobs.items():
        run = p / name
        assert states[job] == state and int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, name_in_source = line.split("  ", 1)
                actual = Path(name_in_source) if Path(name_in_source).is_absolute() else run / kind / name_in_source
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected
                identities.append({"run": name, "kind": kind, "path": name_in_source, "sha256": expected})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "state": state, "returncode": code})
    audit = json.loads((p / "franka-original-rigid-full-prefix865-audit-v1/results/summary.json").read_text())
    assert audit["states_checked"] == len(audit["rows"]) == 865 and not audit["complete_requested_history"]
    assert all(
        row["body_force_equivalent_residual_N"] <= 1e-9 and row["fresh_actual_robot_effort_residual"] <= 1e-9
        for row in audit["rows"]
    )
    numerical = json.loads(
        (p / "franka-original-rigid-step866-homotopy-v1/results/numerical_recovery/summary.json").read_text()
    )
    assert not numerical["recovered_original_equation"]
    assert not (p / "franka-original-rigid-step866-homotopy-v1/results/step00866.npz").exists()
    fields = json.loads((p / "u-nonlinear-fixed-pad5-body-fields-v1/results/summary.json").read_text())
    assert fields["declared_spatial_axis"] == "body_at_fixed_pad"
    assert [row["body_tetrahedra"] for row in fields["rows"]] == [845, 2339, 5669]
    assert all(row["pad_resolution"] == 5 for row in fields["rows"])
    for row in fields["rows"]:
        assert (
            max(
                row["independent_body_residual_N"],
                row["independent_original_mapped_residual_N"],
                row["maximum_recorded_original_residual_N"],
            )
            <= 1e-9
        )
        for index in [1, 2]:
            assert (
                row["parts"][index]["geometry_array_sha256"]
                == fields["rows"][0]["parts"][index]["geometry_array_sha256"]
            )
    for path, expected in fields["input_sha256"].items():
        actual = Path(path)
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected
        extras.add(actual)
    for name in [
        "covered-cad-third-level-factorization-failure-v1.json",
        "original-rigid-step866-recovery-commands-v1.json",
        "original-rigid-step866-homotopy-commands-v1.json",
        "original-rigid-step866-homotopy-audit-commands-v1.json",
        "original-rigid-step866-homotopy-failure-v1.json",
    ]:
        extras.add(p / "meta" / name)
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "independently_qualified_rigid_prefix_states": 865,
        "recovered_original_rigid_step866": False,
        "homotopy_numerical_stages": len(numerical["attempts"]),
        "highest_accepted_artificial_parameter": max(
            row["residual_continuation_parameter"] for row in numerical["attempts"] if row["numerical_stage_accepted"]
        ),
        "fixed_pad_U_body_axis": fields,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"closed-physical-followup-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "All865 retained original rigid states pass separate physical force/moment/robot/motion audits. Step866 direct and residual-continuation recovery failures remain rejected; numerical stages add zero physical states. The171228-tet factorization MemoryError yields no accepted initializer or dynamic state. Three U body/geometry approximations share byte-identical pad geometry/material/actuation, but represented body mass changes at fixed density. Last-pair tensor12.33%, energy0.962% and raw lower peak5.26% do not establish physical convergence. Raw fields, original budgets and all failed records remain intact; full physical/model/feedback/warmed-cost scope stays open.",
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
