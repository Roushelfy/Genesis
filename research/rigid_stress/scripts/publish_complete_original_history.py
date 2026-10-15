"""Publish the complete original330-macro history, independent force/global-field audits and native replays."""

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
    meta = p / f"complete-original-history-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "franka-original-physical-budget-full-history-v1": "716378_0",
        "franka-original-physical-budget-full-audit-v1": "716379_0",
        "franka-original-physical-budget-full-fields-v1": "716380_0",
        "franka-original-physical-budget-full-native-v1": "716381_0",
        "franka-original-physical-budget-full-bounds-v1": "716382_0",
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
    identities, records = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt", p / "meta/full-original-history-readonly-audit-v1.json"}
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            identity = run / f"{kind}.sha256"
            if not identity.exists():
                continue
            assert identity.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in identity.read_text().splitlines():
                expected, path = line.split("  ", 1)
                assert hashlib.sha256((run / kind / path).read_bytes()).hexdigest() == expected
                identities.append({"run": name, "identity": kind, "path": path, "sha256": expected})
        records.append({"run": name, "job": job, "state": states[job], "returncode": 0})
        extras.update(p.glob(f"*{job}.log"))

    def read(name):
        return json.loads((p / name / "results/summary.json").read_text())

    producer, audit, fields, native, bounds = [read(name) for name in jobs]
    assert producer["original_macro_steps_solved"] == producer["available_original_macro_steps"] == 330
    assert len(producer["steps"]) == audit["steps_checked"] == fields["states_checked"] == 2640
    assert audit["requested_steps"] == len(audit["rows"]) == len(fields["rows"]) == 2640
    assert audit["complete_requested_history"] and fields["complete_original_330_macro_history"]
    assert audit["source_closure_kind"] == "natural_producer_exit"
    assert not audit["scheduler_success_inferred_from_wrapper_code"]
    assert [row["step"] for row in audit["rows"]] == list(range(1, 2641))
    for row in audit["rows"]:
        assert row["body_dynamic_residual_N"] <= 1e-9 and row["actual_robot_step_effort_residual"] <= 1e-9
    config_path = p / next(iter(jobs)) / "results/configuration.json"
    config = json.loads(config_path.read_text())
    assert config["macro_steps"] == 330 and config["dt"] == 0.00125
    protocol_path = Path(config["protocol_run"]) / "results/original_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    assert len(controls) == 330 and [row["tick"] for row in controls] == list(range(330))
    extras.add(protocol_path)
    assert [row["step"] for row in fields["rows"]] == list(range(1, 2641))
    for path, expected in fields["input_sha256"].items():
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if path.stat().st_size < 50_000_000:
            extras.add(path)
    assert native["complete_original_330_macro_history"] and len(native["rows"]) == 14
    assert len(native["steps"]) == 7 and {row["mode"] for row in native["rows"]} == {"full_inertia", "rigid_relief"}
    assert all(
        row["native_complete_budget_fraction"] <= 1 and row["native_vs_independent_tensor_relative_L2"] < 1e-6
        for row in native["rows"]
    )
    assert (
        bounds["total_sampled_points"] == sum(row["interior_and_corner_samples"] for row in bounds["rows"]) == 11062080
    )
    assert len(bounds["rows"]) == 334
    observables = {
        "original_macro_steps": 330,
        "physical_substeps": 2640,
        "horizon_s": audit["rows"][-1]["time_s"],
        "maximum_independent_body_residual_N": max(row["body_dynamic_residual_N"] for row in audit["rows"]),
        "maximum_actual_robot_step_effort_residual": max(
            row["actual_robot_step_effort_residual"] for row in audit["rows"]
        ),
        "raw_global_history_peak_lower_Pa": max(row["raw_global_peak_lower_Pa"] for row in fields["rows"]),
        "raw_global_history_peak_upper_Pa": max(row["raw_global_peak_upper_Pa"] for row in fields["rows"]),
        "maximum_native_tensor_relative_L2": max(
            row["native_vs_independent_tensor_relative_L2"] for row in native["rows"]
        ),
        "maximum_native_complete_budget_fraction": max(
            row["native_complete_budget_fraction"] for row in native["rows"]
        ),
        "native_replays": 14,
        "certificate_samples": bounds["total_sampled_points"],
        "maximum_strict_Newton_misses_accepted_at_original_physical_budget": sum(
            not row.get("strict_newton_target_met", True) for row in producer["steps"]
        ),
    }
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "complete_original_macro_protocol": {
            "path": str(protocol_path),
            "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "macro_steps": 330,
        },
        "independent_force_control_audit": audit,
        "all_state_global_fields": fields,
        "actual_native_replays": native,
        "independent_interior_corner_checks": bounds,
        "observables": observables,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"complete-original-history-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "All330 original macro controls and2640 physical states through3.3s, real initial geometry/ground support and unchanged1e-9 body/actual-robot gates are preserved. Recorded artificial solver stages are not physical time steps. Completion follows gapless state counts, matching protocol and independent audits, not legacy prefix labels or wrapper code. Fourteen native nodal replays isolate elastic algebra at seven times under two declared inertia inputs. Full global raw intervals remain intact; interior samples check the independent Bernstein certificates and are not a sampling-only maximum proof. This is one480-tet reference and does not establish physical spatial/contact/geometry/time convergence, finite-footprint model accuracy, matching rigid/deformable feedback or verified warmed cost.",
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
