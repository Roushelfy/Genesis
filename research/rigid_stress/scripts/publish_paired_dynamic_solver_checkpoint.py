"""Archive closed paired Newton diagnostics and full-field solver qualifications, including real failures."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded
from research.rigid_stress.scripts.verify_nonconvex_publication import digest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = f"paired-dynamic-solvers-{args.version}"
    meta = p / f"paired-dynamic-solvers-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "egg-fine-first-dynamic-matrix-v1": ("719038_0", 1),
        "egg-coarse-first-dynamic-matrix-v1": ("719039_0", 1),
        "egg-fine-first-dynamic-matrix-audit-v1": ("719056_0", 0),
        "egg-coarse-first-dynamic-matrix-audit-v1": ("719057_0", 0),
        "egg-coarse-first-dynamic-matrix-v2": ("719060_0", 0),
        "egg-coarse-matrix-capture-equivalence-v1": ("719070_0", 0),
        "egg-cad1920-bordered-metis-candidate-v1": ("719045_0", 0),
        "egg-cad1920-bordered-metis-audit-v1": ("719046_0", 0),
        "egg-cad1920-bordered-metis-fields-v1": ("719047_0", 0),
        "egg-fine-first-dynamic-bordered-factor-v1": ("719064_0", 1),
        "egg-cad1920-schur-metis-candidate-v1": ("719084_0", 0),
        "egg-cad1920-schur-metis-audit-v1": ("719085_0", 0),
        "egg-cad1920-schur-metis-fields-v1": ("719086_0", 0),
        "egg-fine-first-dynamic-schur-factor-v1": ("719087_0", 0),
    }
    runs = [p / name for name in jobs]
    extras, input_records, source_records = set(), {}, []
    cache, parent_sources, parents = {}, {}, []
    for parent_label, expected in [
        ("contact-reference-methods-v1", "68a1349889fa5790b16af4305813ec9b4a99d9205299f6671070561c13f92bf1"),
        ("covered-cad-dynamic-factor-failure-v1", "0015205f170331b1a4f59700f1ef0ddee1c2e3112d9b90f93d50ad1425e1f10b"),
    ]:
        manifest = args.destination_root / parent_label / "manifest.json"
        assert digest(manifest)[0] == expected
        proof = verify_manifest(manifest, True, cache)
        value = json.loads(manifest.read_text())
        stage = p / "publication-staging" / parent_label
        for row in value["files"]:
            original = Path(row["original"])
            if original.is_relative_to(stage):
                original = args.runtime_root / original.relative_to(stage)
            if original.exists() and digest(original) == (row["sha256"], row["bytes"]):
                parent_sources[str(original)] = row["sha256"]
        for row in value["summary"].get("sharded_original_files", []):
            assert digest(Path(row["original"])) == (row["sha256"], row["bytes"])
            parent_sources[row["original"]] = row["sha256"]
        parents.append({"manifest": str(manifest), "sha256": expected, "complete_archive_reverification": proof})

    def retain(path, expected=None):
        actual = digest(path)[0]
        assert expected is None or expected == actual, path
        assert str(path) not in input_records or input_records[str(path)]["sha256"] == actual
        covered = parent_sources.get(str(path)) == actual
        input_records[str(path)] = {"sha256": actual, "covered_by_reverified_parent": covered}
        if not covered and not any(path.is_relative_to(run) for run in runs):
            assert path.is_relative_to(args.runtime_root) and path.stat().st_size < 50_000_000, path
            extras.add(path)

    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job, _ in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    records = []
    for name, (job, code) in jobs.items():
        run = p / name
        expected_state = "COMPLETED" if code == 0 else "FAILED"
        assert states[job] == expected_state and int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                expected, relative = line.split("  ", 1)
                source = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert digest(source)[0] == expected, source
                source_records.append({"run": name, "kind": kind, "path": str(source), "sha256": expected})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "scheduler_state": expected_state, "actual_wrapper_code": code})

    def read(name):
        value = json.loads((p / name / "results/summary.json").read_text())
        for path, expected in value.get("input_sha256", {}).items():
            retain(Path(path), expected)
        return value

    summaries = {name: read(name) for name, (_, code) in jobs.items() if code == 0}
    for name in ["egg-fine-first-dynamic-matrix-v1", "egg-coarse-first-dynamic-matrix-v1"]:
        assert "TypeError: Object of type int32 is not JSON serializable" in (p / name / "capture.log").read_text()
    assert "MemoryError" in (p / "egg-fine-first-dynamic-bordered-factor-v1/audit.log").read_text()
    qualifications = {}
    for kind in ["bordered", "schur"]:
        value = summaries[f"egg-cad1920-{kind}-metis-fields-v1"]
        assert value["states_checked"] == len(value["rows"]) == 5
        assert value["all_original_arrays_and_complete_fields_qualified"]
        assert value["initializer_comparison"]["byte_identical"]
        force = [audit for row in value["rows"] for audit in row["independent_force_audits"]]
        assert len(force) == 10
        maximum_force = max(
            max(row["complete_body_dynamic_residual_N"], row["complete_mapped_dynamic_residual_N"]) for row in force
        )
        assert maximum_force <= 1e-9
        qualifications[kind] = {
            "states": 5,
            "original_initializer_byte_identical": True,
            "maximum_complete_body_or_mapped_force_residual_N": maximum_force,
            "maximum_all_corner_tensor_relative_change": max(
                row["all_corner_tensor_relative_change"] for row in value["rows"]
            ),
            "maximum_all_quadrature_tensor_relative_change": max(
                row["all_quadrature_tensor_relative_change"] for row in value["rows"]
            ),
        }
    fine = summaries["egg-fine-first-dynamic-schur-factor-v1"]
    assert fine["original_matrix_arrays_unchanged"] and not fine["coefficient_pruning_or_regularization_added"]
    assert fine["true_original_linear_residual_N"] <= fine["original_cached_newton_linear_budget_N"]
    assert fine["new_physical_states_accepted"] == 0
    assert fine["captured_matrix_sha256"] == summaries["egg-fine-first-dynamic-matrix-audit-v1"]["matrix_sha256"]

    # Failed captures wrote their numerical arrays before the metadata exception.
    # Retain the same physical capture inputs explicitly, without rewriting those failures.
    for source_name in ["egg-covered-cad-level3-metis-contact-v2", "egg-cad1920-metis-qualification-candidate-v1"]:
        source = p / source_name
        config_path = source / "results/configuration.json"
        retain(config_path)
        closure_path = Path(json.loads(config_path.read_text())["closure_run"]) / "results/configuration.json"
        retain(closure_path)
        capture = Path(json.loads(closure_path.read_text())["capture"])
        report = capture / "results/summary.json"
        retain(report)
        capture_data = json.loads(report.read_text())
        frame = [row for row in capture_data["captures"] if row["mode"] == "stationary_lift_hold"][-1]
        retain(capture / "results" / frame["geometry_snapshot"])
        fingers = {row["link"] for row in frame["fingers"]}
        for row in capture_data["geometry"]:
            if row["link"] in fingers:
                retain(capture / "results" / row["mesh"])
        retain(capture / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz")

    extras.update(
        p / "meta" / name
        for name in [
            "paired-first-dynamic-matrix-capture-commands-v1.json",
            "paired-first-dynamic-matrix-capture-commands-v2.json",
            "paired-first-dynamic-matrix-audit-commands-v1.json",
            "coarse-matrix-capture-equivalence-commands-v1.json",
            "bordered-metis-full-state-qualification-commands-v1.json",
            "fine-first-dynamic-bordered-factor-commands-v1.json",
            "schur-metis-full-state-qualification-commands-v1.json",
            "fine-schur-paired-continuation-commands-v1.json",
        ]
    )
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": source_records,
        "parents": parents,
        "actual_input_sha256_reverified": input_records,
        "complete_original_coarse_qualifications": qualifications,
        "closed_run_summaries": summaries,
        "fine_nonlinear_continuation": {"job": "719100_0", "included_as_successful_evidence": False},
        "historical_fine_factor_limitation_text_correction": "The retained standalone report says only permutation changes. For its Schur run, exact block elimination also changes numerical factorization; every original coefficient and equation is retained. The historical report is preserved unchanged.",
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.update([meta / "summary.json", meta / "jobs.txt", meta / "job-commands.txt"])
    assert all(path.stat().st_size < 50_000_000 for path in extras)
    assert not any(any(path.is_relative_to(run) for run in runs) for path in extras)
    limitations = "All14 closed attempts retain actual successful/failed wrapper codes, scheduler accounting and closed sources. Both numerical candidates pass all five original coarse states, independent original force gates and complete corner/quadrature tensors before fine factorization. The exact fine Schur linear solve is not an accepted nonlinear physical state, a convergence result, a proven diagnosis of the internal MemoryError, a controlled speedup or warmed production cost. Distinct nonsymmetric coupling blocks are retained; no coefficient pruning or physical regularization is added. Failed capture metadata and measured long-double padding differences remain explicit. Full A/B/C physical/model/island/feedback/cost requirements remain open."
    destination = args.destination_root / label
    publish_bounded(
        args.runtime_root, destination, runs, sorted(extras), summary, limitations, p / "publication-staging"
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
            str(p / "publication-staging" / label),
            "--output",
            str(meta / "runtime-verification.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
