"""Publish complete corner-arithmetic/oracle qualifications and matched resolved-load diagnostics."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as element_tree
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
    label = f"corner-arithmetic-model-{args.version}"
    meta = p / f"corner-arithmetic-model-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "native-corner-arithmetic-case2-v1": "719196_0",
        "native-corner-arithmetic-case3-v1": "719197_0",
        "native-corner-arithmetic-case2-cpu-v1": "719198_0",
        "native-corner-arithmetic-case3-cpu-v1": "719204_0",
        "native-corner-difference-form-cpu-v1": "719217_0",
        "native-corner-difference-form-cuda-v1": "719232_0",
        "native-corner-polynomial-decimal-cpu-v1": "719228_0",
        "native-corner-polynomial-decimal-cuda-v1": "719233_0",
        "native-corner-production-cpu-v1": "719248_0",
        "native-corner-production-cuda-v1": "719249_0",
        "native-corner-production-reference-cpu-v2": "719275_0",
        "native-corner-regression-cpu-v1": "719264_0",
        "native-corner-regression-gpu-v1": "719265_0",
        "native-corner-corrected-linear-replay-case2-v1": "719284_0",
        "native-corner-corrected-linear-replay-case3-v1": "719285_0",
        "native-corner-matched-kernel-cost-v1": "719295_0",
        "egg-covered-cad-levels12-linearization-v1": "719205_0",
    }
    runs = [p / name for name in jobs]
    cache, parents, parent_sources = {}, [], {}
    for parent_label, expected in [
        ("egg-spatial-geometry-v1", "e8d6d58c38e40141c8f73beae1539604756e4ad8f71501705f7199a4c282a8a6"),
        ("covered-contact-fields-v1", "823d97f4858532f728257f36b74ccef409d4444cb57d99a1af6c880c86029faa"),
    ]:
        path = args.destination_root / parent_label / "manifest.json"
        assert digest(path)[0] == expected
        proof = verify_manifest(path, True, cache)
        manifest = json.loads(path.read_text())
        stage = p / "publication-staging" / parent_label
        for row in manifest["files"]:
            original = Path(row["original"])
            if original.is_relative_to(stage):
                original = args.runtime_root / original.relative_to(stage)
            if original.exists() and digest(original) == (row["sha256"], row["bytes"]):
                parent_sources[str(original)] = row["sha256"]
        for row in manifest["summary"].get("sharded_original_files", []):
            assert digest(Path(row["original"])) == (row["sha256"], row["bytes"])
            parent_sources[row["original"]] = row["sha256"]
        parents.append({"manifest": str(path), "sha256": expected, "complete_archive_reverification": proof})
    extras, input_records, source_records = set(), {}, []

    def retain(path, expected=None):
        actual = digest(path)[0]
        assert expected is None or actual == expected, path
        assert str(path) not in input_records or input_records[str(path)]["sha256"] == actual
        covered = parent_sources.get(str(path)) == actual
        input_records[str(path)] = {"sha256": actual, "covered_by_reverified_parent": covered}
        if not covered and not any(path.is_relative_to(run) for run in runs):
            assert path.stat().st_size < 50_000_000, path
            if not path.is_relative_to(args.runtime_root):
                target = meta / "external-inputs" / (actual + path.suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                assert digest(target)[0] == actual
                input_records[str(path)]["archived_external_copy"] = str(target)
                path = target
            extras.add(path)

    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,ExitCode,Elapsed,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    records = []
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            if not before.exists():
                continue
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                expected, relative = line.split("  ", 1)
                original = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert digest(original)[0] == expected, original
                source_records.append({"run": name, "kind": kind, "path": str(original), "sha256": expected})
        records.append({"run": name, "job": job, "scheduler_state": "COMPLETED", "actual_wrapper_code": 0})
        extras.update(p.glob(f"*{job}.log"))

    def read(name):
        value = json.loads((p / name / "results/summary.json").read_text())
        for original, expected in value.get("input_sha256", {}).items():
            retain(Path(original), expected)
        return value

    summaries = {name: read(name) for name in jobs if (p / name / "results/summary.json").exists()}
    tests = []
    for backend in ["cpu", "gpu"]:
        xml = p / f"native-corner-regression-{backend}-v1/junit.xml"
        cases = element_tree.parse(xml).getroot().findall(".//testcase")
        assert len(cases) == 13
        assert all(
            case.find("failure") is None and case.find("error") is None and case.find("skipped") is None
            for case in cases
        )
        assert sum("corner_cancellation" in case.attrib["name"] for case in cases) == 2
        tests.append(
            {"backend": backend, "passed_testcases": len(cases), "case_names": [case.attrib["name"] for case in cases]}
        )
    for backend in ["cpu", "cuda"]:
        decimal = summaries[f"native-corner-polynomial-decimal-{backend}-v1"]
        assert decimal["decimal_precision"] == 96 and decimal["exact_rational_ten_monomial_corner_qualification_passed"]
        assert len(decimal["rows"]) == 4
        assert all(
            row["extended_cofactor_vs_decimal_polynomial"]["maximum_existing_component_budget_fraction"] <= 1
            for row in decimal["rows"]
        )
        native = summaries[f"native-corner-production-{backend}-v1"]
        assert native["all_existing_component_budgets_passed"] and len(native["rows"]) == 8
        assert all(row["raw_global_peak_existing_budget_passed"] for row in native["rows"])
    reference = summaries["native-corner-production-reference-cpu-v2"]
    assert reference["all_existing_component_budgets_passed"] and len(reference["rows"]) == 8
    assert all(
        row["extended_reference_vs_decimal_polynomial"]["maximum_existing_component_budget_fraction"] <= 1
        for row in reference["rows"]
    )
    for case in [2, 3]:
        replay = summaries[f"native-corner-corrected-linear-replay-case{case}-v1"]
        assert replay["all_existing_gates_passed"] and len(replay["rows"]) == 2
        assert all(
            row["all_existing_gates_passed"] and row["independent_complete_budget_fraction"] <= 1
            for row in replay["rows"]
        )
    model = summaries["egg-covered-cad-levels12-linearization-v1"]
    assert [row["tetrahedra"] for row in model["rows"]] == [31668, 64716]
    assert all(row["linear_complete_residual_N"] <= 1e-9 for row in model["rows"])
    benchmark = summaries["native-corner-matched-kernel-cost-v1"]
    assert [row["batch"] for row in benchmark["rows"]] == [1024, 2048]
    assert all(row["complete_affine_component_control_passed"] for row in benchmark["rows"])
    for name in [
        "native-corner-arithmetic-commands-v1.json",
        "native-corner-arithmetic-cpu-commands-v1.json",
        "native-arithmetic-and-covered-linearization-commands-v1.json",
        "native-corner-difference-form-commands-v1.json",
        "native-corner-polynomial-decimal-commands-v1.json",
        "native-corner-difference-decimal-cuda-commands-v1.json",
        "native-corner-production-qualification-commands-v1.json",
        "native-corner-production-reference-commands-v2.json",
        "native-corner-regression-commands-v1.json",
        "native-corner-corrected-replay-commands-v1.json",
        "native-corner-matched-kernel-cost-commands-v1.json",
        "covered-cad-three-level-linearization-commands-v1.json",
        "native-solve-before-corner-difference-v1.py",
        "native-corner-difference-production-v1.patch",
        "native-corner-cached-uncached-byte-comparison-v1.json",
    ]:
        retain(p / "meta" / name)
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "parents": parents,
        "source_identity_records": source_records,
        "actual_inputs_reverified": input_records,
        "closed_run_summaries": summaries,
        "regression_tests": tests,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.update([meta / "summary.json", meta / "jobs.txt", meta / "job-commands.txt"])
    assert all(path.stat().st_size < 50_000_000 for path in extras)
    assert not any(any(path.is_relative_to(run) for run in runs) for path in extras)
    destination = args.destination_root / label
    limitations = "All original failed replay exits and float64-oracle discrepancies remain in reverified parent evidence. Original component/force/peak budgets remain unchanged; exact rational polynomial and96-digit decimal geometry checks cover every frozen element/corner/component. CPU/CUDA actual kernels, complete fresh native replays and13 tests per backend close before publication. Matched warmed timings cover only the changed corner/peak stage on an analytic low-mesh control. The covered CAD linearization results remain two conditional6.25ms states with unresolved physical spatial/contact/history/model/island/feedback and warmed application-cost requirements. Full A/B/C scope remains active."
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
