"""Publish source-closed intrinsic path qualifications together with every excluded or intentional failure."""

import argparse
import json
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
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = "native-intrinsic-capacity-v1"
    meta = p / f"{label}-publication-v2"
    meta.mkdir(parents=True, exist_ok=False)
    definitions = [
        ("extended-native-ellipsoid-level1-models-v1", 719781, "FAILED", "1:0", 1, True, False),
        ("extended-native-ellipsoid-level2-models-v1", 719782, "FAILED", "6:0", 134, True, False),
        ("extended-native-capsule-level1-models-v1", 719783, "FAILED", "6:0", 134, True, False),
        ("extended-native-capsule-level2-models-v1", 719784, "FAILED", "6:0", 134, True, False),
        ("intrinsic-path-heap-cpu-regression-v1", 719871, "FAILED", "1:0", 1, True, False),
        ("intrinsic-path-heap-gpu-regression-v1", 719872, "FAILED", "1:0", 0, False, False),
        ("intrinsic-path-heap-cpu-regression-v2", 719886, "COMPLETED", "0:0", 0, True, True),
        ("intrinsic-path-heap-gpu-regression-v2", 719887, "COMPLETED", "0:0", 0, True, True),
        ("intrinsic-path-heap-large-cpu-v1", 719910, "COMPLETED", "0:0", 0, True, True),
        ("intrinsic-path-heap-mutated-edge-regression-v1", 719924, "FAILED", "1:0", 1, True, False),
    ]
    query = ["sacct", "-X", "-n", "-P", "-j", ",".join(str(row[1]) for row in definitions)]
    account = subprocess.check_output([*query, "--format=JobIDRaw,State,ExitCode,Elapsed"], text=True)
    (meta / "jobs.txt").write_text(account)
    states = {line.split("|")[0]: line.split("|")[1:3] for line in account.splitlines() if line}
    records, runs, extras = [], [], set()
    for name, job, state, exit_code, program_code, closure, accepted in definitions:
        run = p / name
        assert states[str(job)] == [state, exit_code]
        assert int((run / "returncode.txt").read_text()) == program_code
        actual_closure = (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        assert actual_closure is closure
        for line in (run / "source.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            assert digest(run / "source" / relative)[0] == expected
        records.append(
            {
                "run": name,
                "job": job,
                "scheduler_state": state,
                "scheduler_exit": exit_code,
                "saved_program_or_wrapper_code": program_code,
                "source_closure_matches": closure,
                "accepted_qualification_evidence": accepted,
            }
        )
        runs.append(run)
        extras.update(p.glob(f"*{job}_0.log"))
    for backend in ["cpu", "gpu"]:
        root = element_tree.fromstring((p / f"intrinsic-path-heap-{backend}-regression-v2/junit.xml").read_bytes())
        suites = list(root.iter("testsuite"))
        assert sum(int(row.attrib["tests"]) for row in suites) == 12
        assert all(int(row.attrib["failures"]) == int(row.attrib["errors"]) == 0 for row in suites)
    mutation_log = (p / "intrinsic-path-heap-mutated-edge-regression-v1/pytest.log").read_text()
    assert "Mismatched elements: 112 / 2880" in mutation_log and "rtol=2e-12, atol=2e-20" in mutation_log
    large = json.loads((p / "intrinsic-path-heap-large-cpu-v1/results/summary.json").read_text())
    assert large["port_count"] == 125710 and len(large["rows"]) == 3 and large["exact_original_restoration"]
    assert all(row["all_support_membership_matches"] for row in large["rows"])
    assert sum(row["all_port_entries_checked"] for row in large["rows"]) == 5656950
    snapshot = p / "source-snapshots/native-intrinsic-heap-v1"
    for line in (snapshot / "snapshot-files.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert digest(snapshot / relative)[0] == expected
    for file in Path("genesis/engine/solvers/rigid/stress").rglob("*.py"):
        expected = digest(file)[0]
        for run in [
            snapshot,
            *(p / f"intrinsic-path-heap-{backend}-regression-v2/source" for backend in ["cpu", "gpu"]),
        ]:
            assert digest(run / file)[0] == expected
    extras.update(file for file in snapshot.rglob("*") if file.is_file())
    parents = []
    for name in ["extended-six-shape-qualification-v1", "extended-finite-ellipsoid-fields-v1"]:
        manifest = args.destination_root / name / "manifest.json"
        parents.append(
            {"manifest": str(manifest), "sha256": digest(manifest)[0], "proof": verify_manifest(manifest, True, {})}
        )
    for original, expected in large["input_sha256"].items():
        assert digest(Path(original))[0] == expected
    names = [
        "native-intrinsic-capacity-publisher-failed-attempt-v1.py",
        "native-intrinsic-capacity-publisher-failed-attempt-v1.json",
        "native-intrinsic-capacity-publication-v1.log",
        "extended-convex-native-model-commands-v1.json",
        "native-finite-failed-dependency-cancellation-v1.json",
        "intrinsic-path-heap-regression-commands-v1.json",
        "intrinsic-path-heap-regression-commands-v2.json",
        "intrinsic-path-heap-v1-source-edit-race.json",
        "intrinsic-path-heap-large-qualification-commands-v1.json",
        "extended-convex-native-model-commands-v2.json",
        "intrinsic-path-heap-edge-mutation-command-v1.json",
        "run-intrinsic-path-heap-edge-mutation-v1.sh",
    ]
    extras.update(p / "meta" / name for name in names)
    summary = {
        "scope": __doc__,
        "runs": records,
        "parents": parents,
        "CPU_and_CUDA_source_closed_regression_cases": 24,
        "large_CPU_qualification": large,
        "intentional_edge_mutation_detected_by_original_profile_assertion": True,
        "excluded_source_race": "GPUv1 numerical program returns0 with12 passing tests, but source closure fails and scheduler returns1. It is retained solely as excluded evidence. CPU/CUDAv2 independently pass all12 cases with source closure.",
        "pending_not_accepted": "Large CUDA qualification719911, four actual native models719912/914/916/918 and fresh independent auditors719913/915/917/919 remain scheduled work.",
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.update([meta / "summary.json", meta / "jobs.txt", Path(__file__)])
    external = Path(__file__).resolve()
    copy = meta / external.name
    copy.write_bytes(external.read_bytes())
    extras.remove(Path(__file__))
    extras.add(copy)
    assert all(file.stat().st_size < 50_000_000 for file in extras)
    destination = args.destination_root / label
    publish_bounded(args.runtime_root, destination, runs, sorted(extras), summary, __doc__, p / "publication-staging")
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
