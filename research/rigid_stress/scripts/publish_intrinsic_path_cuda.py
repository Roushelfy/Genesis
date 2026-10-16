"""Publish the completed large CUDA graph qualification as a supplement to the original capacity checkpoint."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded
from research.rigid_stress.scripts.verify_nonconvex_publication import digest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    label = "native-intrinsic-large-cuda-v1"
    run = p / "intrinsic-path-heap-large-cuda-v1"
    meta = p / f"{label}-publication"
    meta.mkdir(parents=True, exist_ok=False)
    account = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", "719911", "--format=JobIDRaw,State,ExitCode,Elapsed"], text=True
    )
    assert account.split("|")[:3] == ["719911", "COMPLETED", "0:0"]
    assert int((run / "returncode.txt").read_text()) == int((run / "program-returncode.txt").read_text()) == 0
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    for line in (run / "source.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert digest(run / "source" / relative)[0] == expected
    assert "FAILED" not in (run / "snapshot-after-verification.txt").read_text()
    parent = args.destination_root / "native-intrinsic-capacity-v1/manifest.json"
    assert digest(parent)[0] == "9fc23ff51bfd0b6c91c24806c6dd599277140ed31f59f254aecdf855c61fb407"
    parent_proof = verify_manifest(parent, True, {})
    summary = json.loads((run / "results/summary.json").read_text())
    assert summary["port_count"] == 125710 and summary["directed_graph_edges"] == 5074032
    assert summary["exact_original_restoration"] and len(summary["rows"]) == 3
    assert sum(row["all_port_entries_checked"] for row in summary["rows"]) == 5656950
    assert all(row["all_support_membership_matches"] for row in summary["rows"])
    for original, expected in summary["input_sha256"].items():
        assert digest(Path(original))[0] == expected
    for source in Path("genesis/engine/solvers/rigid/stress").rglob("*.py"):
        assert digest(source)[0] == digest(run / "source" / source)[0]
    record = {
        "scope": __doc__,
        "job": 719911,
        "scheduler_state": "COMPLETED",
        "source_and_snapshot_closure": True,
        "parent_manifest_sha256": digest(parent)[0],
        "parent_complete_reverification": parent_proof,
        "large_CUDA_qualification": summary,
        "current_worktree_engine_matches_executed_copy": True,
        "pending_not_accepted": "Actual fine native four-model fields and their separate auditors, remaining physical/model/application/feedback/warmed-cost requirements.",
        "goal_completion": False,
    }
    (meta / "jobs.txt").write_text(account)
    (meta / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    publisher = meta / Path(__file__).name
    shutil.copy2(__file__, publisher)
    extras = [meta / "jobs.txt", meta / "summary.json", publisher, *p.glob("*719911_0.log")]
    destination = args.destination_root / label
    publish_bounded(args.runtime_root, destination, [run], extras, record, __doc__, p / "publication-staging")
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
