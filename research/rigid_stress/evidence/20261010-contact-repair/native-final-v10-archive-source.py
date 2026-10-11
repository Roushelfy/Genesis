"""Publish closed final cases and verify both publication and original hashes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    final = json.loads(args.summary.read_text())
    if args.require_complete and not final["complete"]:
        raise SystemExit("Refusing final publication of incomplete evidence.")
    manifest_path = args.output / "manifest.json"
    previous_bytes = manifest_path.read_bytes()
    known = {entry["runtime_path"] for entry in json.loads(previous_bytes)["artifacts"]}
    names = set()

    def add(path):
        if path.is_file() and str(path) not in known:
            names.add(str(path.relative_to(args.run)))

    for pattern in ("final-native-v8-*/summary.json", "final-native-fullcost-v8-*/summary.json"):
        for path in args.run.glob(pattern):
            summary = json.loads(path.read_text())
            rows = summary["measurements"]
            for row in rows:
                if row["status"] != "completed":
                    continue
                result = Path(row["result"])
                for item in (result, result.with_suffix(".log"), Path(row["memory_samples"])):
                    add(item)
            # Only whole-directory summaries whose expected cases are closed.
            expected = 4 if path.parent.name.startswith("final-native-v8-") else 1
            if len(rows) == expected and all(row["status"] == "completed" for row in rows):
                add(path)
    for pattern in (
        "native-final-episode-*-v8.json",
        "native-final-episode-*-v8-summary.json",
        "native-final-snapshot-profile-*-v8.json",
        "native-final-rollout-profile-*-v8.json",
        "native-indices-*-v1.json",
        "native-indices-*-v1-summary.json",
        "postgraph-*-v1.json",
        "postgraph-*-v1-summary.json",
    ):
        for path in args.run.glob(pattern):
            if path.name.endswith((".postgraph-variant.json", ".indices-variant.json")):
                continue
            # The variant footer also proves the dispatch-count assertion
            # passed after the benchmark/oracle wrote its main result.
            if (
                path.name.startswith("postgraph-")
                and not path.name.endswith("-summary.json")
                and not path.with_suffix(".postgraph-variant.json").is_file()
            ):
                continue
            add(path)
            stem = path.name.removesuffix(".json")
            for suffix in (
                ".log",
                ".postgraph.py",
                ".postgraph-variant.json",
                ".controller.py",
                ".indices-variant.json",
            ):
                add(args.run / (stem + suffix))
    for name in (
        "run_native_final_scopes_v8.sh",
        "run_native_final_fullcost_v8.sh",
        "run_native_final_profile_v8.sh",
        "run_native_final_episode_v8.sh",
        "run_native_final_summary_v8.sh",
        "run_native_indices_v1.sh",
        "run_native_postgraph_v1.sh",
        "native-rollout-profile-frozen-v8.py",
        "archive_native_checkpoint_v9.py",
        "native-checkpoint-v9-selection.json",
        "native-checkpoint-v9-verification.json",
        "native-checkpoint-v9-archive.log",
        "native-fullcost-startup-failure-705228.txt",
    ):
        add(args.run / name)
    add(args.summary)
    add(args.summary.with_suffix(".log"))
    selection_path = args.run / f"native-final-{args.checkpoint}-selection.json"
    assert not selection_path.exists(), "Checkpoint names must identify immutable selections."
    selection_path.write_text(
        json.dumps(
            {
                "names": sorted(names),
                "previous_manifest_sha256": hashlib.sha256(previous_bytes).hexdigest(),
                "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "final_summary": str(args.summary),
                "final_complete": final["complete"],
                "source_revision": args.revision,
                "note": "Only closed case JSON/log/memory files are selected. Active whole-directory summaries are omitted. Source labels and exact prototype/production hashes are retained per artifact.",
            },
            indent=2,
        )
        + "\n"
    )
    names.add(selection_path.name)
    frozen_selector = args.run / f"native-final-{args.checkpoint}-archive-source.py"
    frozen_selector.write_bytes(Path(__file__).read_bytes())
    names.add(frozen_selector.name)
    subprocess.run(
        [
            sys.executable,
            "research/rigid_stress/scripts/archive_completed_evidence.py",
            "--run",
            str(args.run),
            "--output",
            str(args.output),
            "--revision",
            args.revision,
            *sorted(names),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "research/rigid_stress/scripts/verify_contact_evidence.py",
            "--manifest",
            str(manifest_path),
            "--originals",
            "--output",
            str(args.run / f"native-final-{args.checkpoint}-verification.json"),
        ],
        check=True,
    )
    assert all(path.stat().st_size < 100_000_000 for path in args.output.iterdir() if path.is_file())
    print("Closed final cases archived and all hashes verified.", flush=True)


if __name__ == "__main__":
    main()
