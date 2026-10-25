"""Package compact native-candidate receipts without copying runtime arrays or traces."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .artifacts import file_id, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    assert Path(summary["run"]).resolve() == args.run.resolve()
    metadata = args.output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    artifacts = {str(args.summary): file_id(args.summary)}
    outcomes, source_pins = [], []
    cancellations = args.run / "meta" / "cancelled-groups.json"
    cancelled = (
        {(row["campaign"], row["group"]): row for row in json.loads(cancellations.read_text())}
        if cancellations.is_file()
        else {}
    )
    for campaign in summary["campaigns"]:
        name = campaign["name"]
        source = args.run / name
        destination = args.output / "receipts" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("tasks.json", "submission.json"):
            shutil.copy2(source / filename, destination / filename)
            artifacts[str(source / filename)] = file_id(source / filename)
        for path in sorted(source.glob("*source*.json")):
            artifact = file_id(path)
            artifacts[str(path)] = artifact
            if path.name == "analysis-source.json":
                continue
            manifest = json.loads(path.read_text())
            if "revision" in manifest:
                source_pins.append(
                    dict(
                        campaign=name,
                        source=manifest["source"],
                        revision=manifest["revision"],
                        patch_sha256=manifest["patch_sha256"],
                        inventory=artifact,
                    )
                )
        analysis = source / "analysis-files.json"
        if analysis.is_file():
            artifacts[str(analysis)] = file_id(analysis)
        for group_index, group in enumerate(json.loads((source / "tasks.json").read_text())):
            for task in group:
                receipt = source / f"group{group_index}" / f"{task['name']}.returncode.txt"
                if not receipt.is_file() and len(group) == 1:
                    receipt = source / f"group{group_index}" / "returncode.txt"
                code = int(receipt.read_text()) if receipt.is_file() else None
                cancellation = cancelled.get((name, group_index)) if code is None else None
                outcome = "passed" if code == 0 else "failed" if code is not None else "missing_receipt"
                if cancellation is not None:
                    outcome = "cancelled"
                outcomes.append(
                    dict(
                        campaign=name,
                        group=group_index,
                        name=task["name"],
                        module=task["module"],
                        returncode=code,
                        outcome=outcome,
                        receipt=file_id(receipt) if receipt.is_file() else None,
                        cancellation=cancellation,
                    )
                )
    for section in ("benchmarks", "numerics", "memory", "profiles", "traces"):
        write_json(metadata / f"{section}.json", summary[section])
        for row in summary[section]:
            for key in ("source", "benchmark"):
                artifact = row.get(key)
                if isinstance(artifact, dict) and "sha256" in artifact:
                    artifacts[artifact["path"]] = artifact
    for path in sorted((args.run / "meta").glob("*.json")):
        if (
            "preservation" in path.name
            or path.name == "start-state.json"
            or "stop" in path.name
            or "cancel" in path.name
            or "omission" in path.name
        ):
            shutil.copy2(path, metadata / path.name)
            artifacts[str(path)] = file_id(path)
    write_json(metadata / "source-pins.json", source_pins)
    write_json(metadata / "task-outcomes.json", outcomes)
    write_json(metadata / "artifacts.json", list(artifacts.values()))
    counts = dict(Counter(row["outcome"] for row in outcomes))
    write_json(
        metadata / "collection.json",
        dict(
            run=str(args.run),
            summary=file_id(args.summary),
            campaigns=len(summary["campaigns"]),
            tasks=len(outcomes),
            task_outcomes=counts,
            note=(
                "Task exit success is distinct from independent numerical qualification. Missing receipts require "
                "the retained cancellation journal or completion of outstanding work; they are never counted as passes."
            ),
        ),
    )
    files = [
        {**file_id(path), "path": str(path.relative_to(args.output))}
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "packet-manifest.json"
    ]
    write_json(metadata / "packet-manifest.json", files)
    print(json.dumps(dict(campaigns=len(summary["campaigns"]), tasks=len(outcomes), outcomes=counts, files=len(files))))


if __name__ == "__main__":
    main()
