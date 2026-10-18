"""Copy compact evidence and deduplicate exact executed analysis snapshots (stdlib only)."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(payload)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert digest(source) == digest(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.runs.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archives = {}
    source_index = {}
    excluded = []
    for campaign in sorted(p for p in root.iterdir() if p.is_dir()):
        if (campaign / "analysis-files.json").exists():
            entries = json.loads((campaign / "analysis-files.json").read_text())
            source_index[campaign.name] = entries
            for entry in entries:
                path = campaign / "analysis-source" / entry["path"]
                assert digest(path) == {k: entry[k] for k in ("bytes", "sha256")}
                archives[f"analysis-blobs/{entry['sha256']}"] = path
        elif campaign.name == "tools-v1":
            # The first two manual Slurm submissions predate submit.py, but
            # already used this separate frozen Python tree and meta launchers.
            entries = []
            for path in sorted(campaign.rglob("*.py")):
                entry = {"path": str(path.relative_to(campaign)), **digest(path)}
                entries.append(entry)
                archives[f"analysis-blobs/{entry['sha256']}"] = path
            source_index[campaign.name] = entries
        for path in campaign.glob("*"):
            if path.is_file() and path.suffix in (".json", ".patch", ".sh"):
                archives[f"campaigns/{campaign.name}/{path.name}"] = path
        for pattern in ("logs/*.log", "group*/*.log", "group*/returncode.txt"):
            for path in campaign.glob(pattern):
                archives[f"campaigns/{campaign.name}/{path.relative_to(campaign)}"] = path
        for pattern in (
            "group*/*/result.json",
            "group*/*/summary.json",
            "group*/*/surface_geometry_audit.json",
            "group*/*-summary.json",
        ):
            for path in campaign.glob(pattern):
                copy(path, out / "receipts" / path.relative_to(root))
    for path in (root / "meta").iterdir():
        if path.is_file() and path.suffix in (".json", ".sh", ".log"):
            archives[f"meta/{path.name}"] = path
    for path in (root / "hardware-nodes-v1").glob("*.json"):
        archives[f"hardware-nodes-v1/{path.name}"] = path
    with tarfile.open(out / "reproduction.tar.gz", "w:gz") as archive:
        payload = (json.dumps(source_index, indent=2) + "\n").encode()
        info = tarfile.TarInfo("analysis-index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for name, source in sorted(archives.items()):
            archive.add(source, arcname=name, recursive=False)
    for path in (root / "collect-v3/group0/summary").rglob("*"):
        if path.is_file():
            copy(path, out / "derived" / path.relative_to(root / "collect-v3/group0/summary"))
    for path in (root / "figures-v2/group0/figures").glob("*"):
        if path.is_file():
            copy(path, out / "figures" / path.name)
    for path in (root / "figures-v2/group0/export-qa").glob("*"):
        if path.is_file() and not path.name.endswith("-pdf-450.png"):
            copy(path, out / "figures/qa" / path.name)
    patterns = [
        "prepare-inputs-v2/group0/matched-inputs/*",
        "prepare-loads-v2/group0/inputs/*",
        "prepare-sphere-v1/group0/inputs/*",
        "static-inputs-v1/group0/inputs/*",
        "peak-traces-v2/group*/*/trajectory-repeat0.npz",
        "controls-v1/group0/*/fields.npz",
        "contact-ledger-v1/group0/*/trajectory.npz",
        "contact-ledger-v1/group0/*/sheet.obj",
        "diagnostic-controls-v2/group0/drop-n2/trajectory.npz",
        "diagnostic-controls-v2/group0/drop-n2/sheet.obj",
        "static-cost-v1/group*/*/fields.npz",
        "analytic-sphere-v1/group*/*/fields.npz",
        "shell-prescribed-v2/group*/*/fields.npz",
        "analyse-prescribed-v1/group0/audit/*.npz",
        "native-elastic-v2/group0/analytic-plate/*.npz",
    ]
    selected = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                selected.add(path)
                copy(path, out / "raw" / path.relative_to(root))
    for path in root.glob("*/group*/*/*"):
        if path.is_file() and path.suffix in (".npz", ".json") and path not in selected:
            excluded.append({"path": str(path.relative_to(root)), **digest(path)})
    (out / "data-artifacts.json").write_text(
        json.dumps({"root": str(root), "retained_on_data": excluded}, indent=2) + "\n"
    )
    (out / "source-closure.json").write_text(
        json.dumps(
            {
                "campaigns": len(source_index),
                "unique_blob_count": sum(k.startswith("analysis-blobs/") for k in archives),
                "archive": digest(out / "reproduction.tar.gz"),
            },
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
