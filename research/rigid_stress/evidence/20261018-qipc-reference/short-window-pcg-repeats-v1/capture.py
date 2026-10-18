"""Losslessly retain the fresh-process PCG1e-2 repeats; stdlib copying/hashing only."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--prior-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    copies, reconstructions = [], []

    def copy(source, target, compressed=False):
        destination = args.output / target
        assert not destination.exists(), destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if compressed:
            destination.write_bytes(gzip.compress(raw, mtime=0))
            assert gzip.decompress(destination.read_bytes()) == raw
        else:
            shutil.copy2(source, destination)
            assert destination.read_bytes() == raw
        copies.append({"source": identity(source), "selected": target, "lossless_gzip": compressed})

    profile_name = "short-contact-pcg-repeat-v1"
    audit_name = "short-contact-pcg-repeat-audit-v1"
    for campaign in (profile_name, audit_name):
        root = args.fallback / campaign
        assert (root / "group0/returncode.txt").read_text().strip() == "0"
        for name in ("tasks.json", "submission.json", "source.json", "source.patch", "analysis-files.json"):
            compressed = name in ("source.json", "analysis-files.json")
            copy(root / name, f"receipts/{campaign}/{name}" + (".gz" if compressed else ""), compressed)
        for path in sorted((root / "group0").glob("*returncode*")):
            copy(path, f"receipts/{campaign}/group0/{path.name}")
        for path in sorted((root / "group0").glob("*.log")):
            copy(path, f"logs/{campaign}/{path.name}.gz", True)
        rows = json.loads((root / "analysis-files.json").read_text())
        for row in rows:
            source = root / "analysis-source" / row["path"]
            assert identity(source)["sha256"] == row["sha256"]
            base = args.prior_packet / "source" / row["path"]
            if not base.exists() or source.read_bytes() != base.read_bytes():
                copy(source, f"source-variants/{campaign}/{row['path']}")
        reconstructions.append({"campaign": campaign, "verified_source_files": len(rows)})
    profile = args.fallback / profile_name / "group0"
    for path in sorted(profile.glob("*/result.json")):
        folder = path.parent
        assert json.loads(path.read_text())["completed_requested_history"]
        for name in ("result.json", "inputs.json", "rest.npz"):
            copy(folder / name, f"profiles/{folder.name}/{name}")
        for name in ("steps.jsonl", "timings.json"):
            copy(folder / name, f"profiles/{folder.name}/{name}.gz", True)
        for frame in sorted((folder / "frames").glob("*.npz")):
            copy(frame, f"profiles/{folder.name}/frames/{frame.name}")
    for path in sorted((args.fallback / audit_name / "group0").glob("*/result.json")):
        copy(path, f"analyses/{path.parent.name}/result.json")
        for array in sorted(path.parent.glob("*.npz")):
            copy(array, f"analyses/{path.parent.name}/{array.name}")
    plan = args.fallback / "meta/short-contact-pcg-repeat-v1-plan.json"
    copy(plan, "plan.json")
    for row in json.loads(plan.read_text())["code_evidence"]:
        source = Path(row["path"])
        assert identity(source)["sha256"] == row["sha256"]
        copy(source, f"code-evidence/{source.name}")
    frames = list((args.output / "profiles").glob("*/frames/*.npz"))
    assert len(frames) == 9, len(frames)
    (args.output / "selection-provenance.json").write_text(
        json.dumps(
            {
                "prior_packet_manifest": identity(args.prior_packet / "manifest.json"),
                "source_reconstruction": reconstructions,
                "copies": copies,
                "scope": "Exact byte copies or decoded-equal gzip. All9 full-spatial fields from3 completed original-state short runs. Executed source is prior packet source/ plus this packet's campaign-specific source-variants/ overlay, verified against every analysis-file hash. Shared stricter reference states, layout/surface and target binding remain in the prior packet. No scientific calculation runs in this capture helper. Short-window numerical evidence is not full-grasp/physical-reference qualification.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
