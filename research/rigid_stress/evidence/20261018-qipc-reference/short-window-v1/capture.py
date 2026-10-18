"""Copy executed short-window evidence losslessly; stdlib metadata work only."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.fallback, args.output
    output.mkdir(parents=True, exist_ok=True)
    copies = []

    def copy(source, target, compressed=False):
        destination = output / target
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

    campaigns = ("short-contact-profile-v1", "short-contact-followup-v1", "short-contact-minimal-timing-v1")
    for campaign in campaigns:
        prefix = f"profiles/{campaign}"
        copy(root / campaign / "tasks.json", f"{prefix}/tasks.json")
        for path in sorted((root / campaign).glob("group*/*/result.json")):
            folder = path.parent
            assert json.loads(path.read_text())["completed_requested_history"]
            relative = folder.relative_to(root / campaign)
            for name in ("result.json", "inputs.json", "rest.npz"):
                copy(folder / name, f"{prefix}/{relative}/{name}")
            for name in ("steps.jsonl", "timings.json"):
                copy(folder / name, f"{prefix}/{relative}/{name}.gz", compressed=True)
            for frame in sorted((folder / "frames").glob("*.npz")):
                copy(frame, f"{prefix}/{relative}/frames/{frame.name}")
    for campaign in ("short-contact-profile-audit-v1", "short-contact-combined-audit-v1"):
        assert (root / campaign / "group0/returncode.txt").read_text().strip() == "0"
        for name in ("keyframe-fields", "keyframe-reactions", "window-audit"):
            assert (root / campaign / "group0" / name / "result.json").is_file()
        for folder in sorted((root / campaign).glob("group*/*/result.json")):
            copy(folder, f"analyses/{campaign}/{folder.parent.name}/result.json")
            for path in folder.parent.glob("reactions.npz"):
                copy(path, f"analyses/{campaign}/{folder.parent.name}/{path.name}")
    fields = json.loads((root / "short-contact-combined-audit-v1/group0/keyframe-fields/result.json").read_text())
    reactions = json.loads((root / "short-contact-combined-audit-v1/group0/keyframe-reactions/result.json").read_text())
    copy(Path(fields["surface_mesh"]["path"]), "inputs/surface-mesh.npz")
    copy(Path(reactions["layout"]["path"]), "inputs/rigid-layout.npz")
    pose = root / "short-contact-pose-tracking-v1"
    assert (pose / "group0/returncode.txt").read_text().strip() == "0"
    for name in ("result.json", "inputs.json", "prescribed-targets.npz"):
        copy(pose / "group0/target-binding" / name, f"target-binding/{name}")
    copy(pose / "group0/pose-tracking/result.json", "analyses/prescribed-target-tracking.json")
    pose2 = root / "short-contact-pose-tracking-v2"
    assert (pose2 / "group0/returncode.txt").read_text().strip() == "0"
    copy(pose2 / "group0/pose-tracking/result.json", "analyses/prescribed-target-and-egg-pose.json")
    for name in (
        "short-contact-profile-v1-plan.json", "short-contact-followup-v1-plan.json",
        "short-contact-minimal-timing-v1-plan.json",
    ):
        copy(root / "meta" / name, f"plans/{name}")
    copy(root / "short-contact-minimal-timing-v1/kind-label-correction.json", "corrections/kind-label.json")

    baseline = root / "short-contact-pose-tracking-v2/analysis-source"
    for path in sorted(baseline.rglob("*")):
        if path.is_file():
            copy(path, f"source/{path.relative_to(baseline)}")
    for campaign in (*campaigns, "short-contact-profile-audit-v1", "short-contact-combined-audit-v1", "short-contact-pose-tracking-v1", "short-contact-pose-tracking-v2"):
        for name in ("tasks.json", "submission.json", "source.json", "source.patch", "analysis-files.json"):
            compressed = name in ("source.json", "analysis-files.json")
            copy(root / campaign / name, f"receipts/{campaign}/{name}" + (".gz" if compressed else ""), compressed)
        for path in sorted((root / campaign).glob("group*/*returncode*")):
            copy(path, f"receipts/{campaign}/{path.parent.name}/{path.name}")
        for path in sorted((root / campaign / "analysis-source").rglob("*.py")):
            relative = path.relative_to(root / campaign / "analysis-source")
            current = baseline / relative
            if not current.exists() or current.read_bytes() != path.read_bytes():
                copy(path, f"source-variants/{campaign}/{relative}")
    (output / "selection-provenance.json").write_text(json.dumps({
        "copies": copies,
        "scope": "Exact byte copies or lossless gzip with decoded equality verified. All33 full-spatial keyframe states from11 completed short runs are included. Timing and numerical analyses ran in captured Slurm/container workers; this script calculates no scientific metrics. Physics reruns still require pinned external sources/builds/motion/assets recorded in the original inputs. This is a short-window checkpoint, not full-goal closure.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
