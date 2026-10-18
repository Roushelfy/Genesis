"""Retain sparse-grasp and contact-window evidence using stdlib byte operations only."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


CAMPAIGNS = (
    "capped-contact-events-v1",
    "capped-contact-window-v1",
    "capped-contact-window-audit-v1",
    "capped-window-pose-audit-v1",
    "capped-partition-loose-v1",
    "capped-partition-loose-audit-v1",
    "contact-window-protocol-regression-v1",
    "contact-window-protocol-regression-v2",
    "native-sampling-window-v1",
    "native-sampling-window-audit-v1",
    "native-sampling-window-audit-v2",
    "full-grasp-keyframe-plan-v1",
    "native-prescribed-sparse-v1",
    "shell-prescribed-sparse-v1",
    "reference-prescribed-sparse-v1",
    "reference-prescribed-sparse-pcg1e3-v1",
    "prescribed-sparse-fields-v1",
    "prescribed-sparse-equivalence-v1",
    "shell-observer-control-v1",
    "shell-observer-control-audit-v1",
    "prescribed-sparse-fields-nopayload-v1",
    "capped-rigid-binding-v1",
)


def identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--prior-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    copies, sources, blobs = [], [], set()

    def copy(source, target, compressed=False):
        destination = args.output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if destination.exists():
            assert args.extend, destination
            stored = destination.read_bytes()
            assert (gzip.decompress(stored) if compressed else stored) == raw, destination
        elif compressed:
            destination.write_bytes(gzip.compress(raw, mtime=0))
            assert gzip.decompress(destination.read_bytes()) == raw
        else:
            shutil.copy2(source, destination)
            assert destination.read_bytes() == raw
        copies.append({"source": identity(source), "selected": target, "lossless_gzip": compressed})

    for name in CAMPAIGNS:
        campaign = args.fallback / name
        groups = sorted(campaign.glob("group*"))
        assert groups and all((group / "returncode.txt").is_file() for group in groups), name
        for path in sorted(campaign.glob("*")):
            if not path.is_file():
                continue
            compressed = path.name in ("source.json", "analysis-files.json", "source.patch")
            copy(path, f"campaigns/{name}/{path.name}" + (".gz" if compressed else ""), compressed)
        for path in sorted(campaign.glob("logs/*")):
            if path.is_file():
                copy(path, f"campaigns/{name}/logs/{path.name}.gz", True)
        for group in groups:
            for path in sorted(group.rglob("*")):
                if not path.is_file():
                    continue
                assert not path.name.startswith("common-step"), "Retire derived alignment exports first"
                compressed = path.suffix in (".jsonl", ".log") or path.name in ("timings.json", "steps.json")
                target = f"campaigns/{name}/{path.relative_to(campaign)}" + (".gz" if compressed else "")
                copy(path, target, compressed)
        rows = json.loads((campaign / "analysis-files.json").read_text())
        reconstructed = []
        for row in rows:
            path = campaign / "analysis-source" / row["path"]
            assert identity(path)["sha256"] == row["sha256"], path
            prior = args.prior_packet / "source" / row["path"]
            if prior.is_file() and identity(prior)["sha256"] == row["sha256"]:
                location = f"prior/source/{row['path']}"
            else:
                location = f"source-blobs/{row['sha256']}.py"
                if location not in blobs:
                    copy(path, location)
                    blobs.add(location)
            reconstructed.append({"path": row["path"], "sha256": row["sha256"], "stored": location})
        sources.append({"campaign": name, "files": reconstructed})
        for path in sorted((args.fallback / "meta").glob(name + "*")):
            if path.is_file():
                copy(path, f"plans/{path.name}")

    # Shared native/QIPC layout and common-domain inputs are immutable data-side
    # assets. Keep their exact bytes with the selected original reference fields.
    data = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference")
    for relative in (
        "surface-metric-v1/group0/metric/columns.npz",
        "surface-metric-v1/group0/metric/surfaces.npz",
        "roundoff-prescribed-v1/group0/snh/rest.npz",
        "roundoff-prescribed-v1/group0/snh/result.json",
        "reaction-layout-v1/group0/layout/layout.npz",
        "controls-v1/group0/inputs/base1-surface2-wall4.npz",
    ):
        copy(data / relative, "shared/" + relative)
    reference = data / "roundoff-prescribed-v1/group0/snh/frames"
    for step in (1329, 2769, 4019, 4379, 5519):
        copy(reference / f"step{step:06d}.npz", f"shared/reference-frames/step{step:06d}.npz")

    write = args.output / "selection-provenance.json"
    assert args.extend or not write.exists()
    write.write_text(
        json.dumps(
            {
                "prior_packet_manifest": identity(args.prior_packet / "manifest.json"),
                "source_reconstruction": sources,
                "copies": copies,
                "scope": "Exact byte copies or decoded-equal gzip only. Captured analysis sources reconstruct from the frozen short-window-v1 source plus content-addressed source-blobs. Original backend revision/patch/inventory stays in every campaign receipt. Scientific calculations ran only in captured Slurm/container workers. Failed physics, failed equivalence and corrected analysis-only failures are retained. Reproducible derived common-column NPZs were retired with exact hashes under the4GB runtime bound; original complete spatial simulator observations remain. Sparse keyframes do not certify whole-history peaks, common-reference physical accuracy or primary throughput.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
