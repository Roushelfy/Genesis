"""Retain the executed native sampling checks and their exact source overlays."""

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
    copies = []

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

    positive = args.fallback / "native-manual-sampling-v1"
    baseline_tests = positive / "group0/cpu/test-source"
    for path in sorted(baseline_tests.rglob("*")):
        if path.is_file():
            copy(path, f"test-source/{path.relative_to(baseline_tests)}")
    expected_sources = json.loads((positive / "group0/cpu/result.json").read_text())["test_sources"]
    for campaign, label, backends in (
        ("native-manual-sampling-v1", "positive", ("cpu", "cuda")),
        ("native-manual-negative-v1", "negative", ("cuda",)),
    ):
        root = args.fallback / campaign
        assert (root / "group0/returncode.txt").read_text().strip() == "0"
        for name in ("tasks.json", "submission.json", "source.json", "source.patch", "analysis-files.json"):
            compressed = name in ("source.json", "analysis-files.json")
            copy(root / name, f"receipts/{campaign}/{name}" + (".gz" if compressed else ""), compressed)
        for path in sorted((root / "group0").glob("*returncode*")):
            copy(path, f"receipts/{campaign}/{path.name}")
        for backend in backends:
            folder = root / "group0" / backend
            result = json.loads((folder / "result.json").read_text())
            assert result["test_sources"] == expected_sources
            assert result["source_unchanged"]
            if label == "negative":
                assert result["expected_snapshot_failure_observed"] and result["returncode"] == 1
                copy(folder / "negative-control.patch", f"{label}/{backend}/negative-control.patch")
            else:
                assert result["returncode"] == 0
            for name in ("result.json", "report.xml"):
                copy(folder / name, f"{label}/{backend}/{name}")
            copy(folder / "tests.log", f"{label}/{backend}/tests.log.gz", True)
        for row in json.loads((root / "analysis-files.json").read_text()):
            source = root / "analysis-source" / row["path"]
            assert identity(source)["sha256"] == row["sha256"]
            base = args.prior_packet / "source" / row["path"]
            if not base.exists() or source.read_bytes() != base.read_bytes():
                copy(source, f"source-variants/{campaign}/{row['path']}")
    copy(args.fallback / "meta/native-manual-sampling-v1-plan.json", "plan.json")
    (args.output / "selection-provenance.json").write_text(
        json.dumps(
            {
                "copies": copies,
                "prior_packet_manifest": identity(args.prior_packet / "manifest.json"),
                "scope": "Exact copies or decoded-equal gzip. Positive CPU4/CUDA5 cases, expected negative-control failure, exact executed tests and analysis overlays. Engine revision/full inventory/patch remain in receipts. Scientific pytest execution was in Slurm/container workers. This capture performs only metadata/copy/hash work. Application profiling and full-goal qualification remain open.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
