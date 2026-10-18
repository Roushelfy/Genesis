"""Publish existing Slurm outputs and exact sources with lossless encoding; no scientific analysis."""

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
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.fallback, args.output
    output.mkdir(parents=True, exist_ok=True)
    selections = []

    def write(relative, value):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def copy(source, relative, compressed=False):
        target = output / relative
        assert not target.exists(), target
        target.parent.mkdir(parents=True, exist_ok=True)
        original = identity(source)
        if compressed:
            raw = source.read_bytes()
            target.write_bytes(gzip.compress(raw, mtime=0))
            assert gzip.decompress(target.read_bytes()) == raw
        else:
            shutil.copy2(source, target)
            assert identity(target)["sha256"] == original["sha256"]
        selections.append({"source": original, "selected": relative, "lossless_gzip": compressed})

    def reduced(value):
        if isinstance(value, list):
            return [reduced(item) for item in value]
        if isinstance(value, dict):
            return {
                key: reduced(item)
                for key, item in value.items()
                if key not in ("history", "constitutive_at_identical_deformation")
            }
        return value

    results = {
        "material-query-controls": "metric-query-controls-v2/group0/query-controls",
        "observation-grid-controls": "adaptive-linear-budget-audit-v2/group5/grid-admission",
        "ET-admission-controls": "ET-history-admission-v3/group0/journal-controls",
        "historical-ET-audit": "ET-history-admission-v3/group1/histories",
        "adaptive-query-recovered-pair": "adaptive-query-validated-v1/group0/adaptive-prefix-pair",
        "adaptive-budget-pair": "adaptive-linear-budget-audit-v2/group2/linear-budget-contrast",
        "adaptive-budget-vs-surface2": "adaptive-linear-budget-audit-v2/group1/budget-vs-surface2",
        "adaptive-material-range": "adaptive-linear-budget-audit-v2/group3/candidate-material-range",
        "adaptive-budget-reactions": "adaptive-linear-budget-audit-v2/group4/budget-vs-surface2-reactions",
        "adaptive-ET-stopped-run": "adaptive-ET-forensics-v3/group0/adaptive-edge0.0005-wall4",
        "ET-saved-geometry": "ET-and-case7-strict-v1/group0/ET-geometry",
        "ET-stop-and-strict-history": "ET-and-case7-strict-v1/group3/histories",
        "case7-stock": "capped-case7-fields-and-cost-qa-v1/group0/case7-stock",
        "case7-volume-surface2": "capped-case7-fields-and-cost-qa-v1/group1/case7-volume-surface2",
        "case2-stock": "completed-capped-case2-and-wall2-v1/group0/case2-stock",
        "case2-volume-surface2": "completed-capped-case2-and-wall2-v1/group1/case2-volume-surface2",
        "case2-and-wall2-history": "completed-capped-case2-and-wall2-v1/group5/histories",
        "case7-wall2-fields": "completed-capped-case2-and-wall2-v1/group3/case7-wall2-fields",
        "case7-wall2-reactions": "completed-capped-case2-and-wall2-v1/group4/case7-wall2-reactions",
        "case7-strict-fields": "ET-and-case7-strict-v1/group1/case7-strict-fields",
        "case7-strict-reactions": "ET-and-case7-strict-v1/group2/case7-strict-reactions",
        "case2-actual-poses": "completed-capped-case2-and-wall2-v1/group2/case2-actual-poses",
        "case7-actual-poses": "capped-rigid-pose-audit-v1/group0/case7-actual-poses",
        "stock-Shell-B1-full-cost": "shell-b1-full-error-cost-v1/group0/stock-b1-full",
        "volume-surface2-Shell-B1-full-cost": "shell-b1-full-error-cost-v1/group1/volume-surface2-b1-full",
    }
    for case, group in ((0, 0), (2, 1), (7, 2)):
        results[f"unloaded-bank-case{case}"] = f"robot-schedule-comparison-v1/group{group}/case{case}"
    for name, relative in results.items():
        source = root / relative / "result.json"
        copy(source, f"raw-results/{name}.json.gz", compressed=True)
        write(f"summaries/{name}.json", reduced(json.loads(source.read_text())))

    failed = root / "adaptive-ET-forensics-v3/group0"
    copy(failed / "adaptive-edge0.0005-wall4.log", "ET-failure/native-run.log")
    copy(failed / "adaptive-edge0.0005-wall4/contact-failure-capture.json", "ET-failure/capture.json")
    copy(root / "ET-and-case7-strict-v1/group0/ET-geometry/pair0.npz", "ET-failure/reported-pair.npz")
    copy(args.native_source / "qipc/_src/native/solver/edge_triangle_intersect.h", "ET-failure/edge_triangle_intersect.h")
    for filename in ("result.json", "inputs.json", "robot-inputs.json"):
        copy(root / "capped-fresh-coarse-v1/group1/case2" / filename, f"case2-reference/{filename}")
    for relative in (
        "capped-case7-uncertainty-v1/group0/wall2",
        "capped-case7-uncertainty-v1/group3/strict",
    ):
        for filename in ("result.json", "inputs.json"):
            copy(root / relative / filename, f"case7-controls/{Path(relative).name}/{filename}")

    figure = root / "cost-scope-figure-v1/group0/cost-figure"
    for path in sorted(figure.iterdir()):
        if path.is_file():
            copy(path, f"figure/{path.name}")
    qa = root / "capped-case7-fields-and-cost-qa-v1/group2/export-qa"
    for filename in ("result.json", "output-memory-activation-profiles-pdf-preview.png"):
        copy(qa / filename, f"figure/export-qa/{filename}")

    baseline = root / "ET-and-case7-strict-v1/analysis-source"
    for path in sorted(baseline.rglob("*")):
        if path.is_file():
            copy(path, f"source/{path.relative_to(baseline)}")
    campaigns = (
        "adaptive-query-validated-v1", "metric-query-controls-v2", "adaptive-prescribed-linear-budget-v2",
        "adaptive-linear-budget-audit-v2", "ET-history-admission-v3", "adaptive-ET-forensics-v3",
        "ET-geometry-audit-v1", "ET-geometry-audit-v2", "ET-and-case7-strict-v1",
        "capped-case7-fields-and-cost-qa-v1", "completed-capped-case2-and-wall2-v1",
        "capped-case7-reaction-layout-v1", "robot-schedule-native-v1", "robot-schedule-qipc-v1",
        "robot-schedule-comparison-v1", "capped-rigid-binding-v1", "capped-rigid-pose-audit-v1",
        "shell-b1-full-error-cost-v1", "cost-scope-figure-v1", "capped-case7-uncertainty-v1",
        "capped-fresh-coarse-v1",
    )
    for campaign in campaigns:
        for filename in ("tasks.json", "submission.json", "source.json", "source.patch", "analysis-files.json"):
            compressed = filename in ("source.json", "analysis-files.json")
            target = f"receipts/{campaign}/{filename}" + (".gz" if compressed else "")
            copy(root / campaign / filename, target, compressed=compressed)
        for path in sorted((root / campaign).glob("group*/*returncode*")):
            copy(path, f"receipts/{campaign}/{path.parent.name}/{path.name}")
        for path in sorted((root / campaign / "analysis-source").rglob("*.py")):
            relative = path.relative_to(root / campaign / "analysis-source")
            base = baseline / relative
            if not base.exists() or path.read_bytes() != base.read_bytes():
                copy(path, f"source-variants/{campaign}/{relative}")
    write(
        "selection-provenance.json",
        {
            "selections": selections,
            "summary_omitted_keys": ["history", "constitutive_at_identical_deformation"],
            "scope": "Summary JSON selects already computed values. Every full selected result is additionally preserved as exact decoded bytes in lossless gzip. No numerical analysis runs here. Full simulation arrays, native dumps, builds and profiler traces stay at hashed original paths. Live campaign receipts remain provisional; only present terminal task/group receipts are copied. This is an incomplete-goal checkpoint.",
        },
    )


if __name__ == "__main__":
    main()
