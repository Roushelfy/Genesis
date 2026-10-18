"""Select existing Slurm-derived metadata and byte-copy inspected outputs; no numerical analysis."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.fallback, args.output
    output.mkdir(parents=True, exist_ok=True)
    selected_sources, byte_copies = [], []

    def write(relative, value):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def read(relative):
        path = root / relative
        selected_sources.append(identity(path))
        return json.loads(path.read_text())

    def copy(source, relative):
        destination = output / relative
        assert not destination.exists(), destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        original, copied = identity(source), identity(destination)
        assert original["sha256"] == copied["sha256"] and original["bytes"] == copied["bytes"]
        byte_copies.append({"source": original, "destination": relative})

    def reduced_pairs(value):
        result = {key: item for key, item in value.items() if key != "constitutive_at_identical_deformation"}
        result["comparisons"] = [
            {key: item for key, item in row.items() if key != "history"} for row in value["comparisons"]
        ]
        return result

    write(
        "observer-and-admission-controls.json",
        {
            name: read(f"validation-after-storage-v1/group{group}/{name}/result.json")
            for group, name in ((0, "journal-controls"), (1, "shell-independent-max"), (2, "shell-max-application"))
        },
    )
    write("reference-history-audit.json", read("reference-diagnostics-v2/group0/histories/result.json"))
    write("reference-reactions.json", reduced_pairs(read("validation-after-storage-v1/group4/reactions/result.json")))
    write(
        "fixed-wall2-contrast.json",
        reduced_pairs(read("validation-after-storage-v1/group7/fixed-wall2-surface-pair/result.json")),
    )
    write("friction-contrast.json", reduced_pairs(read("expanded-outcomes-v1/group0/friction-pair/result.json")))
    surface2 = read("validation-after-storage-v1/group6/surface2-comparison/result.json")
    write(
        "surface2-common-metrics.json",
        {key: value for key, value in surface2.items() if key not in ("history", "snapshots")},
    )
    write("shell-static-replay.json", read("shell-resolved-replay-audit-v2/group0/replay-audit/result.json"))
    write("adaptive-inputs.json", read("adaptive-input-v2/group0/inputs/result.json"))
    write("adaptive-pilot-history.json", read("adaptive-prescribed-audit-v1/group2/adaptive-history/result.json"))
    write(
        "adaptive-pilot-reactions.json",
        read("adaptive-prescribed-audit-v1/group1/adaptive-prefix-reactions/result.json"),
    )
    write("adaptive-column-gap.json", read("adaptive-metric-gap-v1/group0/metric-gap/result.json"))
    copy(root / "adaptive-prescribed-audit-v1/group0/adaptive-prefix-pair.log", "adaptive-column-failed-analysis.log")
    expanded = []
    for group in range(1, 10):
        name = f"outcomes{group - 1}" if group < 8 else f"capped-case7-{'stock' if group == 8 else 'volume'}"
        value = read(f"expanded-outcomes-v1/group{group}/{name}/result.json")
        value["histories"] = [
            {key: item for key, item in row.items() if key != "history"} for row in value["histories"]
        ]
        expanded.append(value)
    write("expanded-outcomes.json", expanded)
    cost = read("cost-consolidated-v2/group0/cost/result.json")
    for configuration in cost["configurations"]:
        memory = configuration.get("process_memory")
        if memory is not None:
            memory["raw_samples_retained_in_original_source"] = len(memory.pop("samples", []))
    write("cost.json", cost)
    for path in sorted((root / "cost-consolidated-v2/group0/cost").glob("*.csv")):
        copy(path, f"profiles/{path.name}")

    figure = root / "qualification-figure-v3/group0/qualification-figure"
    for path in sorted(figure.iterdir()):
        if path.is_file():
            copy(path, f"figure/{path.name}")
    qa = root / "qualification-export-qa-v1/group0/export-qa"
    for filename in ("result.json", "reference-qualification-pdf-preview.png"):
        copy(qa / filename, f"figure/export-qa/{filename}")

    campaigns = (
        "validation-after-storage-v1",
        "shell-full-max-paired-v1",
        "shell-resolved-full-replay-v1",
        "shell-resolved-full-replay-v2",
        "shell-resolved-replay-audit-v2",
        "adaptive-input-v1",
        "adaptive-input-v2",
        "adaptive-prescribed-pilot-v1",
        "adaptive-prescribed-audit-v1",
        "adaptive-metric-gap-v1",
        "expanded-outcomes-v1",
        "reference-diagnostics-v2",
        "cost-consolidated-v2",
        "qualification-figure-v3",
        "qualification-export-qa-v1",
    )
    for campaign in campaigns:
        for filename in ("tasks.json", "submission.json", "source.json", "source.patch", "analysis-files.json"):
            copy(root / campaign / filename, f"receipts/{campaign}/{filename}")
        for path in sorted((root / campaign).glob("group*/*returncode*")):
            copy(path, f"receipts/{campaign}/{path.parent.name}/{path.name}")
        for path in sorted((root / campaign).glob("correction.json")):
            copy(path, f"receipts/{campaign}/{path.name}")
    # One complete executed analysis capture, with exact later variants retained separately.
    baseline = root / "cost-consolidated-v2/analysis-source"
    for path in sorted(baseline.rglob("*")):
        if path.is_file():
            copy(path, f"source/{path.relative_to(baseline)}")
    for campaign, filename in (
        ("shell-resolved-replay-audit-v2", "audit_shell_replay.py"),
        ("qualification-figure-v3", "qualification_figures.py"),
        ("adaptive-metric-gap-v1", "metric_gap_diagnostic.py"),
    ):
        relative = f"research/rigid_stress/qipc_comparison/{filename}"
        copy(root / campaign / "analysis-source" / relative, f"source-variants/{campaign}/{relative}")
    write(
        "selection-provenance.json",
        {
            "selected_original_json": selected_sources,
            "verified_exact_byte_copies": byte_copies,
            "scope": "Selections of already Slurm-derived JSON values, exact source/receipt/figure copies and byte hashes only. No scientific metric is recalculated on the login node. Complete raw histories and external solver inputs remain at original paths. This is an incomplete-goal checkpoint, not final portable raw/source closure.",
        },
    )


if __name__ == "__main__":
    main()
