"""Show temporal diagnostics separately from short-hold resolved-contact feedback."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports, hashes = [], {}
    for run in [args.temporal, args.feedback]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        source = run / "results/summary.json"
        reports.append(json.loads(source.read_text()))
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    temporal, feedback = reports
    assert len(temporal["adjacent_comparisons"]) == 2
    assert all(len(c["common_times"]) == 40 for c in temporal["adjacent_comparisons"])
    assert len(feedback["comparisons"]) == 80
    contract = {
        "claim_anchor": "coarse-full-arm-temporal-versus-feedback",
        "main_claim": "Temporal and constitutive/feedback differences are separately measured in a short hold at one unresolved spatial setting.",
        "material_passport": hashes,
        "panels": {
            "a": "Common-time objective tensor changes across fixed-law time steps",
            "b": "Resolved-load feedback versus linear/nonlinear body difference",
        },
        "method": "Two line panels at183x82mm; all deterministic time samples, no statistical bands, fitting or smoothing.",
        "alternatives": [
            "A table preserves maxima but hides transient differences.",
            "Bars would hide ordered temporal behaviour.",
        ],
        "open_decisions": [],
        "limitations": "One480-tet fixed polyhedron with unresolved traction concentration. These curves do not establish a spatial tail or original approach/angular-history equivalence.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update(
        {
            "savefig.bbox": None,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "articulated-feedback-20261015",
        }
    )
    figure = plt.figure(figsize=style.mm_to_inches(183, 82))
    axes = [figure.add_axes([0.10, 0.22, 0.36, 0.59]), figure.add_axes([0.59, 0.22, 0.36, 0.59])]
    records = []
    colours, lines = ["#0072B2", "#D55E00"], ["-", "--"]
    for index, pair in enumerate(temporal["adjacent_comparisons"]):
        rows = pair["common_times"]
        x = np.array([r["time_s"] for r in rows]) * 1000
        y = np.array([r["tensor_relative_volume_L2"] for r in rows]) * 100
        assert np.isfinite(y).all() and (y > 0).all()
        label = ["2.5 to 1.25 ms", "1.25 to 0.625 ms"][index]
        axes[0].plot(
            x,
            y,
            color=colours[index],
            linestyle=lines[index],
            linewidth=0.9,
            marker=["o", "s"][index],
            markersize=2,
            markeredgewidth=0.3,
            label=label,
        )
        records.extend(
            {"panel": "a", "group": label, "time_ms": float(t), "tensor_change_percent": float(v)} for t, v in zip(x, y)
        )
    for index, key in enumerate(
        [
            "resolved_load_feedback_tensor_relative_change",
            "deformable_linear_replay_vs_nonlinear_tensor_relative_change",
        ]
    ):
        rows = feedback["comparisons"]
        x = np.array([r["time_s"] for r in rows]) * 1000
        y = np.array([r[key] for r in rows]) * 100
        assert np.isfinite(y).all() and (y > 0).all()
        label = ["Rigid / deformable loads", "Linear / nonlinear body"][index]
        axes[1].plot(x, y, color=colours[index], linestyle=lines[index], linewidth=0.9, label=label)
        records.extend(
            {"panel": "b", "group": label, "time_ms": float(t), "tensor_change_percent": float(v)} for t, v in zip(x, y)
        )
    for index, axis in enumerate(axes):
        axis.set(
            xlabel="Time (ms)",
            ylabel=["Tensor change (%)", "Tensor difference (%)"][index],
            xlim=(0, 103),
            yscale="log",
        )
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.tick_params(which="minor", length=1.5)
        axis.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.015), frameon=False, fontsize=6, handlelength=2.5)
        axis.text(-0.20, 1.14, "ab"[index], transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    outside = []
    visible_text = []
    for axis in axes:
        visible_text.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts, *axis.get_legend().get_texts()])
        visible_text.extend(
            text
            for value, text in zip(axis.get_xticks(), axis.get_xticklabels())
            if axis.get_xlim()[0] <= value <= axis.get_xlim()[1]
        )
        visible_text.extend(
            text
            for value, text in zip(axis.get_yticks(), axis.get_yticklabels())
            if axis.get_ylim()[0] <= value <= axis.get_ylim()[1]
        )
    for text in visible_text:
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if (
                bounds.x0 < -0.1
                or bounds.y0 < -0.1
                or bounds.x1 > figure.bbox.x1 + 0.1
                or bounds.y1 > figure.bbox.y1 + 0.1
            ):
                outside.append(text.get_text())
    assert not outside, outside
    stem = args.output / "franka-temporal-feedback"
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    figure.savefig(args.output / "preview.png", dpi=180, facecolor="white")
    assert "<text" in stem.with_suffix(".svg").read_text()
    with (args.output / "source_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["panel", "group", "time_ms", "tensor_change_percent"])
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "figure_mm": [183, 82],
        "source_rows": len(records),
        "outside_text": outside,
        "svg_editable_text": True,
        "source_sha256": hashes,
        "rendered_preview_inspected": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
