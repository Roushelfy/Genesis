"""Separate regularization moments from a declared concentrated-load spatial control."""

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
    parser.add_argument("--regularization", type=Path, required=True)
    parser.add_argument("--point-control", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports, sources = [], {}
    for run in [args.regularization, args.point_control]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        reports.append(json.loads(path.read_text()))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    regularization, control = reports
    assert len(regularization["cases"]) == 3 and len(control["rows"]) == 4
    contract = {
        "claim_anchor": "separate-contact-regularization-and-point-load-limit",
        "main_claim": "Small contact-distance changes and a growing concentrated-load spatial response are separate diagnostics.",
        "material_passport": sources,
        "panels": {
            "a": "Maximum regularized full-contact moment over80 accepted states",
            "b": "Unaveraged global stress growth under two fixed point forces",
            "c": "Point-load compliance and localized-test lower bound",
        },
        "data_status": "Three closed contact distances and four closed independent linear point-load meshes. Deterministic numerical studies; no statistical uncertainty or fitted rate.",
        "limitations": "Panels a and b/c use separately declared controls, not one matched continuum contact sweep. No finite-patch or full application spatial acceptance.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update(
        {"savefig.bbox": None, "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "contact-limit-20261015"}
    )
    figure = plt.figure(figsize=style.mm_to_inches(183, 82))
    axes = [figure.add_axes([x, 0.23, 0.235, 0.57]) for x in [0.08, 0.405, 0.73]]
    distance = np.array([c["dhat_m"] for c in regularization["cases"]]) * 1e6
    moments = (
        np.array(
            [
                max(np.linalg.norm(r["contact_total_moment_about_body_com_Nm"]) for r in c["rows"])
                for c in regularization["cases"]
            ]
        )
        * 1e9
    )
    elements = np.array([r["tetrahedra"] for r in control["rows"]])
    peaks = np.array([r["raw_global_peak_Pa"] for r in control["rows"]]) / 1e6
    compliance = np.array([r["compliance_J"] for r in control["rows"]]) * 1e6
    bound = np.array([r["localized_test_compliance_lower_bound_J"] for r in control["rows"]]) * 1e6
    assert all(np.isfinite(a).all() and (a > 0).all() for a in [distance, moments, peaks, compliance, bound])
    axes[0].plot(distance, moments, "o-", color="#0072B2", linewidth=0.9, markersize=3)
    axes[0].set(xscale="log", yscale="log", xlabel="Contact distance (µm)", ylabel="Contact moment (nN m)")
    axes[0].set_xticks([0.625, 1.25, 2.5], labels=["0.625", "1.25", "2.5"])
    axes[0].set_yticks([6, 12, 24], labels=["6", "12", "24"])
    axes[1].plot(elements, peaks, "o-", color="#0072B2", linewidth=0.9, markersize=3)
    axes[1].set(xscale="log", yscale="log", xlabel="Tetrahedra", ylabel="Raw global peak (MPa)")
    axes[2].plot(elements, compliance, "o-", color="#0072B2", linewidth=0.9, markersize=3, label="Compliance")
    axes[2].plot(elements, bound, "s--", color="#D55E00", linewidth=0.9, markersize=3, label="Test-work bound")
    axes[2].set(xscale="log", yscale="log", xlabel="Tetrahedra", ylabel="Compliance (µJ)")
    axes[2].legend(loc="lower left", bbox_to_anchor=(-0.02, 1.015), frameon=False, fontsize=6)
    for index, axis in enumerate(axes):
        axis.tick_params(which="minor", length=1.5)
        axis.text(-0.24, 1.14, "abc"[index], transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = []
    for axis in axes:
        texts.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts])
        if axis.get_legend():
            texts.extend(axis.get_legend().get_texts())
        texts.extend(
            t
            for v, t in zip(axis.get_xticks(), axis.get_xticklabels())
            if axis.get_xlim()[0] <= v <= axis.get_xlim()[1]
        )
        texts.extend(
            t
            for v, t in zip(axis.get_yticks(), axis.get_yticklabels())
            if axis.get_ylim()[0] <= v <= axis.get_ylim()[1]
        )
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -0.1 or box.y0 < -0.1 or box.x1 > figure.bbox.x1 + 0.1 or box.y1 > figure.bbox.y1 + 0.1:
            outside.append(text.get_text())
    assert not outside, outside
    stem = args.output / "franka-contact-limit"
    for suffix in [".svg", ".pdf"]:
        figure.savefig(stem.with_suffix(suffix), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    figure.savefig(args.output / "preview.png", dpi=180, facecolor="white")
    records = [
        {"control": "regularized_contact", "metric": "max_contact_moment_nNm", "x": float(x), "value": float(y)}
        for x, y in zip(distance, moments)
    ]
    for name, values in [("raw_peak_MPa", peaks), ("compliance_microJ", compliance), ("test_bound_microJ", bound)]:
        records.extend(
            {"control": "fixed_point_forces", "metric": name, "x": int(x), "value": float(y)}
            for x, y in zip(elements, values)
        )
    with (args.output / "source_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["control", "metric", "x", "value"])
        writer.writeheader()
        writer.writerows(records)
    assert "<text" in stem.with_suffix(".svg").read_text()
    metadata = {
        "figure_mm": [183, 82],
        "source_rows": len(records),
        "outside_text": outside,
        "svg_editable_text": True,
        "source_sha256": sources,
        "rendered_preview_inspected": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
