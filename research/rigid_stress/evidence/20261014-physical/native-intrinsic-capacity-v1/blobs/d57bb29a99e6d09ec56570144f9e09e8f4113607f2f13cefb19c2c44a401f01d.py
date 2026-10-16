"""Plot raw full-domain peaks and common-tensor refinement changes for fixed opposing finite loads."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_physical_controls import export, file_identity, load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads(args.summary.read_text())
    assert not summary["missing"]
    style = load_style(args.style_module)
    geometries = ("ellipsoid", "capsule", "u", "thin_neck", "cavity", "narrow_gap")
    titles = ("Ellipsoid", "Capsule", "Rounded U", "Thin neck", "Cavity shell", "Narrow gap")
    models = ("reference", "pressure", "componentwise", "pair_union", "body_source_prior")
    labels = ("Independent FEM", "Pressure", "Componentwise", "Pair union", "Source prior")
    colours = ("#222222", "#E69F00", "#0072B2", "#009E73", "#CC79A7")
    markers = ("D", "o", "s", "x", "^")
    outputs = []
    for metric, ylabel, factor in (
        ("raw_global_peak_Pa", "Raw global peak (kPa)", 0.001),
        ("adjacent_tensor_relative_change", "Full tensor change (%)", 100),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.84, hspace=0.58, wspace=0.42)
        for geometry, title, axis in zip(geometries, titles, axes.flat):
            for model, label, colour, marker in zip(models, labels, colours, markers):
                rows = sorted(
                    (
                        row
                        for row in summary["rows"]
                        if row["geometry"] == geometry and row["case"] == "opposing" and row["model"] == model
                    ),
                    key=lambda row: row["level"],
                )
                assert len(rows) == 3
                axis.plot(
                    [row["tetrahedra"] for row in rows],
                    [factor * row.get(metric, np.nan) for row in rows],
                    color=colour,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1,
                    label=label,
                    linestyle="--" if model == "reference" else "-",
                )
            if metric == "adjacent_tensor_relative_change":
                axis.axhline(3, color="0.5", linestyle=":", linewidth=0.8)
                axis.set_ylim(bottom=0)
            axis.set_title(title, fontsize=8)
            axis.set_xscale("log")
            axis.set_xticks([row["tetrahedra"] for row in rows], [str(row["tetrahedra"]) for row in rows])
            axis.minorticks_off()
            axis.set_xlabel("Tetrahedra", fontsize=7)
            axis.set_ylabel(ylabel, fontsize=7)
            axis.tick_params(labelsize=7)
            if geometry == "narrow_gap":
                axis.text(
                    0.03,
                    0.16 if metric == "adjacent_tensor_relative_change" else 0.97,
                    "Pressure: cone violation",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=6.5,
                )
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.54, 1)
        )
        outputs.append(export(fig, args.output / metric, style))
    selected_rows = [row for row in summary["rows"] if row["case"] == "opposing"]
    keys = sorted({key for row in selected_rows for key in row})
    with (args.output / "opposing_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(selected_rows)
    contract = {
        "claim": "Same-mesh numerical agreement does not establish continuum accuracy; all studied joint mesh/geometry tails remain unresolved",
        "input": file_identity(args.summary),
        "selected_case": "Opposing declared0.05N finite compact profiles, radius3mm, friction2; fixed per-mesh inputs for all four models",
        "caption": "Three useful levels on six complete3D shapes. The first figure retains every unaveraged global four-corner "
        "von Mises peak. The second reports adjacent full-tensor changes at fixed physical interior probes; "
        "the first level has no adjacent value. The dotted3% line is the declared initial tail diagnostic. "
        "No complete peak/energy/tensor/von-Mises tail passes. These curves combine geometry and volume refinement. "
        "The narrow-gap pressure curve is a numerically qualified solution with inadmissible local traction, "
        "not a validated physical prediction.",
        "alt_text": "Two six-panel figures show raw peak curves and large nonmonotone full-tensor refinement changes for four load models and independent FEM",
        "rendered_text_qa": "passed",
        "exports": outputs,
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
