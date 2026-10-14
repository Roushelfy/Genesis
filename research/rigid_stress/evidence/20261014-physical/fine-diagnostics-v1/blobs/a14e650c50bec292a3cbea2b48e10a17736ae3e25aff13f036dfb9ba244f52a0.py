"""Plot unsmoothed FEM refinement diagnostics and the separate frozen-boundary controls."""

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
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads(args.summary.read_text())
    fixed = json.loads(args.fixed_summary.read_text())
    geometries = list(dict.fromkeys(row["geometry"] for row in summary["rows"]))
    assert len(geometries) in (6, 7)
    titles = {
        "ellipsoid": "Ellipsoid",
        "capsule": "Capsule",
        "u": "Rounded U",
        "thin_neck": "Thin neck",
        "cavity": "Cavity shell",
        "narrow_gap": "Narrow gap",
        "egg": "Full hollow egg",
    }
    style = load_style(args.style_module)
    selected = [row for row in summary["rows"] if row["case"] == "opposing"]
    fixed_rows = [row for row in fixed["volume_rows"] if row["case"] == "opposing"]
    contract = {
        "claim_anchor": "physical-reference-refinement",
        "claim": "Raw peaks and complete common-probe diagnostics respond differently to refinement; geometry changes remain a separate error source",
        "material_passport": [file_identity(path) for path in (args.summary, args.fixed_summary, args.style_module)],
        "data_profile": {
            "selected_case": "Opposing compact finite profiles, force0.05N per patch, radius3mm, friction2",
            "cad_rows": len(selected),
            "frozen_polyhedron_rows": len(fixed_rows),
            "uncertainty": "Deterministic FEM evaluations; no statistical uncertainty or fitted asymptote",
            "outside_probes": {
                g: [r["outside_probe_indices"] for r in selected if r["geometry"] == g] for g in geometries
            },
        },
        "panels": [
            {
                "panel": chr(97 + i),
                "geometry": g,
                "role": titles[g],
                "x": "tetrahedra",
                "plot_type": "ordered line with all measured levels",
            }
            for i, g in enumerate(geometries)
        ],
        "palette": {"remeshed CAD": "#0072B2, solid circles", "frozen polyhedron": "#E69F00, dashed squares"},
        "caption": "Independent FEM opposing-load curves retain the raw global four-corner von Mises maximum. "
        "CAD remeshing changes volume and boundary approximation together. Frozen seed-polyhedron controls "
        "retain volume, boundary area and centre of mass while uniformly refining tetrahedra. The second "
        "figure compares full unsmoothed tensors at unchanged physical interior probes; points outside "
        "either adjacent approximate domain remain explicitly listed in the source data. The3% dotted "
        "line is an initial engineering diagnostic, not continuum-error certification. Energy, displacement "
        "and other loads are retained in the complete CSV/JSON. No smoothed maximum or asymptotic extrapolation is used.",
        "body_interpretation": "The frozen-U raw peak grows on seed reentrant creases while remeshed-U peaks decrease toward the latest range. "
        "Small common-tensor changes coexist with nonmonotone raw peaks on several shapes. A complete operating "
        "range requires the full observable set and resolved-contact comparisons.",
        "open_issues": "Joint CAD/volume changes are not separated for every shape; native finer accuracy/cost and coupled egg/contact acceptance remain required",
    }
    (args.output / "figure_specification.json").write_text(json.dumps(contract, indent=2) + "\n")
    exports = []
    for metric, fixed_metric, ylabel, factor in (
        ("raw_global_peak_Pa", "raw_global_peak_Pa", "Raw global peak (kPa)", 0.001),
        ("adjacent_probe_tensor_relative_change", "adjacent_probe_tensor_change", "Full tensor change (%)", 100),
    ):
        nrows = (len(geometries) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(7.2, 1.62 * nrows))
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.89, hspace=0.70, wspace=0.38)
        for i, (g, axis) in enumerate(zip(geometries, axes.flat)):
            for dataset, key, label, colour, marker, linestyle in (
                (selected, metric, "Remeshed CAD", "#0072B2", "o", "-"),
                (fixed_rows, fixed_metric, "Frozen polyhedron", "#E69F00", "s", "--"),
            ):
                points = sorted((r for r in dataset if r["geometry"] == g), key=lambda r: r["level"])
                if not points:
                    continue
                axis.plot(
                    [r["tetrahedra"] for r in points],
                    [factor * r.get(key, np.nan) for r in points],
                    label=label,
                    color=colour,
                    marker=marker,
                    markersize=3,
                    linewidth=1,
                    linestyle=linestyle,
                )
            axis.set_title(titles[g], fontsize=7)
            axis.text(-0.14, 1.14, chr(97 + i), transform=axis.transAxes, fontsize=8, fontweight="bold")
            axis.set_xscale("log")
            counts = [r["tetrahedra"] for r in selected + fixed_rows if r["geometry"] == g]
            first_power = int(np.ceil(np.log10(min(counts))))
            last_power = int(np.floor(np.log10(max(counts))))
            axis.set_xticks([10**power for power in range(first_power, last_power + 1)])
            axis.set_xlim(0.85 * min(counts), 1.18 * max(counts))
            axis.minorticks_off()
            axis.set_xlabel("Tetrahedra", fontsize=7)
            axis.set_ylabel(ylabel, fontsize=7)
            axis.tick_params(labelsize=7)
            if metric != "raw_global_peak_Pa":
                axis.axhline(3, color="0.5", linestyle=":", linewidth=0.8)
            axis.set_ylim(bottom=0)
        for axis in list(axes.flat)[len(geometries) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7)
        exports.append(export(fig, args.output / metric, style))
    with (args.output / "opposing_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in selected for key in row}))
        writer.writeheader()
        writer.writerows(selected)
    contract.update({"rendered_text_qa": "passed", "exports": exports})
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
