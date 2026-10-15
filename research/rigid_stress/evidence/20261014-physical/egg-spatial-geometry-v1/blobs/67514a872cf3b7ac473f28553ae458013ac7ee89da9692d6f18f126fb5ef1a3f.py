"""Plot independently checked spatial, linearization and CAD geometry diagnostics with exact source tables."""

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
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--linearization", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    reports = []
    for run in [args.comparison, args.linearization, args.geometry]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        reports.append(json.loads(path.read_text()))
    comparison, linearization, geometry = reports
    assert len(comparison["rows"]) == len(linearization["rows"]) == 4
    assert len({row["time_s"] for row in comparison["rows"] + linearization["rows"]}) == 1
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "egg-geometry-spatial-step5"})
    contract = {
        "claim_anchor": "egg-spatial-and-geometry-axes-remain-unresolved",
        "main_claim": "Fixed-polyhedron refinement retains unresolved stress/contact and linear-model changes while a distinct CAD refinement reduces represented mass error.",
        "material_passport": inputs,
        "data_status": "Deterministic saved states at one matched physical time; no statistical replicates, confidence intervals or fitted convergence orders.",
        "panels": {
            "a": "Original global nonlinear peak bounds and independent resolved-load linear peak on four fixed-polyhedron meshes.",
            "b": "Common-point tensor and energy changes for adjacent fixed-polyhedron levels; declared initial3% target.",
            "c": "Independent full-inertia linearization and frozen-objective constitutive discrepancy in volume-weighted tensor norms.",
            "d": "Separate matched-connectivity geometry families at fixed density: CAD and original polyhedron volume/mass error.",
        },
        "chart_choice": "Ordered refinement lines with visible points; separate axes for different physical quantities and geometry families.",
        "colours": {
            "nonlinear_or_CAD": "blue circles/solid",
            "linear_or_polyhedron": "vermillion squares/dashed",
            "reference_bound_or_target": "grey",
        },
        "limitations": "Nonlinear peak intervals are deterministic whole-volume bounds, not sampling uncertainty. The geometry panel does not show stress results or retune density. No panel establishes a physical contact tail or full application/history/feedback/cost acceptance.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    spatial = comparison["rows"]
    counts = np.array([row["tetrahedra"] for row in spatial])
    lower = np.array([row["raw_peak_lower_Pa"] for row in spatial]) / 1e6
    upper = np.array([row["raw_peak_upper_Pa"] for row in spatial]) / 1e6
    linear_peaks = np.array([row["raw_resolved_linear_global_peak_Pa"] for row in linearization["rows"]]) / 1e6
    figure, axes = plt.subplots(2, 2, figsize=style.mm_to_inches(183, 140))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.09, top=0.95, wspace=0.30, hspace=0.45)
    blue, orange = "#0072B2", "#D55E00"
    axes[0, 0].plot(counts, lower, "o-", color=blue, label="Nonlinear lower bound")
    axes[0, 0].vlines(counts, lower, upper, color=blue, linewidth=2.5)
    axes[0, 0].plot(counts, upper, "_", color=blue, markersize=5)
    axes[0, 0].plot(counts, linear_peaks, "s--", color=orange, label="Resolved-load linear")
    axes[0, 0].set(yscale="log", ylabel="Raw global von Mises peak (MPa)")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].set_ylim(0.07, 15000)
    changes = comparison["comparisons"]
    axes[0, 1].plot(
        counts[1:],
        [100 * row["tensor"]["relative_l2_change"] for row in changes],
        "o-",
        color=blue,
        label="Common-point tensor",
    )
    axes[0, 1].plot(
        counts[1:], [100 * row["nonlinear_energy_J"] for row in changes], "s--", color=orange, label="Nonlinear energy"
    )
    axes[0, 1].axhline(3, color="0.4", linestyle=":", label="Initial target: 3%")
    axes[0, 1].set(ylabel="Adjacent-level relative change (%)", ylim=(0, 90))
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].plot(
        counts,
        [100 * row["resolved_linear_vs_nonlinear"]["relative_volume_L2"] for row in linearization["rows"]],
        "o-",
        color=blue,
        label="Resolved-load linear",
    )
    axes[1, 0].plot(
        counts,
        [
            100 * row["frozen_objective_displacement_linear_vs_nonlinear"]["relative_volume_L2"]
            for row in linearization["rows"]
        ],
        "s--",
        color=orange,
        label="Frozen objective field",
    )
    axes[1, 0].set(ylabel="Linear vs nonlinear tensor difference (%)", ylim=(0, 13))
    axes[1, 0].legend(loc="upper left")
    geometry_table = []
    for mode, marker, color in [("CAD", "o-", blue), ("polyhedron", "s--", orange)]:
        rows = [row for row in geometry["rows"] if row["mode"] == mode]
        axes[1, 1].plot(
            [row["tetrahedra"] for row in rows],
            [-100 * row["relative_volume_and_mass_error_from_continuum"] for row in rows],
            marker,
            color=color,
            label=mode,
        )
        geometry_table.extend(
            {
                key: row[key]
                for key in ["mode", "level", "tetrahedra", "mass_kg", "relative_volume_and_mass_error_from_continuum"]
            }
            for row in rows
        )
    axes[1, 1].set(ylabel="Continuum volume/mass deficit (%)", ylim=(0, 15))
    axes[1, 1].legend(loc="upper right")
    for index, axis in enumerate(axes.flat):
        axis.set_xscale("log")
        axis.set_xlabel("Tetrahedra (refinement sequence)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.045, "abcd"[index], transform=axis.transAxes, fontsize=8, fontweight="bold")
    source_tables = {
        "spatial.csv": [
            {
                "tetrahedra": row["tetrahedra"],
                "time_s": row["time_s"],
                "raw_peak_lower_Pa": row["raw_peak_lower_Pa"],
                "raw_peak_upper_Pa": row["raw_peak_upper_Pa"],
                "resolved_linear_peak_Pa": model["raw_resolved_linear_global_peak_Pa"],
                "linearization_relative_volume_L2": model["resolved_linear_vs_nonlinear"]["relative_volume_L2"],
                "frozen_constitutive_relative_volume_L2": model["frozen_objective_displacement_linear_vs_nonlinear"][
                    "relative_volume_L2"
                ],
            }
            for row, model in zip(spatial, linearization["rows"], strict=True)
        ],
        "adjacent.csv": [
            {
                "coarse_tetrahedra": int(counts[row["coarse"]]),
                "fine_tetrahedra": int(counts[row["fine"]]),
                "tensor_relative_L2_change": row["tensor"]["relative_l2_change"],
                "energy_relative_change": row["nonlinear_energy_J"],
            }
            for row in changes
        ],
        "geometry.csv": geometry_table,
    }
    for name, rows in source_tables.items():
        with (args.output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for suffix, dpi in [("svg", None), ("png", 450)]:
        figure.savefig(args.output / f"egg-geometry-spatial.{suffix}", dpi=dpi, bbox_inches=None)
    figure.savefig(args.output / "preview.png", dpi=180, bbox_inches=None)
    assert "<text" in (args.output / "egg-geometry-spatial.svg").read_text()
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    metadata = {
        "figure_mm": [183, 140],
        "input_sha256": inputs,
        "editable_SVG_text_present": True,
        "manual_preview_inspected": False,
        "goal_completion": False,
        "caption": "At6.25ms, four independently checked fixed-polyhedron egg states retain growing raw global stress and substantial adjacent tensor/energy changes. (a) Vertical segments show lower/upper nonlinear whole-volume peak bounds, with independent resolved-load linear peaks. (b) Adjacent common-point tensor and energy changes remain above the declared initial3% engineering target. (c) Resolved-load and frozen-objective tensor discrepancies quantify two distinct linear-model diagnostics. (d) A separate matched-connectivity geometry study retains the original12.79% volume/mass deficit for polyhedral refinement; CAD approximation reduces it to0.219% at30720 tetrahedra. Material/density are fixed and represented mass is reported, not retuned. Deterministic states and bounds carry no statistical uncertainty. These diagnostic sequences do not establish converged contact physics, the full original application or verified accuracy/cost.",
        "body_interpretation": "The finest fixed-polyhedron pair changes tensor/energy by15.52%/9.35%, while its resolved-load linear tensor discrepancy reaches11.22%. CAD geometry error decreases on a distinct sequence, so geometry accuracy cannot be inferred from the fixed-polyhedron stress sweep. The densest contact and stress regions remain unresolved; this figure contains no sampled contact-area boundary or fitted singularity order.",
        "placement": "After the spatial/geometry distinction in the physical-validation report; read alongside the explicit raw-peak and contact-diagnostic table.",
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"panels": 4, "source_tables": len(source_tables), "editable_text": True}), flush=True)


if __name__ == "__main__":
    main()
