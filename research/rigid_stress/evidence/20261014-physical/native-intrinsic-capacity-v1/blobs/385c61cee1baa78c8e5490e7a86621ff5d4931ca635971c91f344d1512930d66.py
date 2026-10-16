"""Plot actual sampled hold motion and temporal load changes without interpolation."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_physical_controls import export, file_identity, load_style
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.audit)
    assert int((args.audit / "returncode.txt").read_text()) == 0
    report = json.loads((args.audit / "results/summary.json").read_text())
    assert all(g["unchanged_geometry_guard_passed"] for run in report["runs"] for g in run["geometry_checks"])
    style = load_style(args.style_module)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.84, wspace=0.38)
    for run, colour, marker in zip(report["runs"], ("#0072B2", "#D55E00", "#009E73"), ("o", "^", "s"), strict=True):
        n = run["substeps"]
        times = [(r["tick"] + r["substep"] / n) * 0.01 for r in run["rows"]]
        angular = [r["angular_speed_rad_s"] for r in run["rows"]]
        axes[0].scatter(
            times, angular, color=colour, marker=marker, s=14, facecolors="none", label=f"dt = {10 / n:g} ms"
        )
        last = run["rows"][-1]
        axes[1].scatter([10 / n - 0.10, 10 / n + 0.10], last["normal_forces_N"], color=colour, marker=marker, s=22)
    axes[0].set_xlim(0.4, 3.8)
    axes[0].set_ylim(0, 0.14)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_yticks([0, 0.05, 0.1])
    axes[0].set_xlabel("Elapsed hold time (s)", fontsize=7)
    axes[0].set_ylabel("Sampled angular speed (rad/s)", fontsize=7)
    axes[0].legend(loc="upper right", fontsize=6, frameon=False)
    axes[1].set_xlim(1, 11.5)
    axes[1].set_ylim(0.29, 0.365)
    axes[1].set_xticks([2.5, 5, 10])
    axes[1].set_yticks([0.30, 0.32, 0.34, 0.36])
    axes[1].set_xlabel("Rigid integration interval (ms)", fontsize=7)
    axes[1].set_ylabel("Last sampled normal force (N)", fontsize=7)
    for index, axis in enumerate(axes):
        axis.tick_params(labelsize=6)
        axis.set_title(("Motion during fixed actuation", "Both fingers, same actuation")[index], fontsize=7)
        axis.text(-0.15, 1.07, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
    export(fig, args.output / "franka_hold_inputs", style)
    contract = {
        "claim_anchor": "actual-franka-inputs-and-unsettled-angular-mode",
        "main_claim": "Actual normal loads depend on rigid temporal resolution and the sampled hold retains angular motion under unchanged actuation.",
        "panel_roles": {
            "a": "Actual sampled angular speeds, no interpolation or frequency inference",
            "b": "Two final finger normal forces at each declared integration interval",
        },
        "material_passport": [file_identity(args.audit / "results/summary.json"), file_identity(args.style_module)],
        "caption": "Panda/full hollow egg under fixed lift-joint and18mm finger targets with original PD gains and4N caps. Markers are deterministic saved observations at the original capture cadence, including distinct physical substeps; no smoothing, statistical interval or oscillation-frequency estimate is applied. Both final finger forces are retained. A stable translational speed does not establish static six-component equilibrium; the observed two-point wrench span has rank5.",
        "alt_text": "Three marker groups retain angular speeds during fixed actuation. Final normal forces rise from about0.304N at10ms to0.353N at5/2.5ms.",
        "limits": "These are rigid application inputs, not a matched deformable-contact or physical accuracy pass. Geometry/pose/mass/control matching, contact-reference convergence and omitted feedback remain required.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
