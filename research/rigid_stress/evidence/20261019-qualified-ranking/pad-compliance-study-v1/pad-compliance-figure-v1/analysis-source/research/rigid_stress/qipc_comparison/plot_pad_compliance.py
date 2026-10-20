"""Display measured pressure-model contrasts with shared scales within each material case."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("pad_compliance_figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    result = json.loads((args.study / "result.json").read_text())
    assert result["passed"]
    panels = []
    for case in (0, 5):
        for mode in ("rigid", "diagonal", "coupled"):
            row = next(r for r in result["records"] if r["case"] == case and r["mode"] == mode
                       and r["order"] == 6 and r["search_contains_actual_elastic_compression"])
            path = Path(row["field"]["path"])
            assert file_id(path) == row["field"]
            with np.load(path) as saved:
                points, pressure = saved["points_m"], saved["pressure_pa"]
            mask = (points[:, 2] > 0) & (pressure > 0)
            panels.append((row, points[mask, :2] * 1000, pressure[mask] / 1000))
    extent = np.ceil(max(np.abs(points).max() for _, points, _ in panels) * 2.2) / 2
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 142), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.10, top=0.89, wspace=0.28, hspace=0.58)
    receipts = []
    for row_index in range(2):
        subset = panels[3 * row_index:3 * row_index + 3]
        maximum = np.ceil(max(pressure.max() for _, _, pressure in subset))
        norm = Normalize(0, maximum)
        for column, (row, points, pressure) in enumerate(subset):
            ax = axes[row_index, column]
            plot = ax.scatter(points[:, 0], points[:, 1], c=pressure, s=2.5, cmap="viridis", norm=norm,
                              linewidths=0, rasterized=True)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xlabel="x (mm)")
            if column == 0:
                ax.set_ylabel("Upper pad: y (mm)")
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            ax.set_title({"rigid": "Rigid gap", "diagonal": "Diagonal compliance", "coupled": "Coupled compliance"}[row["mode"]], pad=7)
            style.style_axes(ax)
            style.add_panel_label(ax, "abcdef"[row_index * 3 + column], x=-0.20, y=1.12)
            receipts.append(row["field"])
        color_axis = fig.add_axes([0.875, 0.59 if row_index == 0 else 0.12, 0.017, 0.27])
        fig.colorbar(plot, cax=color_axis, label="Normal pressure (kPa)")
        case = result["cases"][(0, 5)[row_index]]
        label = f"E = {case['young_pa']/1e9:g} GPa; pad stiffness = {case['pad_stiffness_n_per_m3']:.0e} N/m$^3$"
        fig.text(0.43, 0.97 if row_index == 0 else 0.50, label, ha="center", va="top")
    issues = style.rendered_text_qa(fig)
    exports = []
    for suffix, dpi in (("svg", 450), ("pdf", 450), ("png", 600)):
        path = args.output / f"pad-compliance-pressure.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        exports.append(file_id(path))
    preview = args.output / "pad-compliance-pressure-preview.png"
    fig.savefig(preview, dpi=150, bbox_inches=None)
    plt.close(fig)
    (args.output / "caption.md").write_text(
        "Upper-pad pressure at0.1N from rigid-gap, diagonal-compliance and coupled-compliance predictions. "
        "Rows compare the nominal10GPa link/1e9N/m3 pad and a1GPa link/1e10N/m3 pad. Each row shares its own kPa scale; "
        "all panels share physical coordinates. Every positive degree6 physical quadrature pressure is shown without "
        "smoothing or clipping; marker size is a display aid. The coarse fixed mesh and static normal law are common. "
        "The lower-pad distribution, complete tensors, pressure/field errors and quadrature sensitivity remain in the data. "
        "This figure compares declared models, not continuum accuracy or production throughput.\n")
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": not issues, "source": file_id(args.study / "result.json"), "fields": receipts,
        "style": file_id(args.style), "size_mm": [183, 142], "shared_xy_extent_mm": extent.item(),
        "exports": exports, "preview": file_id(preview), "rendered_text_issues": issues,
        "visual_inspection_pending": True, "whole_goal_complete": False})
    assert not issues, issues


if __name__ == "__main__":
    main()
