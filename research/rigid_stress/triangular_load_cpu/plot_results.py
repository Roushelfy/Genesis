"""Render measured load-proxy accuracy and full CPU pipeline costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def von_mises(values):
    xx, yy, xy = np.moveaxis(values, -1, 0)
    return np.sqrt(np.maximum(0, xx**2 + yy**2 - xx * yy + 3 * xy**2))


def make_plots(output):
    summary = json.loads((output / "summary.json").read_text())
    timing = json.loads((output / "timing_summary.json").read_text())
    by_method = {row["method"]: row for row in summary}
    refinements = [1, 2, 4, 8, 16]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for key, label, marker in (("mean_field", "Mean tensor field", "o"),
                               ("max_field", "Worst tensor field", "s"),
                               ("max_peak", "Worst sampled peak", "^")):
        axes[0].loglog(refinements, [100*by_method[f"tri-k{k}"][key] for k in refinements],
                       marker=marker, linewidth=2, label=label)
    axes[0].axhline(1, color="0.4", linestyle=":", label="1% error")
    axes[0].set(xlabel="Loading-face subdivisions per Q4 edge", ylabel="Relative error (%)",
                title="Same 1,452-DOF Q4 elastic model")
    axes[0].set_xticks(refinements, [str(k) for k in refinements])
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].legend(fontsize=8)
    names = ["Q4"] + [f"tri-k{k}" for k in refinements]
    measured = {row["method"]: row for row in timing}
    bottom = np.zeros(len(names))
    for key, label, color in (("load_ms", "Current-load preparation", "#36789a"),
                              ("solve_ms", "Same elastic solve", "#e9aa41"),
                              ("stress_ms", "Same full stress recovery", "#72a681")):
        values = np.array([measured[name][key] for name in names])
        axes[1].bar(names, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[1].set(ylabel="CPU milliseconds / B16 call", title="Seven paired repeats, one CPU thread")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Planar loading triangles: measured accuracy / cost tradeoff", fontsize=13)
    figure.savefig(output / "accuracy_cost.png", dpi=170)
    plt.close(figure)

    with np.load(output / "selected_fields.npz") as data:
        vertices, quads = data["vertices"], data["quads"]
        fields = {name: data[name] for name in ("Q4", "tri_k1", "tri_k8", "tri_k16")}
    reference = fields["Q4"][0]
    tensors = {name: field[0] for name, field in fields.items()}
    peak_scale = max(von_mises(field).max() for field in tensors.values()) / 1e6
    errors = {name: np.sqrt(np.sum((field-reference)**2 * np.array([1, 1, 2]), axis=-1)) / 1e6
              for name, field in tensors.items()}
    error_scale = max(value.max() for value in errors.values())
    figure = plt.figure(figsize=(13, 7), constrained_layout=True)
    for column, (name, tensor) in enumerate(tensors.items()):
        for row in range(2):
            axis = figure.add_subplot(2, 4, row*4+column+1, projection="3d")
            values = von_mises(tensor).mean(axis=1) / 1e6 if row == 0 else errors[name].mean(axis=1)
            maximum = peak_scale if row == 0 else error_scale
            norm = matplotlib.colors.Normalize(vmin=0, vmax=maximum)
            surface = Poly3DCollection(vertices[quads]*1000, linewidths=0.15, edgecolors=(0, 0, 0, 0.12))
            surface.set_facecolor(plt.cm.viridis(norm(values)) if row == 0 else plt.cm.magma(norm(values)))
            axis.add_collection3d(surface)
            for setter, coordinate in ((axis.set_xlim, 0), (axis.set_ylim, 1), (axis.set_zlim, 2)):
                limits = vertices[:, coordinate].min()*1000, vertices[:, coordinate].max()*1000
                setter(limits)
            axis.set_box_aspect(np.ptp(vertices, axis=0))
            axis.view_init(elev=12, azim=35)
            axis.set_axis_off()
            axis.set_title(name.replace("_", "-"), fontsize=11)
            scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis" if row == 0 else "magma")
            figure.colorbar(scalar, ax=axis, shrink=0.65, pad=0.01,
                            label="von Mises (MPa)" if row == 0 else "Tensor difference (MPa)")
    figure.suptitle("Asymmetric two-contact case, outer side (face means of four Gauss samples)", fontsize=12)
    figure.savefig(output / "stress_comparison.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    make_plots(parser.parse_args().output)
