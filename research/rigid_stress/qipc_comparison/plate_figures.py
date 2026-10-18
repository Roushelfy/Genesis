"""Render traced diagnostic plate mechanics and common-scale material stress maps."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from ..shell_comparison.nature_mpl_style import (
    WONG_PALETTE,
    apply_nature_style,
    figure_size,
    rendered_text_qa,
    style_axes,
)
from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fields", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--dpr", type=Path, required=True)
    cli.add_argument("--shell", type=Path, required=True)
    cli.add_argument("--export-vector", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    apply_nature_style(mpl)
    traces, histories = [], []
    for run in (args.reference, args.dpr, args.shell):
        histories.append([json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()])
        traces.extend(file_id(run / name) for name in ("steps.jsonl", "start.json", "result.json"))
    field_result = json.loads(args.fields.read_text())
    traces.append(file_id(args.fields))
    labels = ("QIPC", "DPR", "Shell")
    colors = [WONG_PALETTE[name] for name in ("black", "blue", "orange")]
    styles = ("-", "--", "-.")
    qa = []

    def export(fig, stem):
        fig.canvas.draw()
        qa.append({"figure": stem, "text_qa": rendered_text_qa(fig)})
        fig.savefig(args.output / f"{stem}.png", dpi=250)
        if args.export_vector:
            vector = args.output / f"{stem}.svg"
            fig.savefig(vector)
            vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
            fig.savefig(args.output / f"{stem}.pdf")
        from PIL import Image

        with Image.open(args.output / f"{stem}.png") as preview:
            preview.convert("L").save(args.output / f"{stem}-grayscale.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=figure_size("double", height_mm=72), layout="constrained")
    for label, color, style, history in zip(labels, colors, styles, histories):
        short = [row for row in history if row["time_s"] <= 0.3 + 1e-12]
        axes[0].plot(
            [row["time_s"] for row in short],
            [row["upper_reaction_n"] for row in short],
            label=label,
            color=color,
            ls=style,
        )
        ends = [row for row in history if "raw_vm_peak_pa" in row] if label == "DPR" else [history[-1]]
        axes[1].plot(
            [row["compression_coordinate_m"] * 1e6 for row in ends],
            [row["upper_reaction_n"] for row in ends],
            color=color,
            ls="none",
            marker={"QIPC": "o", "DPR": "s", "Shell": "^"}[label],
            markerfacecolor="none" if label == "Shell" else color,
            label=label,
        )
    axes[0].plot([0, 0.1, 0.3], [0, 0.1, 0.1], color="0.6", ls=":", label="Command")
    axes[0].set(xlabel="Time (s)", ylabel="Actual upper reaction (N)", xlim=(0, 0.3))
    axes[0].legend(loc="lower right")
    axes[1].set(xscale="log", xlabel="Upper plate travel (µm)", ylabel="Actual upper reaction (N)")
    axes[1].legend(loc="upper left")
    for letter, ax in zip("ab", axes):
        style_axes(ax)
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=8)
    export(fig, "mechanics")

    entries = {row["label"]: row["levels"][0] for row in field_result["comparisons"]}
    fields, points = {}, None
    for label in ("dpr", "shell"):
        field_file = Path(entries[label]["common_field_artifact"]["path"])
        traces.append(file_id(field_file))
        with np.load(field_file) as saved:
            selected = saved["surface_side"] == 1
            coordinates = saved["surface_points_m"][selected]
            if points is None:
                points = coordinates
            np.testing.assert_allclose(coordinates, points, atol=1e-14, rtol=0)
            fields[label] = np.linalg.norm(saved["candidate_surface_tangent_pa"][selected], axis=(-1, -2))
            fields.setdefault(
                "reference", np.linalg.norm(saved["reference_surface_tangent_pa"][selected], axis=(-1, -2))
            )
    all_values = np.concatenate(list(fields.values()))
    assert np.isfinite(all_values).all() and np.all(all_values > 0)
    norm = LogNorm(vmin=all_values.min(), vmax=all_values.max(), clip=False)
    fig, axes = plt.subplots(
        1, 4, figsize=figure_size("double", height_mm=68), layout="constrained", sharex=True, sharey=True
    )
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    for letter, label, ax in zip("abcd", ("reference", "dpr", "scr", "shell"), axes):
        if label in fields:
            artist = ax.scatter(
                azimuth, points[:, 2] * 1000, c=fields[label], cmap="viridis", norm=norm, s=3, linewidths=0
            )
            observed = (
                entries["dpr"]["reference_mechanics"] if label == "reference" else entries[label]["candidate_mechanics"]
            )
            title = (
                "QIPC" if label == "reference" else label.upper() if label != "shell" else "Shell"
            ) + f"\n{observed['upper_reaction_n']:.4f} N"
        else:
            ax.text(
                0.5, 0.5, "Field rejected\n(status40)", transform=ax.transAxes, ha="center", va="center", fontsize=6
            )
            title = "SCR\nNo admitted field"
        ax.set_title(title, fontsize=6)
        ax.text(-0.08, 1.15, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set(xlabel="Azimuth (°)", xticks=[-180, 0, 180], xlim=(-190, 190))
        style_axes(ax)
    axes[0].set_ylabel("Material height (mm)")
    fig.colorbar(artist, ax=axes, shrink=0.85, label="Outer tangential tensor norm (Pa; log scale)")
    export(fig, "outer-stress")
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "sources": traces,
            "claim_anchor": "plates-admission-before-ranking",
            "qa": qa,
            "vector_exports": args.export_vector,
            "scope": "Diagnostic maps share unmodified extrema and a common logarithmic scale. SCR absence is rejection, not zero stress. Shell carries a different actual force and is not admitted for equal-load accuracy. Upper plate travel includes initial clearance/contact regularization, not pure egg elastic indentation.",
            "color_limits_pa": [float(norm.vmin), float(norm.vmax)],
        },
    )


if __name__ == "__main__":
    main()
