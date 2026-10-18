"""Render source-backed pressure/intrinsic comparison figures at final size."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from nature_mpl_style import (
    add_panel_label,
    apply_nature_style,
    figure_size,
    rendered_text_qa,
    save_nature_figure,
    style_axes,
)

PRESSURE = "#0072B2"
INTRINSIC = "#D55E00"
RIGID = "#555555"


def read(path):
    return json.loads(path.read_text())


def export(fig, stem, data_paths):
    fig.canvas.draw()
    issues = rendered_text_qa(fig)
    outputs = save_nature_figure(fig, stem)
    qa = {
        "outputs": outputs,
        "text_qa_issues": issues,
        "size_inches": fig.get_size_inches().tolist(),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths},
        "visual_review": "pending",
        "font_type": {"svg": matplotlib.rcParams["svg.fonttype"], "pdf": matplotlib.rcParams["pdf.fonttype"]},
    }
    stem.with_name(stem.name + "-qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    plt.close(fig)


def performance(args, metadata):
    records = metadata["records"]
    names = ["preflight-b32-v1", "core-b2048-v1", "wide-b8192-v1", "core-b16384-v1", "wide-b32768-v1"]
    widths = [32, 2048, 8192, 16384, 32768]
    fig, axes = plt.subplots(2, 2, figsize=figure_size("double", 137))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.10, top=0.93, hspace=0.43, wspace=0.30)
    styles = [
        ("shared", RIGID, "^", ":", "Rigid"),
        ("pressure-default", PRESSURE, "o", "-", "Pressure"),
        ("intrinsic", INTRINSIC, "s", "--", "Intrinsic"),
    ]
    rows = []
    for ax, scope in zip(axes[0], ["live", "policy"]):
        for profile, color, marker, line, label in styles:
            values, raw = [], []
            for cohort, width in zip(names, widths):
                selected_scope = ("rigid" if scope == "live" else "policy-rigid") if profile == "shared" else scope
                matches = [
                    r
                    for r in records
                    if r["cohort"] == cohort and r["profile"] == profile and r["scope"] == selected_scope
                ]
                assert len(matches) == 1 and matches[0]["headline_eligible"], (cohort, profile, selected_scope)
                row = matches[0]
                values.append(row["median_valid_envstep_s"] / 1000)
                raw.append(np.array(row["raw_rates"]) / 1000)
                rows.extend(
                    {
                        "panel": "a" if scope == "live" else "b",
                        "scope": selected_scope,
                        "profile": profile,
                        "envs": width,
                        "repeat": i,
                        "valid_envstep_s": rate,
                        "raw_sha256": row["raw_sha256"],
                    }
                    for i, rate in enumerate(row["raw_rates"])
                )
            ax.plot(
                widths, values, color=color, marker=marker, linestyle=line, markersize=3.4, linewidth=1.0, label=label
            )
            for width, repeats in zip(widths, raw):
                ax.scatter([width] * len(repeats), repeats, color=color, s=5, zorder=3)
                ax.vlines(width, repeats.min(), repeats.max(), color=color, linewidth=0.6)
        ax.set_yscale("log")
        ax.set_ylim(2, 1600)
        ax.set_yticks([10, 100, 1000])
        ax.set_ylabel("Valid env-step/s ($10^3$)")
        ax.set_xlabel("Environments")
        ax.text(0.03, 0.94, "Scripted" if scope == "live" else "Policy inference", transform=ax.transAxes, va="top")
    for profile, color, marker, line, label in styles[1:]:
        values = []
        for cohort in names:
            row = next(r for r in records if r["cohort"] == cohort and r["profile"] == profile and r["scope"] == "live")
            values.append(row["incremental_stress_step_ms"])
        axes[1, 0].plot(widths, values, color=color, marker=marker, linestyle=line, markersize=3.4, linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(0.75, 400)
    axes[1, 0].set_yticks([1, 10, 100])
    axes[1, 0].set_ylabel("Incremental live step time (ms)")
    axes[1, 0].set_xlabel("Environments")
    for profile, color, marker, line, label in styles:
        memory_widths, memory = [], []
        for width in [32, 2048, 16384, 32768]:
            scope = "rigid" if profile == "shared" else "live"
            matches = [
                r
                for r in records
                if r["cohort"] == f"memory-b{width}-v1" and r["profile"] == profile and r["scope"] == scope
            ]
            assert len(matches) == 1 and matches[0]["memory"]["sampled_peak_GiB"] is not None
            memory_widths.append(width)
            memory.append(matches[0]["memory"]["sampled_peak_GiB"])
        axes[1, 1].plot(memory_widths, memory, color=color, marker=marker, linestyle=line, markersize=3.4, linewidth=1)
    axes[1, 1].set_ylabel("Sampled process peak (GiB)")
    axes[1, 1].set_xlabel("Environments")
    axes[1, 1].set_ylim(bottom=0)
    for ax, letter in zip(axes.flat, "abcd"):
        style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, ["32", "2,048", "8,192", "16,384", "32,768"])
        ax.tick_params(axis="x", labelsize=5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_panel_label(ax, letter, x=-0.16, y=1.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0))
    with (args.output / "performance-source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export(fig, args.output / "performance", [args.metadata])


def errors(args, metadata, models):
    fig, axes = plt.subplots(2, 2, figsize=figure_size("double", 165), gridspec_kw={"height_ratios": [1, 1.35]})
    fig.subplots_adjust(left=0.125, right=0.965, bottom=0.085, top=0.94, hspace=0.42, wspace=0.42)
    fig.legend(
        [
            Line2D([], [], color=PRESSURE, marker="o", linestyle="-", markersize=3.5),
            Line2D([], [], color=INTRINSIC, marker="s", linestyle="--", markersize=3.5),
        ],
        ["Pressure", "Intrinsic source-prior"],
        ncol=2,
        loc="upper center",
        frameon=False,
    )
    costs = {
        model: next(
            r["incremental_stress_step_ms"]
            for r in metadata["records"]
            if r["cohort"] == "preflight-b32-v1" and r["scope"] == "live" and r["profile"] == profile
        )
        for model, profile in [("pressure", "pressure-default"), ("body_source_prior", "intrinsic")]
    }
    eggs = models["egg"]
    colors = {"pressure": PRESSURE, "body_source_prior": INTRINSIC}
    markers = {"pressure": "o", "body_source_prior": "s"}
    steps = sorted({row["step"] for row in eggs})
    for step in steps:
        paired = [
            next(row for row in eggs if row["step"] == step and row["radius_m"] == 0.006 and row["model"] == model)
            for model in colors
        ]
        axes[0, 0].plot(
            [costs[row["model"]] for row in paired],
            [row["M_vs_L"]["relative_L2"] * 100 for row in paired],
            color=".7",
            linewidth=0.55,
            zorder=0,
        )
    for model, color in colors.items():
        rows = [row for row in eggs if row["model"] == model]
        nominal = [row for row in rows if row["radius_m"] == 0.006]
        axes[0, 0].scatter(
            [costs[model]] * len(nominal),
            [row["M_vs_L"]["relative_L2"] * 100 for row in nominal],
            color=color,
            marker=markers[model],
            s=16,
        )
        axes[0, 1].scatter(
            [row["radius_m"] * 1000 for row in rows],
            [row["M_vs_L"]["relative_L2"] * 100 for row in rows],
            color=color,
            marker=markers[model],
            s=16,
        )
    axes[0, 0].set_xlabel("B32 live overhead (ms/batch)")
    axes[0, 0].set_ylabel("Egg tensor change vs L (%)")
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_ylim(0, 30)
    axes[0, 1].set_xlabel("Nominal footprint radius (mm)")
    axes[0, 1].set_ylabel("Egg tensor change vs L (%)")
    axes[0, 1].set_xticks([3, 6])
    axes[0, 1].set_xlim(2.5, 6.5)
    axes[0, 1].set_ylim(0, 30)
    controls = models["controls"]
    matrix, labels = [], []
    for geometry, name in [("ellipsoid", "Convex"), ("narrow_gap", "Gap")]:
        for radius in [0.003, 0.0024]:
            for case, short in [
                ("opposing", "Opp"),
                ("asymmetric", "Asym"),
                ("near_slip", "Slip"),
                ("multiple", "Multi"),
            ]:
                values = []
                for model in colors:
                    row = next(
                        r
                        for r in controls
                        if r["geometry"] == geometry
                        and r["radius_m"] == radius
                        and r["case"] == case
                        and r["model"] == model
                    )
                    values.append(row.get("weighted_volume_tensor_relative_L2", np.nan) * 100)
                matrix.append(values)
                labels.append(f"{name} / {short} / {radius * 1000:g}")
    array = np.array(matrix)
    cmap = matplotlib.colormaps["Greys"].copy()
    cmap.set_bad(".88")
    image = axes[1, 0].pcolormesh(
        np.ma.masked_invalid(array), cmap=cmap, norm=LogNorm(0.1, 100), edgecolors="white", linewidth=0.4
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xticks([0.5, 1.5], ["Pressure", "Intrinsic"])
    axes[1, 0].xaxis.tick_top()
    axes[1, 0].set_yticks(np.arange(16) + 0.5, labels, fontsize=5)
    axes[1, 0].tick_params(length=0)
    for index, pair in enumerate(array):
        for column, value in enumerate(pair):
            if np.isnan(value):
                label, color = "N/A", "black"
            else:
                label = f"{value:.2g}"
                color = "white" if value > 10 else "black"
            axes[1, 0].text(column + 0.5, index + 0.5, label, ha="center", va="center", fontsize=5, color=color)
    bar = fig.colorbar(image, ax=axes[1, 0], fraction=0.065, pad=0.05)
    bar.set_label("Volume tensor change (%)")
    bar.set_ticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    grouped = [("egg", "Egg"), ("ellipsoid", "Convex"), ("narrow_gap", "Gap")]
    for index, (geometry, label) in enumerate(grouped):
        rows = (
            eggs
            if geometry == "egg"
            else [r for r in controls if r["geometry"] == geometry and r["classification"] == "numerically_qualified"]
        )
        supported = (
            {(r["case"], r["radius_m"]) for r in rows if r["model"] == "pressure"} if geometry != "egg" else None
        )
        for shift, model in [(-0.12, "pressure"), (0.12, "body_source_prior")]:
            selected = [
                r
                for r in rows
                if r["model"] == model and (supported is None or (r["case"], r["radius_m"]) in supported)
            ]
            values = [r["traction_audit"]["cones"]["material"]["violating_force_fraction"] * 100 for r in selected]
            offsets = np.linspace(-0.055, 0.055, len(values))
            axes[1, 1].scatter(index + shift + offsets, values, color=colors[model], marker=markers[model], s=14)
    axes[1, 1].set_xticks(range(3), [x[1] for x in grouped])
    axes[1, 1].set_ylabel("Force norm outside local cones (%)")
    axes[1, 1].set_ylim(-2, 55)
    axes[1, 1].axhline(0, color=".6", linewidth=0.5, zorder=0)
    for ax, letter in zip(axes.flat, "abcd"):
        if letter != "c":
            style_axes(ax)
        add_panel_label(ax, letter, x=-0.22 if letter == "c" else -0.17, y=1.12)
    export(fig, args.output / "model-error-cost", [args.metadata, args.models])


def reward(args, rewards):
    fig, axes = plt.subplots(1, 2, figsize=figure_size("double", 82))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.86, wspace=0.30)
    handles, labels = [], []
    styles = [
        ("pressure", 1.0, PRESSURE, "o", True),
        ("pressure", 0.5, PRESSURE, "^", True),
        ("body_source_prior", 1.0, INTRINSIC, "o", False),
        ("body_source_prior", 0.5, INTRINSIC, "^", False),
    ]
    for ax, scope, letter in zip(axes, ["whole_episode", "lift_window"], "ab"):
        for index, condition in enumerate(rewards["conditions"]):
            for shift, (model, radius, color, marker, filled) in zip([-0.21, -0.07, 0.07, 0.21], styles):
                raw = next(
                    x["high_minus_low_peak_Pa"]
                    for x in condition[scope]["raw_differences"]
                    if x["model"] == model and x["radius_scale"] == radius
                )
                ax.scatter(
                    index + shift + np.linspace(-0.012, 0.012, 3),
                    np.array(raw) / 1000,
                    s=15,
                    marker=marker,
                    edgecolors=color,
                    facecolors=color if filled else "none",
                    linewidths=0.8,
                )
        ax.axhline(0, color=".45", linewidth=0.6)
        ax.set_xticks(range(8))
        ax.set_xlabel("Frozen condition")
        ax.set_ylabel("Peak(4 N) − peak(0.2 N) (kPa)")
        ax.set_title("Whole episode" if scope == "whole_episode" else "Lift phase 0.3–0.6", pad=6)
        style_axes(ax)
        add_panel_label(ax, letter, x=-0.17, y=1.12)
    for model, radius, color, marker, filled in styles:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                color=color,
                marker=marker,
                markerfacecolor=color if filled else "none",
                markersize=3.8,
            )
        )
        labels.append(("Pressure" if model == "pressure" else "Intrinsic") + (", 6 mm" if radius == 1 else ", 3 mm"))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.0))
    export(fig, args.output / "reward-phase", [args.rewards])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--rewards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figures", nargs="+", choices=("performance", "errors", "reward"), default=["performance", "errors", "reward"]
    )
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    for figure in args.figures:
        if figure == "performance":
            performance(args, read(args.metadata))
        elif figure == "errors":
            errors(args, read(args.metadata), read(args.models))
        else:
            reward(args, read(args.rewards))


if __name__ == "__main__":
    main()
