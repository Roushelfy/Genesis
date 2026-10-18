"""Render completed output, sampled-memory, fresh-activation and intrusive-profile scopes."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.cost.read_text())
    configurations = data["configurations"]
    palette = WONG_PALETTE

    def select(campaign, name):
        found = [row for row in configurations if row["campaign"].endswith(campaign) and row["configuration"] == name]
        assert len(found) == 1, (campaign, name, len(found))
        return found[0]

    apply_nature_style(mpl)
    mpl.rcParams.update({"savefig.bbox": None, "font.sans-serif": ["DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=figure_size("double", 160))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.91, wspace=0.66, hspace=0.63)
    for index, ax in enumerate(axes.flat):
        style_axes(ax)
        ax.text(-0.28, 1.15, chr(97 + index), transform=ax.transAxes, weight="bold", fontsize=8, va="top")
    tables = {}

    timing = (
        ("DPR\nB4096", "full-field-native-timing-v1", "pressure-{}-s10-b4096", palette["blue"]),
        ("SCR\nB4096", "full-field-native-timing-v1", "intrinsic-{}-s10-b4096", palette["bluish_green"]),
        ("Shell S0\nB32", "shell-full-max-paired-v1", "stock-{}", palette["vermillion"]),
        ("Shell S2\nB32", "shell-full-max-paired-v1", "volume-surface2-{}", palette["orange"]),
    )
    ax = axes[0, 0]
    timing_rows = []
    for index, (label, campaign, pattern, color) in enumerate(timing):
        for output_index, (mode, marker) in enumerate((("full", "o"), ("max", "v"))):
            row = select(campaign, pattern.format(mode))
            assert len(row["raw_repeats"]) == 3
            assert all(
                r["completed_requested_history"] and r["final_output_finite_checked_after_timer"]
                for r in row["raw_repeats"]
            )
            offset = index + (output_index - 0.5) * 0.20
            median = row["median_aggregate_env_steps_per_second"]
            low, high = row["minimum_aggregate_env_steps_per_second"], row["maximum_aggregate_env_steps_per_second"]
            ax.errorbar(
                offset,
                median,
                yerr=[[median - low], [high - median]],
                marker=marker,
                color=color,
                markerfacecolor=color if mode == "full" else "white",
                capsize=2,
                linestyle="none",
            )
            for repeat, raw in enumerate(row["raw_repeats"]):
                rate = raw["aggregate_env_steps_per_second"]
                ax.plot(offset + (repeat - 1) * 0.04, rate, ".", color=color, markersize=2)
                timing_rows.append(
                    (label.replace("\n", " "), mode, repeat, rate, raw["batch_step_ms"], row["identity"]["path"])
                )
    ax.set_xticks(range(4), [row[0] for row in timing])
    ax.set(yscale="log", ylabel="Aggregate env-step/s", title="Full and max-only output", xlim=(-0.5, 3.5))
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="black", label="Full"),
            Line2D([], [], marker="v", linestyle="none", color="black", markerfacecolor="white", label="Max"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncol=2,
        frameon=False,
    )
    ax.set_title("Full and max-only output", pad=24)
    tables["full-max-repeats.csv"] = (
        ["configuration", "output", "repeat", "aggregate_env_steps_per_second", "batch_step_ms", "source"],
        timing_rows,
    )

    ax = axes[0, 1]
    memory_rows = []
    for model, prefix, label, color, marker, linestyle in (
        ("pressure", None, "DPR, s10", palette["blue"], "o", "-"),
        ("intrinsic", None, "SCR, s10", palette["bluish_green"], "D", "-"),
        ("shell", "default-full", "Shell S0, s1", palette["vermillion"], "v", "--"),
        ("shell", "fine-tuned-full", "Shell S1, s10", palette["orange"], "s", ":"),
    ):
        rows = sorted(
            [
                row
                for row in configurations
                if row["campaign"].endswith("fp64-memory-v1")
                and row["args"]["model"] == model
                and (prefix is None or row["configuration"].startswith(prefix))
            ],
            key=lambda row: row["args"]["envs"],
        )
        assert rows and all(
            row["process_memory"]["enabled"] and row["process_memory"]["peak_mib"] is not None for row in rows
        )
        batches = [row["args"]["envs"] for row in rows]
        peak = [row["process_memory"]["peak_mib"] / 1024 for row in rows]
        ax.plot(batches, peak, marker=marker, color=color, linestyle=linestyle, label=label)
        memory_rows.extend(
            (label, batch, value, row["identity"]["path"]) for row, batch, value in zip(rows, batches, peak)
        )
    ax.set(
        xscale="log",
        xlabel="Environments (B)",
        ylabel="Peak process GPU memory (GiB)",
        title="Observed memory feasibility",
        ylim=(0, 25),
    )
    ax.set_xticks([1, 64, 8192], ["1", "64", "8192"])
    ax.legend(loc="upper left", frameon=False)
    tables["sampled-memory.csv"] = (["configuration", "batch", "peak_process_gib", "source"], memory_rows)

    ax = axes[1, 0]
    activation_rows = []
    cold_names = (
        ("rigid-surface0-b1", "Rigid, B1", palette["black"]),
        ("pressure-surface0-b1", "DPR, B1", palette["blue"]),
        ("pressure-tuned-surface0-b1", "DPR tuned, B1", palette["reddish_purple"]),
        ("intrinsic-surface0-b1", "SCR, B1", palette["bluish_green"]),
        ("pressure-surface0-b4096", "DPR, B4096", palette["blue"]),
        ("intrinsic-surface0-b4096", "SCR, B4096", palette["bluish_green"]),
        ("shell-surface0-b1", "Shell S0, B1", palette["vermillion"]),
        ("shell-surface1-b1", "Shell S1, B1", palette["orange"]),
    )
    for index, (name, label, color) in enumerate(cold_names):
        row = select("cold-activation-v1", name)
        cold = row["cold_activation"]
        assert cold["full_fields_finite"]
        seconds = cold["init_build_and_first_two_steps_seconds"]
        ax.plot(seconds, index, "o", color=color)
        activation_rows.append(
            (label, seconds, cold["first_step_seconds"], cold["second_step_seconds"], row["identity"]["path"])
        )
    ax.set_yticks(range(len(cold_names)), [row[1] for row in cold_names])
    ax.invert_yaxis()
    ax.set(xlabel="Initialization, build and first two steps (s)", title="Fresh-cache activation", xlim=(0, 130))
    tables["cold-activation.csv"] = (
        ["configuration", "total_s", "first_step_s", "second_step_s", "source"],
        activation_rows,
    )

    ax = axes[1, 1]
    profile_labels = (
        "DPR B64, s10",
        "DPR B4096, s10",
        "DPR tuned B4096",
        "SCR B4096, s10",
        "Shell S0 B32, s1",
        "Shell S1 B32, s10",
    )
    assert len(data["profiles"]) == len(profile_labels)
    profile_rows = []
    for index, (profile, label) in enumerate(zip(data["profiles"], profile_labels)):
        left = 0
        for rank, (kernel, color) in enumerate(zip(profile["top_kernels"][:3], ("#444444", "#999999", "#dddddd"))):
            value = kernel["fraction_of_summed_kernel_duration"] * 100
            ax.barh(
                index,
                value,
                left=left,
                height=0.58,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=f"Rank {rank + 1}" if index == 0 else None,
            )
            left += value
            profile_rows.append(
                (
                    label,
                    rank + 1,
                    kernel["name"],
                    kernel["calls"],
                    kernel["summed_duration_us"],
                    kernel["fraction_of_summed_kernel_duration"],
                    profile["trace"]["path"],
                )
            )
        assert left <= 100 + 1e-9
    ax.set_yticks(range(len(profile_labels)), profile_labels)
    ax.invert_yaxis()
    ax.set(xlabel="Share of summed kernel duration (%)", title="Largest three individual kernels", xlim=(0, 100))
    ax.legend(loc="upper right", frameon=False)
    tables["profile-top3-kernels.csv"] = (
        [
            "configuration",
            "rank",
            "exact_kernel_name",
            "calls",
            "summed_duration_us",
            "fraction_of_summed_kernel_duration",
            "source_trace",
        ],
        profile_rows,
    )

    for filename, (headers, rows) in tables.items():
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows)
    issues = rendered_text_qa(fig)
    write_json(
        args.output / "rendered-text-qa.json", [{"figure": "output-memory-activation-profiles", "text_issues": issues}]
    )
    stem = args.output / "output-memory-activation-profiles"
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=450)
    fig.savefig(args.output / "output-memory-activation-profiles_preview.png", dpi=150)
    write_json(
        args.output / "result.json",
        {
            "kind": "explicit-full-max-memory-cold-profile-figure",
            "identity": runtime_identity(),
            "claim_anchor": "qipc-cost-scopes-20261018",
            "cost_source": file_id(args.cost),
            "figure_size_mm": (fig.get_size_inches() * 25.4).tolist(),
            "rendered_text_issues": issues,
            "reference_qualified": False,
            "scope": "Matched ten-substep complete-scene full/max timing with all three repeats; numerical native rates and finite diagnostic Shell rates retain different admission scopes. Full Shell is the last-substep field, max covers all physical substeps. Memory is sampled intrusively at tested sizes; s1/s10 are explicit and no capacity extrapolation is made. Cold activation is n1 per setting and excludes preceding imports. Top-three kernel fractions are within each intrusive trace's summed durations, which may overlap and are not scene latency. No fastest-in-error-band or physical accuracy claim.",
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
