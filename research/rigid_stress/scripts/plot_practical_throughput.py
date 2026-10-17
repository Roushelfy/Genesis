"""Plot closed four-scope measurements and separately observed process memory.

Run rendering under Slurm. The style helper is an explicitly supplied, archived
dependency; the input is the metadata-only throughput summarizer's JSON.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--style-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-family", default="baseline-four-scopes-b")
    parser.add_argument("--memory-family", default="baseline-memory-four-scopes-b")
    parser.add_argument("--series-label", default="baseline", help="Source/configuration label for the caption.")
    parser.add_argument("--export-pdf", action="store_true", help="Use after the editable preview has passed review.")
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    source = args.summary.read_bytes()
    summary = json.loads(source)
    spec = importlib.util.spec_from_file_location("figure_style", args.style_helper)
    assert spec is not None and spec.loader is not None
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    scopes = ("rigid", "live", "policy-rigid", "policy")
    labels = ("Rigid", "Rigid + stress", "Policy + rigid", "Policy + rigid + stress")
    colours = ("black", "blue", "orange", "reddish_purple")
    markers = ("o", "s", "^", "D")
    linestyles = ("-", "-", "--", "--")
    rates = {
        (row["width"], row["scope"]): row
        for row in summary["points"]
        if row["name"].startswith(args.rate_family) and row["eligible_primary_headline"]
    }
    memory = {
        (row["width"], row["scope"]): row
        for row in summary["points"]
        if row["name"].startswith(args.memory_family)
        and row["job_and_wrapper_passed"]
        and row["source_closed"]
        and row["condition_pair_closed"]
        and row["same_primary_physics_and_options"]
        and row["intrusive_diagnostic"]
        and row["sampled_process_memory"] is not None
    }
    rate_widths = sorted(
        width
        for width in {key[0] for key in rates}
        if all((width, scope) in rates for scope in scopes)
    )
    memory_widths = sorted(
        width for width in {key[0] for key in memory} if all((width, scope) in memory for scope in scopes)
    )
    assert len(rate_widths) >= 2 and len(memory_widths) >= 2
    identities = {
        (row["snapshot_manifest_sha256"], row["gpu"], row["gpu_total_bytes"])
        for row in (*rates.values(), *memory.values())
    }
    assert len(identities) == 1, "Each figure must use one source manifest and comparable GPU allocations."
    widths = sorted(set(rate_widths) | set(memory_widths))
    # Keep every observation; sparse ticks leave the dense capacity end readable.
    tick_widths = [width for width in widths if width == widths[0] or width & (width - 1) == 0]
    if widths[-1] not in tick_widths:
        if len(tick_widths) > 1 and widths[-1] / tick_widths[-1] < 2:
            tick_widths.pop()
        tick_widths.append(widths[-1])
    rows = []
    for width in widths:
        for scope in scopes:
            rate = rates.get((width, scope)) if width in rate_widths else None
            observed = memory.get((width, scope)) if width in memory_widths else None
            repeats = rate["repeat_rates_valid_env_steps_per_second"] if rate else [None, None, None]
            assert len(repeats) == 3
            peak = observed["sampled_process_memory"] if observed else {}
            if observed:
                assert peak["maximum_observed_compute_process_bytes"] is not None
            rows.append(
                {
                    "width": width,
                    "scope": scope,
                    "median_valid_env_steps_per_second": rate["median_rate"] if rate else None,
                    "repeat_min": min(repeats) if rate else None,
                    "repeat_max": max(repeats) if rate else None,
                    "repeat_1": repeats[0],
                    "repeat_2": repeats[1],
                    "repeat_3": repeats[2],
                    "observed_peak_compute_process_bytes": peak.get("maximum_observed_compute_process_bytes"),
                    "memory_sampling_interval_ms": peak.get("sampling_interval_ms"),
                    "memory_samples_at_observed_maximum": peak.get("samples_at_observed_maximum"),
                    "headline_native_buffer_bytes_at_end": rate["native_buffer_bytes_at_end"] if rate else None,
                    "memory_pass_native_buffer_bytes_at_end": observed["native_buffer_bytes_at_end"] if observed else None,
                    "rate_source": rate["path"] if rate else None,
                    "rate_source_sha256": rate["sha256"] if rate else None,
                    "memory_source": observed["path"] if observed else None,
                    "memory_source_sha256": observed["sha256"] if observed else None,
                    "memory_samples_sha256": peak.get("sha256"),
                    "snapshot_manifest_sha256": next(iter(identities))[0],
                }
            )
    with (args.output / "source_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contract = {
        "claim_anchor": "practical_throughput_closed_scaling",
        "main_claim": "The closed robust workload scales sublinearly while observed process memory grows with batch width.",
        "scope": "Each panel retains every width with all four closed scopes for that measurement. Missing observations are blank, not zero or interpolated. Both panels use one source manifest and comparable GPU allocations.",
        "material_passport": {
            "source": str(args.summary),
            "sha256": hashlib.sha256(source).hexdigest(),
            "kind": "recorded measurements",
            "analysis_status": "source/scheduler/wrapper closed",
            "uncertainty": "Rates: three repeats, raw values and observed range. Memory: one separately sampled pass,100ms lower-bound peak; no confidence interval.",
        },
        "panels": [
            {
                "panel": "a",
                "claim": "Aggregate valid throughput gains diminish over the shown batch range.",
                "data": "source_data.csv",
                "plot_type": "log-log line with individual repeats and range",
                "x": "width",
                "y": "valid_env_steps_per_second",
                "group": "scope",
            },
            {
                "panel": "b",
                "claim": "Observed process VRAM increases with width and enabled stress recovery.",
                "data": "source_data.csv",
                "plot_type": "line with observed points",
                "x": "width",
                "y": "observed_peak_compute_process_bytes / 2**30",
                "group": "scope",
            },
        ],
        "widths": widths,
        "rate_widths": rate_widths,
        "memory_widths": memory_widths,
        "tick_widths": tick_widths,
        "series_label": args.series_label,
        "size_mm": [183, 86],
        "target_stage": "technical report draft",
        "open_decisions": "Update with the final selected implementation and capacity observations before claiming a final plateau.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=style.mm_to_inches(183, 86))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.82, wspace=0.34)
    for axis, letter in zip(axes, ("a", "b")):
        style.style_axes(axis)
        style.add_panel_label(axis, letter, x=-0.18, y=1.08)
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator(tick_widths))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
        axis.minorticks_off()
        axis.set_xlim(widths[0] / 1.5, widths[-1] * 1.4)
        axis.set_xlabel("Environments")
    handles = []
    for scope, label, colour, marker, linestyle in zip(scopes, labels, colours, markers, linestyles):
        selected = [row for row in rows if row["scope"] == scope and row["rate_source"] is not None]
        observed = [row for row in rows if row["scope"] == scope and row["memory_source"] is not None]
        colour = style.WONG_PALETTE[colour]
        medians = [row["median_valid_env_steps_per_second"] for row in selected]
        axes[0].plot(rate_widths, medians, color=colour, linestyle=linestyle, label=label)
        handles.append(Line2D([], [], color=colour, linestyle=linestyle, marker=marker, markersize=3))
        axes[0].errorbar(
            rate_widths,
            medians,
            yerr=[
                [median - row["repeat_min"] for median, row in zip(medians, selected)],
                [row["repeat_max"] - median for median, row in zip(medians, selected)],
            ],
            fmt="none",
            ecolor=colour,
            elinewidth=0.6,
            capsize=2,
        )
        for row in selected:
            axes[0].scatter(
                [row["width"]] * 3,
                [row[f"repeat_{index}"] for index in (1, 2, 3)],
                s=8,
                marker=marker,
                color=colour,
                alpha=0.65,
                linewidths=0,
            )
        axes[1].plot(
            memory_widths,
            [row["observed_peak_compute_process_bytes"] / 2**30 for row in observed],
            color=colour,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
        )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: {1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M"}.get(value, ""))
    )
    axes[0].minorticks_off()
    axes[0].set_ylim(1e3, 1.1e6)
    axes[0].set_ylabel("Valid throughput (env-step/s)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Observed peak compute VRAM (GiB)")
    figure.legend(
        handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 0.995), handlelength=2.8, columnspacing=2.0
    )
    issues = style.rendered_text_qa(figure)
    figure.savefig(args.output / "scaling.svg")
    figure.savefig(args.output / "scaling.png", dpi=450)
    figure.savefig(args.output / "scaling_preview.png", dpi=150)
    if args.export_pdf:
        assert not issues, issues
        figure.savefig(args.output / "scaling.pdf")
    svg = (args.output / "scaling.svg").read_text()
    assert "<text" in svg, "SVG must retain editable text."
    qa = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_helper_sha256": hashlib.sha256(args.style_helper.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(source).hexdigest(),
        "editable_svg_text": True,
        "rendered_text_issues": issues,
        "visual_inspection_pending": True,
        "pdf_exported": args.export_pdf,
        "goal_completion": False,
    }
    (args.output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    caption = (
        f"Closed robust egg/Franka {args.series_label} on RTX PRO6000 Blackwell. "
        "(a) Aggregate valid environment transitions per second: lines show medians of three 2400-step repeats; "
        "points retain individual repeats and whiskers their observed range, after 900 warmup steps. "
        "(b) Maximum observed compute-process VRAM from separate 100 ms NVML observations, including setup/warmup "
        "and one 2400-step measurement pass per scope. These are sampled lower bounds, not native-buffer byte counts "
        "or confidence intervals. Policy scopes include inference and rollout. The same eight-condition bank, "
        "480-tet full shell, body_source_prior law and numerical budgets are retained. "
        "Each panel retains its own closed widths; missing observations are left blank. Separate process peaks "
        "include allocator behavior and sampling, so differences do not isolate policy-memory costs. "
        "The rightmost point does not establish the final capacity limit."
    )
    (args.output / "caption.md").write_text(
        caption + "\n\n"
        "Interpretation: Stress-enabled throughput grows more slowly than rigid-only throughput over the shown "
        "range, while observed memory grows. The dense information is at the smaller widths; only a few wide "
        "points are available. This supports further large-batch/capacity measurement, not extrapolation to an "
        "unmeasured plateau or full RL-training throughput.\n"
    )
    print(json.dumps({"output": str(args.output), "widths": widths, "qa_issues": issues}, indent=2))


if __name__ == "__main__":
    main()
