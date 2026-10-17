"""Compare closed robust throughput and separate sampled-memory measurements.

Run through Slurm/gs-srun. History0 is a labeled configuration choice, while
the original and selected history4 series retain the same history setting.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history4-summary", type=Path, required=True)
    parser.add_argument("--history0-summary", type=Path, required=True)
    parser.add_argument("--style-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-pdf", action="store_true", help="Use after reviewing the editable preview.")
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    summaries = [json.loads(path.read_text()) for path in (args.history4_summary, args.history0_summary)]
    assert [summary["reference_history"] for summary in summaries] == [4, 0]
    assert summaries[0]["baseline_contract_sha256"] == summaries[1]["baseline_contract_sha256"]
    spec = importlib.util.spec_from_file_location("figure_style", args.style_helper)
    assert spec is not None and spec.loader is not None
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    series = (
        ("Original, history4", summaries[0], "baseline", "black", "o", "--"),
        ("Selected, history4", summaries[0], "packed-history-surface", "blue", "s", "-"),
        ("Selected, history0", summaries[1], "history0", "reddish_purple", "^", ":"),
    )
    records, groups, identities = [], [], set()
    for label, summary, family, colour, marker, linestyle in series:
        rates = {
            (row["width"], row["scope"]): row
            for row in summary["points"]
            if row["name"].startswith(f"{family}-four-scopes-b") and row["eligible_primary_headline"]
        }
        memory = {
            (row["width"], row["scope"]): row
            for row in summary["points"]
            if row["name"].startswith(f"{family}-memory-four-scopes-b")
            and row["job_and_wrapper_passed"]
            and row["source_closed"]
            and row["condition_pair_closed"]
            and row["same_primary_physics_and_options"]
            and row["intrusive_diagnostic"]
            and row["sampled_process_memory"] is not None
        }
        # Retain only widths whose complete four-scope measurement has closed.
        scopes = ("rigid", "live", "policy-rigid", "policy")
        rate_widths = sorted(w for w in {key[0] for key in rates} if all((w, s) in rates for s in scopes))
        memory_widths = sorted(w for w in {key[0] for key in memory} if all((w, s) in memory for s in scopes))
        assert len(rate_widths) >= 3 and len(memory_widths) >= 3, label
        manifests = {row["snapshot_manifest_sha256"] for row in (*rates.values(), *memory.values())}
        assert len(manifests) == 1, label
        identities.update((row["gpu"], row["gpu_total_bytes"]) for row in (*rates.values(), *memory.values()))
        selected = []
        for scope in ("live", "policy"):
            for width in sorted(set(rate_widths) | set(memory_widths)):
                rate = rates.get((width, scope)) if width in rate_widths else None
                observed = memory.get((width, scope)) if width in memory_widths else None
                repeats = rate["repeat_rates_valid_env_steps_per_second"] if rate else [None] * 3
                assert len(repeats) == 3
                peak = observed["sampled_process_memory"] if observed else {}
                record = {
                    "series": label,
                    "history": summary["reference_history"],
                    "scope": scope,
                    "width": width,
                    "median_valid_env_steps_per_second": rate["median_rate"] if rate else None,
                    "repeat_1": repeats[0],
                    "repeat_2": repeats[1],
                    "repeat_3": repeats[2],
                    "batch_latency_ms": rate["median_batch_latency_ms"] if rate else None,
                    "observed_peak_compute_process_bytes": peak.get("maximum_observed_compute_process_bytes"),
                    "rate_source": rate["path"] if rate else None,
                    "rate_source_sha256": rate["sha256"] if rate else None,
                    "memory_source": observed["path"] if observed else None,
                    "memory_source_sha256": observed["sha256"] if observed else None,
                    "memory_samples_sha256": peak.get("sha256"),
                    "snapshot_manifest_sha256": next(iter(manifests)),
                }
                selected.append(record)
                records.append(record)
        groups.append((label, colour, marker, linestyle, selected))
    assert len(identities) == 1, "Comparable GPU model and CUDA-visible capacity are required."
    with (args.output / "source_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    contract = {
        "claim_anchor": "practical_throughput_baseline_final_comparison",
        "main_claim": "The optimized robust workload provides greater measured throughput; history0 is a separate qualified tuning with lower observed wide-batch process memory.",
        "material_passport": [
            {
                "source": str(path),
                "sha256": digest(path),
                "kind": "closed scheduler-backed metadata",
                "reference_history": history,
            }
            for path, history in ((args.history4_summary, 4), (args.history0_summary, 0))
        ],
        "panels": {
            "a": "Live rigid+stress throughput, three repeats and observed range.",
            "b": "Policy+rigid+stress inference/rollout throughput, same repeated protocol.",
            "c": "Live process memory, separate100ms NVML observations.",
            "d": "Policy process memory, separate100ms NVML observations.",
        },
        "scope": "One physical workload; same-history code comparison and explicitly separate history tuning. Different comparable allocations; same-GPU paired gains are reported separately. No extrapolation to unmeasured widths.",
        "uncertainty": "Three raw timing repeats; ranges are not confidence intervals. Memory is a sampled lower-bound peak including setup/growth in a separate pass.",
        "size_mm": [183, 132],
        "target_stage": "repository technical report supplement",
        "open_decisions": "Visual review is required before PDF export.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    figure, axes = plt.subplots(2, 2, figsize=style.mm_to_inches(183, 132), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.84, wspace=0.30, hspace=0.26)
    for axis, letter in zip(axes.flat, "abcd"):
        style.style_axes(axis)
        style.add_panel_label(axis, letter, x=-0.17, y=1.04)
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator([32, 256, 2048, 8192, 32768]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
        axis.minorticks_off()
        axis.set_xlim(24, 45000)
    axes[0, 0].set_title("Rigid + stress", fontsize=8)
    axes[0, 1].set_title("Policy + rigid + stress", fontsize=8)
    axes[0, 0].set_ylabel("Valid throughput (env-step/s)")
    axes[1, 0].set_ylabel("Observed process VRAM (GiB)")
    for axis in axes[0]:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1000:g}k" if value else "0"))
    for axis in axes[1]:
        axis.set_xlabel("Environments")
    handles = []
    for label, colour, marker, linestyle, selected in groups:
        colour = style.WONG_PALETTE[colour]
        handles.append(Line2D([], [], color=colour, marker=marker, linestyle=linestyle, markersize=3))
        for column, scope in enumerate(("live", "policy")):
            rows = [row for row in selected if row["scope"] == scope and row["rate_source"] is not None]
            medians = [row["median_valid_env_steps_per_second"] for row in rows]
            axes[0, column].plot([row["width"] for row in rows], medians, color=colour, linestyle=linestyle)
            for row, median in zip(rows, medians):
                repeats = [row[f"repeat_{index}"] for index in (1, 2, 3)]
                axes[0, column].errorbar(
                    row["width"],
                    median,
                    yerr=[[median - min(repeats)], [max(repeats) - median]],
                    fmt="none",
                    ecolor=colour,
                    elinewidth=0.6,
                    capsize=2,
                )
                axes[0, column].scatter(
                    [row["width"]] * 3, repeats, color=colour, marker=marker, s=9, alpha=0.7, linewidths=0
                )
            rows = [row for row in selected if row["scope"] == scope and row["memory_source"] is not None]
            axes[1, column].plot(
                [row["width"] for row in rows],
                [row["observed_peak_compute_process_bytes"] / 2**30 for row in rows],
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=3,
            )
    for axis in axes.flat:
        axis.set_ylim(bottom=0)
    rate_max = max(row[f"repeat_{i}"] for row in records if row["rate_source"] for i in (1, 2, 3))
    memory_max = max(row["observed_peak_compute_process_bytes"] / 2**30 for row in records if row["memory_source"])
    assert all(axis.get_ylim()[1] >= rate_max for axis in axes[0])
    assert all(axis.get_ylim()[1] >= memory_max for axis in axes[1])
    figure.legend(
        handles,
        [group[0] for group in groups],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.98),
        columnspacing=1.4,
        handlelength=2.8,
    )
    issues = style.rendered_text_qa(figure)
    figure.savefig(args.output / "comparison.svg")
    figure.savefig(args.output / "comparison.png", dpi=450)
    figure.savefig(args.output / "comparison_preview.png", dpi=150)
    if args.export_pdf:
        assert not issues, issues
        figure.savefig(args.output / "comparison.pdf")
    assert "<text" in (args.output / "comparison.svg").read_text()
    qa = {
        "script_sha256": digest(Path(__file__)),
        "style_helper_sha256": digest(args.style_helper),
        "editable_svg_text": True,
        "all_plotted_data_within_axes": True,
        "rendered_text_issues": issues,
        "visual_inspection_pending": True,
        "pdf_exported": args.export_pdf,
        "goal_completion": False,
    }
    (args.output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    (args.output / "caption.md").write_text(
        "Baseline and selected robust egg/Franka configurations on comparable RTX PRO6000 Blackwell allocations. "
        "(a,b) Aggregate valid throughput without and with the frozen inference/rollout policy. "
        "Lines show medians of three2400-step repeats after900 warmup steps; markers retain raw repeats and whiskers their observed range. "
        "(c,d) Separate100ms NVML sampled compute-process peaks including setup/growth and one2400-step timing pass. "
        "Peaks are sampled lower bounds, not confidence intervals or native-buffer counts. "
        "The480-tet/2430-DOF full shell, body_source_prior contact law and original numerical budgets remain unchanged. "
        "Original/selected history4 isolate the code setting; selected history0 is separately qualified configuration tuning. "
        "Curves span comparable allocations, while same-GPU paired gains are reported separately. "
        "Only closed observations are shown; missing measurements remain blank and lines connect observations without extrapolating capacity. "
        "Independent process peaks do not isolate policy memory. Full RL training and fracture calibration are outside this measurement.\n"
    )
    print(json.dumps({"output": str(args.output), "records": len(records), "qa_issues": issues}))


if __name__ == "__main__":
    main()
