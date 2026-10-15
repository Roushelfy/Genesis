"""Plot every audited barrier-history stage and retain exact data, scope and editable artwork."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import apply_nature_style, mm_to_inches, rendered_text_qa, save_preview
from matplotlib import font_manager

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--export-pdf", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.comparison)
    assert int((args.comparison / "returncode.txt").read_text()) == 0
    source = args.comparison / "results/summary.json"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = json.loads(source.read_text())
    assert result["stages_compared_per_level"] == len(result["rows"]) == 744 and result["levels"] == 3
    rows = result["rows"]
    assert [row["ordinal"] for row in rows] == list(range(1, 745))
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "claim_anchor": "u-barrier-history-initialization-versus-final-state",
        "main_claim": "Small final stress differences do not summarize barrier sensitivity over the entire declared initialization/loading path.",
        "material_passport": {
            "path": str(source),
            "sha256": source_sha,
            "status": "Closed three-level comparison after every-stage independent current-friction and original force-gate audits",
        },
        "panels": [
            {
                "panel": "a",
                "claim": "The maximum over744 stages occurs during initialization, and activation depends on barrier distance.",
                "plot_type": "all-stage lines and explicit activation inset",
                "x": "quasi-static stage ordinal",
                "y": "raw global corner lower/whole-volume upper bounds",
                "data": "all744 rows, every level",
            },
            {
                "panel": "b",
                "claim": "Absolute objective tensor differences vary over the full path.",
                "plot_type": "all-stage lines",
                "x": "stage ordinal",
                "y": "volume-weighted tensor absolute L2, Pa m^1.5",
                "data": "all744 rows, every pair",
            },
            {
                "panel": "c",
                "claim": "Final adjacent tensor differences increase under halving barrier distance.",
                "plot_type": "individual deterministic points",
                "x": "declared barrier comparison",
                "y": "final relative tensor L2, percent",
                "data": "final row, every pair",
            },
        ],
        "evidence_hierarchy": "Original complete fields and independent forces, then material-frame volume comparisons, then deterministic plotting; no statistical replicates or uncertainty estimates.",
        "design": "183 by140mm; editable SVG/PDF; colour-blind-safe colours with line-style redundancy; all744 stages retained without smoothing/downsampling; inset338–350 is supplemental.",
        "interpretation_limit": "Original quasi-static stages are not physical times. One coarse845-tet/pad5 barrier family does not establish spatial/contact-pressure/history or full application convergence.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    with (args.output / "all_stage_source_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stage",
                "phase",
                "phase_step",
                *[
                    f"level{i}_{key}"
                    for i in range(3)
                    for key in [
                        "barrier_distance_m",
                        "raw_global_peak_lower_Pa",
                        "raw_global_peak_upper_Pa",
                        "volume_weighted_tensor_L2_Pa_m1p5",
                        "body_nonlinear_energy_J",
                    ]
                ],
                *[
                    f"pair{i}_{key}"
                    for i in range(3)
                    for key in [
                        "objective_tensor_absolute_L2_Pa_m1p5",
                        "objective_tensor_relative_L2",
                        "finer_tensor_reference_L2_Pa_m1p5",
                    ]
                ],
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["ordinal"],
                    row["phase"],
                    row["step"],
                    *[
                        level[key]
                        for level in row["levels"]
                        for key in [
                            "barrier_distance_m",
                            "raw_global_peak_lower_Pa",
                            "raw_global_peak_upper_Pa",
                            "volume_weighted_tensor_L2_Pa_m1p5",
                            "body_nonlinear_energy_J",
                        ]
                    ],
                    *[
                        pair[key]
                        for pair in row["pairs"]
                        for key in [
                            "objective_tensor_absolute_L2_Pa_m1p5",
                            "objective_tensor_relative_L2",
                            "finer_tensor_reference_L2_Pa_m1p5",
                        ]
                    ],
                ]
            )
    apply_nature_style(matplotlib)
    font_manager.findfont("Liberation Sans", fallback_to_default=False)
    matplotlib.rcParams.update(
        {
            "font.sans-serif": ["Liberation Sans"],
            "savefig.bbox": None,
            "svg.hashsalt": "u-barrier-history-20261015",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )
    fig = plt.figure(figsize=mm_to_inches(183, 140))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.1, 1], left=0.105, right=0.97, bottom=0.14, top=0.85, hspace=0.78, wspace=0.48
    )
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    inset = a.inset_axes([0.09, 0.43, 0.30, 0.52])
    stages = np.arange(1, 745)
    level_colours = ["#000000", "#0072B2", "#E69F00"]
    pair_colours = ["#0072B2", "#E69F00", "#CC79A7"]
    styles = ["-", "--", ":"]
    for index in range(3):
        low = np.array([row["levels"][index]["raw_global_peak_lower_Pa"] for row in rows])
        high = np.array([row["levels"][index]["raw_global_peak_upper_Pa"] for row in rows])
        label = ["2.5 µm", "1.25 µm", "0.625 µm"][index]
        a.plot(stages, low / 1e6, color=level_colours[index], ls=styles[index], label=label)
        a.fill_between(stages, low / 1e6, high / 1e6, color=level_colours[index], alpha=0.12, linewidth=0)
        selected = (stages >= 338) & (stages <= 350)
        inset.plot(
            stages[selected],
            low[selected] / 1000,
            color=level_colours[index],
            ls=styles[index],
            marker=["o", "s", "^"][index],
            markersize=2,
        )
        absolute = [row["pairs"][index]["objective_tensor_absolute_L2_Pa_m1p5"] for row in rows]
        b.plot(
            stages,
            absolute,
            color=pair_colours[index],
            ls=styles[index],
            label=["2.5 to1.25", "1.25 to0.625", "2.5 to0.625"][index],
        )
        value = 100 * rows[-1]["pairs"][index]["objective_tensor_relative_L2"]
        c.plot(index, value, color=pair_colours[index], marker=["o", "s", "^"][index], linestyle="none", markersize=4)
        c.annotate(f"{value:.3f}", (index, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.28), ncol=3, title="Barrier distance")
    for axis in [a, b]:
        axis.set_xlim(0, 744)
        axis.set_xticks([0, 200, 400, 600, 744])
        axis.set_xlabel("Quasi-static stage")
        for boundary in [12.5, 616.5, 680.5]:
            axis.axvline(boundary, color="0.6", lw=0.5, ls="--", zorder=0)
    for position, label in [(314, "Initialization"), (648, "Held"), (712, "Free")]:
        a.text(position, 1.07, label, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
    a.set_ylim(0, 1.9)
    a.set_ylabel("Raw global stress bounds (MPa)")
    inset.set_xlim(338, 350)
    inset.set_ylim(bottom=0)
    inset.set_xticks([338, 344, 350])
    inset.set_ylabel("Stress (kPa)", fontsize=5)
    inset.set_xlabel("Stage", fontsize=5)
    inset.tick_params(labelsize=5)
    b.set_ylim(bottom=0)
    b.set_ylabel(r"Tensor difference (Pa m$^{1.5}$)")
    b.legend(loc="upper left", fontsize=5)
    c.set_xlim(-0.5, 2.5)
    c.set_ylim(0, 0.22)
    c.set_xticks([0, 1, 2], ["2.5 to\n1.25", "1.25 to\n0.625", "2.5 to\n0.625"])
    c.set_xlabel("Barrier comparison (µm)")
    c.set_ylabel("Final tensor difference (%)")
    for label, axis in zip("abc", [a, b, c], strict=True):
        axis.text(-0.10, 1.16, label, transform=axis.transAxes, fontweight="bold", fontsize=8)
    for axis in [a, b, c, inset]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    stem = args.output / "u_barrier_history"
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    preview = Path(save_preview(fig, args.output / "u_barrier_history_preview.png", dpi=150))
    issues = rendered_text_qa(fig)
    if args.export_pdf:
        assert args.accepted_source_qa is not None and not issues
        accepted = json.loads(args.accepted_source_qa.read_text())
        assert accepted["visual_qa_passed"] and accepted["source_summary_sha256"] == source_sha
        assert accepted["editable_svg_sha256"] == hashlib.sha256(svg.read_bytes()).hexdigest()
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=1000)
    for axis in [a, b, c, inset]:
        for line in axis.lines:
            line.set_color("black")
            line.set_markerfacecolor("black")
            line.set_markeredgecolor("black")
        for collection in axis.collections:
            collection.set_facecolor("0.5")
    save_preview(fig, args.output / "u_barrier_history_grayscale_preview.png", dpi=150)
    caption = "All744 original quasi-static stages of the845-tet/pad5 U case at2.5/1.25/0.625µm barrier distances. (a) Complete raw global corner lower and certified whole-volume upper stress bounds; the tight envelopes are bounds on the computed field, not statistical uncertainty. The inset expands stages338–350 without relocating data. Vertical separators follow12 approach,604 contact-initialization,64 held-force and64 free-body stages. (b) Absolute material-frame, volume-weighted complete quadrature tensor differences for each declared pair. (c) Final tensor differences use the finer member's norm; the third point spans the full family. All744 stages, absolute errors near zero, original force gates and source wrenches are retained. These are not physical times or a demonstrated spatial/contact/hard-contact continuum tail. No statistical error bars or replicate inference are implied."
    (args.output / "caption.md").write_text(caption + "\n")
    (args.output / "body_interpretation.md").write_text(
        "Figure role: distinguish final-state stability from the full initialization/loading path. Strongest final agreement:2.5 to1.25µm,0.06999% tensor difference. The next adjacent final difference rises to0.11381%, motivating a fourth barrier distance. Largest absolute path discrepancy for the full span occurs at stage601 during initialization; global raw peaks occur at616, before force control. The approach carries zero stress. Activation produces very large relative quotients against small reference norms, so panel b retains absolute errors; no stages are discarded. The material-frame quadrature norm is not an exact nonlinear continuum integral. Coarse contact-pressure/spatial/history convergence, full Franka matching and verified physical accuracy/cost remain open.\n"
    )
    (args.output / "alt_text.md").write_text(
        "Three panels separate the original U initialization/loading path from its final state. Raw global stress peaks near the end of initialization at about1.7MPa and falls under force control. Absolute tensor differences are largest during initialization. Final adjacent tensor differences are0.070% and0.114%, with0.183% over the full barrier span. All stages remain visible, and an inset shows barrier-dependent first contact.\n"
    )
    qa = {
        "scope": __doc__,
        "source_summary": str(source),
        "source_summary_sha256": source_sha,
        "plotted_stages": 744,
        "levels": 3,
        "size_mm": [183, 140],
        "statistical_uncertainty_claimed": False,
        "text_qa_issues": issues,
        "editable_svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
        "preview": str(preview),
        "pdf_exported_after_accepted_source": args.export_pdf,
        "font": "Liberation Sans",
        "goal_completion": False,
    }
    (args.output / "qa-summary.json").write_text(json.dumps(qa, indent=2) + "\n")
    assert not issues
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha


if __name__ == "__main__":
    main()
