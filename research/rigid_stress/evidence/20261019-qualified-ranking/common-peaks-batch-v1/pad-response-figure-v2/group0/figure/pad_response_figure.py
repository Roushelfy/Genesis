"""Plot actual normal force/closure and qualified common-location stress peaks."""

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def read(relative):
        path = args.data_root / relative
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    normal = read("pad-normal-qualification-v2/group0/qualification/result.json")
    peaks = read("pad-common-peak-qualification-v2/group0/qualification/result.json")
    assert normal["normal_reference_change_targets_passed"] and peaks["common_peak_change_targets_passed"]
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    methods = [
        ("dpr", "DPR, 2 mm", style.WONG_PALETTE["blue"], "o"),
        ("scr", "SCR, 2 mm", style.WONG_PALETTE["vermillion"], "s"),
        ("shell", "Shell, repaired", style.WONG_PALETTE["bluish_green"], "^"),
    ]
    response = []
    for group, (method, _, _, _) in enumerate(methods):
        relative = (
            "compliant-pad-fine-coarse-history-v1/group0/cycle"
            if method == "shell"
            else f"native-pad-full-source-stability-v1/group{group}/native"
        )
        start = read(relative + "/start.json")
        journal = args.data_root / relative / "steps.jsonl"
        receipts.append(file_id(journal))
        history = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(history) == 1000
        for level in range(5):
            row = next(row for row in reversed(history) if row["level"] == level)
            response.append(
                {
                    "method": method,
                    "level": level,
                    "command_n": row["command_force_n"],
                    "upper_reaction_n": row["upper_reaction_n"],
                    "lower_reaction_n": row["lower_reaction_n"],
                    "closure_m": row["compression_coordinate_m"] - 2 * start["args"]["clearance"],
                    "initial_clearance_per_plate_m": start["args"]["clearance"],
                }
            )
    native = [[r for r in response if r["method"] == method] for method in ("dpr", "scr")]
    motion_difference = max(abs(a["closure_m"] - b["closure_m"]) for a, b in zip(*native))
    assert motion_difference < 1e-10, "Do not combine distinct native mechanics in the response legend"
    reference = read("compliant-pad-joint-reference-v2/group0/reference/result.json")
    for row in reference["cases"]:
        if row["quadrature_order"] == 10:
            assert row["pad_equations_admitted"]
            lower, upper = row["footprints"]
            response.append(
                {
                    "method": "reference",
                    "level": len([r for r in response if r["method"] == "reference"]),
                    "command_n": row["load_n"],
                    "upper_reaction_n": upper["normal_force_n"],
                    "lower_reaction_n": lower["normal_force_n"],
                    "closure_m": row["plate_travel_m"],
                    "initial_clearance_per_plate_m": 0.0,
                }
            )
    with (args.output / "response.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(response)
    write_json(args.output / "peaks.json", peaks)
    write_json(
        args.output / "figure-contract.json",
        {
            "claim_anchor": "normal-pad-response-and-common-peaks",
            "main_claim": "Native rigid mechanics changes force/closure; common-location peak errors have metric/load-dependent ordering, including an unresolved0.1N Mises comparison.",
            "panels": [
                "Measured equilibrium force versus plate-gap closure",
                "Common tangent Mises peak error",
                "Common tangent tensile peak error",
            ],
            "chart_choice": "Connected equilibrium observations show response; point/range plots retain conditional uncertainty without statistical aggregation.",
            "material_passport": receipts,
            "target": "English repository supplement,183mm by75mm, editable SVG and PDF with raster preview.",
            "missing_materials": "None for these normal-case panels. Full spatial maps and friction comparison are separate deliverables.",
            "limits": "Peak intervals assume the observed refinement envelope bounds unresolved error. Response connecting lines are guides, not resolved transient paths. Plate-gap closure includes physical pad compression, not just egg deformation.",
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=style.mm_to_inches(183, 75))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.73, wspace=0.46)
    ax = axes[0]
    for method, label, color, marker in [
        ("dpr", "DPR / SCR mechanics", methods[0][2], "o"),
        ("shell", "Shell, repaired", methods[2][2], "^"),
        ("reference", "P2 reference", "0.25", "x"),
    ]:
        rows = [r for r in response if r["method"] == method]
        loading = rows[:3]
        ax.plot(
            [r["closure_m"] * 1e6 for r in loading],
            [r["upper_reaction_n"] for r in loading],
            color=color,
            marker=marker,
            markersize=3,
            label=label,
        )
        if len(rows) == 5:
            ax.plot(
                [r["closure_m"] * 1e6 for r in rows[2:]],
                [r["upper_reaction_n"] for r in rows[2:]],
                color=color,
                marker=marker,
                markerfacecolor="white",
                markersize=5,
                linestyle="--",
            )
    ax.set_xlabel("Plate-gap closure (µm)")
    ax.set_ylabel("Measured normal force (N)")
    ax.set_title("Equilibrium response", pad=10)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)
    for ax, metric, title in zip(axes[1:], ("mises", "tension"), ("Common Mises peak", "Common tensile peak")):
        for offset, (method, label, color, marker) in zip((-0.13, 0, 0.13), methods):
            rows = [
                next(
                    r for r in peaks["peaks"] if r["method"] == method and r["metric"] == metric and r["load_n"] == load
                )
                for load in (0.1, 0.3, 1.0)
            ]
            values, lower, upper = [
                np.array([r[k] for r in rows]) * 100
                for k in ("relative_error", "conditional_lower", "conditional_upper")
            ]
            assert np.isfinite([values, lower, upper]).all() and np.all(lower <= values) and np.all(values <= upper)
            ax.errorbar(
                np.arange(3) + offset,
                values,
                yerr=[values - lower, upper - values],
                linestyle="none",
                marker=marker,
                color=color,
                markerfacecolor="white",
                markeredgewidth=0.7,
                elinewidth=0.7,
                capsize=2,
                label=label,
            )
        ax.set_xticks(range(3), ["0.1", "0.3", "1.0"])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Normal load (N)")
        ax.set_ylabel("Relative peak error (%)")
        ax.set_title(title, pad=10)
    for i, ax in enumerate(axes):
        style.style_axes(ax)
        style.add_panel_label(ax, chr(97 + i), x=-0.24, y=1.17)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.99), ncol=3, fontsize=6)
    qa = style.rendered_text_qa(fig)
    paths = style.save_nature_figure(fig, args.output / "normal-response-peaks")
    shutil.copy2(args.style, args.output / "nature_mpl_style.py")
    shutil.copy2(__file__, args.output / "pad_response_figure.py")
    (args.output / "caption.md").write_text(
        "Claim anchor: normal-pad-response-and-common-peaks.\n\n"
        "Normal physical-pad benchmark: (a) measured equilibrium upper reaction versus plate-gap closure; solid lines connect loading observations0.1/0.3/1N, dashed lines and open symbols show unloading to0.3/0.1N. Lines are guides between equilibria, not transient paths. Closure subtracts the two initial22µm clearances from each trajectory's compression coordinate. The reference starts at zero clearance. Closure includes both pad compression and elastic deformation; native DPR/SCR share rigid mechanics and their overlapping response is drawn once. (b,c) errors in common tangent-plane Mises and nonnegative tensile peaks, at identical material locations with complete one-sided through-wall traces and25 angular samples per face. Whiskers propagate spatial/contact/angular refinement conditionally, not statistical confidence or certified continuum bounds.\n\n"
        "All cases use the free full egg,h=500µm,E=10GPa,nu=0.3 and physical pad stiffness1e9N/m3. Native footprint radius is the predeclared2mm; Shell is the repaired250µm contact/elastic variant; the reference is125µm/three-wall P2 at contact order10. Raw native/solid maxima remain separately reported. At0.1N the common Mises errors overlap across all three methods; the common tensile errors separate native from Shell. No universal peak ordering follows. At1N the conditional tensile comparison separates DPR from SCR only within the fixed footprint specification; the larger independent footprint sensitivity prevents a parameter-robust superiority claim.\n\n"
        "Place after the normal common-peak table. Source arrays, hashes, plotting code and style accompany the figure. Response curves have no uncertainty bars and do not alone certify continuum displacement accuracy. Friction spatial qualification and matched cost remain separate requirements.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "receipts": receipts,
            "paths": paths,
            "programmatic_qa_issues": qa,
            "native_response_difference_m": motion_difference,
            "editable_svg_text": "<text" in Path(paths["svg"]).read_text(),
            "visual_qa_required": True,
        },
    )
    assert not qa, "Inspect retained layout warnings before publication"


if __name__ == "__main__":
    main()
