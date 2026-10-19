"""Plot admitted normal-pad field errors with conditional resolution ranges."""

import argparse
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
    cli.add_argument("--qualification", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.qualification.read_text())
    assert data["normal_reference_change_targets_passed"] and not data["certified_continuum_error_bound"]
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    panels = [
        (
            "tangent_relative_l2",
            "Tangential stress",
            "Complete physical-volume field separates the native variants from Shell.",
        ),
        ("membrane_relative_l2", "Membrane resultant", "Overlapping conditional ranges do not separate the methods."),
        (
            "bending_relative_l2",
            "Bending resultant",
            "The Shell discrepancy is concentrated in bending rather than membrane.",
        ),
    ]
    methods = [
        ("dpr", "DPR, 2 mm", style.WONG_PALETTE["blue"], "o"),
        ("scr", "SCR, 2 mm", style.WONG_PALETTE["vermillion"], "s"),
        ("shell", "Shell, repaired", style.WONG_PALETTE["bluish_green"], "^"),
    ]
    write_json(
        args.output / "figure-contract.json",
        {
            "claim_anchor": "normal-pad-metric-specific-conditional-ranking",
            "main_claim": "On the declared physical-pad normal case, observed reference refinement separates native and repaired-Shell tangent/bending errors, while membrane and DPR/SCR comparisons overlap.",
            "material_passport": file_id(args.qualification),
            "evidence_hierarchy": "Executed source receipts, complete history admission and independent surface/wall/contact/integration changes precede this derived figure.",
            "panels": [
                {"panel": chr(97 + i), "metric": metric, "title": title, "claim": claim}
                for i, (metric, title, claim) in enumerate(panels)
            ],
            "chart_choice": "Individual points with conditional ranges at three loads; no statistical aggregation. Bars would obscure overlap; a table remains available as errors.csv.",
            "data_profile": {"methods": 3, "loads_n": [0.1, 0.3, 1.0], "metrics": 3, "points": 27, "missing": 0},
            "target": "Repository supplement, English,183mm by65mm; no venue-specific submission claim.",
            "open_decisions": "Empirical refinement is not a certified continuum bound. Sustained friction, common-spatial peak convergence and primary performance remain separate.",
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=style.mm_to_inches(183, 65))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.72, wspace=0.32)
    loads = (0.1, 0.3, 1.0)
    points = []
    for index, (ax, (metric, title, _)) in enumerate(zip(axes, panels)):
        for offset, (method, label, color, marker) in zip((-0.13, 0.0, 0.13), methods):
            rows = [
                next(
                    r for r in data["errors"] if r["method"] == method and r["metric"] == metric and r["load_n"] == load
                )
                for load in loads
            ]
            values = np.array([r["relative_error"] for r in rows]) * 100
            lower = np.array([r["conditional_lower"] for r in rows]) * 100
            upper = np.array([r["conditional_upper"] for r in rows]) * 100
            assert np.isfinite([values, lower, upper]).all()
            assert np.all(lower <= values) and np.all(values <= upper)
            ax.errorbar(
                np.arange(3) + offset,
                values,
                yerr=[values - lower, upper - values],
                linestyle="none",
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.7,
                color=color,
                elinewidth=0.7,
                capsize=2,
                label=label,
            )
            points.extend(rows)
        ax.set_xticks(range(3), ["0.1", "0.3", "1.0"])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Normal load (N)")
        ax.set_title(title, pad=10)
        style.style_axes(ax)
        style.add_panel_label(ax, chr(97 + index), x=-0.18, y=1.17)
    axes[0].set_ylabel("Relative error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=3, fontsize=6)
    qa = style.rendered_text_qa(fig)
    paths = style.save_nature_figure(fig, args.output / "normal-field-errors")
    assert "<text" in Path(paths["svg"]).read_text()
    shutil.copy2(args.style, args.output / "nature_mpl_style.py")
    shutil.copy2(args.qualification, args.output / "qualification.json")
    shutil.copy2(args.qualification.with_name("errors.csv"), args.output / "errors.csv")
    shutil.copy2(__file__, args.output / "pad_normal_figure.py")
    write_json(args.output / "plotted-values.json", points)
    (args.output / "caption.md").write_text(
        "Claim anchor: normal-pad-metric-specific-conditional-ranking.\n\n"
        "Normal physical-pad benchmark: complete tangential stress (a), membrane resultant (b) and bending resultant (c) errors at0.1,0.3 and1N. The egg remains free, h=500um,E=10GPa,nu=0.3; the reference is the125um polar/three-wall P2 mesh with contact quadrature10. Native DPR/SCR use all actual sources with the predeclared2mm footprint; Shell is the explicitly repaired250um contact/elastic variant. Symbols show measured relative errors; horizontal offsets separate methods at the same load. Whiskers propagate observed independent wall/surface/joint refinement and quadrature changes through the triangle inequality, conditional on those changes bounding reference error. They are neither statistical confidence intervals nor certified continuum bounds. All normal histories and reference-change gates pass.\n\n"
        "Interpretation: native variants cluster near7.4–8.0% tangent and10.3–10.9% bending error, while Shell is near24.0–26.0% and41.7–42.0%, respectively. Membrane errors cluster near4.3–5.8% and the conditional ranges overlap. DPR/SCR remain unresolved; no universal method order follows. There are three deterministic load cases per method, without statistical replicates in this figure. Radius sensitivity is a separate parameter uncertainty.\n\n"
        "Placement: after the normal reference qualification table. Source data, executed code and qualification receipts accompany the figure. This figure does not close sustained-friction qualification or matched primary cost/batching requirements.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "input": file_id(args.qualification),
            "style": file_id(args.style),
            "paths": paths,
            "programmatic_qa_issues": qa,
            "editable_svg_text": True,
            "visual_qa_required": True,
        },
    )
    assert not qa, "Inspect retained layout warnings before publishing"


if __name__ == "__main__":
    main()
