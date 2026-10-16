"""Show all eight-level raw finite-load maxima and the worst original adjacent observable changes."""

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

from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-run", type=Path, required=True)
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.summary_run)
    assert int((args.summary_run / "returncode.txt").read_text()) == 0
    path = args.summary_run / "results/summary.json"
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = json.loads(path.read_text())
    assert len(summary["rows"]) == 240 and len(summary["tails"]) == 30
    assert summary["initial_diagnostic_target"] == 0.03
    assert all(not row["physical_convergence_accepted"] for row in summary["tails"])
    shapes = ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]
    names = ["Ellipsoid", "Capsule", "Thin neck", "Rounded U", "Narrow gap", "Cavity"]
    metrics = [
        ("Probe tensor", "probe_tensor"),
        ("Probe von Mises", "probe_vm"),
        ("Probe displacement", "probe_displacement"),
        ("Raw peak", "raw_peak"),
        ("Strain energy", "energy"),
    ]
    columns = {
        "probe_tensor": "adjacent_probe_tensor_relative_change",
        "probe_vm": "adjacent_probe_vm_relative_change",
        "probe_displacement": "adjacent_probe_displacement_relative_change",
        "raw_peak": "adjacent_raw_peak_relative_change",
        "energy": "adjacent_energy_relative_change",
    }
    data = {(row["geometry"], row["case"], row["level"]): row for row in summary["rows"]}
    assert len(data) == 240
    envelope = []
    for shape in shapes:
        for level in range(1, 8):
            for label, key in metrics:
                values = [data[shape, case, level][columns[key]] for case in CASES]
                assert all(value is not None and np.isfinite(value) and value >= 0 for value in values)
                worst = max(values)
                envelope.append(
                    {
                        "geometry": shape,
                        "level": level,
                        "observable": label,
                        "tetrahedra": data[shape, CASES[0], level]["tetrahedra"],
                        "maximum_adjacent_change_percent": 100 * worst,
                        "maximizing_cases": [case for case, value in zip(CASES, values, strict=True) if value == worst],
                    }
                )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("all_original_eight_level_rows.csv", summary["rows"]),
        ("worst_original_adjacent_changes.csv", envelope),
    ]:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
                for row in rows
            )
    contract = {
        "claim_anchor": "eight-level-interior-probe-stability-does-not-certify-raw-global-peaks",
        "main_claim": "All original cases and eight levels retain unresolved raw maxima, even where the original probe/energy changes become small.",
        "material_passport": inputs,
        "intake": {
            "purpose": "Physical-validation report and repository supplement",
            "language": "English",
            "evidence_gaps": [],
            "target_venue": "Repository technical figures using established Nature-style defaults",
        },
        "data_profile": {
            "geometries": 6,
            "loads_per_geometry": 5,
            "levels_per_load": 8,
            "raw_rows": 240,
            "adjacent_case_rows": 210,
            "missing_adjacent_metrics": 0,
            "statistics": "Deterministic individual solves. No replicates, fitted limits, confidence intervals or inferred continuum error bounds.",
        },
        "chart_choice": {
            "primary": "Faceted raw-peak lines with every ordered mesh observation",
            "supplement": "Faceted maximum adjacent-change lines across the same five fixed cases",
            "alternatives": [
                "Last-pair point matrix loses the earlier nonmonotone history",
                "Twenty-five adjacent curves per panel would obscure cases and observables",
            ],
        },
        "panels": [
            {
                "panel": chr(97 + i),
                "geometry": shape,
                "title": name,
                "raw_role": "Retain the full raw whole-domain peak history for this distinct geometry",
                "change_role": "Show the worst original load for each observable without asserting convergence",
            }
            for i, (shape, name) in enumerate(zip(shapes, names, strict=True))
        ],
        "export_targets": [
            "editable SVG",
            "PDF after accepted exact SVG/preview hashes",
            "1000dpi RGB PNG after acceptance",
            "colour and monochrome previews",
        ],
        "figure_size_mm": [183, 170],
        "body_text_handoff": "Use the raw curves to retain nonmonotonicity and the change panels to distinguish probe/energy stability from global extrema. The U diagnostic is strongest, while smooth convex, narrow-gap and cavity raw peaks retain important changes. Eight deterministic levels do not by themselves establish a tail. Local geometry/volume/contact controls and resolved physical comparisons remain necessary.",
        "limitations": "CAD geometry and volume/surface approximation remain coupled. Probe norms use the original physical points and pairwise validity intersections, not volume integrals. Maximum across the five cases is an explicitly supplemental worst-case diagnostic, not uncertainty. All original rows, outside masks, raw peak positions and energies remain in the CSV and reverified input. No zero is dropped by the symmetric-log change axis. The3percent line is the original initial diagnostic, not continuum acceptance. The referenced shared notation file is absent locally; figure labels use plain English/units and standard log ticks, with direct vector and rendered text QA.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    font_contract = apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "finite-eight-level-references-20261016"})
    palette = ["#0072B2", "#D55E00", "#333333", "#CC79A7", "#009E73"]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v"]
    qa = json.loads(args.accepted_source_qa.read_text()) if args.accepted_source_qa else None
    if qa is not None:
        assert qa["visual_qa_passed"]
        inputs[str(args.accepted_source_qa)] = hashlib.sha256(args.accepted_source_qa.read_bytes()).hexdigest()
    results = []
    for kind, stem_name in [("raw", "finite_raw_peaks_eight_levels"), ("change", "finite_worst_adjacent_eight_levels")]:
        fig = plt.figure(figsize=mm_to_inches(183, 170))
        grid = fig.add_gridspec(
            3, 2, left=0.10, right=0.98, bottom=0.095, top=0.92 if kind == "raw" else 0.875, hspace=0.52, wspace=0.40
        )
        axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(6)]
        for i, (ax, shape, name) in enumerate(zip(axes, shapes, names, strict=True)):
            if kind == "raw":
                for j, case in enumerate(CASES):
                    rows = [data[shape, case, level] for level in range(8)]
                    ax.plot(
                        [row["tetrahedra"] for row in rows],
                        [row["raw_global_peak_Pa"] / 1000 for row in rows],
                        label=case.replace("_", " ").capitalize(),
                        color=palette[j],
                        linestyle=styles[j],
                        marker=markers[j],
                        markersize=2.6,
                        markerfacecolor="white",
                        markeredgewidth=0.6,
                        linewidth=0.8,
                    )
                ax.set_ylim(bottom=0)
            else:
                for j, (label, _) in enumerate(metrics):
                    rows = [row for row in envelope if row["geometry"] == shape and row["observable"] == label]
                    ax.plot(
                        [row["tetrahedra"] for row in rows],
                        [row["maximum_adjacent_change_percent"] for row in rows],
                        label=label,
                        color=palette[j],
                        linestyle=styles[j],
                        marker=markers[j],
                        markersize=2.6,
                        markerfacecolor="white",
                        markeredgewidth=0.6,
                        linewidth=0.8,
                    )
                ax.axhline(3, color="0.55", linestyle="--", linewidth=0.65, label="Initial 3% diagnostic")
                ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
                ax.set_ylim(bottom=0)
            ax.set_xscale("log")
            lower = min(row["tetrahedra"] for row in summary["rows"]) / 1.2
            upper = max(row["tetrahedra"] for row in summary["rows"]) * 1.2
            ax.set_xlim(lower, upper)
            exponents = np.arange(np.ceil(np.log10(lower)), np.floor(np.log10(upper)) + 1)
            ax.set_xticks(10.0**exponents)
            if kind == "change":
                upper_change = (
                    max(row["maximum_adjacent_change_percent"] for row in envelope if row["geometry"] == shape) * 1.15
                )
                ax.set_ylim(0, upper_change)
                ticks = [value for value in [0, 0.1, 1, 3, 10, 30, 100, 300, 1000] if value <= upper_change]
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(value) for value in ticks])
            ax.set_title(name, loc="left", fontsize=7, pad=4)
            ax.text(-0.22, 1.08, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
            if i % 2 == 0:
                ax.set_ylabel("Raw peak von Mises (kPa)" if kind == "raw" else "Max. adjacent change (%)")
            if i // 2 == 2:
                ax.set_xlabel("Tetrahedra" if kind == "raw" else "Finer mesh tetrahedra")
            else:
                ax.tick_params(labelbottom=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=5 if kind == "raw" else 3,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            fontsize=6,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.2,
        )
        stem = args.output / stem_name
        preview = stem.with_name(stem.name + "_preview.png")
        save_preview(fig, preview, dpi=160)
        issues = rendered_text_qa(fig)
        (args.output / f"{stem_name}_text_qa.json").write_text(json.dumps({"issues": issues}, indent=2) + "\n")
        assert not issues, (stem_name, issues)
        fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
        svg_sha = hashlib.sha256(stem.with_suffix(".svg").read_bytes()).hexdigest()
        preview_sha = hashlib.sha256(preview.read_bytes()).hexdigest()
        if qa is not None:
            assert qa["figures"][stem_name]["editable_svg_sha256"] == svg_sha
            assert qa["figures"][stem_name]["preview_png_sha256"] == preview_sha
            fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
            fig.savefig(stem.with_suffix(".png"), dpi=1000)
        for ax in axes:
            for line in ax.get_lines():
                line.set_color("0.25")
        for legend in fig.legends:
            for handle in legend.legend_handles:
                handle.set_color("0.25")
        save_preview(fig, stem.with_name(stem.name + "_monochrome_preview.png"), dpi=160)
        plt.close(fig)
        results.append(
            {
                "stem": stem_name,
                "text_qa_issues": issues,
                "editable_svg_sha256": svg_sha,
                "preview_png_sha256": preview_sha,
                "exports_after_exact_visual_source_qa": qa is not None,
            }
        )
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "input_sha256": inputs,
                "font_contract": font_contract,
                "figures": results,
                "full_original_rows_retained": 240,
                "worst_adjacent_case_observable_rows": 210,
                "goal_completion": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
