"""Plot the qualified shared-initialization U tail together with every original raw-peak stage."""

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

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--history-audit", type=Path, nargs=3, required=True)
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def report(run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    comparison = report(args.comparison)
    assert comparison["common_material_points"] == 75
    assert comparison["declared_spatial_axis"] == "body_at_fixed_pad"
    assert [row["body_tetrahedra"] for row in comparison["rows"]] == [845, 2339, 5669]
    histories = [report(run) for run in args.history_audit]
    assert [row["audit"] for row in comparison["independent_all_stage_audits"]] == list(map(str, args.history_audit))
    for history in histories:
        assert history["stages_checked"] == len(history["rows"]) == 744
        assert history["all_retained_stages_independent_current_friction_and_original_force_gate"]
        assert history["all_retained_stages_full_corner_quadrature_and_raw_bounds"]
        assert history["requested_initialization_coordinates_independently_matched"]
        assert all(row["independent_complete_constrained_residual_N"] <= 1e-9 for row in history["rows"])
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "claim_anchor": "shared-initialization-does-not-close-U-spatial-tail-or-history-peaks",
        "main_claim": "The same physical initializer and pad geometry leave an unresolved final tensor/peak tail; the complete original loading history has much larger, nonmonotone initializer raw maxima.",
        "material_passport": inputs,
        "data_status": "Deterministic independently audited three744-stage histories, all75 common material points; no fitted limit or uncertainty interval.",
        "panels": {
            "a": "Both adjacent final-field changes for all five original observables; point pairs compare metrics without implying an ordered continuous x variable.",
            "b": "All2232 original raw global lower/upper bounds on a linear MPa scale; zero-valued stages are retained and phase boundaries are explicit.",
        },
        "placement": "After the three-level final comparison and before discussing remaining contact/volume/geometry axes.",
        "limitations": "The final norms use the original75 equally weighted material points, not continuum volume integrals. Body geometry/mass and volume/contact trace remain coupled at fixed pad5. The three histories share physical initialization coordinates and12/604/64/64 original quasi-static stages; stage ordinal is not physical time. Shading denotes computed-field lower/upper bounds, not statistical uncertainty. Every original raw maximum remains; no singularity or continuum-pressure acceptance is inferred.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "shared-U-history-tail-20261016"})
    fig = plt.figure(figsize=mm_to_inches(183, 126))
    grid = fig.add_gridspec(2, 1, left=0.10, right=0.97, bottom=0.11, top=0.94, hspace=0.55, height_ratios=[1, 1.1])
    first, second = [fig.add_subplot(grid[index]) for index in range(2)]
    metrics = [
        ("Full tensor\n75 points", "tensor_Pa"),
        ("von Mises\n75 points", "vm_Pa"),
        ("Objective\ndisplacement", "objective_displacement_m"),
        ("Raw global\nlower peak", "raw_global_peak_lower_Pa"),
        ("Body strain\nenergy", "body_nonlinear_energy_J"),
    ]
    final_rows = []
    for offset, change, marker, color in zip(
        [-0.10, 0.10], comparison["adjacent_changes"], ["o", "s"], ["#0072B2", "#D55E00"], strict=True
    ):
        values = [
            100 * (change[key]["relative_change"] if isinstance(change[key], dict) else change[key])
            for _, key in metrics
        ]
        label = f"{change['previous_body_tetrahedra']:,} → {change['body_tetrahedra']:,} tets"
        first.plot(np.arange(len(metrics)) + offset, values, linestyle="none", marker=marker, color=color, label=label)
        final_rows.extend(
            {
                "first_tetrahedra": change["previous_body_tetrahedra"],
                "second_tetrahedra": change["body_tetrahedra"],
                "observable": label_text.replace("\n", " "),
                "relative_change_percent": value,
            }
            for (label_text, _), value in zip(metrics, values, strict=True)
        )
    first.axhline(3, color="0.5", linestyle="--", linewidth=0.7)
    first.text(
        0.995,
        3.25,
        "Initial 3% diagnostic",
        transform=first.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.5,
    )
    first.set(
        yscale="log",
        ylim=(0.65, 70),
        xlim=(-0.5, 4.5),
        xticks=np.arange(len(metrics)),
        xticklabels=[label for label, _ in metrics],
        yticks=[1, 3, 10, 30, 60],
        yticklabels=["1", "3", "10", "30", "60"],
        ylabel="Final adjacent change (%)",
    )
    first.minorticks_off()
    first.legend(loc="upper right", ncol=2, fontsize=5.5)
    history_rows = []
    upper_max = max(row["raw_global_peak_upper_Pa"] for history in histories for row in history["rows"]) / 1e6
    for history, color, style in zip(histories, ["#0072B2", "#D55E00", "#000000"], ["-", "--", ":"], strict=True):
        x = np.array([row["ordinal"] for row in history["rows"]])
        lower = np.array([row["raw_global_peak_lower_Pa"] for row in history["rows"]]) / 1e6
        upper = np.array([row["raw_global_peak_upper_Pa"] for row in history["rows"]]) / 1e6
        second.plot(x, lower, color=color, linestyle=style, label=f"{history['body_tetrahedra']:,} tets")
        second.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        history_rows.extend(
            {
                "body_tetrahedra": history["body_tetrahedra"],
                "ordinal": row["ordinal"],
                "phase": row["phase"],
                "step_in_phase": row["step"],
                "raw_global_lower_MPa": row["raw_global_peak_lower_Pa"] / 1e6,
                "raw_global_upper_MPa": row["raw_global_peak_upper_Pa"] / 1e6,
            }
            for row in history["rows"]
        )
    for boundary in [12, 616, 680]:
        second.axvline(boundary, color="0.7", linewidth=0.6, linestyle=":", zorder=0)
    for x, label in [(314, "Contact initialization"), (648, "Held"), (712, "Free")]:
        second.text(x, upper_max * 1.055, label, fontsize=5.5, ha="center", va="center")
    second.set(
        xlim=(0, 754),
        ylim=(0, upper_max * 1.12),
        xticks=[0, 200, 400, 600, 744],
        xlabel="Original quasi-static stage ordinal",
        ylabel="Raw global von Mises bound (MPa)",
    )
    second.legend(loc="upper left", ncol=3, fontsize=5.5)
    for label, axis in zip("ab", [first, second], strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        position = axis.get_position()
        fig.text(position.x0 - 0.06, position.y1 + 0.015, label, fontsize=8, fontweight="bold")
    for name, records in [
        ("final_original_observables.csv", final_rows),
        ("complete_original_raw_peak_histories.csv", history_rows),
    ]:
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    stem = args.output / "shared_u_history_tail"
    preview = stem.with_name(stem.name + "_preview").with_suffix(".png")
    save_preview(fig, preview, dpi=180)
    issues = rendered_text_qa(fig)
    (args.output / "rendered_text_qa.json").write_text(json.dumps({"issues": issues}, indent=2) + "\n")
    assert not issues, issues
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    if args.accepted_source_qa:
        inputs[str(args.accepted_source_qa)] = hashlib.sha256(args.accepted_source_qa.read_bytes()).hexdigest()
        qa = json.loads(args.accepted_source_qa.read_text())
        assert qa["visual_qa_passed"]
        assert qa["editable_svg_sha256"] == hashlib.sha256(stem.with_suffix(".svg").read_bytes()).hexdigest()
        assert qa["preview_png_sha256"] == hashlib.sha256(preview.read_bytes()).hexdigest()
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=1000)
    plt.close(fig)
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "input_sha256": inputs,
                "text_qa_issues": issues,
                "figure_size_mm": [183, 126],
                "history_stages_retained": len(history_rows),
                "publication_exports_after_visual_qa": bool(args.accepted_source_qa),
                "goal_completion": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
