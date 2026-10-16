"""Plot matched near-surface tensor diagnostics and isolated coarse-mesh contact-distance sensitivity."""

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
    parser.add_argument("--spatial-probes", type=Path, required=True)
    parser.add_argument("--distance-probes", type=Path, required=True)
    parser.add_argument("--distance-fields", type=Path, required=True)
    parser.add_argument("--distance-sources", type=Path, required=True)
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, reports = {}, []
    for run in [args.spatial_probes, args.distance_probes, args.distance_fields, args.distance_sources]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        reports.append(json.loads(path.read_text()))
    spatial, distance, fields, sources = reports
    assert spatial["supplemental_probes"] == 576
    assert distance["exact_original576_probe_coordinates_and_depths_retained"]
    assert distance["qualified_baseline_final_probe_tensor_reproduced"]
    assert fields["all_geometry_and_contact_map_arrays_exactly_equal"]
    assert len(fields["rows"]) == len(sources["rows"]) == 15
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "claim_anchor": "matched-near-surface-points-expose-unresolved-spatial-contact-tail",
        "main_claim": "On the same576 supplemental material points, spatial/contact changes exceed fixed-coarse-mesh distance sensitivity; neither diagnostic establishes continuum contact acceptance.",
        "material_passport": inputs,
        "data_status": "Deterministic independently audited states at6.25ms; no fitted order, replicates or statistical uncertainty.",
        "panels": {
            "a": "All six nominal depths and both spatial and both distance-adjacent pairs on identical material points. All relative norms remain their original equal-point diagnostics.",
            "b": "Complete volume tensor, raw whole-volume lower peak, energy and complete nodal contact-force changes for both distance pairs at fixed31668-tet geometry.",
            "c": "Per-source active deformed lumped areas at all three distances; node counts are annotated and complete below-threshold forces remain in the parent audit.",
        },
        "limitations": "Depth is measured from original baseline polyhedral boundary facets, not the deformed or exact CAD surface. Original158340 probes and all raw extrema remain separately retained. Coarse-mesh regularization sensitivity and coupled spatial/geometry/contact changes cannot be subtracted as an additive error budget. Active lumped areas are discrete threshold diagnostics, not continuum contact areas or pressure convergence. All full histories remain in the qualified parents; the figure shows their6.25ms endpoints only.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "contact-depth-distance-20261016"})
    fig = plt.figure(figsize=mm_to_inches(183, 136))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.18, 1], left=0.10, right=0.97, bottom=0.12, top=0.92, hspace=0.63, wspace=0.48
    )
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    axis = axes[0]
    profiles = []
    for index, (pair, style, color) in enumerate(
        zip([(31668, 64716), (64716, 171228)], ["o-", "s-"], ["#0072B2", "#D55E00"], strict=True)
    ):
        selected = [
            row
            for row in spatial["adjacent_depth_changes"]
            if row["source_zero_means_both"] == 0 and (row["coarse_tetrahedra"], row["fine_tetrahedra"]) == pair
        ]
        selected.sort(key=lambda row: row["facet_normal_depth_m"])
        assert len(selected) == 6
        x = [row["facet_normal_depth_m"] * 1e6 for row in selected]
        y = [row["tensor_equal_point_weight_diagnostic"]["relative_l2_change"] * 100 for row in selected]
        label = f"Spatial {pair[0]:,} → {pair[1]:,}"
        axis.plot(x, y, style, color=color, label=label)
        profiles.extend(
            {"axis": "Spatial", "pair": index + 1, "nominal_depth_um": a, "relative_tensor_change_percent": b}
            for a, b in zip(x, y, strict=True)
        )
    for index, (pair, style, color) in enumerate(
        zip([(2.5e-6, 1.25e-6), (1.25e-6, 6.25e-7)], ["^--", "D:"], ["#000000", "#7A7A7A"], strict=True)
    ):
        selected = [
            row
            for row in distance["adjacent_depth_changes"]
            if row["source_zero_means_both"] == 0
            and row["step"] == 5
            and (row["first_barrier_distance_m"], row["second_barrier_distance_m"]) == pair
        ]
        selected.sort(key=lambda row: row["facet_normal_depth_m"])
        assert len(selected) == 6
        x = [row["facet_normal_depth_m"] * 1e6 for row in selected]
        y = [row["tensor_equal_point_weight_diagnostic"]["relative_l2_change"] * 100 for row in selected]
        axis.plot(x, y, style, color=color, label=f"Distance {pair[0] * 1e6:g} → {pair[1] * 1e6:g} µm")
        profiles.extend(
            {"axis": "Distance", "pair": index + 1, "nominal_depth_um": a, "relative_tensor_change_percent": b}
            for a, b in zip(x, y, strict=True)
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(0.85, 60),
        ylim=(1, 110),
        xticks=[1, 2, 5, 10, 20, 50],
        xticklabels=["1", "2", "5", "10", "20", "50"],
        yticks=[1, 3, 10, 30, 100],
        yticklabels=["1", "3", "10", "30", "100"],
        xlabel="Nominal depth below baseline facet (µm)",
        ylabel="Supplemental tensor change (%)",
    )
    axis.minorticks_off()
    axis.legend(loc="center left", ncol=2, fontsize=5.5)
    axis = axes[1]
    final = [row for row in fields["comparisons"] if row["step"] == 5]
    assert len(final) == 2
    global_data = []
    metrics = [
        (
            "Volume tensor",
            "o-",
            "#0072B2",
            [100 * row["complete_quadrature_tensor"]["relative_volume_L2_change"] for row in final],
        ),
        (
            "Raw lower peak",
            "s--",
            "#D55E00",
            [100 * row["raw_global_peak_lower_Pa"]["relative_change"] for row in final],
        ),
        ("Energy", "^:", "#000000", [100 * row["nonlinear_energy_J"]["relative_change"] for row in final]),
        (
            "Nodal contact force",
            "D-.",
            "#7A7A7A",
            [100 * row["complete_nodal_contact_force_relative_L2_change"] for row in final],
        ),
    ]
    for label, style, color, values in metrics:
        axis.plot([1, 2], values, style, color=color, label=label)
        global_data.extend(
            {"observable": label, "pair": i + 1, "change_percent": value} for i, value in enumerate(values)
        )
    axis.axhline(3, color="0.6", linestyle=":", linewidth=0.7)
    axis.set(
        xlim=(0.85, 2.15),
        ylim=(0, 4.3),
        xticks=[1, 2],
        xlabel="Adjacent distance pair",
        ylabel="Complete-field change (%)",
    )
    axis.legend(loc="upper left", ncol=2, fontsize=5)
    axis = axes[2]
    area_data = []
    final_sources = [row for row in sources["rows"] if row["step"] == 5]
    distances = [row["barrier_distance_m"] for row in fields["rows"] if row["step"] == 5]
    assert len(final_sources) == len(distances) == 3
    for source, style, color in [(1, "o-", "#0072B2"), (2, "s--", "#D55E00")]:
        samples = [row["sources"][source - 1]["normal_body_density"] for row in final_sources]
        values = [sample["active_lumped_area_m2"] * 1e12 for sample in samples]
        axis.plot(np.array(distances) * 1e6, values, style, color=color, label=f"Source {source}")
        for d, value, sample in zip(distances, values, samples, strict=True):
            count = sample["active_samples_above_1em8_N"]
            axis.annotate(
                str(count),
                (d * 1e6, value),
                xytext=(0, 6 if source == 1 else -9),
                textcoords="offset points",
                ha="center",
                fontsize=5,
            )
            area_data.append(
                {
                    "source": source,
                    "barrier_distance_um": d * 1e6,
                    "active_lumped_area_um2": value,
                    "active_samples": count,
                }
            )
    axis.set(
        xscale="log",
        xlim=(2.9, 0.54),
        ylim=(11000, 16400),
        xticks=[2.5, 1.25, 0.625],
        xticklabels=["2.5", "1.25", "0.625"],
        yticks=[11000, 13000, 15000],
        xlabel="Barrier distance (µm)",
        ylabel="Active lumped area (µm²)",
    )
    axis.minorticks_off()
    axis.legend(loc="upper right", fontsize=5.5)
    for label, axis in zip("abc", axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        position = axis.get_position()
        fig.text(position.x0 - 0.055, position.y1 + 0.025, label, fontsize=8, fontweight="bold")
    for name, records in [
        ("matched_depth_profiles.csv", profiles),
        ("complete_distance_changes.csv", global_data),
        ("discrete_source_areas.csv", area_data),
    ]:
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    stem = args.output / "contact_depth_distance"
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
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "input_sha256": inputs,
                "text_qa_issues": issues,
                "figure_size_mm": [183, 136],
                "publication_exports_after_visual_qa": bool(args.accepted_source_qa),
                "goal_completion": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
