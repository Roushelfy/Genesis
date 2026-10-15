"""Show matched local-reference stress sections and unresolved refinement curves."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from .plot_franka_spatial import section
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupled", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    matplotlib.rcParams["svg.hashsalt"] = "franka-local-20261015"
    reports = []
    for run in [args.coupled, args.frozen]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        reports.append(json.loads((run / "results/summary.json").read_text()))
    coupled, frozen = reports
    assert len(coupled["cases"]) == len(frozen["cases"]) == 3
    (args.output / "figure_contract.json").write_text(
        json.dumps(
            {
                "claim_anchor": "unresolved-local-panda-stress-concentration",
                "main_claim": "Local stress and raw peak remain unresolved despite stable contact resultants.",
                "material_passport": [str(args.coupled), str(args.frozen)],
                "panels": {
                    "a": "Coarse stress location and raw peak",
                    "b": "First local stress redistribution",
                    "c": "Finer stress concentration",
                    "d": "Raw global peak growth on both discretization paths",
                    "e": "Common-material-point tensor tail remains above declared diagnostic",
                },
                "data_status": "Three closed deterministic discretizations and complete original force gates per path; no fitted order or statistical uncertainty.",
                "exports": ["editable SVG", "embedded-font PDF", "450dpi PNG", "180dpi preview"],
                "limitations": "Fixed polyhedron; element envelopes and projected locations; no physical convergence, singularity or full-arm acceptance.",
            },
            indent=2,
        )
        + "\n"
    )
    figure = plt.figure(figsize=style.mm_to_inches(183, 160))
    grid = figure.add_gridspec(
        2, 6, height_ratios=[1.45, 1], left=0.09, right=0.88, bottom=0.11, top=0.93, hspace=0.56, wspace=1.5
    )
    norm = Normalize(0, 3.2)
    records = []
    plane = coupled["cases"][0]["final_lower_peak_reference_position_m"][0]
    for index, case in enumerate(coupled["cases"]):
        axis = figure.add_subplot(grid[0, 2 * index : 2 * index + 2])
        run = Path(case["run"])
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        with np.load(run / "results/reference_geometry.npz") as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
        with np.load(run / "results/body_fields.npz") as saved:
            raw = saved["vm_Pa"]
            vm = raw.max(axis=1) / 1e6
        np.testing.assert_allclose(raw.max(), case["final_peak_lower_Pa"], rtol=0, atol=1e-8)
        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000)
        with np.load(run / f"results/step{config['steps']:04d}.npz") as saved:
            force = saved["contact_force_N"][: reference.basis.N].reshape(-1, 3)
        contacts = reference.coordinates[np.linalg.norm(force, axis=1) > 1e-8]
        polygons, elements = section(vertices[tets], plane)
        collection = PolyCollection(
            polygons, array=vm[elements], cmap="cividis", norm=norm, edgecolors="none", linewidths=0
        )
        axis.add_collection(collection)
        axis.scatter(
            contacts[:, 1] * 1000,
            contacts[:, 2] * 1000,
            marker="s",
            s=14,
            facecolors="none",
            edgecolors="#D55E00",
            linewidths=0.65,
            label="Projected contact nodes",
        )
        peak = np.array(case["final_lower_peak_reference_position_m"])
        axis.scatter(
            peak[1] * 1000,
            peak[2] * 1000,
            marker="*",
            s=35,
            color="black",
            edgecolors="white",
            linewidths=0.35,
            label="Projected global peak",
        )
        axis.set(
            xlim=(-25, 25),
            ylim=(-32, 35),
            xlabel="Reference y (mm)",
            title=f"{len(tets)} tetrahedra\nPeak {raw.max() / 1e6:.3f} MPa",
        )
        axis.set_aspect("equal")
        if index == 0:
            axis.set_ylabel("Reference z (mm)")
        else:
            axis.tick_params(labelleft=False)
        axis.text(-0.16, 1.12, "abc"[index], transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        records.append(
            {
                "run": str(run),
                "tetrahedra": len(tets),
                "section_elements": len(elements),
                "projected_contact_nodes": len(contacts),
                "raw_global_peak_Pa": float(raw.max()),
                "peak_reference_position_m": peak.tolist(),
            }
        )
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.48, 0.51), ncol=2, fontsize=5.5)
    colour_axis = figure.add_axes([0.92, 0.565, 0.018, 0.31])
    figure.colorbar(collection, cax=colour_axis, label="Element corner-max von Mises (MPa)")
    peak_axis, error_axis = figure.add_subplot(grid[1, :3]), figure.add_subplot(grid[1, 3:])
    for report, colour, marker, linestyle, label in [
        (coupled, "#0072B2", "o", "-", "Volume + contact"),
        (frozen, "#D55E00", "x", "--", "Fixed contact"),
    ]:
        levels = [r["local_levels"] for r in report["cases"]]
        peaks = [r["final_peak_lower_Pa"] / 1e6 for r in report["cases"]]
        errors = [100 * r["common_time_changes"]["tensor"]["relative_l2_change"] for r in report["comparisons"]]
        peak_axis.plot(
            levels,
            peaks,
            marker=marker,
            linestyle=linestyle,
            color=colour,
            markersize=4.5,
            markerfacecolor="none",
            label=label,
        )
        error_axis.plot(
            [0, 1],
            errors,
            marker=marker,
            linestyle=linestyle,
            color=colour,
            markersize=4.5,
            markerfacecolor="none",
            label=label,
        )
    peak_axis.set(xlabel="Local refinement rounds", ylabel="Raw global peak (MPa)", xticks=[0, 3, 6], ylim=(0, 3.25))
    peak_axis.legend(loc="upper left", fontsize=5.5)
    error_axis.axhline(3, color="0.4", linestyle=":", linewidth=0.7, label="3% diagnostic target")
    error_axis.set(
        xlabel="Adjacent refinement rounds",
        ylabel="Full tensor change (%)",
        xticks=[0, 1],
        xticklabels=["0 to 3", "3 to 6"],
        ylim=(0, 50),
    )
    error_axis.legend(loc="lower left", fontsize=5.5)
    for axis, letter in [(peak_axis, "d"), (error_axis, "e")]:
        axis.text(-0.13, 1.06, letter, transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    stem = args.output / "franka-local-refinement"
    for suffix in ["svg", "pdf"]:
        figure.savefig(stem.with_suffix("." + suffix), bbox_inches=None)
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches=None)
    figure.savefig(args.output / "preview.png", dpi=180, bbox_inches=None)
    assert "<text" in stem.with_suffix(".svg").read_text()
    metadata = {
        "claim_anchor": "unresolved-local-panda-stress-concentration",
        "material_passport": records,
        "comparison_runs": [str(args.coupled), str(args.frozen)],
        "figure_mm": [183, 160],
        "colour_limits_MPa": [0, 3.2],
        "section_plane_reference_x_m": plane,
        "sample_count": "Three deterministic discretizations per path; no statistical replicates or fitted order.",
        "caption": "Matched full3D reference-geometry sections at0.1s. Maps use shared0–3.2MPa limits and unaveraged tetrahedron corner-maximum envelopes, not smoothed nodal fields or exact cut-plane stress. Squares and stars project active contact nodes and the global lower peak. Curves retain original raw extrema and fixed2400-material-point/10-time tensor diagnostics. Nearly coincident coupled/fixed-contact curves show unresolved volume sensitivity; the3% line is the declared initial diagnostic, not acceptance. Two active load nodes persist. No convergence or singularity claim follows.",
        "manual_preview_inspected": False,
        "goal_completion": False,
    }
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
