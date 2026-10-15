"""Show matched-time raw stress envelopes and unsmoothed contact-node force density."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from .plot_franka_spatial import section
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.comparison)
    assert int((args.comparison / "returncode.txt").read_text()) == 0
    report = json.loads((args.comparison / "results/summary.json").read_text())
    assert len(report["rows"]) == 3 and len({r["time_s"] for r in report["rows"]}) == 1
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "matched-spatial-step26"})
    contract = {
        "claim_anchor": "unresolved-contact-field-after-resolving-more-nodes",
        "main_claim": "At a matched physical time, resolving more contact nodes does not establish stress or traction convergence.",
        "material_passport": [str(args.comparison)],
        "data_status": "Three deterministic solved states with independent original1e-9N force gates. No statistical replicates or fitted order.",
        "panels": {
            letter: {"role": role, "case": index}
            for index in range(3)
            for letter, role in [
                ("abc"[index], "Full reference-section element corner-max stress envelope"),
                ("def"[index], "Matched local contact vertices and unsmoothed normal force per lumped area"),
            ]
        },
        "units": {
            "stress": "MPa",
            "normal_force_per_area": "MPa",
            "whole_section_axes": "mm",
            "contact_axes": "micrometres",
        },
        "shared_colour_limits": {"stress_MPa": [1e-3, 3e3], "normal_force_per_area_MPa": [1e-3, 2e3]},
        "colour_role": "Cividis sequential logarithmic quantity scales; black stars mark projected raw global peaks.",
        "limitations": "One fixed time, fixed polyhedron and held initialization; no completed fine history or physical tail. Projected element envelopes are not exact section stresses. Marker area does not represent contact area.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    figure = plt.figure(figsize=style.mm_to_inches(183, 155))
    grid = figure.add_gridspec(
        2, 3, height_ratios=[1.55, 1], left=0.10, right=0.86, bottom=0.10, top=0.93, hspace=0.48, wspace=0.28
    )
    stress_norm, density_norm = LogNorm(1e-3, 3e3), LogNorm(1e-3, 2e3)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_under("#eeeeee")
    records = []
    contact_side = np.sign(report["rows"][-1]["raw_peak_reference_position_m"][1])
    assert contact_side != 0
    for index, row in enumerate(report["rows"]):
        with np.load(args.comparison / f"results/case{index:02d}.npz") as saved:
            data = {k: saved[k].copy() for k in saved.files}
        polygons, elements = section(data["vertices"][data["tetrahedra"]], 0.0)
        envelope = np.asarray(data["raw_corner_vm_Pa"].max(axis=1), dtype=float) / 1e6
        np.testing.assert_allclose(envelope.max() * 1e6, row["raw_peak_lower_Pa"], rtol=1e-14)
        positions = np.asarray(data["surface_positions_m"], dtype=float)
        active = data["active_body_contact_nodes"]
        axis = figure.add_subplot(grid[0, index])
        collection = PolyCollection(
            polygons,
            array=np.maximum(envelope[elements], 1e-30),
            cmap=cmap,
            norm=stress_norm,
            edgecolors="none",
            linewidths=0,
        )
        axis.add_collection(collection)
        axis.scatter(
            positions[active, 1] * 1000,
            positions[active, 2] * 1000,
            marker="s",
            s=9,
            facecolors="none",
            edgecolors="#D55E00",
            linewidths=0.6,
        )
        peak = np.asarray(row["raw_peak_reference_position_m"])
        axis.scatter(
            peak[1] * 1000, peak[2] * 1000, marker="*", s=28, color="black", edgecolors="white", linewidths=0.25
        )
        axis.set(
            xlim=(-25, 25),
            ylim=(-32, 35),
            xlabel="Reference y (mm)",
            title=f"{row['tetrahedra']} tetrahedra\nPeak {row['raw_peak_lower_Pa'] / 1e6:.3g} MPa",
        )
        axis.set_aspect("equal")
        if index == 0:
            axis.set_ylabel("Reference z (mm)")
        else:
            axis.tick_params(labelleft=False)
        axis.text(-0.17, 1.08, "abc"[index], transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        inset = axis.inset_axes([0.22, 0.58, 0.62, 0.24])
        inset.add_collection(
            PolyCollection(
                polygons,
                array=np.maximum(envelope[elements], 1e-30),
                cmap=cmap,
                norm=stress_norm,
                edgecolors="none",
                linewidths=0,
            )
        )
        zoom_centre = np.asarray(report["rows"][-1]["raw_peak_reference_position_m"])[1:] * 1000
        inset.set(
            xlim=(zoom_centre[0] - 0.08, zoom_centre[0] + 0.08), ylim=(zoom_centre[1] - 0.08, zoom_centre[1] + 0.08)
        )
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.text(0.5, 1.07, "0.16 mm window", transform=inset.transAxes, ha="center", fontsize=5)
        for spine in inset.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("0.5")
        selected = active[contact_side * positions[active, 1] > 0]
        assert len(selected) > 0
        if index == 0:
            centre = positions[selected][:, [0, 2]].mean(axis=0)
        offsets = (positions[selected][:, [0, 2]] - centre) * 1e6
        density = np.linalg.norm(data["normal_surface_force_N"], axis=1) / data["surface_vertex_lumped_area_m2"] / 1e6
        contact_axis = figure.add_subplot(grid[1, index])
        dots = contact_axis.scatter(
            offsets[:, 0],
            offsets[:, 1],
            c=density[selected],
            cmap=cmap,
            norm=density_norm,
            s=22,
            edgecolors="black",
            linewidths=0.35,
        )
        contact_axis.set(
            xlim=(-90, 90),
            ylim=(-90, 90),
            xticks=[-80, 0, 80],
            yticks=[-80, 0, 80],
            xlabel="Offset x (µm)",
            title=f"{'−' if contact_side < 0 else '+'}y contact: {len(selected)} {'vertex' if len(selected) == 1 else 'vertices'}",
        )
        contact_axis.set_aspect("equal")
        if index == 0:
            contact_axis.set_ylabel("Offset z (µm)")
        else:
            contact_axis.tick_params(labelleft=False)
        contact_axis.text(-0.17, 1.10, "def"[index], transform=contact_axis.transAxes, fontsize=8, fontweight="bold")
        contact_axis.spines[["top", "right"]].set_visible(False)
        records.append(
            {
                "case": index,
                "run": row["run"],
                "time_s": row["time_s"],
                "raw_global_peak_MPa": row["raw_peak_lower_Pa"] / 1e6,
                "selected_contact_side": int(contact_side),
                "selected_contact_nodes": len(selected),
                "total_active_body_nodes": len(active),
                "local_contact_nodes_outside_axes": int(np.any(abs(offsets) > 90, axis=1).sum()),
                "section_elements": len(elements),
                "all_raw_peak_reference_position_m": peak.tolist(),
            }
        )
        assert records[-1]["local_contact_nodes_outside_axes"] == 0
    stress_cb = figure.add_axes([0.90, 0.55, 0.018, 0.32])
    density_cb = figure.add_axes([0.90, 0.14, 0.018, 0.25])
    figure.colorbar(collection, cax=stress_cb, label="Corner-max von Mises (MPa)", extend="min", ticks=[1e-3, 1, 1e3])
    figure.colorbar(dots, cax=density_cb, label="Normal force / lumped area (MPa)", extend="min", ticks=[1e-3, 1, 1e3])
    stem = args.output / "matched-spatial-step26"
    for suffix in ["svg", "pdf"]:
        figure.savefig(stem.with_suffix("." + suffix), bbox_inches=None)
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches=None)
    figure.savefig(args.output / "preview.png", dpi=180, bbox_inches=None)
    assert "<text" in stem.with_suffix(".svg").read_text()
    caption = "Matched32.5ms states on the same full hollow polyhedron, physical material/mass and two-jaw protocol. Top: fixed reference x=0 sections colored by unsmoothed tetrahedron corner-max von Mises envelopes with common logarithmic limits; squares and stars project active contact nodes and the raw global peak. Insets show a common0.16x0.16mm reference y-z window centered on the finest global-peak position. Bottom: a common180x180µm tangential x-z projection of the negative-y contact, chosen from the finest global-peak side and centered on the original coarse node; fixed-size dots show contact vertices, colored by unsmoothed normal force divided by current barycentric lumped area. The full3D global search remains unrestricted. Marker size is not physical area. Density and stress use distinct labeled color scales. All raw maxima are retained. The finest state is an independently recovered step after its original25-state prefix, not a completed fine trajectory. More contact nodes do not establish spatial/contact convergence."
    metadata = {
        "claim_anchor": contract["claim_anchor"],
        "material_passport": records,
        "figure_mm": [183, 155],
        "caption": caption,
        "section_reference_x_m": 0.0,
        "manual_preview_inspected": False,
        "goal_completion": False,
    }
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "caption.md").write_text(caption + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
