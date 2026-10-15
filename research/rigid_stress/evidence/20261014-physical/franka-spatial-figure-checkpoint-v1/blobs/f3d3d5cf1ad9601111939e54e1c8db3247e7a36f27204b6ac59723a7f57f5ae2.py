"""Plot matched reference-geometry sections of unaveraged tetrahedral corner maxima."""

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from .summarize_fine_references import closed_source


def section(corners, plane):
    polygons, selected = [], []
    for element, vertices in enumerate(corners):
        distance = vertices[:, 0] - plane
        points = [v[1:] for v, d in zip(vertices, distance) if abs(d) <= 1e-13]
        for first, second in combinations(range(4), 2):
            a, b = distance[first], distance[second]
            if a * b < 0:
                point = vertices[first] + a / (a - b) * (vertices[second] - vertices[first])
                points.append(point[1:])
        unique = []
        for point in points:
            if all(np.linalg.norm(point - old) > 1e-12 for old in unique):
                unique.append(point)
        if len(unique) < 3:
            continue
        polygon = np.array(unique)
        relative = polygon - polygon.mean(axis=0)
        polygon = polygon[np.argsort(np.arctan2(relative[:, 1], relative[:, 0]))]
        area = abs(np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])) / 2
        if area <= 1e-20:
            continue
        polygons.append(polygon * 1000)
        selected.append(element)
    return polygons, np.array(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=2, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["svg.hashsalt"] = "franka-spatial-20261015"
    data = []
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        report = json.loads((run / "results/summary.json").read_text())
        with np.load(run / "results/reference_geometry.npz") as mesh:
            vertices, tets = mesh["vertices"], mesh["tetrahedra"]
        with np.load(run / "results/body_fields.npz") as saved:
            vm = saved["vm_Pa"]
        with np.load(run / f"results/step{config['steps']:04d}.npz") as saved:
            count = len(saved["body_component_permutation"])
            force = saved["contact_force_N"][:count].reshape(-1, 3)
        # Contact coordinates use the scalar P2 geometry, in the saved interleaved ordering.
        with np.load(run / "results/step0001.npz") as saved:
            assert len(force) * 3 == len(saved["body_component_permutation"])
        from .skfem_reference import SkfemElasticReference

        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000)
        corners = vertices[tets]
        element, corner = np.unravel_index(vm.argmax(), vm.shape)
        peak = corners[element, corner]
        active = reference.coordinates[np.linalg.norm(force, axis=1) > 1e-8]
        data.append((run, corners, vm.max(axis=1) / 1000, peak, active, report))
    plane = data[0][3][0]
    figure, axes = plt.subplots(1, 2, figsize=style.mm_to_inches(183, 120), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.095, right=0.86, bottom=0.18, top=0.91, wspace=0.18)
    norm = Normalize(0, 240)
    records = []
    for index, (axis, values) in enumerate(zip(axes, data)):
        run, corners, vm, peak, active, report = values
        polygons, selected = section(corners, plane)
        collection = PolyCollection(
            polygons, array=vm[selected], cmap="cividis", norm=norm, edgecolors="none", linewidths=0
        )
        axis.add_collection(collection)
        axis.scatter(
            active[:, 1] * 1000,
            active[:, 2] * 1000,
            marker="s",
            s=10,
            facecolors="none",
            edgecolors="#D55E00",
            linewidths=0.6,
            label="Projected contact nodes",
        )
        axis.scatter(
            peak[1] * 1000,
            peak[2] * 1000,
            marker="*",
            s=35,
            color="black",
            edgecolors="white",
            linewidths=0.35,
            label="Global lower peak",
        )
        axis.set_aspect("equal")
        axis.set_xlim(-25, 25)
        axis.set_ylim(-32, 35)
        axis.set_xlabel("Reference y (mm)")
        axis.set_title(f"{len(corners)} tetrahedra; peak {report['body_peak_lower_Pa'] / 1000:.1f} kPa")
        axis.text(-0.13, 1.045, "ab"[index], transform=axis.transAxes, fontweight="bold", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        records.append(
            {
                "run": str(run),
                "section_elements": len(selected),
                "projected_contact_nodes": len(active),
                "global_lower_peak_Pa": report["body_peak_lower_Pa"],
                "global_upper_peak_Pa": report["body_peak_upper_Pa"],
                "peak_reference_position_m": peak.tolist(),
            }
        )
    axes[0].set_ylabel("Reference z (mm)")
    cax = figure.add_axes([0.9, 0.22, 0.022, 0.63])
    figure.colorbar(collection, cax=cax, label="Tetrahedron corner-max von Mises (kPa)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.47, 0.035), ncol=2)
    stem = args.output / "franka-first-spatial"
    for suffix in ["svg", "pdf"]:
        figure.savefig(stem.with_suffix("." + suffix), bbox_inches=None)
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches=None)
    figure.savefig(args.output / "preview.png", dpi=180, bbox_inches=None)
    assert "<text" in stem.with_suffix(".svg").read_text()
    output = {
        "claim_anchor": "franka-first-spatial-field-sensitivity",
        "material_passport": records,
        "plane_reference_x_m": float(plane),
        "colour_limits_kPa": [0, 240],
        "figure_mm": [183, 120],
        "svg_text_editable": True,
        "caption": "Reference-geometry sections of the full3D fixed-polyhedron Panda inertial pilot at0.1s. Each intersected tetrahedron is colored by its unaveraged four-corner von Mises maximum; this is an element envelope, not interpolated cut-plane stress or a smoothed nodal field. Shared0–240kPa limits preserve the measured difference. Squares project all body contact nodes with force norm above1e-8N; stars mark the independently audited global corner lower peak. Whole-volume nonlinear peak upper bounds remain separate. Two levels do not establish convergence.",
        "manual_preview_inspected": False,
        "goal_completion": False,
    }
    (args.output / "figure_metadata.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
