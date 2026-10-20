"""Show complete outer-boundary coverage of the two admitted load-budget diagnostics."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .contact_fem_exact_columns import surface_quadrature


def mises(stress):
    dev = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * (dev**2).sum(axis=(-2, -1)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("scr_correction_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    audit_path = args.data_root / "nearby-scr-group-comparison-v1/group0/comparison/result.json"
    audit = json.loads(audit_path.read_text())
    assert audit["passed"]
    panels, receipts, plot_rows = [], [file_id(audit_path)], []
    bary, _ = surface_quadrature(81)
    maximum, delta_maximum = 0., 0.
    for group, budget in ((0, 0.01), (2, 0.001)):
        directory = args.data_root / "nearby-scr-fit-accuracy-v1" / f"group{group}/study"
        start = json.loads((directory / "start.json").read_text())
        row = json.loads((directory / "result.json").read_text())["rows"][0]
        candidate = next(b for b in row["budgets"] if b["budget"] == budget)
        assert candidate["accepted"] and not candidate["engineering_stress_accepted"]
        for receipt in (start["state"], candidate["field"]):
            assert file_id(receipt["path"]) == receipt
            receipts.append(receipt)
        with np.load(start["state"]["path"]) as state, np.load(candidate["field"]["path"]) as field:
            vertices, tets = field["vertices"], field["tetrahedra"]
            boundary = state["mesh.vertices"][state["mesh.surface_nodes"][:, :3]]
            distance, faces = cKDTree(vertices).query(boundary)
            assert distance.max() < 1e-13
            owners, corners = [], []
            for face in faces:
                match = (tets[:, :, None] == face[None, None]).any(axis=-1)
                owner = np.flatnonzero(match.sum(axis=1) == 3)
                assert len(owner) == 1
                owners.append(owner[0])
                corners.append((tets[owner[0], :, None] == face[None]).argmax(axis=0))
            owners, corners = np.array(owners), np.array(corners)
            position = np.einsum("qc,fci->fqi", bary, vertices[faces]).reshape(-1, 3)
            longitude = np.degrees(np.arctan2(position[:, 1], position[:, 0]))
            latitude = np.degrees(np.arctan2(position[:, 2], np.linalg.norm(position[:, :2], axis=1)))
            values = []
            for key in ("baseline_stress_pa", "stress_pa", "delta_stress_pa"):
                stress = field[key]
                trace = np.einsum("qc,fcij->fqij", bary, stress[owners[:, None], corners])
                values.append(mises(trace).reshape(-1) / 1000)
                if key == "delta_stress_pa":
                    delta_maximum = max(delta_maximum, float(mises(stress).max() / 1000))
                else:
                    maximum = max(maximum, float(mises(stress).max() / 1000))
        panels.append((row["environment"], budget, candidate["complete_stress_relative_change"], longitude, latitude, values))
        for index, point in enumerate(position):
            plot_rows.append({"environment": row["environment"], "budget": budget,
                "x_m": point[0], "y_m": point[1], "z_m": point[2],
                "longitude_deg": longitude[index], "latitude_deg": latitude[index],
                "baseline_vm_kpa": values[0][index], "candidate_vm_kpa": values[1][index],
                "correction_tensor_vm_kpa": values[2][index]})
    norms = (Normalize(0, np.ceil(maximum / 5) * 5), Normalize(0, np.ceil(delta_maximum)))
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 125), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.86, wspace=0.24, hspace=0.65)
    plots = []
    for i, (env, budget, change, longitude, latitude, values) in enumerate(panels):
        for j, value in enumerate(values):
            ax = axes[i, j]
            plot = ax.scatter(longitude, latitude, c=value, s=0.8, cmap="viridis" if j < 2 else "magma",
                              norm=norms[int(j == 2)], linewidths=0, rasterized=True)
            ax.set(xlim=(-180, 180), ylim=(-90, 90), aspect="equal")
            ax.set_xticks([-180, 0, 180])
            ax.set_yticks([-90, 0, 90])
            ax.set_title(("Prior load", "Cone-feasible load", "Correction tensor")[j], pad=6)
            if j == 0:
                ax.set_ylabel("Latitude (deg)")
            if i == 1:
                ax.set_xlabel("Longitude (deg)")
            style.style_axes(ax)
            style.add_panel_label(ax, "abcdef"[3 * i + j], x=-0.15, y=1.12)
            plots.append(plot)
        fig.text(0.53, 0.965 if i == 0 else 0.52,
                 f"Env {env}: aggregate budget {100*budget:g}%; complete stress change {100*change:.2f}%",
                 ha="center", va="top")
    fig.colorbar(plots[0], cax=fig.add_axes([0.15, 0.085, 0.43, 0.022]), orientation="horizontal",
                 label="Prior and candidate von Mises stress (kPa)")
    fig.colorbar(plots[2], cax=fig.add_axes([0.73, 0.085, 0.20, 0.022]), orientation="horizontal",
                 label="Correction tensor (kPa)")
    issues = style.rendered_text_qa(fig)
    exports = []
    for suffix, dpi in (("svg", 450), ("pdf", 450), ("png", 600)):
        path = args.output / f"scr-group-correction.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        exports.append(file_id(path))
    preview = args.output / "scr-group-correction-preview.png"
    fig.savefig(preview, dpi=150, bbox_inches=None)
    plt.close(fig)
    with (args.output / "plotted-data.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    (args.output / "caption.md").write_text(
        "Complete angular coverage of the original80 outer boundary faces for env9 and env330. "
        "Every81-point/facet display sample uses its owning tetrahedron's affine P2 stress trace; "
        "all points are shown without smoothing or stress clipping. Longitude/latitude are rest-frame coordinate directions. "
        "The first two columns share one global kPa scale; the third shares a separate global scale and displays "
        "von Mises of the correction tensor, not the difference of scalar von Mises values. "
        "The1%/0.1% aggregate force/moment budgets correspond to8.76%/14.66% complete3D physical-volume stress changes. "
        "Full resultants, unsmoothed3D maxima/locations, source changes and sampling sensitivities are separate numerical outputs. "
        "Absolute fields use the declared contact-only prior and candidate; the unchanged unrecorded inertial field cancels "
        "only in their linear correction. This is a load-budget diagnostic, not continuum accuracy or engineering acceptance.\n")
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": not issues, "inputs": receipts, "style": file_id(args.style), "size_mm": [183, 125],
        "display_samples": len(plot_rows), "exports": exports, "preview": file_id(preview),
        "rendered_text_issues": issues, "visual_inspection_pending": True, "whole_goal_complete": False})
    assert not issues, issues


if __name__ == "__main__":
    main()
