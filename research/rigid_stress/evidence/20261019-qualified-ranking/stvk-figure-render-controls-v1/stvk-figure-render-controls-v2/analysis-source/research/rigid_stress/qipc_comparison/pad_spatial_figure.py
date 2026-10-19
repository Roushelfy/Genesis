"""Retain full-surface stress maps and actual unsmoothed contact-pressure samples."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .artifacts import file_id, runtime_identity, write_json


def tangent_mises(tensor):
    # Embedded3D tangent tensor with the third row/column zero.
    return np.sqrt(
        np.maximum(1.5 * np.sum(tensor * tensor, axis=(-2, -1)) - 0.5 * np.trace(tensor, axis1=-2, axis2=-1) ** 2, 0)
    )


def angular_polygons(unit, faces):
    result = []
    for tri in unit[faces]:
        theta = np.arccos(np.clip(tri[:, 2], -1, 1))
        phi = np.arctan2(tri[:, 1], tri[:, 0])
        pole = np.flatnonzero(np.abs(tri[:, 2]) > 1 - 1e-12)
        if len(pole):
            others = [i for i in range(3) if i != pole[0]]
            azimuths = np.unwrap(phi[others])
            polygon = np.array(
                [
                    [azimuths[0], theta[others[0]]],
                    [azimuths[1], theta[others[1]]],
                    [azimuths[1], theta[pole[0]]],
                    [azimuths[0], theta[pole[0]]],
                ]
            )
        else:
            polygon = np.column_stack((np.unwrap(phi), theta))
        result.append(np.rad2deg(polygon))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--fields", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def path(relative):
        p = args.data_root / relative
        receipts.append(file_id(p))
        return p

    qualification = json.loads(path("pad-normal-qualification-v2/group0/qualification/result.json").read_text())
    assert qualification["normal_reference_change_targets_passed"]
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    mesh = path("compliant-pad-joint-refinement-mesh-v1/group0/meshes/land-contact-0.npz")
    with np.load(mesh) as saved:
        unit, faces = saved["material_unit"], saved["mid_faces"]
    unit = unit / np.linalg.norm(unit, axis=1)[:, None]
    polygons = angular_polygons(unit, faces)
    values, positions = {}, None
    for group, method in enumerate(("dpr", "scr", "shell")):
        root = args.fields / f"group{group}/{method}-0"
        receipt = root / "result.json"
        receipts.append(file_id(receipt))
        assert json.loads(receipt.read_text())["surface_quadrature_points_per_face"] == 1
        p = root / "surface-fields-0.1-0.npz"
        receipts.append(file_id(p))
        with np.load(p) as saved:
            current = saved["outer_rest_position_m"]
            if positions is None:
                positions = current
            else:
                np.testing.assert_array_equal(positions, current)
            for side in ("outer", "inner"):
                pair = tangent_mises(saved[side + "_tangent_pa"])
                assert pair.shape == (2, len(faces)) and np.isfinite(pair).all()
                values[method + "_" + side] = pair[0]
                key = "reference_" + side
                if key in values:
                    np.testing.assert_allclose(values[key], pair[1], rtol=1e-12, atol=1e-8)
                values[key] = pair[1]
    np.savez_compressed(args.output / "stress-map-data.npz", unit=unit, faces=faces, **values)
    stress_max = max(v.max() for v in values.values()) / 1000
    stress_norm = matplotlib.colors.Normalize(0, stress_max)
    fig, axes = plt.subplots(2, 4, figsize=style.mm_to_inches(183, 120))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.90, wspace=0.22, hspace=0.32)
    titles = ("DPR, 2 mm", "SCR, 2 mm", "Shell, repaired", "P2 reference")
    for row, side in enumerate(("outer", "inner")):
        for col, method in enumerate(("dpr", "scr", "shell", "reference")):
            ax = axes[row, col]
            for shift in (-360, 0, 360):
                collection = PolyCollection(
                    [p + [shift, 0] for p in polygons],
                    array=values[method + "_" + side] / 1000,
                    cmap="viridis",
                    norm=stress_norm,
                    edgecolors="none",
                    antialiased=False,
                    rasterized=True,
                )
                ax.add_collection(collection)
            ax.set_xlim(-180, 180)
            ax.set_ylim(180, 0)
            ax.set_xticks([-180, 0, 180])
            ax.set_yticks([0, 45, 90, 135, 180])
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(side.capitalize() + " polar angle (°)")
            if row:
                ax.set_xlabel("Azimuth (°)")
            else:
                ax.set_title(titles[col], pad=9)
            style.add_panel_label(ax, chr(97 + row * 4 + col), x=-0.13, y=1.10)
    cax = fig.add_axes([0.91, 0.23, 0.018, 0.54])
    fig.colorbar(collection, cax=cax, label="Tangent Mises stress (kPa)")
    stress_qa = style.rendered_text_qa(fig)
    stress_paths = style.save_nature_figure(fig, args.output / "normal-complete-surface-stress")
    plt.close(fig)

    samples = []
    for method, relative, keys in (
        (
            "native",
            "native-pad-full-source-stability-v1/group0/native/frames/level0.npz",
            ("points_world_m", "contact_force_n", "projected_area_m2"),
        ),
        (
            "shell",
            "compliant-pad-fine-coarse-history-v1/group0/cycle/frames/level0.npz",
            ("pad_points_world_m", "pad_point_force_n", "pad_projected_area_m2"),
        ),
        (
            "reference",
            "compliant-pad-joint-reference-v2/group0/reference/order-10/load-0.1.npz",
            ("traction_points_m", "traction_point_forces_n", "traction_quadrature_area_m2"),
        ),
    ):
        with np.load(path(relative)) as saved:
            points, force, area = (saved[key] for key in keys)
        assert np.isfinite(points).all() and np.isfinite(force).all() and np.isfinite(area).all()
        assert np.linalg.norm(force[:, :2]) < 1e-8
        positive = np.abs(force[:, 2]) > 0
        assert np.all(area[positive] > 0)
        pressure = np.zeros(len(points))
        pressure[positive] = np.abs(force[positive, 2]) / area[positive]
        assert np.max(np.linalg.norm(points[positive, :2], axis=1)) < 0.0032, "View must include the complete footprint"
        np.savez_compressed(
            args.output / f"traction-{method}.npz", points_m=points, force_n=force, area_m2=area, pressure_pa=pressure
        )
        samples.append((method, points, force, pressure))
    pressure_max = max(p.max() for _, _, _, p in samples) / 1000
    pressure_norm = matplotlib.colors.Normalize(0, pressure_max)
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 125))
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.11, top=0.91, wspace=0.20, hspace=0.35)
    footprints = []
    for col, (method, points, force, pressure) in enumerate(samples):
        for row, upper in enumerate((True, False)):
            ax = axes[row, col]
            selected = (points[:, 2] > 0) == upper
            reaction = np.abs(force[selected, 2].sum())
            assert abs(reaction - 0.1) < 1e-6
            active = selected & (pressure > 0)
            scatter = ax.scatter(
                points[active, 0] * 1000,
                points[active, 1] * 1000,
                c=pressure[active] / 1000,
                s=3 if method == "native" else 0.7,
                cmap="viridis",
                norm=pressure_norm,
                linewidths=0,
                rasterized=True,
            )
            ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", xticks=[-3, 0, 3], yticks=[-3, 0, 3])
            if row:
                ax.set_xlabel("x (mm)")
            else:
                ax.set_title(("DPR / SCR contact", "Shell, repaired", "P2 reference")[col], pad=10)
            if col == 0:
                ax.set_ylabel(("Upper" if upper else "Lower") + ": y (mm)")
            style.add_panel_label(ax, chr(97 + row * 3 + col), x=-0.18, y=1.06)
            footprints.append(
                {
                    "method": method,
                    "upper": upper,
                    "reaction_n": float(reaction),
                    "active_points": int(active.sum()),
                    "maximum_pressure_pa": float(pressure[selected].max()),
                }
            )
    cax = fig.add_axes([0.91, 0.23, 0.018, 0.54])
    fig.colorbar(scatter, cax=cax, label="Actual pad pressure (kPa)")
    traction_qa = style.rendered_text_qa(fig)
    traction_paths = style.save_nature_figure(fig, args.output / "normal-actual-traction")
    shutil.copy2(args.style, args.output / "nature_mpl_style.py")
    shutil.copy2(__file__, args.output / "pad_spatial_figure.py")
    write_json(
        args.output / "figure-contract.json",
        {
            "claim_anchor": "normal-full-surface-stress-and-actual-footprints",
            "main_claim": "Identical color scales expose complete-surface stress differences and the distinct actual normal-pressure sampling/footprints; a recovery footprint parameter is not the measured physical contact footprint.",
            "target": "English repository supplement; editable text/axes in SVG/PDF, dense data artists rasterized at export resolution.",
            "chart_choice": "Unwrapped full-surface face maps show both poles and all azimuths without selecting a meridian; actual quadrature samples show pressure without interpolation or smoothing.",
            "data_profile": {
                "load_n": 0.1,
                "surface_faces": len(faces),
                "stress_methods": 4,
                "surface_sides": 2,
                "contact_methods": 3,
                "plates": 2,
            },
            "limits": "Stress display is one unsmoothed common centroid per reference face, not the9/25-sample peak definition. The angular chart is not equal-area and does not define error weights. Full-domain errors and qualified peaks remain separate. Pressure markers represent samples, not marker-sized contact areas. No continuum peak/field claim is derived from raster images.",
        },
    )
    (args.output / "caption.md").write_text(
        "Claim anchor: normal-full-surface-stress-and-actual-footprints.\n\n"
        "At0.1N normal load, outer and inner tangent-plane Mises stress is sampled at identical angular material locations on the125µm/three-wall reference grid. Each face receives its unsmoothed centroid value; all faces, azimuths and both poles are shown with one common color scale. Polar angle0°/180° denotes the upper/lower pole. The angular chart is not equal-area. Dense face artists are rasterized, while labels and axes remain editable. This display does not replace complete-volume errors or qualified9/25-point common/raw maxima.\n\n"
        "Actual upper/lower pad-pressure maps show every force-bearing quadrature sample, with one pressure scale across native, repaired Shell and P2 reference. Pressure is absolute vertical point force divided by its own projected quadrature area; all reactions are independently checked against0.1N and all active samples fit inside the displayed window. Native DPR/SCR share actual contact mechanics. Marker size is only a visual aid, not physical contact area. No pressure smoothing, fitted support or percentile clipping is applied. Reference points are in the declared rest material projection; native/Shell points use actual world positions in this normal equilibrium. All models use the declared1e9N/m3 pad stiffness.\n\n"
        "Place alongside the complete-field error table and force/closure response. Exact tensors/triangles, actual point-force/area samples, source hashes, code and style accompany the figures. Original hard-plate/Shell failures, sustained friction and spatial uncertainty of individual3D-only components remain separate.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "receipts": receipts,
            "stress_paths": stress_paths,
            "traction_paths": traction_paths,
            "programmatic_qa_issues": stress_qa + traction_qa,
            "footprints": footprints,
            "common_stress_maximum_kpa": float(stress_max),
            "common_pressure_maximum_kpa": float(pressure_max),
            "visual_qa_required": True,
        },
    )
    assert not stress_qa and not traction_qa, "Inspect retained layout warnings before publishing"


if __name__ == "__main__":
    main()
