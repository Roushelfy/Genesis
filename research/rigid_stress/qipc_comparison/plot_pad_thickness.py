"""Compare thickness-dependent physical pressure on one common outer geometry."""

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

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--studies", type=Path, nargs=3, required=True)
    cli.add_argument("--thicknesses", type=float, nargs=3, default=[0.00025, 0.0005, 0.001])
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--contract", type=Path, required=True)
    cli.add_argument("--accepted-svg", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads(args.contract.read_text())
    assert contract["claim_anchor"] == "pad-thickness-coupling"
    spec = importlib.util.spec_from_file_location("pad_thickness_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "pad-thickness-coupling"})
    studies, receipts, outer_reference, faces_reference = [], [], None, None
    for directory, thickness in zip(args.studies, args.thicknesses):
        start, result = (json.loads((directory / name).read_text()) for name in ("start.json", "result.json"))
        assert result["passed"]
        assert file_id(Path(start["mesh"]["path"])) == start["mesh"]
        receipts.extend((file_id(directory / "result.json"), file_id(directory / "start.json"), start["mesh"]))
        with np.load(start["mesh"]["path"]) as saved:
            outer, inner, faces, vertices, tets = (
                saved[k] for k in ("outer_vertices", "inner_vertices", "mid_faces", "vertices", "tetrahedra")
            )
            np.testing.assert_allclose(np.linalg.norm(outer - inner, axis=1), thickness, rtol=0, atol=1e-15)
            if outer_reference is not None:
                np.testing.assert_array_equal(outer, outer_reference)
                np.testing.assert_array_equal(faces, faces_reference)
            outer_reference, faces_reference = outer, faces
            volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
        studies.append((result, volumes))
    panels, model_errors, plotted = [], [], {}
    for stiffness in (1e9, 1e10):
        for column, ((result, volumes), thickness) in enumerate(zip(studies, args.thicknesses)):
            case = next(
                i
                for i, c in enumerate(result["cases"])
                if c == {"young_pa": 1e10, "pad_stiffness_n_per_m3": stiffness, "load_n": 0.1, "stiffness_gradient": 0}
            )
            coupled = next(
                r
                for r in result["records"]
                if r["case"] == case
                and r["order"] == 6
                and r["mode"] == "coupled"
                and r["search_contains_actual_elastic_compression"]
            )
            assert coupled["full_elastic_pad_law_relative_error"] <= 1e-7
            path = Path(coupled["field"]["path"])
            assert file_id(path) == coupled["field"]
            with np.load(path) as saved:
                points, pressure, area, stress = (saved[k] for k in ("points_m", "pressure_pa", "area_m2", "stress_pa"))
            assert np.isfinite(pressure).all() and pressure.min() >= 0
            panels.append((thickness, stiffness, points, pressure, coupled))
            index = len(panels) - 1
            plotted[f"panel{index}_points_m"], plotted[f"panel{index}_pressure_pa"] = points, pressure
            plotted[f"panel{index}_area_m2"] = area
            receipts.append(coupled["field"])
            for mode in ("rigid", "diagonal"):
                row = next(
                    r
                    for r in result["records"]
                    if r["case"] == case
                    and r["order"] == 6
                    and r["mode"] == mode
                    and r["search_contains_actual_elastic_compression"]
                )
                assert file_id(Path(row["field"]["path"])) == row["field"]
                with np.load(row["field"]["path"]) as saved:
                    np.testing.assert_array_equal(saved["points_m"], points)
                    np.testing.assert_array_equal(saved["area_m2"], area)
                    pressure_error = np.sqrt(
                        np.sum(area * (saved["pressure_pa"] - pressure) ** 2) / np.sum(area * pressure**2)
                    )
                    stress_error = field_norm(saved["stress_pa"] - stress, volumes) / field_norm(stress, volumes)
                expected = next(
                    r for r in result["comparisons"] if r["case"] == case and r["order"] == 6 and r["mode"] == mode
                )
                np.testing.assert_allclose(
                    pressure_error, expected["pressure_area_relative_l2"], rtol=1e-10, atol=1e-13
                )
                np.testing.assert_allclose(
                    stress_error, expected["complete_stress_volume_relative_l2"], rtol=1e-10, atol=1e-13
                )
                model_errors.append(
                    {
                        "thickness_m": thickness,
                        "pad_stiffness_n_per_m3": stiffness,
                        "mode": mode,
                        "pressure_area_relative_l2": pressure_error.item(),
                        "complete_stress_volume_relative_l2": stress_error.item(),
                        "coupled_plate_travel_m": coupled["plate_travel_m"],
                        "coupled_maximum_displacement_gradient": coupled["maximum_displacement_gradient"],
                    }
                )
                receipts.append(row["field"])
    assert len(model_errors) == 12
    with (args.output / "model-errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_errors[0]))
        writer.writeheader()
        writer.writerows(model_errors)
    data_path = args.output / "plotted-pressure.npz"
    np.savez_compressed(data_path, **plotted)
    extent = (
        np.ceil(
            max(
                np.abs(points[(points[:, 2] > 0) & (pressure > 0), :2]).max() * 1000
                for _, _, points, pressure, _ in panels
            )
            * 2.2
        )
        / 2
    )
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 142), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.10, top=0.89, wspace=0.28, hspace=0.58)
    for row_index in range(2):
        subset = panels[row_index * 3 : (row_index + 1) * 3]
        maximum = np.ceil(max(p[(x[:, 2] > 0) & (p > 0)].max() / 1000 for _, _, x, p, _ in subset))
        norm = Normalize(0, maximum)
        for column, (thickness, stiffness, points, pressure, _) in enumerate(subset):
            ax = axes[row_index, column]
            mask = (points[:, 2] > 0) & (pressure > 0)
            artist = ax.scatter(
                points[mask, 0] * 1000,
                points[mask, 1] * 1000,
                c=pressure[mask] / 1000,
                s=2.5,
                cmap="viridis",
                norm=norm,
                linewidths=0,
            )
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xlabel="x (mm)")
            if column == 0:
                ax.set_ylabel("Upper pad: y (mm)")
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            ax.set_title(f"h = {thickness * 1e6:g} " + r"$\mu$m", pad=7)
            style.style_axes(ax)
            style.add_panel_label(ax, "abcdef"[row_index * 3 + column], x=-0.20, y=1.12)
        color_axis = fig.add_axes([0.875, 0.59 if row_index == 0 else 0.12, 0.017, 0.27])
        fig.colorbar(artist, cax=color_axis, label="Normal pressure (kPa)")
        label = f"Pad stiffness = {subset[0][1]:.0e} N/m" + r"$^3$"
        fig.text(0.43, 0.97 if row_index == 0 else 0.50, label, ha="center", va="top")
    issues = style.rendered_text_qa(fig)
    assert not issues, issues
    svg = args.output / "pad-thickness-pressure.svg"
    fig.savefig(svg, dpi=450, bbox_inches=None, metadata={"Date": None})
    exports = [file_id(svg)]
    if args.accepted_svg is not None:
        assert svg.read_bytes() == args.accepted_svg.read_bytes(), (
            "Final editable source must match the inspected draft"
        )
        for suffix, dpi in (("pdf", 450), ("png", 600)):
            path = args.output / f"pad-thickness-pressure.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches=None)
            exports.append(file_id(path))
    preview = args.output / "pad-thickness-pressure-preview.png"
    fig.savefig(preview, dpi=150, bbox_inches=None)
    plt.close(fig)
    (args.output / "caption.md").write_text(
        "Coupled upper-pad pressure at0.1N on identical outer geometry, E=10GPa and three inward wall thicknesses. "
        "Columns show250,500 and1000um. Rows use physical pad stiffness1e9 and1e10N/m3, with a shared kPa scale within "
        "each row and common physical coordinates throughout. Every positive degree6 quadrature pressure is displayed "
        "without smoothing or clipping. Marker size is a display aid. The lower pad and complete fields remain in the data. "
        "These deterministic linear-elastic discrete-model controls isolate thickness at fixed outer triangulation and "
        "through-wall layer count. They do not supply spatial convergence or a universal physical regime bound.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "contract": file_id(args.contract),
            "claim_anchor": "pad-thickness-coupling",
            "source_receipts": receipts,
            "model_errors": model_errors,
            "plot_data": file_id(data_path),
            "size_mm": [183, 142],
            "style": file_id(args.style),
            "exports": exports,
            "preview": file_id(preview),
            "editable_source_matches_inspected_draft": args.accepted_svg is not None,
            "rendered_text_issues": issues,
            "visual_inspection_pending": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
