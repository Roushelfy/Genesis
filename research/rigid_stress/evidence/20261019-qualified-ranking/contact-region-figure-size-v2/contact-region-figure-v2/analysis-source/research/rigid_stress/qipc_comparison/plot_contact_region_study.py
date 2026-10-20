"""Show the complete pressure-bearing polar region and retain inner loading separately."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("contact_region_figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    source_paths = [args.study / f"group{g}/study/result.json" for g in (0, 1)]
    results = [json.loads(p.read_text()) for p in source_paths]
    assert all(r["passed"] for r in results)
    cases = [next(c for c in r["double_spreading"] if c["level"] == 0) for r in results]
    assert [c["method"] for c in cases] == ["dpr", "scr"]
    assert cases[0]["mesh"] == cases[1]["mesh"]
    mesh = Path(cases[0]["mesh"]["path"])
    assert file_id(mesh) == cases[0]["mesh"]
    with np.load(mesh) as saved:
        faces = saved["mid_faces"]
        boundary = np.concatenate((saved["outer_vertices"][faces], saved["inner_vertices"][faces]))
    maps = []
    for c in cases:
        path = Path(c["face_maps"]["path"])
        assert file_id(path) == c["face_maps"]
        with np.load(path) as saved:
            maps.append({k: saved[k] for k in saved.files})
    a, b = maps
    np.testing.assert_array_equal(a["points_material_m"], b["points_material_m"])
    input_difference = np.linalg.norm(a["original_forces_world_n"] - b["original_forces_world_n"]) / np.linalg.norm(a["original_forces_world_n"])
    assert input_difference <= 1e-8, "The displayed common original must match both actual inputs"
    upper = a["points_material_m"][:, 2] > 0
    outer = a["is_outer_boundary"]
    definitions = [
        ("Measured outer", a["original_forces_world_n"], upper & outer),
        ("DPR outer", a["reconstructed_forces_world_n"], upper & outer),
        ("DPR inner", a["reconstructed_forces_world_n"], upper & ~outer),
        ("SCR outer", b["reconstructed_forces_world_n"], upper & outer),
    ]
    total = -a["original_forces_world_n"][upper, 2].sum()
    panels, maximum, extent = [], 0.0, 0.0
    for title, force, selected in definitions:
        values = -force[:, 2]
        assert np.all(values[selected] >= 0), "A signed pressure map needs a diverging scale"
        pressure = np.divide(values, a["projected_area_m2"], out=np.zeros_like(values), where=a["projected_area_m2"] > 0) / 1000
        active = selected & (values != 0)
        assert active.any()
        maximum = max(maximum, pressure[selected].max())
        extent = max(extent, np.abs(boundary[active, :, :2]).max() * 1000)
        panels.append((title, selected, pressure, values[selected].sum() / total))
    extent = np.ceil(1.05 * extent * 2) / 2
    maximum = np.ceil(maximum)
    contract = {
        "claim_anchor": "fixed-radius-distributed-pad-double-spreading",
        "main_claim": "The saved fixed-radius reconstruction redistributes an already physical pressure field; DPR also loads the disconnected inner boundary.",
        "material_passport": [file_id(p) for p in source_paths] + [file_id(mesh)] + [c["face_maps"] for c in cases],
        "panels": ["Measured outer physical pressure", "DPR remaining outer pressure", "DPR inner-boundary transfer", "SCR outer redistribution"],
        "style": file_id(args.style), "size_mm": [183, 145], "level": 0,
        "full_nonzero_support_visible": True, "symmetric_polar_limits_mm": [-extent, extent],
        "common_pressure_scale_kpa": [0, maximum], "uncertainty": "One paired saved normal endpoint; no statistical error bars or continuum qualification.",
        "dpr_scr_original_force_relative_difference": input_difference.item(),
        "no_interpolation_or_smoothing": True, "cell_averages_on_actual_parent_facets": True,
    }
    write_json(args.output / "figure-contract.json", contract)
    fig, axes = plt.subplots(2, 2, figsize=style.mm_to_inches(183, 145), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.83, bottom=0.08, top=0.94, wspace=0.30, hspace=0.32)
    norm = Normalize(0, maximum)
    for index, (ax, (title, selected, pressure, fraction)) in enumerate(zip(axes.reshape(-1), panels)):
        active = selected & (pressure != 0)
        assert np.abs(boundary[active, :, :2]).max() * 1000 < extent
        ax.set_facecolor(matplotlib.colormaps["Blues"](0.0))
        collection = PolyCollection(boundary[active, :, :2] * 1000, array=pressure[active],
                                    cmap="Blues", norm=norm, edgecolors="none", linewidths=0)
        ax.add_collection(collection)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_title(f"{title} ({100*fraction:.2f}% load)", pad=7)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        style.style_axes(ax)
        style.add_panel_label(ax, "abcd"[index], x=-0.18, y=1.12)
    color_axis = fig.add_axes([0.88, 0.25, 0.02, 0.46])
    fig.colorbar(collection, cax=color_axis, label="Normal pressure (kPa)")
    issues = style.rendered_text_qa(fig)
    outputs = []
    for suffix, dpi in (("svg", 450), ("pdf", 450), ("png", 600)):
        path = args.output / f"contact-pressure.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        outputs.append(file_id(path))
    preview = args.output / "contact-pressure-preview.png"
    fig.savefig(preview, dpi=150, bbox_inches=None)
    plt.close(fig)
    svg = (args.output / "contact-pressure.svg").read_text()
    assert "<text" in svg and "viewBox" in svg
    caption = (
        "Upper polar pressure at the saved0.1N normal endpoint. Panels show measured outer-face centroid pressure, "
        "DPR outer and inner boundary pressure, and SCR outer pressure, using the actual production reconstruction samples "
        "summed onto their exact parent facets. All panels use the same kPa scale and physical axes; the complete nonzero "
        "support is visible. Percentages are each panel's integrated normal load divided by the measured upper reaction. "
        "Original inner loading is zero. The two native paths' original inputs agree within the stated numerical gate. "
        "These are cell averages without spatial interpolation, not resolved pointwise pressure or a fine-reference stress error. "
        "Both methods use the retained2mm/manual baseline and full physical centroid inputs."
    )
    (args.output / "caption.md").write_text(caption + "\n")
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": not issues, "contract": file_id(args.output / "figure-contract.json"),
        "exports": outputs, "preview": file_id(preview), "caption": file_id(args.output / "caption.md"),
        "rendered_text_issues": issues, "svg_text_editable": True, "visual_inspection_pending": True,
        "whole_goal_complete": False})
    assert not issues, issues


if __name__ == "__main__":
    main()
