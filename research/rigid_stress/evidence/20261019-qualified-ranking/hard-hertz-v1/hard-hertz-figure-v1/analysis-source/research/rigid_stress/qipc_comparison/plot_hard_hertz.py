"""Show why ellipticity and inexpensive interaction corrections matter for hard contact."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--interaction", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("hard_hertz_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    study = json.loads((args.study / "result.json").read_text())
    correction = json.loads((args.interaction / "result.json").read_text())
    assert study["passed"] and correction["passed"]
    assert correction["source"] == file_id(args.study / "result.json")
    selected = [
        next(r for r in study["rows"] if r["case"] == case and r["grid"] == 54) for case in ("ellipse", "two_regions")
    ]
    corrected = next(r for r in correction["rows"] if r["grid"] == 54)
    assert corrected["reference"] == selected[1]["arrays"]
    records, passports = [], [file_id(args.study / "result.json"), file_id(args.interaction / "result.json")]
    for row in selected:
        path = Path(row["arrays"]["path"])
        assert file_id(path) == row["arrays"]
        passports.append(file_id(path))
        with np.load(path) as saved:
            records.append(
                {
                    k: saved[k]
                    for k in (
                        "points_m",
                        "reference_force_n",
                        "elliptic_hertz_force_n",
                        "circular_geometric_curvature_force_n",
                    )
                }
            )
    path = Path(corrected["arrays"]["path"])
    assert file_id(path) == corrected["arrays"]
    passports.append(file_id(path))
    with np.load(path) as saved:
        adjusted_force = saved["force_n"]
    definitions = [
        (0, "Elliptical obstacle: BEM", records[0]["reference_force_n"], None),
        (
            0,
            "Equivalent circle",
            records[0]["circular_geometric_curvature_force_n"],
            selected[0]["variants"]["circular_geometric_curvature"]["pressure_area_l2_relative"],
        ),
        (
            0,
            "Elliptic Hertz",
            records[0]["elliptic_hertz_force_n"],
            selected[0]["variants"]["elliptic_hertz"]["pressure_area_l2_relative"],
        ),
        (1, "Two regions: BEM", records[1]["reference_force_n"], None),
        (
            1,
            "Independent Hertz regions",
            records[1]["elliptic_hertz_force_n"],
            selected[1]["variants"]["elliptic_hertz"]["pressure_area_l2_relative"],
        ),
        (1, "One interaction correction", adjusted_force, corrected["pressure_area_l2_relative"]),
    ]
    contract = {
        "claim_anchor": "cheap-hard-contact-ellipticity-and-interaction",
        "main_claim": "A scalar equivalent radius loses elliptical pressure geometry. A single far-field Taylor correction tests a cheap remedy for interaction between two known symmetric regions.",
        "material_passport": passports,
        "style": file_id(args.style),
        "size_mm": [183, 127],
        "panels": [
            {
                "panel": chr(97 + i),
                "title": title,
                "quantity": "cell-average pressure (MPa)",
                "pressure_l2_relative": error,
            }
            for i, (_, title, _, error) in enumerate(definitions)
        ],
        "uncertainty": "One deterministic load per geometry, 54x54 BEM grid. The report retains the separate 36x36 comparison. No statistical error bars or shell qualification.",
        "full_nonzero_support_visible": True,
        "smoothing": False,
        "alternatives": "A pressure cross-section would hide the rotated elliptical support; six pressure maps retain its orientation and both support islands.",
    }
    write_json(args.output / "figure-contract.json", contract)
    maximum = max(
        force.max() / ((np.unique(records[row]["points_m"][:, 0])[1] - records[row]["points_m"][0, 0]) ** 2) / 1e6
        for row, _, force, _ in definitions
    )
    norm = Normalize(0, np.ceil(maximum))
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 127), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.93, wspace=0.32, hspace=0.35)
    for index, (ax, (row, title, force, error)) in enumerate(zip(axes.flat, definitions)):
        points = records[row]["points_m"]
        width = np.unique(points[:, 0])[1] - points[0, 0]
        extent = (
            points[:, 0].min() - width / 2,
            points[:, 0].max() + width / 2,
            points[:, 1].min() - width / 2,
            points[:, 1].max() + width / 2,
        )
        pressure = force.reshape(54, 54).T / width**2 / 1e6
        assert np.all(pressure >= -1e-9)
        mappable = ax.imshow(
            pressure, extent=np.array(extent) * 1e6, origin="lower", interpolation="none", cmap="viridis", norm=norm
        )
        ax.set(xlabel="x (µm)", ylabel="y (µm)", xticks=[-100, 0, 100], yticks=[-100, 0, 100])
        ax.set_title(
            title + (f"\nPressure error {100 * error:.2f}%" if error is not None else "\nIndependent contact solve"),
            pad=5,
        )
        ax.text(-0.22, 1.15, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=8)
    color_ax = fig.add_axes([0.91, 0.25, 0.015, 0.5])
    fig.colorbar(mappable, cax=color_ax, label="Cell-average pressure (MPa)")
    exports = []
    for suffix, dpi in (("svg", 150), ("pdf", 600), ("png", 600)):
        path = args.output / f"hard-hertz.{suffix}"
        fig.savefig(path, dpi=dpi)
        exports.append(file_id(path))
    preview = args.output / "preview.png"
    fig.savefig(preview, dpi=150)
    assert "<text" in (args.output / "hard-hertz.svg").read_text()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "exports": exports,
            "preview": file_id(preview),
            "editable_svg_text": True,
            "contract": file_id(args.output / "figure-contract.json"),
            "visual_review_required": True,
        },
    )


if __name__ == "__main__":
    main()
