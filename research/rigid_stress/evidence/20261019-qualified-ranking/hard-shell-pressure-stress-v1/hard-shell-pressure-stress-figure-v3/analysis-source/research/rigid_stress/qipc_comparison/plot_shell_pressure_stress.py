"""Plot the retained single-case independent-pressure shell comparison."""

import argparse
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
from PIL import Image

from .artifacts import file_id, runtime_identity, write_json


def measures(tensor):
    principal = np.linalg.eigvalsh(tensor)[..., -1]
    deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return principal / 1e6, np.sqrt(1.5 * np.sum(deviator**2, axis=(-1, -2))) / 1e6


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--inspection", type=Path, required=True)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("retained_nature_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "savefig.bbox": None})
    inspection = json.loads(args.inspection.read_text())
    assert inspection["passed"] and inspection["conditional_single_case_reference_targets_met"]
    inputs = [args.inspection, args.pressure, args.style, args.study / "queries.npz"]
    inputs += [args.study / (name + "-fields.npz") for name in ["coarse", "medium", "fine"]]
    contract = {
        "claim_anchor": "For this fixed normal polar case, a finite-body regular stress response is needed; the cheap central-source residual approximates the independent pressure reference within the declared conditional budgets.",
        "material_passport": [file_id(path) for path in inputs],
        "placement": "Repository supplement to the independent hollow-shell pressure/stress report.",
        "language": "English",
        "venue_stage": "repository supplement",
        "panels": {
            "a": "Independently solved97-grid pressure; dashed analytic Hertz support.",
            "b": "Fine reference maximum principal surface stress on the retained41x41 probe grid; global sampled peak locations overlaid.",
            "c": "Surface x-axis principal stress, with observed reference-variation envelope.",
            "d": "Axis von Mises stress in the first150um of the500um wall.",
        },
        "statistics": "Deterministic calculations; no replicates or statistical confidence intervals. The gray band sums observed pointwise body/pressure/source/integration changes and the measured edge-sampling peak change.",
        "limits": "One normal load, fixed faceted polar land, free tilt and consistent-mass inertial relief. Complete-field errors are volume weighted; plotted slices are descriptive. No general contact/friction or native cost claim.",
        "exports": "Editable SVG, embedded-font PDF,600dpi RGB PNG,150dpi color and grayscale previews; source arrays and caption retained.",
    }
    write_json(args.output / "figure-contract.json", contract)
    with np.load(args.study / "queries.npz") as data:
        points = data["common_points_m"]
        nv, nc, npb = len(data["map_volume"]), len(data["map_columns"]), len(data["map_probes"])
    start = nv + nc
    probes = np.arange(start, start + npb)
    ring = np.arange(start + npb, len(points))
    surface = probes[: 41**2]
    xy = points[surface, :2].reshape(41, 41, 2) * 1e6
    axis_surface = np.r_[surface, ring]
    axis_surface = axis_surface[np.isclose(points[axis_surface, 1], 0, rtol=0, atol=1e-16)]
    axis_surface = axis_surface[np.isclose(points[axis_surface, 2], 0.03, rtol=0, atol=1e-16)]
    axis_surface = axis_surface[np.argsort(points[axis_surface, 0])]
    axis = probes[np.linalg.norm(points[probes, :2], axis=1) < 1e-16]
    depth = (0.03 - points[axis, 2]) * 1e6
    axis = axis[(depth >= -1e-8) & (depth <= 150)]
    axis = axis[np.argsort(-points[axis, 2])]
    _, first = np.unique(points[axis, 2], return_index=True)
    axis = axis[np.sort(first)]
    selection = np.unique(np.r_[surface, axis_surface, axis])
    fields = {}
    for mesh, labels in {
        "coarse": ["hertz-point", "hertz-only"],
        "medium": ["reference97"],
        "fine": ["reference97", "reference65", "reference97-sources3", "reference97-lower-integration"],
    }.items():
        with np.load(args.study / (mesh + "-fields.npz")) as data:
            for label in labels:
                fields[(mesh, label)] = measures(data[label + "-common_tensor_pa"][selection])
    selected = {int(index): i for i, index in enumerate(selection)}

    def lookup(indices):
        return np.array([selected[int(index)] for index in indices])

    grid, surf, axial = lookup(surface), lookup(axis_surface), lookup(axis)
    reference = fields[("fine", "reference97")]
    ref_row = next(row for row in inspection["rows"] if row["mesh"] == "fine" and row["variant"] == "reference97")
    point_row = next(row for row in inspection["rows"] if row["mesh"] == "coarse" and row["variant"] == "hertz-point")
    variation = np.zeros(len(selection))
    for key in [
        ("medium", "reference97"),
        ("fine", "reference65"),
        ("fine", "reference97-sources3"),
        ("fine", "reference97-lower-integration"),
    ]:
        variation += np.abs(fields[key][0] - reference[0])
    variation += inspection["contact_edge_sampling_change"]["tensile_peak_relative"] * ref_row["tensile_max_pa"] / 1e6
    with np.load(args.pressure) as data:
        pressure = data["grid97-sources5-pressure_pa"].reshape(97, 97) / 1e6
    hertz_axes = np.array([6.993279565275541e-5, 5.11374189492572e-5])
    pressure_xy = np.meshgrid(
        *[np.linspace(-1.3 * width, 1.3 * width, 97) * 1e6 for width in hertz_axes], indexing="ij"
    )
    fig = plt.figure(figsize=style.mm_to_inches(183, 153))
    layout = fig.add_gridspec(2, 2, left=0.095, right=0.89, bottom=0.15, top=0.94, wspace=0.48, hspace=0.38)
    axes = [fig.add_subplot(layout[index // 2, index % 2]) for index in range(4)]
    a, b, c, d = axes
    image = a.pcolormesh(*pressure_xy, pressure, cmap="viridis", shading="nearest", rasterized=False)
    a.add_patch(
        Ellipse(
            (0, 0),
            *(2 * hertz_axes * 1e6),
            fill=False,
            edgecolor=style.WONG_PALETTE["orange"],
            linestyle="--",
            linewidth=0.9,
        )
    )
    fig.colorbar(image, ax=a, fraction=0.046, pad=0.035, ticks=[0, 5, 10, 15, 20, 25]).set_label("Pressure (MPa)")
    principal = reference[0][grid].reshape(41, 41)
    norm = (
        TwoSlopeNorm(vcenter=0, vmin=float(principal.min()), vmax=float(principal.max()))
        if principal.min() < 0 < principal.max()
        else None
    )
    image = b.pcolormesh(
        xy[..., 0],
        xy[..., 1],
        principal,
        cmap="RdBu_r" if norm else "magma",
        norm=norm,
        shading="nearest",
        rasterized=False,
    )
    fig.colorbar(image, ax=b, fraction=0.046, pad=0.035, ticks=[-20, -10, 0, 2]).set_label(r"$\sigma_1$ (MPa)")
    for row, marker, color in [(ref_row, "o", "black"), (point_row, "x", style.WONG_PALETTE["blue"])]:
        coordinate = np.array(row["tensile_location_m"])[:2] * 1e6
        options = {"facecolors": "none", "edgecolors": color} if marker == "o" else {"color": color}
        b.scatter(*coordinate, marker=marker, s=17, linewidths=0.8, **options)
    for ax in [a, b]:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        ax.set_yticks([-50, 0, 50])
    a.set_xticks([-75, 0, 75])
    b.set_xticks([-100, -50, 0, 50, 100])
    series = [
        ("fine", "reference97", "Reference", "black", "-"),
        ("coarse", "hertz-point", "Hertz + shell residual", style.WONG_PALETTE["blue"], "-."),
        ("coarse", "hertz-only", "Hertz half-space", style.WONG_PALETTE["orange"], "--"),
    ]
    x = points[axis_surface, 0] * 1e6
    z = (0.03 - points[axis, 2]) * 1e6
    c.fill_between(
        x,
        reference[0][surf] - variation[surf],
        reference[0][surf] + variation[surf],
        color="0.8",
        linewidth=0,
        label="Observed reference variation",
    )
    arrays = {
        "pressure_mpa": pressure,
        "pressure_x_um": pressure_xy[0],
        "pressure_y_um": pressure_xy[1],
        "surface_xy_um": xy,
        "surface_reference_sigma1_mpa": principal,
        "profile_x_um": x,
        "axis_depth_um": z,
        "surface_variation_mpa": variation[surf],
    }
    for mesh, label, name, color, line in series:
        tension, vm = fields[(mesh, label)]
        c.plot(x, tension[surf], label=name, color=color, linestyle=line)
        d.plot(z, vm[axial], color=color, linestyle=line)
        arrays[f"{mesh}-{label}-surface_sigma1_mpa"] = tension[surf]
        arrays[f"{mesh}-{label}-axis_vm_mpa"] = vm[axial]
    c.set_xlabel(r"Surface $x$ ($\mu$m), $y=0$")
    c.set_ylabel(r"$\sigma_1$ (MPa)")
    c.set_yticks([-20, -15, -10, -5, 0])
    c.set_xticks([-100, -50, 0, 50, 100])
    handles, labels = c.get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        fontsize=5,
    )
    d.set_xlabel(r"Depth ($\mu$m), $x=y=0$")
    d.set_ylabel("von Mises stress (MPa)")
    d.set_xlim(0, 150)
    for index, ax in enumerate(axes):
        style.style_axes(ax)
        style.add_panel_label(ax, "abcd"[index], x=-0.18, y=1.10)
    issues = style.rendered_text_qa(fig)
    stem = args.output / "shell-pressure-stress"
    fig.savefig(stem.with_suffix(".svg"))
    style.save_preview(fig, args.output / "shell-pressure-stress_preview.png")
    with Image.open(args.output / "shell-pressure-stress_preview.png") as preview:
        preview.convert("L").save(args.output / "shell-pressure-stress_grayscale.png")
    root = ET.parse(stem.with_suffix(".svg")).getroot()
    texts = list(root.iter("{http://www.w3.org/2000/svg}text"))
    assert len(texts) >= 20
    if not issues:
        style.save_nature_figure(fig, stem, png_dpi=600, preview_dpi=150)
    np.savez_compressed(args.output / "plotted-data.npz", **arrays)
    np.savetxt(
        args.output / "surface-profile.csv",
        np.column_stack(
            [
                x,
                reference[0][surf],
                fields[("coarse", "hertz-point")][0][surf],
                fields[("coarse", "hertz-only")][0][surf],
                variation[surf],
            ]
        ),
        delimiter=",",
        header="x_um,reference_sigma1_mpa,coarse_point_sigma1_mpa,hertz_sigma1_mpa,observed_variation_mpa",
        comments="",
    )
    caption = "Single normal polar contact on the fixed hollow shell (Fn=0.2N, E=10GPa, nu=0.3, wall500um, rigid principal curvatures50/80 per m). (a) Independent97-grid bilinear pressure and analytic Hertz support (orange dashed). (b) Fine-reference maximum principal stress on the retained41x41 surface grid, without smoothing. Open circle and blue cross mark reference and coarse central-source candidate global sampled tensile maxima, respectively, using the full corner/probe/contact-edge bank. (c) Surface x-axis maximum principal stress, including edge-ring probes. Gray shading sums observed pointwise medium/fine body,65/97 pressure,3/5 source and256/512-near9/17-far integration variations plus the1.32494% reference tensile-peak edge-sampling change; this is a conditional numerical variation band, not a confidence interval. (d) Axis von Mises stress over the first150um of the500um wall; curves join retained samples. The Hertz plus shell residual uses one central finite-body regular response and the coarse research body mesh. All calculations use free tilt and consistent-mass inertial relief. Results apply to this one faceted polar-land normal case; neither native throughput nor arbitrary contact/friction accuracy is established."
    (args.output / "caption.txt").write_text(caption + "\n")
    (args.output / "interpretation.txt").write_text(
        "The surface tensile response is concentrated near the contact boundary, while the von Mises maximum lies below the contact center. A half-space-only description misses the shell's regular stress response: complete-field error78.24%, compared with2.19% for the coarse central-source correction. The associated sampled tensile/VM peak errors are5.02%/0.77%; adding measured reference variations gives conditional8.01%/0.82% envelopes. This is an offline single-case result; the figure does not establish generality or production cost. Symmetric tensile peak coordinates can switch sides; raw locations and near-maximum-set diagnostics remain in the numerical report.\n"
    )
    (args.output / "alt-text.txt").write_text(
        "Four panels show an elliptical normal-pressure patch, a surface principal-stress map with boundary maxima, surface stress profiles, and subsurface von Mises profiles. Adding the finite-shell regular response brings the Hertz estimate close to the independent reference. The half-space-only surface profile differs substantially.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "passed": not issues,
            "identity": runtime_identity(require_gpu_inventory=False),
            "text_qa_issues": issues,
            "editable_svg_text_elements": len(texts),
            "material_passport": contract["material_passport"],
            "claim_anchor": contract["claim_anchor"],
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )
    assert not issues, issues


if __name__ == "__main__":
    main()
