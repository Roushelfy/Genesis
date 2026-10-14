"""Plot resolved-contact and native finite-footprint loads without smoothing."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_physical_controls import export, file_identity, load_style
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.mapping)
    assert int((args.mapping / "returncode.txt").read_text()) == 0
    report = json.loads((args.mapping / "results/summary.json").read_text())
    replay = Path(report["replay"])
    closed_source(replay)
    contact = json.loads((replay / "results/summary.json").read_text())
    normal = [r for r in contact["normal_stencils"] if r["shoe"] == 0]
    friction = [r for r in contact["friction_stencils"] if r["shoe"] == 0]
    points, forces, normals, areas = [], [], [], []
    for first, second in zip(normal, friction, strict=True):
        assert first["nodes"] == second["nodes"]
        force = np.array(first["body_resultant_N"])
        points.append(first["reference_application_point_m"])
        forces.append(force + second["body_resultant_N"])
        normals.append(force / np.linalg.norm(force))
        areas.append(first["weight"])
    clouds = [("Resolved contact stencils", np.array(points), np.array(forces), np.array(normals), np.array(areas))]
    passports = [file_identity(args.style_module), file_identity(args.mapping / "results/summary.json")]
    for name, title in (("componentwise", "Componentwise"), ("body_source_prior", "Body source prior")):
        row = next(r for r in report["rows"] if r["model"] == name)
        assert row["classification"] == "numerically_qualified"
        path = args.mapping / f"results/{name}_traction.npz"
        passports.append(file_identity(path))
        with np.load(path) as saved:
            selected = saved["source"] == 0
            clouds.append(
                (
                    title,
                    saved["positions_m"][selected],
                    saved["forces_N"][selected],
                    saved["normals"][selected],
                    saved["areas_m2"][selected],
                )
            )
    passports.append(file_identity(replay / "results/summary.json"))
    style = load_style(args.style_module)
    values = []
    for _, _, force, inward, area in clouds:
        pressure = np.sum(force * inward, axis=1) / area
        shear = np.linalg.norm(force - pressure[:, None] * area[:, None] * inward, axis=1) / area
        assert pressure.min() >= -1e-8
        assert np.max(shear - 0.8 * pressure) < 1e-6
        values.append((pressure / 1000, shear / 1000))
    contract = {
        "claim_anchor": "resolved-contact-finite-footprint-load-model-sensitivity",
        "main_claim": "Identical source wrenches leave the U peak nearly unchanged while the declared native finite-footprint law spreads a more concentrated resolved-contact load and changes its full stress field.",
        "material_passport": passports,
        "panel_roles": {
            "pressure_row": "Left-shoe material-coordinate normal samples on one scale",
            "shear_row": "Corresponding tangential samples on a separate shared scale",
            "redistribution": "Both physical shoes' signed local force changes under the aggregate prior",
        },
        "caption": "Left-shoe contact samples in material coordinates at0.05N per jaw and friction0.8. Each dot is an actual integration sample; no interpolation, element averaging or pressure smoothing is applied. Resolved contact values divide stencil forces by IPC stencil weights; native values use the production quadrature's physical areas. These are different declared measures, so dots do not establish a converged continuum contact-edge pressure. Pressure and shear each use shared limits. Componentwise and pair_union agree for these separated single-source pairs; pressure rejects the original free contact moments. The second plot retains signed local force shifts for both shoes while the original aggregate wrench is conserved.",
        "body_text_handoff": "At fixed3mm native radius, the full volume tensor difference is3.27% for componentwise and3.49% for body_source_prior, despite a nearly unchanged componentwise global peak. The reference contact and pad compliance still need spatial, regularization and nonlinear validation.",
        "alt_text": "Six scatter panels compare concentrated resolved normal and shear samples with broader native footprints. A companion signed dot plot shows the aggregate prior moving force between two physical shoes.",
        "limits": "One deterministic U gravity case; no statistical confidence interval, physical convergence, general model ranking or Franka equivalence is established.",
        "open_decisions": "Repository supplement, established Matplotlib style; no journal template or figure number supplied.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.13, top=0.9, wspace=0.16, hspace=0.32)
    for quantity in range(2):
        limit = max(v[quantity].max() for v in values)
        for column, (title, position, _, _, _) in enumerate(clouds):
            axis = axes[quantity, column]
            scatter = axis.scatter(
                position[:, 1] * 1000,
                position[:, 2] * 1000,
                c=values[column][quantity],
                cmap="magma",
                vmin=0,
                vmax=limit,
                s=17 if column == 0 else 4,
                linewidths=0,
            )
            axis.set_xlim(14.7, 20.5)
            axis.set_ylim(-2.3, 2.3)
            axis.set_xticks([16, 18, 20])
            axis.set_yticks([-2, 0, 2])
            axis.set_xlabel("y (mm)", fontsize=6)
            axis.tick_params(labelsize=6)
            if column == 0:
                axis.set_ylabel("z (mm)", fontsize=6)
            else:
                axis.set_yticklabels([])
            if quantity == 0:
                axis.set_title(title, fontsize=7)
            axis.text(
                0.04,
                0.94,
                chr(97 + 3 * quantity + column),
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
            )
            axis.text(
                0.96,
                0.05,
                f"max {values[column][quantity].max():.1f} kPa",
                transform=axis.transAxes,
                ha="right",
                fontsize=5.5,
            )
        colour = fig.colorbar(scatter, cax=fig.add_axes((0.92, 0.57 if quantity == 0 else 0.13, 0.015, 0.33)))
        colour.set_label("Pressure (kPa)" if quantity == 0 else "Shear (kPa)", fontsize=6)
        colour.ax.tick_params(labelsize=6)
    export(fig, args.output / "resolved_native_traction", style)
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.94)
    row = next(r for r in report["rows"] if r["model"] == "body_source_prior")
    for offset, source, marker, colour in ((-0.08, 0, "o", "#0072B2"), (0.08, 1, "^", "#D55E00")):
        shift = np.array(row["source_charts"][source]["force_shift_from_resolved_prior_N"]) * 1000
        axis.scatter(np.arange(3) + offset, shift, marker=marker, color=colour, s=20, label=f"Shoe {source + 1}")
    axis.axhline(0, color="black", linewidth=0.5)
    axis.set_ylim(-0.65, 0.65)
    axis.set_yticks([-0.5, 0, 0.5])
    axis.set_xticks(range(3), ("x", "y", "z"))
    axis.set_xlabel("Material-frame component", fontsize=6)
    axis.set_ylabel("Local force change (mN)", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.legend(fontsize=6, frameon=False, loc="upper left")
    export(fig, args.output / "resolved_native_redistribution", style)


if __name__ == "__main__":
    main()
