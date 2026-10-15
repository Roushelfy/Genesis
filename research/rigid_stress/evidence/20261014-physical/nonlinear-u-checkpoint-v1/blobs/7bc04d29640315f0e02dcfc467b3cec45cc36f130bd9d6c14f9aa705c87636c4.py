"""Show nonlinear U mesh changes alongside separately defined resolved-load model diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .neohookean_reference import NeoHookeanBulk
from .plot_physical_controls import export, file_identity, load_style
from .qualify_articulated_precision import tensor_from_f
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def von_mises(tensor):
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.sum(dev**2, axis=(-2, -1)))


def source_redistribution_figure(models, output, style):
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 90 / 25.4))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.93, wspace=0.27, hspace=0.45)
    values_csv = []
    for row_index, (key, scale, unit, prefix, limit) in enumerate(
        [
            ("force_shift_from_resolved_prior_N", 1000, "mN", "F", 0.6),
            ("moment_shift_from_resolved_prior_Nm", 1e6, "µN m", "M", 7.2),
        ]
    ):
        for component in range(3):
            axis = axes[row_index, component]
            for report, offset, colour, marker, label, face in zip(
                models,
                [-0.10, 0.10],
                ["#0072B2", "#D55E00"],
                ["o", "s"],
                ["845 tets", "2339 tets"],
                ["#0072B2", "white"],
            ):
                model = next(item for item in report["rows"] if item["model"] == "body_source_prior")
                shifts = [chart[key][component] * scale for chart in model["source_charts"]]
                assert max(abs(value) for value in shifts) < limit
                axis.scatter(
                    np.arange(2) + offset,
                    shifts,
                    s=23,
                    marker=marker,
                    edgecolor=colour,
                    facecolor=face,
                    linewidth=0.8,
                    label=label,
                )
                values_csv.extend(
                    {
                        "panel": "redistribution_" + chr(97 + row_index * 3 + component),
                        "metric": prefix + "xyz"[component],
                        "case": label + "_" + side,
                        "value": value,
                        "units": unit,
                    }
                    for side, value in zip(["left", "right"], shifts)
                )
            axis.axhline(0, color="#777777", linewidth=0.6)
            axis.set_xlim(-0.45, 1.45)
            axis.set_ylim(-limit, limit)
            axis.set_xticks([0, 1], ["Left", "Right"])
            axis.set_title(prefix + "xyz"[component], fontsize=7)
            axis.text(
                -0.14,
                1.05,
                chr(97 + row_index * 3 + component),
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
            )
            if component == 0:
                axis.set_ylabel(("Force" if row_index == 0 else "Moment") + " shift (" + unit + ")")
    axes[0, 2].legend(frameon=False, loc="lower center", fontsize=6)
    return export(fig, output / "source_wrench_redistribution", style), values_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs=2, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.audit, *args.models]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    audit = json.loads((args.audit / "results/summary.json").read_text())
    assert len(audit["rows"]) == 2 and audit["common_material_points"] == 75
    models = [json.loads((run / "results/summary.json").read_text()) for run in args.models]
    assert all(row["physical_radius_m"] == 0.003 and row["objective_nonlinear_input"] for row in models)
    args.output.mkdir(parents=True, exist_ok=False)
    style = load_style(args.style_module)
    passports = [file_identity(args.style_module), file_identity(args.audit / "results/summary.json")]
    passports.extend(file_identity(run / "results/summary.json") for run in args.models)
    panels = []
    for row in audit["rows"]:
        directory = Path(row["run"]) / "solution"
        config = json.loads((directory / "configuration.json").read_text())
        with np.load(config["mesh"]) as saved:
            body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(directory / "fields.npz") as saved:
            vector = np.empty(body.basis.N)
            vector[saved["body_component_permutation"]] = saved["displacement"][: body.basis.N]
            force = saved["contact_force_N"][: body.basis.N].reshape(-1, 3)
        bulk = NeoHookeanBulk(body)
        faces = body.mesh.facets[:, body.mesh.boundary_facets()].T
        owners = body.mesh.f2t[0, body.mesh.boundary_facets()]
        local = np.argmax(body.mesh.t.T[owners, :, None] == faces[:, None, :], axis=1)
        barycentric = np.zeros((len(faces), 4))
        barycentric[np.arange(len(faces))[:, None], local] = 1 / 3
        # F is affine, but nonlinear Cauchy stress is not: evaluate at each
        # boundary centroid rather than averaging corner stresses.
        face_f = np.einsum("nc,ncij->nij", barycentric, bulk.corner_deformation_gradients(vector)[owners])
        values = np.asarray(von_mises(tensor_from_f(face_f, body.lame)), dtype=float) / 1000
        centres = []
        for side in [-1, 1]:
            weight = np.linalg.norm(force, axis=1) * (side * body.coordinates[:, 0] > 0)
            assert weight.sum() > 0
            centres.append(weight @ body.coordinates / weight.sum())
        panels.append((body.mesh.p.T, faces, values, np.array(centres), row))
        passports.extend(file_identity(path) for path in [Path(config["mesh"]), directory / "fields.npz"])
    contract = {
        "claim_anchor": "physical_nonlinear_u_spatial_and_load_models",
        "main_claim": "A small constitutive/numerical replay discrepancy does not close the large nonlinear U spatial change; finite-footprint model differences are separate conditional diagnostics.",
        "material_passport": passports,
        "panel_roles": {
            "a": "Complete845-tet nonlinear reference, boundary-centroid von Mises and unrestricted raw peak marker",
            "b": "Corresponding2339-tet reference on the same units, camera and colour limits",
            "c": "Separate deterministic75-probe, displacement, energy and raw-peak changes, normalized by the finer result",
            "d": "Three qualified native finite-footprint models at identical within-case source wrenches; two meshes remain separate",
        },
        "analysis_status": "Completed independent equilibrium/field audits and original native residuals; physical convergence remains unproven",
        "uncertainty": "Deterministic cases, no statistical confidence intervals or fitted order",
        "figure_size_mm": [183, 148],
        "colour_roles": {
            "stress": "cividis, sequential, shared0–120kPa",
            "coarse": "Wong blue filled circle",
            "medium": "Wong vermilion open square",
            "refinement": "neutral grey",
        },
        "required_or_high_value_materials": "Available: raw meshes/fields, independent audit, matched model reports and established style",
        "missing_materials": "No journal template or figure number supplied; repository supplement convention used",
        "can_proceed_without_missing_materials": True,
        "limitations": "Two levels do not establish convergence. Panels a–c use original final states; panel d uses the corresponding single additional constant-load holds. Surface colours evaluate each facet centroid directly and are not averaged across elements. The unrestricted raw global lower maximum and upper bound remain separate. Panel c probe norms and panel d volume-weighted model norms are distinct diagnostics, not an additive error budget. A3% guide is shown only for initial refinement diagnostics, not model acceptance.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    grid = [
        {
            "panel": key,
            "claim": value,
            "plot_type": "3D boundary stress"
            if key in ["a", "b"]
            else "horizontal bars"
            if key == "c"
            else "paired categorical points",
        }
        for key, value in contract["panel_roles"].items()
    ]
    (args.output / "panel_spec.json").write_text(json.dumps(grid, indent=2) + "\n")
    fig = plt.figure(figsize=(183 / 25.4, 148 / 25.4))
    layout = fig.add_gridspec(
        2, 2, left=0.13, right=0.91, bottom=0.12, top=0.90, wspace=0.48, hspace=0.38, height_ratios=[1.18, 1]
    )
    values_csv = []
    for index, (vertices, faces, values, centres, row) in enumerate(panels):
        axis = fig.add_subplot(layout[0, index], projection="3d", computed_zorder=False)
        collection = Poly3DCollection(
            vertices[faces] * 1000, linewidths=0, edgecolors="none", cmap="cividis", norm=Normalize(0, 120)
        )
        collection.set_array(values)
        axis.add_collection3d(collection)
        axis.scatter(
            *np.array(row["raw_peak_reference_position_m"]) * 1000,
            marker="*",
            s=38,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=10,
        )
        axis.scatter(
            *centres.T * 1000, marker="o", s=18, facecolor="white", edgecolor="#0072B2", linewidth=0.8, zorder=10
        )
        axis.set_xlim(-12, 12)
        axis.set_ylim(-12, 25)
        axis.set_zlim(-2, 2)
        axis.set_box_aspect([24, 37, 4])
        axis.view_init(elev=65, azim=-60)
        axis.set_xticks([-10, 0, 10])
        axis.set_yticks([-10, 0, 10, 20])
        axis.set_zticks([0])
        axis.set_xlabel("x (mm)", labelpad=0)
        axis.set_ylabel("y (mm)", labelpad=0)
        axis.set_zlabel("z (mm)", labelpad=0)
        axis.tick_params(labelsize=6, pad=0)
        axis.grid(False)
        for dimension in [axis.xaxis, axis.yaxis, axis.zaxis]:
            dimension.pane.set_visible(False)
        axis.set_title(
            f"{row['body_tetrahedra']} tetrahedra\nRaw peak lower: {row['raw_global_peak_lower_Pa'] / 1000:.2f} kPa",
            fontsize=7,
            pad=5,
        )
        axis.text2D(-0.12, 1.03, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
        values_csv.extend(
            [
                {
                    "panel": chr(97 + index),
                    "metric": key,
                    "case": row["body_tetrahedra"],
                    "value": row[key],
                    "units": "Pa",
                }
                for key in ["raw_global_peak_lower_Pa", "raw_global_peak_upper_Pa"]
            ]
        )
    bar = fig.colorbar(collection, cax=fig.add_axes([0.93, 0.58, 0.014, 0.23]))
    bar.set_label("von Mises stress (kPa)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    axis = fig.add_subplot(layout[1, 0])
    change = audit["adjacent_changes"][0]
    metrics = [
        ("Full tensor", "tensor_Pa"),
        ("von Mises", "vm_Pa"),
        ("Displacement", "objective_displacement_m"),
        ("Energy", "body_nonlinear_energy_J"),
        ("Raw peak", "raw_global_peak_lower_Pa"),
    ]
    relative = [
        100 * (change[key]["relative_change"] if isinstance(change[key], dict) else change[key]) for _, key in metrics
    ]
    axis.barh(np.arange(5), relative, height=0.54, color="#777777")
    axis.set_yticks(range(5), [label for label, _ in metrics])
    axis.invert_yaxis()
    axis.set_xlim(0, 52)
    axis.set_xlabel("Adjacent refinement change (%)")
    axis.set_title("Nonlinear reference fields", fontsize=7)
    axis.axvline(3, color="black", linestyle="--", linewidth=0.7)
    for position, ((_, key), value) in enumerate(zip(metrics, relative)):
        axis.text(value + 0.8, position, f"{value:.1f}", va="center", fontsize=6)
        values_csv.append({"panel": "c", "metric": key, "case": "845_to2339", "value": value, "units": "percent"})
    axis.text(-0.26, 1.04, "c", transform=axis.transAxes, fontsize=8, fontweight="bold")
    axis = fig.add_subplot(layout[1, 1])
    names = ["componentwise", "pair_union", "body_source_prior"]
    for report, offset, colour, marker, label, face in zip(
        models, [-0.10, 0.10], ["#0072B2", "#D55E00"], ["o", "s"], ["845 tets", "2339 tets"], ["#0072B2", "white"]
    ):
        chosen = [next(row for row in report["rows"] if row["model"] == name) for name in names]
        assert all(row["classification"] == "numerically_qualified" for row in chosen)
        values = [100 * row["volume_l2_tensor_relative_difference_from_resolved_contact"] for row in chosen]
        axis.scatter(
            np.arange(3) + offset,
            values,
            s=23,
            marker=marker,
            edgecolor=colour,
            facecolor=face,
            linewidth=0.8,
            label=label,
        )
        values_csv.extend(
            {"panel": "d", "metric": name, "case": label, "value": value, "units": "percent"}
            for name, value in zip(names, values)
        )
    axis.set_xticks(range(3), ["Component", "Pair union", "Source prior"])
    axis.set_xlim(-0.45, 2.45)
    axis.set_ylim(0, 4.2)
    axis.set_ylabel("Load-model tensor difference (%)")
    axis.set_title("Fixed 3 mm native footprint", fontsize=7)
    axis.legend(frameon=False, loc="lower right", fontsize=6)
    axis.text(-0.24, 1.04, "d", transform=axis.transAxes, fontsize=8, fontweight="bold")
    fig.text(
        0.14,
        0.035,
        "Star: raw global maximum; circle: contact-force centre. Dashed line: initial 3% refinement guide.",
        fontsize=6,
    )
    exports = export(fig, args.output / "nonlinear_u_comparison", style)
    redistribution_exports, redistribution_values = source_redistribution_figure(models, args.output, style)
    values_csv.extend(redistribution_values)
    with (args.output / "source_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["panel", "metric", "case", "value", "units"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(values_csv)
    caption = "Nonlinear U contact on two body/pad discretizations. (a,b) Reference-geometry boundary-centroid Cauchy von Mises fields share0–120kPa limits; stars retain the unrestricted raw global lower maxima, whose whole-volume upper bounds remain in the source data. Circles are nodal contact-force-magnitude centres on each arm. (c) Deterministic changes across75 common material probes, objective displacement, nonlinear body energy and raw peak, normalized by the finer result. The initial 3% refinement guide does not certify a two-level tail. (d) Separate volume-weighted linear-FEM tensor differences for three native finite-footprint models relative to objective resolved nodal loads, at fixed3mm radius and original within-case source wrenches. These use one additional constant-load hold, not the original final-state comparison. Pressure does not support the original free contact moments. Probe and model norms are distinct, and the results do not establish physical convergence or a validated operating range."
    (args.output / "caption.md").write_text(caption + "\n")
    redistribution_caption = "Signed local wrench shifts under the native body_source_prior model, relative to each original resolved nonlinear-contact source prior. Force and moment components use the reference material frame, with moments about the reference link COM. Opposing changes between the two shoes preserve the aggregate wrench while changing local loading. Both references use fixed3mm native footprints and friction0.8. The two discretizations and deterministic source samples do not establish a physical tail, statistical interval or general model ranking."
    (args.output / "caption_redistribution.md").write_text(redistribution_caption + "\n")
    metadata = {
        "scope": __doc__,
        "claim_anchor": contract["claim_anchor"],
        "material_passport": passports,
        "exports": exports,
        "redistribution_exports": redistribution_exports,
        "redistribution_figure_size_mm": [183, 90],
        "figure_size_mm": [183, 148],
        "rendered_visual_qa_performed": False,
        "goal_completion": False,
        "body_text_handoff": "The full probe tensor changes44.58% while the raw peak changes11.69%, so a scalar peak is insufficient to qualify spatial accuracy. At the separately resolved nonlinear holds, native linearization errors remain about0.003% and native finite-footprint model differences about3%; these separate diagnostics do not close the much larger physical spatial uncertainty.",
    }
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"exports": exports, "figure_size_mm": [183, 148]}))


if __name__ == "__main__":
    main()
