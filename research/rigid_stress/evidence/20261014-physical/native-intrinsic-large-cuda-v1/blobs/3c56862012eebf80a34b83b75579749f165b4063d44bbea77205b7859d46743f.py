"""Plot paired raw penetration and force under fixed-step contact regularization."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_physical_controls import export, file_identity, load_style
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.audit)
    assert int((args.audit / "returncode.txt").read_text()) == 0
    data_path = args.audit / "results/summary.json"
    report = json.loads(data_path.read_text())
    assert report["fixed_integration_interval_s"] == 0.00125
    rows = []
    for case in report["cases"]:
        final = case["final_geometric_checks"][-1]
        assert final["phase"] == "released_free_body_fixed_hand"
        for contact in final["contacts"]:
            assert contact["support_vertex_projects_inside_finite_box_face"]
            rows.append(
                {
                    "tau_ms": 1000 * case["contact_timeconst_s"],
                    "source_geom": contact["source_geom"],
                    "raw_penetration_mm": 1000 * contact["raw_collider_penetration_m"],
                    "normal_force_N": contact["normal_force_N"],
                    "time_s": final["time_s"],
                }
            )
    assert len(rows) == 6 and len({r["time_s"] for r in rows}) == 1
    with (args.output / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contract = {
        "claim_anchor": "franka-contact-regularization-before-feedback",
        "main_claim": "Contact regularization changes geometric penetration and normal loads at fixed physical inputs, integration interval and recorded weld parameters.",
        "material_passport": [file_identity(data_path), file_identity(args.style_module)],
        "panel_roles": {
            "a": "Both source-contact penetration depths compared with the physical wall thickness",
            "b": "Both source-contact forces at the same three numerical time constants",
        },
        "panel_spec": [
            {"panel": "a", "plot_type": "paired scatter", "x": "tau_ms", "y": "raw_penetration_mm"},
            {"panel": "b", "plot_type": "paired scatter", "x": "tau_ms", "y": "normal_force_N"},
        ],
        "caption": "Actual compound Panda fingers and full hollow egg, fixed1.25ms integration interval, identical52s control protocol, friction0.6 and temporary-weld parameters. Symbols retain both source contacts16/23 at the same51.99875s instant. No smoothing, intermediate-value estimate or statistical interval is drawn. The dashed line in panel a is the0.5mm wall thickness; plotted penetration is rigid geometric overlap, not material deformation. The body retains angular motion, and these observations do not establish a zero-regularization or deformable-contact convergence tail.",
        "alt_text": "Raw overlap decreases from about0.461mm to0.161mm and0.043mm as the contact time constant decreases from10ms to5ms and2.5ms. Normal forces increase from about0.353N to0.383N and0.395N.",
        "body_argument": "The largest overlap is close to the wall thickness. Smaller contact time constants reduce that numerical overlap and increase the rigid-controller loads. Material-deformation feedback must be separated from this contact-law effect before comparing stress accuracy.",
        "uncertainty": "Six paired deterministic values from three cases; no confidence interval or population estimate.",
        "limits": "A coupled deformable reference and matched supports/history/contact regularization remain required. This is not a physical operating-range claim.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    style = load_style(args.style_module)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.22, top=0.84, wspace=0.4)
    for geom, colour, marker, label in ((16, "#0072B2", "o", "Left contact"), (23, "#D55E00", "s", "Right contact")):
        selected = sorted([row for row in rows if row["source_geom"] == geom], key=lambda r: r["tau_ms"])
        for axis, key in zip(axes, ("raw_penetration_mm", "normal_force_N"), strict=True):
            axis.scatter(
                [r["tau_ms"] for r in selected],
                [r[key] for r in selected],
                color=colour,
                marker=marker,
                s=25,
                facecolors="none",
                linewidths=1,
                label=label,
            )
    axes[0].axhline(0.5, color="0.45", linestyle="--", linewidth=0.7)
    axes[0].text(8.6, 0.515, "Wall thickness", fontsize=6, ha="center")
    axes[0].set_ylim(0, 0.56)
    axes[0].set_yticks([0, 0.2, 0.4, 0.5])
    axes[0].set_ylabel("Rigid geometric overlap (mm)", fontsize=7)
    axes[0].set_title("Contact penetration", fontsize=7)
    axes[1].set_ylim(0.34, 0.41)
    axes[1].set_yticks([0.35, 0.37, 0.39, 0.41])
    axes[1].set_ylabel("Normal force (N)", fontsize=7)
    axes[1].set_title("Load under unchanged actuation", fontsize=7)
    axes[1].legend(loc="lower left", fontsize=6, frameon=False)
    for index, axis in enumerate(axes):
        axis.set_xlim(1.5, 11)
        axis.set_xticks([2.5, 5, 10])
        axis.set_xlabel("Contact time constant (ms)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.text(-0.12, 1.07, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
    export(fig, args.output / "franka_contact_regularization", style)


if __name__ == "__main__":
    main()
