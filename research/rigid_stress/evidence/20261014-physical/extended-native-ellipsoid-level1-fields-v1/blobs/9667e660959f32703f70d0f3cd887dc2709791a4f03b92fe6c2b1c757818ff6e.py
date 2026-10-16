"""Plot the unresolved three-level CAD contact tail with every plotted value traced to closed audits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import apply_nature_style, mm_to_inches, rendered_text_qa, save_preview
from matplotlib.collections import PolyCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    reports = []
    for run in [args.comparison, args.source_audit, args.localization]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        retain(path)
        reports.append(json.loads(path.read_text()))
    fields, sources, localization = reports
    assert [row["tetrahedra"] for row in fields["rows"]] == [31668, 64716, 171228]
    assert (
        localization["all_original_probes_retained"]
        and localization["original_complete_weighted_tensor_changes_reproduced"]
    )
    assert fields["common_physical_probes"] == localization["complete_probe_count"] == 158340
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "claim_anchor": "covered-contact-tail-remains-localized-and-unresolved",
        "main_claim": "Refinement increases tensor and raw-peak differences while discrete contact forces concentrate near a fixed sharp finger feature.",
        "material_passport": inputs,
        "panels": {
            "a-c": "Matched outer-front boundary maps at31668/64716/171228 tetrahedra, each colored by maximum of that face's three element-side corner values; identical0–60MPa scale and0.4mm insets. Whole-volume raw peak remains an independent star and title value.",
            "d": "Both adjacent full-probe tensor/peak/energy changes, retaining their distinct behaviors and predeclared3% diagnostic target.",
            "e": "All-node maximum discrete body normal force/lumped-area for both captured finger sources; these are not continuous pressures.",
            "f": "Cumulative complete weighted squared tensor changes by distance to the union of original nonzero normal samples, including all158340 probes.",
        },
        "data_status": "Deterministic closed numerical fields, no fitted order, statistical uncertainty or independent replicates.",
        "chart_choice": "Matched spatial maps plus adjacent changes, separate source concentration and exact additive diagnostic localization. Colors have marker/line redundancy.",
        "limitations": "Boundary color is a facewise corner diagnostic, not smoothing or a replacement global maximum. Reference-coordinate sample distances are not continuous patch boundaries. No original probe lies within25um of those samples, so the localization curves do not resolve the innermost contact field. Seed-volume weights are diagnostic, not exact cross-mesh integration. Coupled geometry/bulk/contact changes and a measured sharp edge do not alone prove a continuum singularity or convergence.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "covered-cad-tail-20261015"})
    fig, axes = plt.subplots(2, 3, figsize=mm_to_inches(183, 151))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.10, top=0.94, wspace=0.52, hspace=0.69)
    norm, cmap = matplotlib.colors.Normalize(0, 60), plt.get_cmap("viridis")
    center = np.asarray(fields["rows"][0]["com_reference_m"])
    zoom = 1000 * (np.asarray(fields["rows"][-1]["raw_peak_reference_position_m"]) - center)
    surface_data, concentration = [], []
    local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    for index, row in enumerate(fields["rows"]):
        path = args.comparison / f"results/case{index:02d}.npz"
        retain(path)
        with np.load(path) as data:
            vertices, tets, corner_vm = [data[key].copy() for key in ["vertices", "tetrahedra", "raw_corner_vm_Pa"]]
        np.testing.assert_allclose(corner_vm.max(), row["raw_peak_lower_Pa"], rtol=1e-12, atol=0)
        faces = tets[:, local_faces].reshape(-1, 3)
        _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
        boundary = first[count == 1]
        faces, owners, local = faces[boundary], boundary // 4, local_faces[boundary % 4]
        edges = np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1)
        graph = sparse.coo_matrix(
            (np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(len(vertices),) * 2
        )
        _, labels = connected_components(graph, directed=False)
        components = np.unique(labels[np.unique(faces)])
        assert len(components) == 2
        outer = max(
            components, key=lambda component: np.linalg.norm(vertices[labels == component] - center, axis=1).mean()
        )
        selected = (labels[faces[:, 0]] == outer) & (vertices[faces].mean(axis=1)[:, 1] > center[1])
        faces, owners, local = faces[selected], owners[selected], local[selected]
        vm = np.asarray(corner_vm[owners[:, None], local].max(axis=1), dtype=float) / 1e6
        projected = 1000 * (vertices[faces] - center)[..., [0, 2]]
        path = args.source_audit / f"results/case{index:02d}-step{row['step']:04d}.npz"
        retain(path)
        with np.load(path) as data:
            rest, groups, normal = [
                data[key].copy() for key in ["rest_surface_positions_m", "groups", "normal_force_by_source_N"]
            ]
        active = (groups == 0) & np.any(normal[1] != 0, axis=1)
        contacts = 1000 * (rest[active] - center)
        peak = 1000 * (np.asarray(row["raw_peak_reference_position_m"]) - center)
        axis = axes[0, index]
        inset = axis.inset_axes([0.56, 0.10, 0.43, 0.36])
        for target in [axis, inset]:
            target.add_collection(
                PolyCollection(projected, facecolors=cmap(norm(vm)), edgecolors="none", antialiased=False)
            )
            target.scatter(
                contacts[:, 0],
                contacts[:, 2],
                s=3 if target is axis else 1.1,
                facecolors="white",
                edgecolors="black",
                linewidths=0.25,
            )
            target.scatter(
                peak[0],
                peak[2],
                marker="*",
                s=20 if target is axis else 14,
                color="#D55E00",
                edgecolors="black",
                linewidths=0.25,
            )
        axis.set(
            xlim=(-26, 26),
            ylim=(-36, 36),
            aspect="equal",
            xlabel="x (mm)",
            ylabel="z (mm)",
            xticks=[-20, 0, 20],
            yticks=[-30, 0, 30],
        )
        axis.set_title(
            f"{row['tetrahedra']:,} tets\nRaw max {row['raw_peak_lower_Pa'] / 1e6:.2f} MPa", fontsize=6, pad=4
        )
        inset.set(
            xlim=(zoom[0] - 0.2, zoom[0] + 0.2),
            ylim=(zoom[2] - 0.2, zoom[2] + 0.2),
            aspect="equal",
            xticks=[],
            yticks=[],
        )
        inset.set_title("0.4 mm window", fontsize=5, pad=2, backgroundcolor="white")
        surface_data.extend(
            {"tetrahedra": row["tetrahedra"], "owner_tetrahedron": int(owner), "corner_maximum_MPa": float(value)}
            for owner, value in zip(owners, vm, strict=True)
        )
        selected_source = [
            item for item in sources["rows"] if item["run"] == row["run"] and item["step"] == row["step"]
        ]
        assert len(selected_source) == 1
        concentration.extend(
            {
                "tetrahedra": row["tetrahedra"],
                "source": item["source"],
                "maximum_discrete_normal_density_MPa": item["normal_body_density"][
                    "maximum_sample_force_over_lumped_area_Pa"
                ]
                / 1e6,
            }
            for item in selected_source[0]["sources"]
        )
    colour_axis = fig.add_axes([0.29, 0.527, 0.42, 0.011])
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colour_axis,
        orientation="horizontal",
        label="Boundary corner maximum (MPa)",
    )
    bar.solids.set_rasterized(False)
    axis = axes[1, 0]
    changes = []
    for key, label, style, colour in [
        ("tensor", "Tensor", "o-", "#0072B2"),
        ("raw_peak_lower_Pa", "Raw peak", "s--", "#D55E00"),
        ("nonlinear_energy_J", "Energy", "^:", "#000000"),
    ]:
        values = [
            100 * (pair[key]["relative_l2_change"] if key == "tensor" else pair[key]) for pair in fields["comparisons"]
        ]
        axis.plot([1, 2], values, style, color=colour, label=label)
        changes.extend({"pair": i + 1, "observable": label, "change_percent": value} for i, value in enumerate(values))
    axis.axhline(3, color="0.5", linestyle=":", linewidth=0.7)
    axis.set(xlim=(0.85, 2.15), ylim=(0, 20), xticks=[1, 2], xlabel="Adjacent refinement pair", ylabel="Change (%)")
    axis.legend(loc="upper left", ncol=1, fontsize=5)
    axis = axes[1, 1]
    for source, style, colour in [(1, "o-", "#0072B2"), (2, "s--", "#D55E00")]:
        points = [item for item in concentration if item["source"] == source]
        axis.plot(
            [1, 2, 3],
            [item["maximum_discrete_normal_density_MPa"] for item in points],
            style,
            color=colour,
            label=f"Finger source {source}",
        )
    axis.set(
        xlim=(0.85, 3.15),
        ylim=(0, 410),
        yticks=[0, 100, 200, 300, 400],
        xticks=[1, 2, 3],
        xlabel="CAD refinement level",
        ylabel="Max discrete normal density (MPa)",
    )
    axis.legend(loc="upper left", fontsize=5)
    axis = axes[1, 2]
    distance_data = []
    for index, (row, style, colour) in enumerate(
        zip(localization["rows"], ["o-", "s--"], ["#0072B2", "#D55E00"], strict=True)
    ):
        bins = row["distance_bins"][:-1]
        distance = [item["upper_distance_m_exclusive"] * 1e6 for item in bins]
        cumulative = 100 * np.cumsum([item["fraction_of_complete_squared_tensor_change"] for item in bins])
        axis.plot(distance, cumulative, style, color=colour, label=f"Pair {index + 1}")
        distance_data.extend(
            {"pair": index + 1, "upper_distance_um": x, "cumulative_squared_change_percent": float(y)}
            for x, y in zip(distance, cumulative, strict=True)
        )
    axis.set(
        xscale="log",
        xlim=(22, 560),
        ylim=(0, 110),
        xticks=[25, 100, 500],
        xticklabels=["25", "100", "500"],
        xlabel="Distance to normal samples (µm)",
        ylabel="Cumulative squared change (%)",
    )
    axis.minorticks_off()
    axis.legend(loc="upper left", fontsize=5)
    for index, axis in enumerate(axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        position = axis.get_position()
        fig.text(
            position.x0 - 0.035,
            0.97 if index < 3 else position.y1 + 0.025,
            "abcdef"[index],
            fontsize=8,
            fontweight="bold",
        )
    for name, records in [
        ("boundary_corner_values.csv", surface_data),
        ("adjacent_changes.csv", changes),
        ("source_concentration.csv", concentration),
        ("distance_localization.csv", distance_data),
    ]:
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    stem = args.output / "covered_cad_tail"
    preview = stem.with_name(stem.name + "_preview").with_suffix(".png")
    save_preview(fig, preview, dpi=180)
    issues = rendered_text_qa(fig)
    (args.output / "rendered_text_qa.json").write_text(json.dumps({"issues": issues}, indent=2) + "\n")
    assert not issues, issues
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    if args.accepted_source_qa:
        retain(args.accepted_source_qa)
        accepted = json.loads(args.accepted_source_qa.read_text())
        assert accepted["visual_qa_passed"]
        assert hashlib.sha256(stem.with_suffix(".svg").read_bytes()).hexdigest() == accepted["editable_svg_sha256"]
        assert hashlib.sha256(preview.read_bytes()).hexdigest() == accepted["preview_png_sha256"]
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=1000)
    plt.close(fig)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "input_sha256": inputs,
                "text_qa_issues": issues,
                "figure_size_mm": [183, 151],
                "publication_exports_after_visual_qa": bool(args.accepted_source_qa),
                "goal_completion": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
