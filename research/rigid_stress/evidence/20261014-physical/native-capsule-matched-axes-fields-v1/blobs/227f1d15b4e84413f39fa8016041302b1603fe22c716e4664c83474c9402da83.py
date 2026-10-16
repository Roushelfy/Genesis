"""Plot covered CAD stress fields and separated U refinement diagnostics with complete source trace."""

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
from matplotlib.ticker import NullLocator
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .qualify_articulated_precision import tensor_from_f
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-fields", type=Path, required=True)
    parser.add_argument("--u-pad-fields", type=Path, required=True)
    parser.add_argument("--u-body-fields", type=Path, required=True)
    parser.add_argument("--export-pdf", action="store_true")
    parser.add_argument("--accepted-source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    reports = []
    for run in [args.cad_fields, args.u_pad_fields, args.u_body_fields]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "results/summary.json"
        retain(path)
        reports.append(json.loads(path.read_text()))
    cad, pad, body_axis = reports
    assert [r["tetrahedra"] for r in cad["rows"]] == [31668, 64716]
    assert pad["declared_spatial_axis"] == "pad_at_fixed_body" and len(pad["rows"]) == 3
    assert body_axis["declared_spatial_axis"] == "body_at_fixed_pad" and len(body_axis["rows"]) == 3
    args.output.mkdir(parents=True, exist_ok=False)
    apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "covered-contact-axes-20261015"})
    figure, axes = plt.subplots(2, 2, figsize=mm_to_inches(183, 146))
    figure.subplots_adjust(left=0.095, right=0.96, bottom=0.10, top=0.94, hspace=0.66, wspace=0.43)
    norm, cmap = matplotlib.colors.Normalize(0, 50), plt.get_cmap("viridis")
    surfaces = []
    for index, row in enumerate(cad["rows"]):
        run = Path(row["run"])
        geometry, state, fields = (
            run / "results/reference_geometry.npz",
            run / f"results/step{row['step']:04d}.npz",
            args.cad_fields / f"results/case{index:02d}.npz",
        )
        for path in [geometry, state, fields]:
            retain(path)
        with np.load(geometry) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(state) as saved:
            vector = np.empty(reference.basis.N, dtype=np.longdouble)
            vector[saved["body_component_permutation"]] = saved["displacement"][: reference.basis.N]
        corners = WidePiolaBulk(reference).corner_deformation_gradients(vector)
        with np.load(fields) as saved:
            np.testing.assert_allclose(
                tensor_from_f(corners, reference.lame), saved["raw_corner_tensor_Pa"], rtol=1e-11, atol=1e-8
            )
            active = saved["active_body_contact_nodes"]
            contacts = np.asarray(saved["surface_positions_m"][active], dtype=float)
        boundary = reference.mesh.boundary_facets()
        faces, owners = reference.mesh.facets[:, boundary].T, reference.mesh.f2t[0, boundary]
        edges = np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1)
        graph = sparse.coo_matrix(
            (np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(reference.mesh.nvertices,) * 2
        )
        _, labels = connected_components(graph, directed=False)
        components = np.unique(labels[np.unique(faces)])
        assert len(components) == 2
        outer = max(
            components,
            key=lambda component: np.linalg.norm(
                reference.mesh.p.T[labels == component] - reference.com, axis=1
            ).mean(),
        )
        selected = (labels[faces[:, 0]] == outer) & (reference.mesh.p.T[faces].mean(axis=1)[:, 1] > reference.com[1])
        faces, owners = faces[selected], owners[selected]
        local = np.argmax(reference.mesh.t.T[owners, :, None] == faces[:, None, :], axis=1)
        tensor = tensor_from_f(corners[owners[:, None], local].mean(axis=1), reference.lame)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        vm = np.asarray(np.sqrt(1.5 * np.sum(dev**2, axis=(-2, -1))), dtype=float) / 1e6
        assert vm.max() * 1e6 <= row["raw_peak_upper_Pa"] + 1e-5
        projected = 1000 * (reference.mesh.p.T[faces] - reference.com)[..., [0, 2]]
        contact_points = 1000 * (contacts[contacts[:, 1] > reference.com[1]] - reference.com)
        peak = 1000 * (np.asarray(row["raw_peak_reference_position_m"]) - reference.com)
        axis = axes[0, index]
        inset = axis.inset_axes([0.62, 0.06, 0.35, 0.35])
        for target in [axis, inset]:
            target.add_collection(
                PolyCollection(projected, facecolors=cmap(norm(vm)), edgecolors="none", antialiased=False)
            )
            target.scatter(
                contact_points[:, 0],
                contact_points[:, 2],
                s=5 if target is axis else 1.5,
                facecolors="white",
                edgecolors="black",
                linewidths=0.3,
            )
            target.scatter(
                peak[0],
                peak[2],
                marker="*",
                s=22 if target is axis else 12,
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
            f"{row['tetrahedra']:,} tetrahedra; raw max {row['raw_peak_lower_Pa'] / 1e6:.2f} MPa", fontsize=6
        )
        inset.set(
            xlim=(peak[0] - 0.2, peak[0] + 0.2),
            ylim=(peak[2] - 0.2, peak[2] + 0.2),
            aspect="equal",
            xticks=[],
            yticks=[],
        )
        inset.set_title("Peak region", fontsize=5, pad=2, backgroundcolor="white")
        surfaces.extend(
            {"mesh": row["tetrahedra"], "boundary_facet": int(face), "centroid_vm_MPa": float(value)}
            for face, value in zip(boundary[selected], vm, strict=True)
        )
    colour_axis = figure.add_axes([0.34, 0.525, 0.33, 0.012])
    colour_bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colour_axis,
        orientation="horizontal",
        label="Centroid von Mises stress (MPa)",
    )
    colour_bar.solids.set_rasterized(False)
    change = cad["comparisons"][0]
    names = ["Tensor", "von Mises", "Energy", "Raw peak", "Displacement"]
    values = [
        100 * change["tensor"]["relative_l2_change"],
        100 * change["vm"]["relative_l2_change"],
        100 * change["nonlinear_energy_J"],
        100 * change["raw_peak_lower_Pa"],
        100 * change["objective_displacement"]["relative_l2_change"],
    ]
    axis = axes[1, 0]
    axis.plot(values, range(5), "o", color="#0072B2")
    for position, value in enumerate(values):
        axis.text(value + 0.35, position, f"{value:.3g}", fontsize=6, va="center")
    axis.axvline(3, color="0.45", linestyle=":", label="Initial 3% target")
    axis.set(xlim=(0, 16), ylim=(4.6, -0.6), yticks=range(5), yticklabels=names, xlabel="Adjacent CAD change (%)")
    axis.legend(loc="lower right", fontsize=5.2, frameon=False)
    table = []
    axis = axes[1, 1]
    for report, colour, marker, label in [
        (pad, "#0072B2", "o-", "Pad mesh"),
        (body_axis, "#D55E00", "s--", "Body + geometry"),
    ]:
        changes = report["adjacent_changes"]
        ys = [100 * r["tensor_Pa"]["relative_change"] for r in changes]
        axis.plot([0, 1], ys, marker, color=colour, label=label)
        table.extend(
            {
                "axis": label,
                "pair": i + 1,
                "tensor_change_percent": y,
                "energy_change_percent": 100 * r["body_nonlinear_energy_J"],
                "raw_peak_change_percent": 100 * r["raw_global_peak_lower_Pa"],
            }
            for i, (r, y) in enumerate(zip(changes, ys, strict=True))
        )
    axis.axhline(3, color="0.45", linestyle=":", label="Initial 3% target")
    axis.set(
        xlim=(-0.1, 1.1),
        ylim=(0.8, 65),
        yscale="log",
        xticks=[0, 1],
        xticklabels=["First pair", "Last pair"],
        ylabel="U tensor change (%)",
    )
    axis.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.legend(loc="lower left", fontsize=5.2, frameon=False)
    for index, axis in enumerate(axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        figure.text([0.055, 0.565][index % 2], [0.985, 0.455][index // 2], "abcd"[index], fontsize=8, fontweight="bold")
    for name, rows in [
        ("surface_centroids.csv", surfaces),
        (
            "cad_changes.csv",
            [{"observable": n, "adjacent_change_percent": v} for n, v in zip(names, values, strict=True)],
        ),
        ("u_axis_changes.csv", table),
    ]:
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    contract = {
        "claim_anchor": "energy-stability-does-not-close-contact-stress-convergence",
        "main_claim": "Small displacement/energy changes coexist with unresolved stress and geometry/contact axes.",
        "material_passport": inputs,
        "data_status": "Deterministic audited fields; no replicates, uncertainty bars or fitted convergence order.",
        "panels": {
            "a": "31668-tet directly evaluated outer-front facet-centroid stress with raw whole-volume maximum and actual contact samples.",
            "b": "Matched64716-tet map on the same0–50MPa colour scale, with the independently computed raw maximum.",
            "c": "Every declared CAD adjacent observable, retaining the tiny displacement change and large tensor/peak changes.",
            "d": "Three-level U pad and body/geometry sequences, shown separately without adding their norms.",
        },
        "chart_choice": "Matched maps and paired deterministic diagnostic points/lines; no statistical aggregation or dual axes.",
        "limitations": "Maps evaluate stress directly at facet centroids; colour does not replace whole-volume raw bounds. Each inset spans0.4mm by0.4mm around the independently found raw peak. CAD bulk, surface geometry and body trace remain coupled. U pad pairs are5→7→9; U body/geometry pairs are845→2339→5669 at fixed resolution5 pads. Two CAD levels and a conditional pad tail do not establish physical convergence.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    stem = args.output / "covered_contact_axes"
    save_preview(figure, stem.with_name(stem.name + "_preview").with_suffix(".png"))
    issues = rendered_text_qa(figure)
    assert not issues, issues
    figure.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    if args.export_pdf:
        assert args.accepted_source_qa is not None
        accepted = json.loads(args.accepted_source_qa.read_text())
        assert accepted["visual_qa_passed"]
        assert hashlib.sha256(stem.with_suffix(".svg").read_bytes()).hexdigest() == accepted["editable_svg_sha256"]
        assert (
            hashlib.sha256(stem.with_name(stem.name + "_preview").with_suffix(".png").read_bytes()).hexdigest()
            == accepted["preview_png_sha256"]
        )
        figure.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    assert inputs == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    (args.output / "summary.json").write_text(
        json.dumps(
            {"inputs": inputs, "text_qa_issues": issues, "figure_size_mm": [183, 146], "goal_completion": False},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
