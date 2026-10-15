"""Plot matched fixed-bulk egg stress maps and independently checked contact-integration diagnostics."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--stencils", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--export-pdf", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for run in [args.comparison, args.stencils]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        retain(run / "results/summary.json")
    comparison = json.loads((args.comparison / "results/summary.json").read_text())
    observations = json.loads((args.stencils / "results/summary.json").read_text())["actual_state_observations"]
    rows = comparison["rows"]
    assert [row["contact_refinement"] for row in rows] == [0, 1, 2]
    assert len({row["time_s"] for row in rows}) == 1
    args.output.mkdir(parents=True, exist_ok=False)
    retain(args.style)
    spec = importlib.util.spec_from_file_location("figure_style", args.style)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "egg-body-contact-quadrature"})
    contract = {
        "claim_anchor": "body-trace-field-tail-does-not-close-physical-contact",
        "main_claim": "A small fixed-bulk stress tail coexists with growing discrete force concentration and an unchanged finger-vertex integration weight.",
        "material_passport": inputs,
        "data_status": "Three deterministic independently force-audited states at6.25ms; no statistical uncertainty or fitted order.",
        "panels": {
            "a-c": "Matched reference geometry and shared MPa colour scale, with directly evaluated outer-front facet-centroid stress, actual front contact samples and original whole-volume raw maxima.",
            "d": "Common-volume-probe tensor and nonlinear-energy adjacent changes, with the declared initial3% diagnostic target.",
            "e": "Growing discrete lumped normal-force density norm on the body surface.",
            "f": "Actual per-stencil area weights distinguish a fixed finger vertex from a body vertex refined against the other finger.",
        },
        "chart_choice": "Matched spatial maps plus ordered marker/line diagnostics; separate physical units and no dual axes.",
        "limitations": "Maps show direct centroid evaluations on the front outer boundary, not smoothed stresses or the global extrema. Every raw global maximum is retained separately. Discrete density and area weights are not a continuous pressure reconstruction or physical contact patch. Bulk, finger, geometry and regularization remain unresolved.",
    }
    figure, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 140))
    figure.subplots_adjust(left=0.075, right=0.97, bottom=0.10, top=0.94, wspace=0.38, hspace=0.54)
    cmap, norm = plt.get_cmap("viridis"), matplotlib.colors.Normalize(0, 0.36)
    scalars, surfaces = [], []
    for index, row in enumerate(rows):
        run = Path(row["run"])
        closed_source(run)
        geometry = run / "results/reference_geometry.npz"
        state = run / f"results/step{row['step']:04d}.npz"
        field = args.comparison / f"results/case{index:02d}.npz"
        for path in [geometry, state, field]:
            retain(path)
        with np.load(geometry) as saved:
            body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(state) as saved:
            vector = np.empty(body.basis.N, dtype=np.longdouble)
            vector[saved["body_component_permutation"]] = saved["displacement"][: body.basis.N]
        bulk = WidePiolaBulk(body)
        corners = bulk.corner_deformation_gradients(vector)
        with np.load(field) as saved:
            np.testing.assert_allclose(
                tensor_from_f(corners, body.lame), saved["raw_corner_tensor_Pa"], rtol=1e-11, atol=1e-8
            )
            active = saved["active_body_contact_nodes"]
            contact_points = np.array(saved["surface_positions_m"][active], dtype=float)
        boundary = body.mesh.boundary_facets()
        faces, owners = body.mesh.facets[:, boundary].T, body.mesh.f2t[0, boundary]
        edges = np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1)
        graph = sparse.coo_matrix(
            (np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(body.mesh.nvertices,) * 2
        )
        _count, labels = connected_components(graph, directed=False)
        components = np.unique(labels[np.unique(faces)])
        assert len(components) == 2
        outer = max(
            components, key=lambda label: np.linalg.norm(body.mesh.p.T[labels == label] - body.com, axis=1).mean()
        )
        front = (labels[faces[:, 0]] == outer) & (body.mesh.p.T[faces].mean(axis=1)[:, 1] > body.com[1])
        faces, owners = faces[front], owners[front]
        local = np.argmax(body.mesh.t.T[owners, :, None] == faces[:, None, :], axis=1)
        face_f = corners[owners[:, None], local].mean(axis=1)
        tensors = tensor_from_f(face_f, body.lame)
        dev = tensors - np.trace(tensors, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.asarray(np.sqrt(1.5 * np.sum(dev**2, axis=(-2, -1))), dtype=float) / 1e6
        projected = 1000 * (body.mesh.p.T[faces] - body.com)[..., [0, 2]]
        axis = axes[0, index]
        axis.add_collection(PolyCollection(projected, facecolors=cmap(norm(vm)), edgecolors="none", rasterized=False))
        visible = contact_points[:, 1] > body.com[1]
        samples = 1000 * (contact_points[visible] - body.com)
        axis.scatter(samples[:, 0], samples[:, 2], s=10, facecolors="white", edgecolors="black", linewidths=0.45)
        peak = 1000 * (np.array(row["raw_peak_reference_position_m"]) - body.com)
        axis.scatter(peak[0], peak[2], marker="*", s=24, c="#D55E00", edgecolors="black", linewidths=0.3)
        axis.set(xlim=(-26, 26), ylim=(-36, 36), aspect="equal", xlabel="x (mm)", ylabel="z (mm)")
        axis.set_xticks([-20, 0, 20])
        axis.set_yticks([-30, 0, 30])
        axis.set_title(f"q = {index}; raw max {row['raw_peak_lower_Pa'] / 1e6:.6f} MPa", fontsize=6)
        surfaces.extend(
            {"level": index, "boundary_face": number, "centroid_vm_MPa": float(value)}
            for number, value in enumerate(vm)
        )
        scalars.append(
            {
                key: row[key]
                for key in [
                    "contact_refinement",
                    "raw_peak_lower_Pa",
                    "nonlinear_energy_J",
                    "active_lumped_contact_area_m2",
                    "lumped_body_normal_force_density_L2_N_per_m",
                ]
            }
        )
    colour_axis = figure.add_axes([0.36, 0.535, 0.30, 0.012])
    figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colour_axis,
        orientation="horizontal",
        label="Centroid von Mises stress (MPa)",
    )
    blue, orange = "#0072B2", "#D55E00"
    changes = comparison["comparisons"]
    axes[1, 0].plot(
        [1, 2], [100 * row["tensor"]["relative_l2_change"] for row in changes], "o-", color=blue, label="Tensor"
    )
    axes[1, 0].plot([1, 2], [100 * row["nonlinear_energy_J"] for row in changes], "s--", color=orange, label="Energy")
    axes[1, 0].axhline(3, color="0.45", linestyle=":", label="Initial 3% target")
    axes[1, 0].set(ylabel="Adjacent change (%)", ylim=(0, 10), xticks=[1, 2], xlabel="Finer body trace level")
    axes[1, 0].legend(loc="upper right", fontsize=5.5)
    axes[1, 1].plot([0, 1, 2], [row["lumped_body_normal_force_density_L2_N_per_m"] for row in rows], "o-", color=blue)
    axes[1, 1].set(
        ylabel="Lumped normal density norm (N/m)", ylim=(0, 600), xticks=[0, 1, 2], xlabel="Body trace level"
    )
    weights = []
    for group, marker, colour, label in [
        (1, "o-", blue, "Finger 1 vertex"),
        (2, "s--", orange, "Body vertex at finger 2"),
    ]:
        values = []
        for row in rows:
            observation = next(
                item
                for item in observations
                if item["source_run"] == row["run"] and Path(item["state"]).name == "step0005.npz"
            )
            stencil = next(item for item in observation["stencils"] if group in item["groups"])
            values.append(stencil["weight"] * 1e6)
            weights.append(
                {
                    "body_trace_level": row["contact_refinement"],
                    "finger_source": group,
                    "vertex_area_weight_m2": stencil["weight"],
                }
            )
        axes[1, 2].plot([0, 1, 2], values, marker, color=colour, label=label)
    axes[1, 2].set(
        yscale="log", ylabel="Vertex area weight (mm²)", xlabel="Body trace level", xticks=[0, 1, 2], ylim=(0.1, 20)
    )
    axes[1, 2].legend(loc="upper right", fontsize=5.2)
    axes[1, 2].set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    axes[1, 2].yaxis.set_minor_locator(NullLocator())
    for index, axis in enumerate(axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.19, 1.10, "abcdef"[index], transform=axis.transAxes, fontsize=8, fontweight="bold")
    for name, table in [
        ("metrics.csv", scalars),
        ("surface_centroids.csv", surfaces),
        ("stencil_weights.csv", weights),
    ]:
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    stem = args.output / "contact_quadrature"
    style.save_preview(figure, stem.with_name(stem.name + "_preview").with_suffix(".png"))
    issues = style.rendered_text_qa(figure)
    assert not issues, issues
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    if args.export_pdf:
        figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    (args.output / "summary.json").write_text(
        json.dumps({"inputs": inputs, "text_qa_issues": issues, "goal_completion": False}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
