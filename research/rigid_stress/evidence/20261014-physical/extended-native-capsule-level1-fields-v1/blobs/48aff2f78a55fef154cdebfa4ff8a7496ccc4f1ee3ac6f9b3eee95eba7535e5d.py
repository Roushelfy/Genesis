"""Show full3D contact-load replay agreement and the pad strain limitation."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from skfem import MeshTet

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_physical_controls import export, file_identity, load_style
from .plot_profile_references import von_mises
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gravity-replay", type=Path, required=True)
    parser.add_argument("--zero-replay", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    style = load_style(args.style_module)
    reports = []
    for run in (args.gravity_replay, args.zero_replay):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        reports.append(json.loads((run / "results/summary.json").read_text()))
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    mesh = MeshTet(vertices.T, tets.T, sort_t=False)
    boundary = mesh.boundary_facets()
    faces, owners = mesh.facets[:, boundary].T, mesh.f2t[0, boundary]
    local = np.argmax(tets[owners, :, None] == faces[:, None, :], axis=1)
    with np.load(args.gravity_replay / "results/contact_hold.npz") as saved:
        reference = saved["body_tensor_Pa"]
        n_body_nodes = len(saved["body_force_N"]) // 3
        positions = saved["rest_positions"][:n_body_nodes]
        forces = saved["source_normal_force_N"][:, :n_body_nodes]
        weights = np.linalg.norm(forces, axis=-1)
        centres = np.einsum("sn,nj->sj", weights, positions) / weights.sum(axis=1)[:, None]
    with np.load(args.gravity_replay / "results/native_nodal_replay.npz") as saved:
        raw = saved["raw_native_tensor_Pa"]
        native = np.zeros((*raw.shape[:2], 3, 3))
        i, j = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]
        native[..., i, j] = raw
        native[..., j, i] = raw
    limit = max(von_mises(reference).max(), von_mises(native).max()) / 1000
    fig = plt.figure(figsize=(7.2, 4.25))
    fig.subplots_adjust(left=0.02, right=0.85, bottom=0.08, top=0.89, wspace=0.04, hspace=0.02)
    for view, (elevation, azimuth) in enumerate(((60, -55), (78, 125))):
        for column, (title, tensor) in enumerate(
            (("Coupled FEM contact", reference), ("Native resolved-load replay", native))
        ):
            axis = fig.add_subplot(2, 2, 2 * view + column + 1, projection="3d", computed_zorder=False)
            values = von_mises(tensor[owners[:, None], local].mean(axis=1)) / 1000
            triangles = Poly3DCollection(
                vertices[faces] * 1000, linewidth=0.04, edgecolor=(0, 0, 0, 0.15), cmap="magma"
            )
            triangles.set_array(values)
            triangles.set_clim(0, limit)
            axis.add_collection3d(triangles)
            vm = von_mises(tensor)
            element, corner = np.unravel_index(vm.argmax(), vm.shape)
            peak = vertices[tets[element, corner]] * 1000
            axis.scatter(*peak, marker="*", s=40, color="white", edgecolor="black", linewidth=0.6, zorder=10)
            axis.scatter(
                *centres.T * 1000, marker="o", s=17, color="#56B4E9", edgecolor="black", linewidth=0.4, zorder=10
            )
            for dim, setter in enumerate((axis.set_xlim, axis.set_ylim, axis.set_zlim)):
                setter(vertices[:, dim].min() * 1000 - 0.5, vertices[:, dim].max() * 1000 + 0.5)
            axis.set_box_aspect(np.ptp(vertices, axis=0))
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_xticks([-10, 0, 10])
            axis.set_yticks([-10, 0, 20])
            axis.set_zticks([0])
            axis.set_xlabel("x (mm)", fontsize=6, labelpad=-3)
            axis.set_ylabel("y (mm)", fontsize=6, labelpad=-1)
            axis.set_zlabel("z (mm)", fontsize=6, labelpad=1)
            axis.tick_params(labelsize=5, pad=0)
            axis.grid(False)
            if view == 0:
                axis.set_title(title, fontsize=7)
                axis.text2D(0.02, 0.94, chr(97 + column), transform=axis.transAxes, fontsize=8, fontweight="bold")
    colour = fig.colorbar(triangles, cax=fig.add_axes((0.89, 0.2, 0.015, 0.56)))
    colour.set_label("von Mises stress (kPa)", fontsize=7)
    colour.ax.tick_params(labelsize=6)
    fig.text(0.12, 0.015, "White star: raw global maximum; blue circle: normal-load centre", fontsize=6)
    export(fig, args.output / "resolved_u_stress", style)
    fig, axis = plt.subplots(figsize=(3.5, 2.65))
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.24, top=0.93)
    for offset, report, marker, title in ((-0.07, reports[0], "o", "Gravity"), (0.07, reports[1], "^", "Zero gravity")):
        values = [p["maximum_symmetric_strain_norm"] for p in report["part_small_strain_diagnostics"]]
        axis.scatter(
            np.arange(3) + offset, values, s=23, marker=marker, facecolor="none", edgecolor="black", label=title
        )
    axis.set_yscale("log")
    axis.set_ylim(5e-6, 0.12)
    axis.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    axis.set_xticks(range(3), ("U link", "Left pad", "Right pad"), fontsize=7)
    axis.set_ylabel("Maximum symmetric strain norm", fontsize=7)
    axis.tick_params(labelsize=6)
    axis.legend(loc="center left", fontsize=6, frameon=False)
    export(fig, args.output / "resolved_u_strain", style)
    contract = {
        "claim_anchor": "resolved-u-contact-replay-and-pad-strain-limit",
        "main_claim": "Converged full3D U contact loads reproduce the raw field in native nodal recovery; 5–6% local pad strain limits physical interpretation of the chosen linear pads.",
        "panel_roles": {
            "stress_left": "Independent coupled FEM contact field",
            "stress_right": "Actual native engine full tensor on the identical resolved nodal loads",
            "second_view": "Expose remaining material-boundary faces and the inner bend",
            "strain": "Display deterministic body/pad strain disparity without invented statistical uncertainty",
        },
        "material_passport": [
            file_identity(args.mesh),
            file_identity(args.style_module),
            *[file_identity(run / "results/summary.json") for run in (args.gravity_replay, args.zero_replay)],
            file_identity(args.gravity_replay / "results/contact_hold.npz"),
            file_identity(args.gravity_replay / "results/native_nodal_replay.npz"),
        ],
        "caption": "Full3D rounded U at fixed0.05N per backplate and Coulomb coefficient0.8, following one constant-load hold. Colours show exact face-centroid stresses on a shared scale, with no averaging between elements. White stars mark original unaveraged corner maxima; blue circles mark equivalent normal-load centres in material coordinates. A second view exposes other boundary faces. The strain plot retains single deterministic maxima for each part; no confidence intervals apply.",
        "body_text_handoff": "The inner bend carries the global peak away from the arm contacts. Native raw tensor and independent contact-body stress agree under direct nodal replay. Pads strain much more than the10GPa link, exposing an assumption to investigate before claiming physical operating-range accuracy.",
        "alt_text": "Two views of matching U-shaped stress fields with the inner-bend maximum and two arm load centres marked. A logarithmic dot plot shows link strain near10^-5 and pad strain near0.05–0.06.",
        "placement": "Repository contact-validation progress report, beside numerical and physical scope limitations",
        "open_decisions": "No journal template or final figure number supplied; repository supplement exports follow the established style.",
        "limitations": "Direct nodal replay isolates elastic recovery. Finite-footprint mapping, regularization/spatial convergence, nonlinear pad response, contact-feedback differences and actual Franka matching remain required.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
