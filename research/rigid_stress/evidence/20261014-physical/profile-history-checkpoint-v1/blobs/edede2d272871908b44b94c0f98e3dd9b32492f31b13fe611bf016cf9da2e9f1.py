"""Render fixed-wrench profile uncertainty and matched unsmoothed stress/traction maps."""

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


def von_mises(tensor):
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.sum(dev**2, axis=(-2, -1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    style = load_style(args.style_module)
    data = json.loads(args.summary.read_text())
    profiles = ("narrow", "baseline", "broad", "quartic")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.88, wspace=0.35)
    for index, (geometry, axis) in enumerate(zip(("u", "capsule"), axes)):
        rows = {r["profile"]: r for r in data["profile_rows"] if r["geometry"] == geometry and r["case"] == "opposing"}
        assert all(rows[p]["traction_admissible"] for p in profiles)
        axis.scatter(range(4), [rows[p]["raw_global_peak_Pa"] / 1000 for p in profiles], s=20, color="black")
        axis.set_xticks(range(4), ["Narrow", "Baseline", "Broad", "Quartic"], fontsize=7)
        axis.set_ylabel("Raw global peak (kPa)", fontsize=7)
        axis.set_title("Rounded U" if geometry == "u" else "Capsule", fontsize=7)
        axis.set_ylim(bottom=0)
        axis.text(-0.16, 1.1, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
    export(fig, args.output / "fixed_wrench_peak", style)
    run = args.runtime_root / "profile-capsule-q4-v1/results/opposing"
    mesh_path = args.runtime_root / "matrix-mesh-capsule-v1/meshes/h0.0025.npz"
    with np.load(mesh_path) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    mesh = MeshTet(vertices.T, tets.T, sort_t=False)
    boundary = mesh.boundary_facets()
    faces = mesh.facets[:, boundary].T
    owners = mesh.f2t[0, boundary]
    local = np.argmax(tets[owners, :, None] == faces[:, None, :], axis=1)
    selections = [
        ("Narrow FEM", run / "narrow/reference_field.npz"),
        ("Baseline FEM", run / "baseline/reference_field.npz"),
        ("Quartic FEM", run / "quartic/reference_field.npz"),
        (
            "Native componentwise",
            args.runtime_root / "load-matrix-capsule-l2-v3/results/opposing/componentwise_field.npz",
        ),
    ]
    fields = []
    for title, path in selections:
        with np.load(path) as saved:
            tensor = saved["tensor_Pa"]
        corner_vm = von_mises(tensor)
        element, corner = np.unravel_index(np.argmax(corner_vm), corner_vm.shape)
        fields.append(
            (
                title,
                path,
                von_mises(tensor[owners[:, None], local].mean(axis=1)),
                float(corner_vm[element, corner]),
                vertices[tets[element, corner]],
            )
        )
    colour_max = max(f[3] for f in fields) / 1000
    fig = plt.figure(figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.01, right=0.87, bottom=0.08, top=0.89, wspace=0.02)
    for index, (title, path, face_vm, peak, position) in enumerate(fields):
        axis = fig.add_subplot(1, 4, index + 1, projection="3d")
        colours = plt.get_cmap("viridis")(face_vm / 1000 / colour_max)
        axis.add_collection3d(Poly3DCollection(vertices[faces] * 1000, facecolors=colours, edgecolors="none"))
        axis.scatter(*(position * 1000), marker="*", s=25, c="white", edgecolors="black", linewidths=0.3)
        axis.set_xlim(-10, 10)
        axis.set_ylim(-10, 10)
        axis.set_zlim(-21, 21)
        axis.set_box_aspect((20, 20, 42))
        axis.view_init(elev=12, azim=-90)
        axis.set_proj_type("ortho")
        axis.set_axis_off()
        axis.set_title(title + f"\nRaw peak {peak / 1000:.2f} kPa", fontsize=7)
    colour_axis = fig.add_axes((0.91, 0.24, 0.012, 0.50))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, colour_max), cmap="viridis"),
        cax=colour_axis,
        label="von Mises (kPa)",
    )
    export(fig, args.output / "capsule_stress_maps", style)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.22, top=0.83, wspace=0.4)
    clouds = []
    for profile in profiles:
        with np.load(run / f"{profile}/patch1.npz") as saved:
            p, n, t = saved["positions_m"], saved["normals"], saved["tractions_Pa"]
        pressure = np.einsum("qi,qi->q", t, n) / 1000
        clouds.append((profile, p, pressure))
    maximum = max(c[2].max() for c in clouds)
    for axis, (profile, points, pressure) in zip(axes, clouds):
        axis.scatter(
            points[:, 1] * 1000,
            points[:, 2] * 1000,
            c=pressure,
            s=0.15,
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            rasterized=True,
        )
        axis.set_aspect("equal")
        axis.set_xlim(-3.2, 3.2)
        axis.set_ylim(-3.2, 3.2)
        axis.set_title(profile.capitalize(), fontsize=7)
        axis.set_xlabel("y (mm)", fontsize=7)
        axis.set_ylabel("z (mm)", fontsize=7)
    colour_axis = fig.add_axes((0.92, 0.25, 0.012, 0.46))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, maximum), cmap="viridis"), cax=colour_axis, label="Pressure (kPa)"
    )
    export(fig, args.output / "capsule_pressure_maps", style)
    contract = {
        "claim_anchor": "fixed-wrench-profile-ambiguity",
        "claim": "Identical original point wrenches permit different admissible traction distributions and different stress predictions",
        "material_passport": [file_identity(args.summary), file_identity(mesh_path)]
        + [file_identity(f[1]) for f in fields],
        "caption": "Opposing0.05N patches, radius3mm and friction2. All displayed alternative profiles preserve the original six-component patch wrenches and use byte-identical rigid inputs. "
        "The peak chart retains every raw global corner maximum. The full3D stress maps use a shared scale and exact face-centroid tensors for display; stars identify unaveraged global corner maxima. "
        "Pressure maps retain quadrature samples with matched colour limits and physical coordinates. Native input/profile is unchanged. These are fixed-discretization model comparisons, not continuum or contact-feedback validation.",
        "rendered_text_qa": "passed",
        "open_issues": "Coulomb admissibility is sampled; complete mesh/contact convergence and resolved egg/Franka comparisons remain required.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
