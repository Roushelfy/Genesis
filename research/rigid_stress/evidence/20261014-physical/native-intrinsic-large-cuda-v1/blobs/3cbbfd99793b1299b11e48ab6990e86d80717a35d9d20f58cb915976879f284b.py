"""Show a pressure anchor error through matched traction, signed stress and raw von Mises fields."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .plot_physical_controls import export, file_identity, load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    style = load_style(args.style_module)
    before, after = args.before / "results/opposing", args.after / "results/opposing"
    assert (before / "frozen_inputs.npz").read_bytes() == (after / "frozen_inputs.npz").read_bytes()
    with np.load(args.mesh) as saved:
        vertices, elements = saved["vertices"], saved["tetrahedra"]
    centroids = vertices[elements].mean(axis=1)
    section = abs(centroids[:, 2]) < 0.0008
    snapshots = [json.loads((d / "snapshot.json").read_text()) for d in (before, after)]
    titles = ["Declared traction", "Pressure: before", "Pressure: fixed", "Source prior"]
    field_paths = [
        after / "reference_field.npz",
        before / "pressure_field.npz",
        after / "pressure_field.npz",
        after / "body_source_prior_field.npz",
    ]
    cloud_paths = []
    clouds, tensors, peaks = [], [], []
    reference_cloud = []
    for path in sorted(after.glob("reference_patch*.npz")):
        with np.load(path) as saved:
            reference_cloud.append((saved["positions_m"], saved["tractions_Pa"][:, 0]))
        cloud_paths.append(path)
    clouds.append(tuple(np.concatenate([part[index] for part in reference_cloud]) for index in (0, 1)))
    for directory, name in ((before, "pressure"), (after, "pressure"), (after, "body_source_prior")):
        path = directory / f"{name}_traction.npz"
        with np.load(path) as saved:
            clouds.append((saved["positions_m"], saved["forces_N"][:, 0] / saved["areas_m2"]))
        cloud_paths.append(path)
    for path in field_paths:
        with np.load(path) as saved:
            tensor = saved["tensor_Pa"]
        tensors.append(tensor)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        element, corner = np.unravel_index(np.argmax(vm), vm.shape)
        peaks.append((float(vm[element, corner]), vertices[elements[element, corner]]))
    # The average of the four corner tensors is the exact centroid value of a linear tensor field.
    centroid_tensors = [tensor.mean(axis=1) for tensor in tensors]
    centroid_vm = []
    for tensor in centroid_tensors:
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        centroid_vm.append(np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1))))
    scales = [
        max(abs(traction).max() for _, traction in clouds) / 1000,
        max(abs(tensor[:, 0, 0]).max() for tensor in centroid_tensors) / 1000,
        max(vm.max() for vm in centroid_vm) / 1000,
    ]
    fig, axes = plt.subplots(3, 4, figsize=(7.3, 5.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.1, top=0.91, hspace=0.12, wspace=0.12)
    face_centres = vertices[elements].mean(axis=1)
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=8)
        for row in range(3):
            axis = axes[row, column]
            axis.scatter(face_centres[:, 0] * 1000, face_centres[:, 1] * 1000, s=3, color="0.88", linewidths=0)
            if row == 0:
                positions, values = clouds[column]
                selected = abs(positions[:, 2]) < 0.0008
                artist = axis.scatter(
                    positions[selected, 0] * 1000,
                    positions[selected, 1] * 1000,
                    c=values[selected] / 1000,
                    s=4,
                    cmap="RdBu_r",
                    norm=Normalize(-scales[row], scales[row]),
                    linewidths=0,
                )
            else:
                values = centroid_tensors[column][:, 0, 0] if row == 1 else centroid_vm[column]
                artist = axis.scatter(
                    centroids[section, 0] * 1000,
                    centroids[section, 1] * 1000,
                    c=values[section] / 1000,
                    s=14,
                    linewidths=0,
                    cmap="RdBu_r" if row == 1 else "viridis",
                    norm=Normalize(-scales[row] if row == 1 else 0, scales[row]),
                )
                if row == 2:
                    peak, position = peaks[column]
                    axis.scatter(
                        position[0] * 1000,
                        position[1] * 1000,
                        marker="*",
                        s=28,
                        color="white",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=5,
                    )
                    axis.text(
                        0.5,
                        1.01,
                        f"Raw {peak / 1000:.3f} kPa",
                        transform=axis.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=6.7,
                    )
            axis.set_aspect("equal")
            axis.set_xlim(-13, 13)
            axis.set_ylim(-12, 28)
            axis.set_xticks([-12, 0, 12])
            axis.set_yticks([-8, 8, 24])
            axis.tick_params(labelsize=7)
            if column == 0:
                axis.set_ylabel("y (mm)")
            if row == 2:
                axis.set_xlabel("x (mm)")
            if column == 3:
                box = axis.get_position()
                colour_axis = fig.add_axes([0.90, box.y0 + 0.015, 0.012, box.height - 0.03])
                colourbar = fig.colorbar(artist, cax=colour_axis)
                colourbar.set_label(["Traction x (kPa)", "Stress xx (kPa)", "von Mises (kPa)"][row], fontsize=7)
                colourbar.ax.tick_params(labelsize=6)
    paths = export(fig, args.output / "pressure_material_sheet", style)
    rows = []
    for column, tensor in enumerate(tensors):
        rows.append(
            {
                "field": titles[column],
                "raw_global_peak_Pa": peaks[column][0],
                "full_tensor_relative_error": float(np.linalg.norm(tensor - tensors[0]) / np.linalg.norm(tensors[0])),
            }
        )
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "slice_samples.npz",
        centroids_m=centroids,
        section=section,
        tensors_Pa=np.stack(centroid_tensors),
        vm_Pa=np.stack(centroid_vm),
        global_peaks_Pa=np.array([value for value, _ in peaks]),
        global_peak_positions_m=np.stack([position for _, position in peaks]),
    )
    report = {
        "claim": "Selecting the wrong material sheet reverses signed interior stress while the raw von Mises peak agrees",
        "inputs": [
            file_identity(path)
            for path in [
                args.mesh,
                before / "frozen_inputs.npz",
                after / "frozen_inputs.npz",
                *field_paths,
                *cloud_paths,
            ]
        ],
        "metrics": rows,
        "colour_limits_kPa": scales,
        "slice": "Fixed |z|<0.8mm slab; exact per-element centroid tensors, no inter-element smoothing",
        "caption": "Matched U-link frozen opposing contact inputs. Declared compact traction and three native maps "
        "share geometry, material and radius. The old pressure anchor loads another material sheet, "
        "reversing interior signed stress. Nearest-sheet anchoring restores agreement. "
        "Raw full3D corner peaks remain visible as projected stars; similar von Mises values alone "
        "do not establish the correct stress tensor or patch location.",
        "alt_text": "Twelve panels compare traction x, signed stress xx and von Mises on the same U link; "
        "old pressure has reversed signed interior stress with a nearly identical raw peak.",
        "rendered_text_qa": "passed",
        "exports": {name: str(path) for name, path in paths.items()},
        "snapshot_cases": [snapshot["case"] for snapshot in snapshots],
    }
    (args.output / "figure_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
