"""Plot complete-domain sampled stresses and tensor errors for a two-contact case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from threadpoolctl import threadpool_limits

from research.rigid_stress.shell_cpu.core import quad_shape

from .fixtures import cases_manifest, make_cases, unit
from .loads import assemble_sources, force_for_source
from .model import Mother
from .selection import build_bank


def sampled_mises(local: np.ndarray) -> np.ndarray:
    xx, yy, xy = local[..., 0], local[..., 1], local[..., 2]
    return np.sqrt(np.maximum(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy, 0.0))


def draw_map(ax, longitude, latitude, values, normalization, cmap, title, sources):
    artist = ax.scatter(longitude, latitude, c=values, s=10, marker="s", linewidths=0,
                        cmap=cmap, norm=normalization, rasterized=True)
    for index, source in enumerate(sources):
        direction = unit(source.direction)
        x, y = np.arctan2(direction[1], direction[0]), np.arcsin(direction[2])
        ax.scatter([x], [y], s=85, marker="*", c="white", edgecolors="black", linewidths=0.8, zorder=8)
        ax.annotate(f"C{index + 1}", (x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color="black", bbox={"facecolor": "white", "alpha": 0.85, "pad": 1})
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
    return artist


def study(output: Path, order: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    mother = Mother()
    bank = build_bank(mother)
    case = next(case for case in make_cases() if case.name == "two-far-asymmetric")
    raw = assemble_sources(mother, case.sources, order=order, audit=True)
    balanced = mother.balance(raw)
    q = {"Mother reference": mother.solve(np.sum(balanced, axis=1))}
    strategies = {"Continuous blend": "continuous", "Nearest one": "nearest", "Narrow transition": "narrow"}
    group_counts = {}
    for label, strategy in strategies.items():
        groups = {}
        for column, source in enumerate(case.sources):
            indices, coefficients = bank.select(source, strategy)
            for index, coefficient in zip(indices, coefficients):
                if int(index) not in groups:
                    groups[int(index)] = np.zeros(mother.n_dof)
                groups[int(index)] += coefficient * balanced[:, column]
        total = np.zeros(mother.n_dof)
        for index, force in groups.items():
            space = bank.spaces[index]
            total += space.lift(space.solve(force))
        q[label] = total
        group_counts[label] = len(groups)
    stress = {label: (mother.S @ value).reshape((2, len(mother.model.quads), 4, 3)) for label, value in q.items()}
    reference = stress["Mother reference"]
    mises = {label: sampled_mises(value) / 1e6 for label, value in stress.items()}
    tensor_errors = {label: np.sqrt(np.sum((value - reference)**2 * np.array([1.0, 1.0, 2.0]), axis=-1)) / 1e6
                     for label, value in stress.items()}
    shape = np.array([quad_shape(xi, eta)[0] for xi, eta in mother.model.gauss_points])
    directions = unit(np.einsum("gn,enp->egp", shape, mother.geometry.directions[mother.model.quads])).reshape((-1, 3))
    positions = np.einsum("gn,enp->egp", shape, mother.model.vertices[mother.model.quads])
    longitude = np.arctan2(directions[:, 1], directions[:, 0])
    latitude = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    stress_norm = Normalize(vmin=0.0, vmax=max(float(np.max(value)) for value in mises.values()))
    error_norm = Normalize(vmin=0.0, vmax=max(float(np.max(value)) for value in tensor_errors.values()))
    panels = tuple(stress)
    figure_paths = []
    for side_index, side in enumerate(("top", "bottom")):
        fig, axes = plt.subplots(2, 4, figsize=(19.5, 8.2), subplot_kw={"projection": "mollweide"})
        for index, label in enumerate(panels):
            title = label if label == "Mother reference" else f"{label} ({group_counts[label]} solve groups)"
            stress_artist = draw_map(axes[0, index], longitude, latitude, mises[label][side_index].reshape(-1),
                                     stress_norm, "viridis", title, case.sources)
            error_artist = draw_map(axes[1, index], longitude, latitude,
                                    tensor_errors[label][side_index].reshape(-1), error_norm, "magma",
                                    "Full tensor error" if index else "Reference error = 0", case.sources)
        fig.subplots_adjust(left=0.035, right=0.92, bottom=0.15, top=0.87, hspace=0.20, wspace=0.12)
        cax_stress = fig.add_axes([0.94, 0.54, 0.012, 0.29])
        cax_error = fig.add_axes([0.94, 0.20, 0.012, 0.29])
        fig.colorbar(stress_artist, cax=cax_stress, label="von Mises stress [MPa]")
        fig.colorbar(error_artist, cax=cax_error, label="Tensor Frobenius error [MPa]")
        fig.suptitle(f"Two asymmetric contacts: complete {side} surface stress and error", fontsize=17, y=0.97)
        loads = "; ".join(f"C{i+1}: radius {source.radius*1e3:g} mm, Fn {source.normal_force:g} N, "
                          f"Ft {source.friction*source.normal_force:g} N" for i, source in enumerate(case.sources))
        fig.text(0.04, 0.09, loads, fontsize=10)
        fig.text(0.04, 0.05, "All 3,456 mother Gauss points shown in material longitude/latitude. "
                 "White stars mark contact centers. Identical color scales on top and bottom.", fontsize=10)
        fig.text(0.04, 0.02, "Tensor error includes all in-plane components, with shear counted twice. "
                 "This is a same-discretization reference, not a continuum contact solution.", fontsize=9)
        path = output / f"two_contact_{side}.png"
        fig.savefig(path, dpi=170, metadata={"Software": "Matplotlib reconstructed CPU shell study"})
        plt.close(fig)
        figure_paths.append(path)
    arrays = {"positions": positions, "material_directions": directions}
    arrays.update({label.lower().replace(" ", "_"): value for label, value in stress.items()})
    np.savez_compressed(output / "fields.npz", **arrays)
    forces = [force_for_source(mother, source).tolist() for source in case.sources]
    result = {"case": cases_manifest((case,))[0], "quadrature_order": order,
              "global_forces_N": forces, "solve_groups": group_counts,
              "sample_points_per_side": len(longitude), "stress_range_MPa": [stress_norm.vmin, stress_norm.vmax],
              "tensor_error_range_MPa": [error_norm.vmin, error_norm.vmax],
              "figures": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in figure_paths],
              "derived_array_file": "fields.npz (ignored by Git, regenerated by this command)"}
    (output / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = study(args.output, args.order)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
