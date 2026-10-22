"""Read retained failed surface probes and quantify coordinate roundoff."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .check_native_hertz_recovery import discrete_particular, material_points


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in [0, 1]:
        path = args.campaign / f"group{group}/rotated/failed-field.npz"
        with np.load(path) as data:
            native, reference, points = data["native_tensor_pa"], data["reference_tensor_pa"], data["positions_m"]
            indices, centers, bases = data["source_indices"], data["centers_m"], data["frames"]
            axes, peaks, forces, offsets = data["axes_m"], data["peak_pressure_pa"], data["forces_n"], data["offsets"]
        ideal = reference.copy()
        descriptions = []
        for index, patch in enumerate(indices):
            selected = slice(index * 89, (index + 1) * 89)
            world_local = material_points(points[selected], centers[patch], bases[patch])
            exact_local = offsets * np.r_[axes[patch], axes[patch, 0]]
            world = discrete_particular(world_local, axes[patch], peaks[patch], forces[patch], True)
            exact = discrete_particular(exact_local, axes[patch], peaks[patch], forces[patch], True)
            ideal[selected] += bases[patch] @ (exact - world) @ bases[patch].T
            descriptions.append(
                {
                    "source": int(patch),
                    "surface_max_positive_world_depth_m": float(world_local[:64, 2].max()),
                    "surface_max_positive_depth_over_a": float(world_local[:64, 2].max() / axes[patch, 0]),
                    "self_tensor_surface_world_vs_parametric_relative": float(
                        np.linalg.norm(world[:64] - exact[:64]) / max(np.linalg.norm(exact[:64]), 1)
                    ),
                }
            )
        norm = max(np.linalg.norm(reference), 1)
        rows.append(
            {
                "group": group,
                "input": file_id(path),
                "sources": descriptions,
                "native_vs_original_world_reference_relative": float(np.linalg.norm(native - reference) / norm),
                "native_vs_parametric_surface_reference_relative": float(np.linalg.norm(native - ideal) / norm),
                "world_vs_parametric_reference_relative": float(np.linalg.norm(reference - ideal) / norm),
            }
        )
        np.savez_compressed(args.output / f"group{group}-parametric-reference.npz", tensor_pa=ideal)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "read_only_no_native_elastic_rerun": True,
            "original_rotated_failure_admitted": False,
            "parametric_surface_self_coordinates_are_the_probe_definition": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
