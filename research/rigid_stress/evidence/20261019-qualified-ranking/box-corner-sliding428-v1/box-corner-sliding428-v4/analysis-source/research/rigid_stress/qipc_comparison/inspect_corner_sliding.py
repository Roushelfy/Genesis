"""Audit two-face surface metric strain and preserve the original normal-contact control."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .inspect_box_corner_surface import inspect


def surface_metric(path):
    with np.load(path) as saved:
        points, rest, position, force = [
            saved[key] for key in ["points_m", "rest_m", "deformed_relative_m", "normal_force_n"]
        ]
    x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
    shape = (len(x), len(y), 3)
    gradients = []
    for value in [rest, position]:
        tangents = np.stack(np.gradient(value.reshape(shape), x, y, axis=(0, 1), edge_order=2), axis=-1)
        gradients.append(np.einsum("...ia,...ib->...ab", tangents, tangents))
    eigenvalues, eigenvectors = np.linalg.eigh(gradients[0])
    assert eigenvalues.min() > 0
    inverse = (eigenvectors / np.sqrt(eigenvalues)[..., None, :]) @ eigenvectors.swapaxes(-1, -2)
    stretch_squared = np.linalg.eigvalsh(inverse @ gradients[1] @ inverse).reshape(-1, 2)
    assert stretch_squared.min() >= 0
    strain = (stretch_squared - 1) / 2
    active = force > 1e-10 * force.sum()
    stretch = np.sqrt(stretch_squared[active])
    return {
        "active_principal_surface_stretch_range": [float(stretch.min()), float(stretch.max())],
        "active_principal_green_strain_range": [float(strain[active].min()), float(strain[active].max())],
        "force_weighted_green_strain_rms": float(np.sqrt(force @ np.sum(strain**2, axis=1) / force.sum())),
        "scope": "Cell-grid finite differences compare the deformed and rest surface metrics. Principal Green strain removes rigid rotations. This is a discrete surface diagnostic; it supplies no continuum volume-strain or material-accuracy bound.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--controls", type=Path, nargs="+", required=True)
    cli.add_argument("--normal-baseline", type=Path)
    cli.add_argument("--target-orders", type=int, nargs="*", default=[])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [file_id(root / name) for root in args.controls for name in ["result.json", "fields.npz"]]
    if args.normal_baseline:
        inputs.append(file_id(args.normal_baseline))
    write_json(
        args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False), "inputs": inputs}
    )
    rows = []
    for root in args.controls:
        reference = json.loads((root / "result.json").read_text())
        assert reference["passed"]
        unchanged = None
        if reference["measured_sliding_control"] is None and args.normal_baseline:
            with np.load(root / "fields.npz") as actual, np.load(args.normal_baseline) as original:
                assert set(actual.files) == set(original.files)
                for key in original.files:
                    np.testing.assert_array_equal(actual[key], original[key], err_msg=key)
            unchanged = True
        rows.append(
            {
                "control": str(root),
                "normal_baseline_arrays_exact": unchanged,
                "finite_face_membership_passed": reference["finite_face_membership_passed"],
                "surface_ordering": inspect(root / "fields.npz"),
                "surface_metric": surface_metric(root / "fields.npz"),
            }
        )
        if args.target_orders:
            assert min(args.target_orders) > 0
            rows[-1]["fixed_force_reintegrations"] = [
                inspect(root / "fields.npz", order) for order in args.target_orders
            ]
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
