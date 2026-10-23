"""Check exact rectangle stresses before the single-edge shell-field study."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_fields import composite_tensor
from .edge_normal_batch import composite_normal
from .hertz_reference import halfspace_stress
from .rectangle_normal_stress import RectangleNormalStress, rectangle_tensor
from .study_edge_shell_pressure import cell_tensor


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"direct_relative": 1e-9, "surface_traction": 1e-6, "saved_field_relative": 1e-7, "far_relative": 1e-5}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.pressure / name) for name in ["grid48.npz", "candidate.npz", "start.json"]],
            "targets": targets,
            "scope": "Uniform-cell normal stress integration only; no new contact or shell qualification.",
        },
    )
    widths = np.array([0.7, 1.3])
    centers, force = np.array([[0.0, 0.0], [1.7, -0.6]]), np.array([0.4, 0.8])
    points = np.array([[x, y, z] for x in [-0.4, 0, 0.2, 1.5] for y in [-0.6, 0.3] for z in [0.4, 1.5]])
    actual = rectangle_tensor(points, centers, widths, force)
    direct = []
    for order in [32, 48]:
        knots, weights = np.polynomial.legendre.leggauss(order)
        offsets = np.stack(np.meshgrid(*(knots * widths[i] / 2 for i in (0, 1)), indexing="ij"), axis=-1).reshape(-1, 2)
        sources = (centers[:, None] + offsets).reshape(-1, 2)
        loads = (force[:, None] * np.outer(weights, weights).reshape(-1) / 4).reshape(-1)
        direct.append(halfspace_stress(points, sources, loads))
    direct_error = float(np.linalg.norm(actual - direct[-1]) / np.linalg.norm(direct[-1]))
    rule_change = float(np.linalg.norm(direct[0] - direct[1]) / np.linalg.norm(direct[1]))
    assert max(direct_error, rule_change) < targets["direct_relative"]
    np.testing.assert_allclose(
        rectangle_tensor(points * 1e-6, centers * 1e-6, widths * 1e-6, force * 1e-12), actual, rtol=1e-10, atol=1e-12
    )
    surface = np.array([[0, 0, 1e-9], [0.1, 0.2, 1e-9], [0.8, 0.8, 1e-9]])
    stress = rectangle_tensor(surface, np.zeros((1, 2)), widths, np.array([widths.prod()]))
    expected = np.zeros((3, 3))
    expected[:2, 2] = -1
    surface_error = float(np.max(np.abs(stress[:, :, 2] - expected)))
    assert surface_error < targets["surface_traction"]
    with np.load(args.pressure / "grid48.npz") as saved:
        model = RectangleNormalStress(saved["points_m"], saved["widths_m"], saved["sources5-force_n"])
        prior = saved["sources5-particular_pa"]
    with np.load(args.pressure / "candidate.npz") as saved:
        query = saved["local_queries_m"]
    current = model.near(query)
    current[:, 2, :2] *= -1
    current[:, :2, 2] *= -1
    saved_error = float(np.linalg.norm(current - prior) / np.linalg.norm(prior))
    assert saved_error < targets["saved_field_relative"]
    start = json.loads((args.pressure / "start.json").read_text())
    along = start["candidate"]["along"]
    edge = EdgeScales(**start["candidate"])
    close = np.array(
        [
            [x * edge.width, y * along, depth * edge.width]
            for x in [-2, 0, 0.5, 2]
            for y in [-1.1, 0, 0.8, 1.1]
            for depth in [0.001, 0.1, 1, 10]
        ]
    )
    scalar = composite_tensor(close, edge, start["problem"]["load_n"], np.zeros(2), 32)
    batched = composite_normal(close, edge, start["problem"]["load_n"], 32)
    composite_error = float(np.linalg.norm(batched - scalar) / np.linalg.norm(scalar))
    np.testing.assert_allclose(batched, scalar, rtol=1e-7, atol=1e-5)
    far = np.array([[x, y, z] for x in [-4, 0, 4] for y in [-4, 0, 4] for z in [4, 8]]) * along
    reference = model.near(far)
    low, high = model.far(far, 9), model.far(far, 17)
    far_error = float(np.linalg.norm(high - reference) / np.linalg.norm(reference))
    far_change = float(np.linalg.norm(low - high) / np.linalg.norm(high))
    assert max(far_error, far_change) < targets["far_relative"]
    # Independently integrated actual very anisotropic cells at resolved depth.
    integrated = cell_tensor(query[:12], model.centers, model.widths, model.force, 6)
    controls = model.near(query[:12])
    np.testing.assert_allclose(controls, integrated, rtol=1e-7, atol=1e-4)
    np.savez_compressed(
        args.output / "controls.npz",
        points=points,
        exact=actual,
        direct=direct[-1],
        surface=stress,
        far_points=far,
        far_exact=reference,
        far9=low,
        far17=high,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "direct_relative": direct_error,
            "direct_rule_change": rule_change,
            "surface_traction_error": surface_error,
            "actual_saved_field_relative": saved_error,
            "far_relative": far_error,
            "far9_far17_relative_change": far_change,
            "batched_composite_relative": composite_error,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
