"""Integrate finite-depth displacement under the captured sharp-edge pressure estimate."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .edge_fields import stretched_line_rule
from .inspect_edge_shell_tilt import point_displacement


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--orders", type=int, nargs="+", default=[48, 96])
    cli.add_argument("--depth-widths", type=float, nargs="+", default=[0.1, 0.01])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.moments, args.hard]],
        },
    )
    source = json.loads(args.moments.read_text())
    assert source["passed"] and len(source["rows"]) == 1
    model = source["rows"][0]
    with np.load(args.hard) as saved:
        patch = model["region"] * saved["profile"].shape[0] + source["environment"]
        poisson, reduced = saved["edge_poisson"][patch], saved["edge_modulus"][patch]
    young = reduced * (1 - poisson**2)
    force = np.array(model["local_force_n"])
    width, along, exponent, mean = [model[key] for key in ["width_m", "half_length_m", "exponent", "pressure_mean_x_m"]]
    rows = []
    for depth in args.depth_widths:
        point = np.array([mean, 0, depth * width])
        for order in args.orders:
            knots, weights_x = roots_jacobi(order, exponent, -exponent)
            u, weights_x = (knots + 1) / 2, weights_x / weights_x.sum()
            y, measure = stretched_line_rule(point, width, along, order)
            profile = np.maximum(1 - (y / along) ** 2, 0)
            positions = np.column_stack(
                [(width * profile[:, None] * u).reshape(-1), np.repeat(y, order), np.zeros(len(y) * order)]
            )
            weights = ((3 * profile * measure / (4 * along))[:, None] * weights_x).reshape(-1)
            np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(weights @ positions, [mean, 0, 0], rtol=0, atol=1e-12 * along)
            displacement = weights @ point_displacement(point - positions, np.zeros(3), np.eye(3), force, poisson, young)
            rows.append(
                {
                    "order": order,
                    "depth_over_width": depth,
                    "position_from_edge_m": point.tolist(),
                    "displacement_local_m": displacement.tolist(),
                    "displacement_norm_m": float(np.linalg.norm(displacement)),
                    "integrated_weight": float(weights.sum()),
                    "integrated_centroid_m": (weights @ positions).tolist(),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Independent Boussinesq/Cerruti displacement integration of the reconstructed pressure and proportional-shear estimate at declared positive depths. This estimates the self-contact displacement scale; it does not validate pressure/contact physics, the exact surface limit or the failed original load against a friction cone. Normalization and centroid are checked, with quadrature refinement retained.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
