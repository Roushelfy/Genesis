"""Resolve the worst retained single-precision baseline error against pressure integration."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_fields import composite_tensor
from .study_edge_farfield import evaluate_rule, pressure_rule


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--native", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=False), "inputs": [file_id(args.native)]},
    )
    with np.load(args.native) as f:
        positions, inputs, expected, native = f["points_m"], f["inputs"], f["expected_pa"], f["native_pa"]
    relative = np.linalg.norm(native - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    index = relative.argmax()
    point, values = positions[index : index + 1], inputs[index]
    width, along, exponent, amplitude, force, tx, ty, poisson = values
    source, weights = pressure_rule(width, along, exponent, 24, 40)
    reference = evaluate_rule(point, source, weights, force, np.array([tx, ty]), poisson)[0]
    rounded = values.astype(np.float32).astype(float)
    model = EdgeScales(*rounded[:4])
    quantized = composite_tensor(
        point.astype(np.float32).astype(float), model, rounded[4], rounded[5:7], poisson=rounded[7]
    )[0]
    far = np.linalg.norm(positions, axis=1) >= 8 * np.maximum(inputs[:, 0], inputs[:, 1])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "maximum_relative_error": float(relative.max()),
            "worst_index": int(index),
            "worst_point_m": point[0].tolist(),
            "parameters": values.tolist(),
            "distance_over_half_length": float(np.linalg.norm(point) / along),
            "native_pa": native[index].tolist(),
            "baseline64_pa": expected[index].tolist(),
            "continuous_pa": reference.tolist(),
            "rounded_input_composite64_pa": quantized.tolist(),
            "input_rounding_relative_change": float(
                np.linalg.norm(quantized - expected[index]) / np.linalg.norm(expected[index])
            ),
            "baseline_model_error": float(np.linalg.norm(reference - expected[index]) / np.linalg.norm(reference)),
            "near_maximum_relative_error": float(relative[~far].max()),
            "far_maximum_relative_error": float(relative[far].max()),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
