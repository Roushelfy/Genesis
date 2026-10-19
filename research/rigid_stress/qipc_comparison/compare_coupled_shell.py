"""Measure complete same-mesh Shell field sensitivity to timestep, repeat and load history."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--frames", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    values, rows = [], []
    reference_rest = None
    for path in [args.reference, *args.frames]:
        with np.load(path.parent.parent / "rest.npz") as saved:
            vertices, faces, masses = saved["vertices"], saved["faces"], saved["masses_kg"]
        if reference_rest is None:
            reference_rest = (vertices, faces, masses)
        else:
            for actual, expected in zip((vertices, faces, masses), reference_rest):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-16)
        triangle = vertices[faces]
        normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        area = np.linalg.norm(normal, axis=1) / 2
        normal /= 2 * area[:, None]
        first = triangle[:, 1] - triangle[:, 0]
        first /= np.linalg.norm(first, axis=1)[:, None]
        basis = np.stack((first, np.cross(normal, first)), axis=-1)
        with np.load(path) as saved:
            rotation = corotation(vertices, saved["vertices_world_m"], masses)
            tensor = rotation.T @ saved["cauchy_world_pa"] @ rotation
            tangent = basis[:, None].transpose(0, 1, 3, 2) @ tensor @ basis[:, None]
            values.append(
                {
                    "tangent": tangent,
                    "membrane": saved["membrane_n_per_m"],
                    "bending": saved["bending_n"],
                    "raw_vm": float(saved["vm_pa"].max()),
                    "raw_tension": float(np.linalg.eigvalsh(tensor)[..., -1].max()),
                }
            )
    for path, actual in zip(args.frames, values[1:]):
        reference = values[0]
        errors = {}
        for key in ("tangent", "membrane", "bending"):
            # Simpson weights integrate exactly the square of the linear fiber field.
            weights = area[:, None] * np.array([1, 4, 1])[None] / 6 if key == "tangent" else area
            numerator = np.sum(weights * np.sum((actual[key] - reference[key]) ** 2, axis=(-2, -1)))
            denominator = np.sum(weights * np.sum(reference[key] ** 2, axis=(-2, -1)))
            errors[key + "_relative_l2"] = float(np.sqrt(numerator / denominator))
        for key in ("raw_vm", "raw_tension"):
            errors[key + "_relative_change"] = abs(actual[key] / reference[key] - 1)
        rows.append(
            {
                "frame": file_id(path),
                **errors,
                "raw_vm_pa": actual["raw_vm"],
                "raw_tension_pa": actual["raw_tension"],
                "field_and_peak_sensitivity_passed": bool(
                    max(errors[key + "_relative_l2"] for key in ("tangent", "membrane", "bending")) <= 0.005
                    and max(errors[key + "_relative_change"] for key in ("raw_vm", "raw_tension")) <= 0.01
                ),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "reference": file_id(args.reference),
            "comparisons": rows,
            "physical_accuracy_qualified": False,
            "scope": "Complete area-weighted same-mesh fields, rigid corotation, exact linear-fiber square integration; raw peaks unmodified. Targets 0.5% fields and 1% peaks, separately from mechanical admission and continuum reference qualification.",
        },
    )


if __name__ == "__main__":
    main()
