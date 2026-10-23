"""Resolve the sharp particular-field energy with a smooth full-domain subtraction identity."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_normal_batch import composite_normal
from .edge_volume_window import quadrature, window
from .rectangle_normal_stress import RectangleNormalStress
from .study_shell_pressure_stress import P2Readout


def squared(field, weights):
    return float(weights @ np.sum(field**2, axis=(1, 2)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--studies", type=Path, nargs=2, required=True)
    cli.add_argument("--samplings", type=Path, nargs=2, required=True)
    cli.add_argument("--particulars", type=Path, nargs=2, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--orders", type=int, nargs=3, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.pressure / "start.json").read_text())
    model, load = EdgeScales(**start["candidate"]), start["problem"]["load_n"]
    inputs = [file_id(args.mesh), file_id(args.pressure / "start.json")]
    pressures, boundaries = {}, []
    for count in [36, 48]:
        p = args.pressure / f"grid{count}.npz"
        inputs.append(file_id(p))
        with np.load(p) as saved:
            pressures[count] = RectangleNormalStress(saved["points_m"], saved["widths_m"], saved["sources5-force_n"])
            y = np.unique(saved["points_m"][:, 1])
            boundaries.extend([y - saved["widths_m"][1] / 2, y + saved["widths_m"][1] / 2])
    for study, sampling, particular in zip(args.studies, args.samplings, args.particulars):
        previous = json.loads((study / "start.json").read_text())
        for receipt in previous["inputs"] + previous["particular_reuse"]:
            assert file_id(receipt["path"]) == receipt
        assert previous["candidate"] == start["candidate"] and previous["problem"] == start["problem"]
        assert file_id(args.pressure / "grid48.npz") in previous["inputs"]
        assert file_id(sampling / "queries.npz") in previous["inputs"]
        assert file_id(particular / "particular.npz") in previous["particular_reuse"]
        inputs.extend(
            file_id(path)
            for path in [
                study / "start.json",
                study / "result.json",
                study / "coarse-fields.npz",
                study / "fine-fields.npz",
                sampling / "queries.npz",
                particular / "particular.npz",
            ]
        )
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": inputs,
            "orders": args.orders,
            "candidate": start["candidate"],
            "problem": start["problem"],
            "scope": "Q_full[g - chi*g_particular] + Q_local[chi*g_particular]. The compact smooth window lies entirely "
            "inside the original material. No removed volume or stress-law change; original regular and cross terms "
            "remain in Q_full. Polar transverse/depth quadrature cancels the leading edge-energy singularity, "
            "and along-edge intervals include both pressure grids' cell boundaries.",
        },
    )
    # No nonplanar material boundary may enter the entire support box.
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
        triangles = vertices[saved["surface_triangles"]]
    lower = np.array([-4 * model.width, -2 * model.along, 0.03 - 8 * model.width])
    upper = np.array([4 * model.width, 2 * model.along, 0.03])
    overlap = np.all(triangles.max(axis=1) >= lower, axis=1) & np.all(triangles.min(axis=1) <= upper, axis=1)
    assert overlap.any() and np.all(triangles[overlap, :, 2] == 0.03)
    center = (lower + upper)[None] / 2
    P2Readout(vertices, tets, center)
    points, weights, controls = quadrature(model.width, model.along, np.concatenate(boundaries), *args.orders)
    write_json(
        args.output / "quadrature.json",
        {
            **controls,
            "support_box_m": [lower.tolist(), upper.tolist()],
            "overlapping_boundary_triangles": int(overlap.sum()),
            "only_flat_top_boundary_overlaps": True,
        },
    )
    tensors, seconds = {}, {}
    for label, count in [("reference48", 48), ("reference36", 36), ("candidate", None)]:
        begun = perf_counter()
        tensor = np.empty((len(points), 3, 3))
        for first in range(0, len(points), 4096):
            current = points[first : first + 4096]
            tensor[first : first + len(current)] = (
                pressures[count].near(current) if count else composite_normal(current, model, load, 32)
            )
            write_json(
                args.output / "progress.json", {"field": label, "completed": first + len(current), "total": len(points)}
            )
        tensors[label] = tensor
        seconds[label] = perf_counter() - begun
    local_energy = {
        "reference": squared(tensors["reference48"], weights),
        "candidate_error": squared(tensors["candidate"] - tensors["reference48"], weights),
        "pressure_change": squared(tensors["reference36"] - tensors["reference48"], weights),
    }
    np.savez_compressed(
        args.output / "window-fields.npz",
        points_m=points,
        weights_m3=weights,
        **{key + "_local_pa": value for key, value in tensors.items()},
    )
    rows = []
    for study, sampling, particular in zip(args.studies, args.samplings, args.particulars):
        report = json.loads((study / "result.json").read_text())
        with np.load(sampling / "queries.npz") as saved:
            common, common_weights = saved["volume_points_m"], saved["volume_weights_m3"]
        nv = len(common)
        common[:, 2] = 0.03 - common[:, 2]
        selected_weights = common_weights * window(common, model.width, model.along)
        with np.load(particular / "particular.npz") as saved:
            rp, cp, lp = [saved[label + "_pa"][:nv].copy() for label in ["reference48", "candidate", "reference36"]]
        subtracted = {
            "reference": squared(rp, selected_weights),
            "candidate_error": squared(cp - rp, selected_weights),
            "pressure_change": squared(lp - rp, selected_weights),
        }
        with np.load(study / "fine-fields.npz") as saved:
            reference, lower_pressure = saved["reference48-tensor_pa"][:nv], saved["reference36-tensor_pa"][:nv]
        with np.load(study / "coarse-fields.npz") as saved:
            candidate = saved["candidate-point-tensor_pa"][:nv]
        original = {
            "reference": squared(reference, common_weights),
            "candidate_error": squared(candidate - reference, common_weights),
            "pressure_change": squared(lower_pressure - reference, common_weights),
        }
        np.testing.assert_allclose(
            (original["candidate_error"] / original["reference"]) ** 0.5,
            report["candidate_errors"]["coarse"]["complete_tensor_relative"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            (original["pressure_change"] / original["reference"]) ** 0.5,
            report["reference_changes"]["reference36"]["complete_tensor_relative"],
            rtol=1e-12,
        )
        corrected = {key: original[key] - subtracted[key] + local_energy[key] for key in original}
        assert min(corrected.values()) > 0
        errors = {
            body: float(
                np.sqrt(
                    (
                        value["complete_tensor_relative"] ** 2 * original["reference"]
                        - subtracted["candidate_error"]
                        + local_energy["candidate_error"]
                    )
                    / corrected["reference"]
                )
            )
            for body, value in report["candidate_errors"].items()
        }
        changes = {
            label: float(
                np.sqrt(value["complete_tensor_relative"] ** 2 * original["reference"] / corrected["reference"])
            )
            for label, value in report["reference_changes"].items()
        }
        changes["reference36"] = float(np.sqrt(corrected["pressure_change"] / corrected["reference"]))
        rows.append(
            {
                "study": str(study),
                "original_squared_norms": original,
                "subtracted_window_squared_norms": subtracted,
                "corrected_squared_norms": corrected,
                "candidate_errors": errors,
                "reference_changes": changes,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "quadrature": controls,
            "seconds": seconds,
            "local_window_squared_norms": local_energy,
            "rows": rows,
            "quadrature_convergence_checked": False,
            "complete_volume_error_qualified": False,
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
