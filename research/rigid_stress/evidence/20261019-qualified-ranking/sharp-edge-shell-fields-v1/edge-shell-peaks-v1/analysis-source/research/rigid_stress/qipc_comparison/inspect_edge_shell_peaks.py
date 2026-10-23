"""Refine observed positive-depth peaks without removing any field-integration region."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_normal_batch import composite_normal
from .rectangle_normal_stress import RectangleNormalStress
from .study_edge_shell_fields import global_tensor
from .study_hertz_nearfield import stress_measures
from .study_shell_pressure_stress import P2Readout


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fields", type=Path, required=True)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--parents", type=Path, nargs=3, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert (args.fields.parent / "returncode.txt").read_text().strip() == "0"
    result = json.loads((args.fields / "result.json").read_text())
    start = json.loads((args.fields / "start.json").read_text())
    for receipt in start["inputs"]:
        assert file_id(receipt["path"]) == receipt
    model, load = EdgeScales(**start["candidate"]), start["problem"]["load_n"]
    variants = {
        "reference48": ("fine", "reference48", 48),
        "reference36": ("fine", "reference36", 36),
        "source3": ("fine", "reference48-source3", 48),
        "medium": ("medium", "reference48", 48),
        "candidate": ("coarse", "candidate-point", None),
    }
    inputs = [file_id(args.fields / name) for name in ["start.json", "result.json"]]
    meshes, nodal, peaks = {}, {}, {}
    for body, parent in zip(["coarse", "medium", "fine"], args.parents):
        info = json.loads((parent / "start.json").read_text())
        assert file_id(info["mesh"]["path"]) == info["mesh"]
        inputs.extend([file_id(args.fields / f"{body}-fields.npz"), info["mesh"]])
        with np.load(info["mesh"]["path"]) as mesh:
            meshes[body] = (mesh["vertices"], mesh["tetrahedra"])
        row = next(row for row in result["rows"] if row["body"] == body)
        with np.load(args.fields / f"{body}-fields.npz") as saved:
            for name, (owner, label, _) in variants.items():
                if owner == body:
                    nodal[name] = saved[label + "-regular_displacement_m"]
                    peaks[name] = [
                        {
                            "cutoff_m": peak["cutoff_m"],
                            **{
                                key: peak["all_" + key]
                                for key in ["tensile_pa", "tensile_location_m", "vm_pa", "vm_location_m"]
                            },
                        }
                        for peak in row["variants"][label]["peaks"]
                    ]
    pressure = {}
    for count in [36, 48]:
        path = args.pressure / f"grid{count}.npz"
        assert file_id(path) in start["inputs"]
        inputs.append(file_id(path))
        with np.load(path) as saved:
            pressure[count] = RectangleNormalStress(saved["points_m"], saved["widths_m"], saved["sources5-force_n"])
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": inputs,
            "candidate": start["candidate"],
            "scope": "Local 3D peak refinement seeded by the complete-domain volume, column, plane and "
            "unsmoothed own-corner banks. Fixed positive-depth cutoffs; all original maxima retained. "
            "Observed refinement is not a certified global optimization or finite surface maximum.",
        },
    )
    initial = json.loads(json.dumps(peaks))
    offsets = np.stack(np.meshgrid(*([np.linspace(-1, 1, 5)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    history, saved_points, saved_fields = [], [], {name: [] for name in variants}
    for iteration in range(8):
        level = iteration // 2
        banks = []
        for rows in peaks.values():
            for row in rows:
                for quantity in ["tensile", "vm"]:
                    center = np.array(row[quantity + "_location_m"])
                    depth = 0.03 - center[2]
                    step = np.array([model.width / 4, model.along / 40, max(model.width, depth / 4)]) / 2**level
                    points = center + offsets * step
                    points[:, 2] = np.minimum(points[:, 2], 0.03 - row["cutoff_m"])
                    banks.append(points)
        points = np.unique(np.concatenate(banks), axis=0)
        local = points.copy()
        local[:, 2] = 0.03 - local[:, 2]
        assert np.all(local[:, 2] > 0) and np.linalg.norm(local, axis=1).max() < 4 * model.along
        particular = {count: global_tensor(value.near(local)) for count, value in pressure.items()}
        particular[None] = global_tensor(composite_normal(local, model, load, 32))
        values = {}
        for body, (vertices, tets) in meshes.items():
            readout = P2Readout(vertices, tets, points)
            for name, (owner, _, count) in variants.items():
                if owner == body:
                    regular, _ = readout.field(nodal[name])
                    values[name] = regular + particular[count]
            del readout
        previous = json.loads(json.dumps(peaks))
        for name, field in values.items():
            measures = dict(zip(["tensile", "vm"], stress_measures(field)))
            for row in peaks[name]:
                indices = np.flatnonzero(local[:, 2] + 64 * np.finfo(float).eps * 0.03 >= row["cutoff_m"])
                for quantity, measure in measures.items():
                    index = indices[measure[indices].argmax()]
                    if measure[index] > row[quantity + "_pa"]:
                        row[quantity + "_pa"] = float(measure[index])
                        row[quantity + "_location_m"] = points[index].tolist()
            saved_fields[name].append(field)
        saved_points.append(points)
        changes = {
            name: [
                {quantity: abs(row[quantity + "_pa"] / old[quantity + "_pa"] - 1) for quantity in ["tensile", "vm"]}
                for row, old in zip(peaks[name], previous[name])
            ]
            for name in peaks
        }
        history.append({"iteration": iteration, "level": level, "queries": len(points), "changes": changes})
        write_json(args.output / "progress.json", {"history": history, "peaks": peaks})
    records = []
    for index, reference in enumerate(peaks["reference48"]):
        row = {"cutoff_m": reference["cutoff_m"], "reference": reference, "checks": {}}
        for quantity in ["tensile", "vm"]:
            value = reference[quantity + "_pa"]
            error = abs(peaks["candidate"][index][quantity + "_pa"] / value - 1)
            variations = {
                name: abs(peaks[name][index][quantity + "_pa"] / value - 1)
                for name in ["reference36", "source3", "medium"]
            }
            last_changes = [entry["changes"][name][index][quantity] for entry in history[-2:] for name in peaks]
            row["checks"][quantity] = {
                "candidate_relative_error": error,
                "reference_changes": variations,
                "change_from_original_reference_bank": abs(value / initial["reference48"][index][quantity + "_pa"] - 1),
                "maximum_last_level_change": max(last_changes),
                "observed_variation_sum": error + sum(variations.values()) + sum(last_changes),
                "conditional_targets_met": bool(
                    max(variations.values()) <= 0.05
                    and max(last_changes) <= 0.01
                    and error + sum(variations.values()) + sum(last_changes) <= 0.10
                ),
            }
        records.append(row)
    np.savez_compressed(
        args.output / "fields.npz",
        points_m=np.concatenate(saved_points),
        **{name + "_pa": np.concatenate(fields) for name, fields in saved_fields.items()},
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "initial_peaks": initial,
            "refined_peaks": peaks,
            "history": history,
            "comparisons": records,
            "finite_surface_global_peak": False,
            "certified_global_maximum": False,
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
