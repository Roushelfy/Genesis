"""Replace Gauss sampling of the saved regular shell stress with exact column integrals."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .layered_column_integrals import LayeredColumnIntegrals
from .study_edge_shell_fields import resultants
from .study_hertz_residual import field_difference
from .study_shell_pressure_stress import P2Readout


def compare(candidate, reference, areas):
    return {
        name + "_relative": float(field_difference(first, second, areas))
        for name, first, second in zip(["membrane", "bending"], candidate, reference)
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fields", type=Path, required=True, help="Higher-partition three-body saved fields.")
    cli.add_argument("--sampling", type=Path, required=True)
    cli.add_argument("--particular", type=Path, required=True)
    cli.add_argument("--parents", type=Path, nargs=3, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.fields / "start.json").read_text())
    old = json.loads((args.particular / "start.json").read_text())
    assert old["inputs"][0] == file_id(args.sampling / "queries.npz")
    assert old["candidate"] == start["candidate"] and old["problem"] == start["problem"]
    for receipt in start["inputs"] + old["inputs"]:
        assert file_id(receipt["path"]) == receipt
    meshes, inputs = (
        [],
        [
            file_id(args.fields / "start.json"),
            file_id(args.fields / "result.json"),
            file_id(args.sampling / "queries.npz"),
            file_id(args.particular / "particular.npz"),
        ],
    )
    for body, parent in zip(["coarse", "medium", "fine"], args.parents):
        info = json.loads((parent / "start.json").read_text())
        mesh = Path(info["mesh"]["path"])
        assert file_id(mesh) == info["mesh"]
        meshes.append((body, mesh))
        inputs.extend([info["mesh"], file_id(args.fields / f"{body}-fields.npz")])
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": inputs,
            "candidate": start["candidate"],
            "problem": start["problem"],
            "scope": "Same full midsurface column bank and particular stresses; exact Cartesian tetrahedron intersections "
            "integrate the saved piecewise-linear regular stress. No new elastic or pressure solve, smoothing, "
            "removed contact region, changed stress law or native qualification.",
        },
    )
    with np.load(args.sampling / "queries.npz") as saved:
        queries = {key: saved[key] for key in saved.files if key.startswith("column_")}
        nv = len(saved["volume_points_m"])
    nc, areas = len(queries["column_points_m"]), queries["column_areas_m2"]
    particulars = {}
    with np.load(args.particular / "particular.npz") as saved:
        for label in ["candidate", "reference48", "reference36", "reference48-lower"]:
            particulars[label] = resultants(saved[label + "_pa"][nv : nv + nc], queries)
    rows, fields = [], {}
    for body, path in meshes:
        begun = perf_counter()
        with np.load(path) as saved:
            vertices, tets, outer, inner, faces = [
                saved[key] for key in ["vertices", "tetrahedra", "outer_vertices", "inner_vertices", "mid_faces"]
            ]
        integrator = LayeredColumnIntegrals(
            vertices,
            tets,
            outer,
            inner,
            faces,
            queries["column_outer_m"],
            queries["column_inner_m"],
            queries["column_projection"],
        )
        readout = P2Readout(vertices, tets, vertices[tets[:1]].mean(axis=1))
        variants = ["candidate-point", "reference48"]
        if body == "fine":
            variants += ["reference36", "reference48-source3", "reference48-lower"]
        current, saved_arrays = {}, {}
        checks = None
        with np.load(args.fields / f"{body}-fields.npz") as saved:
            for label in variants:
                _, corner = readout.field(saved[label + "-regular_displacement_m"])
                corner = corner.reshape(len(tets), 4, 3, 3)
                if label == "reference48":
                    checks = integrator.controls(corner)
                regular = integrator.integrate(corner)
                key = {"candidate-point": "candidate", "reference48-source3": "reference48"}.get(label, label)
                combined = tuple(a + b for a, b in zip(regular, particulars[key]))
                current[label] = combined
                for name, value, reg, part in zip(
                    ["membrane_n_per_m", "bending_n"], combined, regular, particulars[key]
                ):
                    saved_arrays[label + "-" + name] = value
                    saved_arrays[label + "-regular-" + name] = reg
                    saved_arrays[label + "-particular-" + name] = part
                write_json(args.output / "progress.json", {"body": body, "variant": label})
        np.savez_compressed(args.output / f"{body}-columns.npz", **saved_arrays)
        fields[body] = current
        rows.append(
            {
                "body": body,
                "seconds": perf_counter() - begun,
                "geometry": integrator.diagnostics,
                "controls": checks,
                "squared_norms": {
                    label: {
                        name: float(areas @ np.sum(value**2, axis=(1, 2)))
                        for name, value in zip(["membrane", "bending"], pair)
                    }
                    for label, pair in current.items()
                },
            }
        )
        del integrator, readout
    reference = fields["fine"]["reference48"]
    errors = {body: compare(value["candidate-point"], reference, areas) for body, value in fields.items()}
    changes = {
        body: compare(value["reference48"], reference, areas) for body, value in fields.items() if body != "fine"
    }
    changes.update(
        {
            label: compare(fields["fine"][label], reference, areas)
            for label in ["reference36", "reference48-source3", "reference48-lower"]
        }
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "candidate_errors": errors,
            "reference_changes": changes,
            "spatial_and_particular_integral_convergence_checked": False,
            "complete_volume_error_qualified": False,
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
