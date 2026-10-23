"""Complete shell fields for independently solved single-edge normal pressure."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_normal_batch import composite_normal
from .rectangle_normal_stress import RectangleNormalStress, point_sources_tensor
from .study_edge_farfield import moment_rule
from .study_edge_shell_pressure import source_average
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import field_difference
from .study_shell_pressure_stress import P2Readout


def global_tensor(tensor):
    tensor[:, 2, :2] *= -1
    tensor[:, :2, 2] *= -1
    return tensor


def particular_fields(points, pressure, model, load, output):
    local = points.copy()
    local[:, 2] = 0.03 - local[:, 2]
    assert local[:, 2].min() > 0
    near = np.linalg.norm(local, axis=1) < 4 * model.along
    tensors, records = {}, {}
    for count in [48, 36]:
        saved = pressure[count]
        reference = RectangleNormalStress(saved["points_m"], saved["widths_m"], saved["sources5-force_n"])
        begun = perf_counter()
        field = np.empty((len(points), 3, 3))
        field[~near] = reference.far(local[~near], 17)
        field[near] = reference.near(local[near])
        tensors[f"reference{count}"] = global_tensor(field)
        records[f"reference{count}"] = {
            "seconds": perf_counter() - begun,
            "near_points": int(near.sum()),
            "far_points": int((~near).sum()),
        }
        if count == 48:
            lower = np.empty_like(field)
            lower[near] = tensors["reference48"][near]
            lower[~near] = global_tensor(reference.far(local[~near], 9))
            tensors["reference48-lower"] = lower
        write_json(output / "particular-progress.json", records)
    begun = perf_counter()
    candidate_near = np.linalg.norm(local, axis=1) < 8 * model.along
    candidate = np.empty((len(points), 3, 3))
    sources, weights = moment_rule(model.width, model.along, model.exponent)
    candidate[~candidate_near] = point_sources_tensor(local[~candidate_near], sources, weights * load)
    selected = np.flatnonzero(candidate_near)
    for start in range(0, len(selected), 512):
        indices = selected[start : start + 512]
        candidate[indices] = composite_normal(local[indices], model, load, 32)
        write_json(
            output / "candidate-progress.json", {"completed": min(start + 512, len(selected)), "total": len(selected)}
        )
    tensors["candidate"] = global_tensor(candidate)
    records["candidate"] = {
        "seconds": perf_counter() - begun,
        "near_points": int(candidate_near.sum()),
        "far_points": int((~candidate_near).sum()),
    }
    return tensors, records


def resultants(tensor, queries):
    count = len(queries["column_areas_m2"])
    membrane, bending = np.zeros((count, 3, 3)), np.zeros((count, 3, 3))
    index = queries["column_index"]
    for start in range(0, len(tensor), 65536):
        chosen = index[start : start + 65536]
        projection = queries["column_projection"][chosen]
        tangent = projection @ tensor[start : start + 65536] @ projection
        np.add.at(membrane, chosen, tangent * queries["column_weights_m"][start : start + 65536, None, None])
        np.add.at(bending, chosen, tangent * queries["column_moment_weights_m2"][start : start + 65536, None, None])
    return membrane, bending


def summarize(tensor, queries, points, corner_tensor=None, corner_points=None):
    nv, nc = len(queries["volume_points_m"]), len(queries["column_points_m"])
    volume = tensor[:nv]
    membrane, bending = resultants(tensor[nv : nv + nc], queries)
    tension, vm = stress_measures(tensor)
    corner_tension, corner_vm = stress_measures(corner_tensor) if corner_tensor is not None else (None, None)
    depth = 0.03 - points[:, 2]
    peak_rows = []
    for cutoff in queries["peak_cutoffs_m"]:
        selected = depth + 64 * np.finfo(float).eps * 0.03 >= cutoff
        dedicated = selected.copy()
        dedicated[: nv + nc] = False
        row = {"cutoff_m": float(cutoff), "admissible_queries": int(selected.sum())}
        for bank, mask in [("all", selected), ("dedicated", dedicated)]:
            indices = np.flatnonzero(mask)
            for name, values in [("tensile", tension), ("vm", vm)]:
                index = indices[values[indices].argmax()]
                row[f"{bank}_{name}_pa"] = float(values[index])
                row[f"{bank}_{name}_location_m"] = points[index].tolist()
                if bank == "all" and corner_tensor is not None:
                    eligible = 0.03 - corner_points[:, 2] + 64 * np.finfo(float).eps * 0.03 >= cutoff
                    corner_indices = np.flatnonzero(eligible)
                    corner_values = corner_tension if name == "tensile" else corner_vm
                    corner_index = corner_indices[corner_values[corner_indices].argmax()]
                    if corner_values[corner_index] > values[index]:
                        row[f"{bank}_{name}_pa"] = float(corner_values[corner_index])
                        row[f"{bank}_{name}_location_m"] = corner_points[corner_index].tolist()
        peak_rows.append(row)
    weights, areas = queries["volume_weights_m3"], queries["column_areas_m2"]
    return {
        "volume": volume,
        "membrane": membrane,
        "bending": bending,
        "peaks": peak_rows,
        "volume_squared_norm_pa2_m3": float(weights @ np.sum(volume**2, axis=(1, 2))),
        "membrane_squared_norm_n2": float(areas @ np.sum(membrane**2, axis=(1, 2))),
        "bending_squared_norm_n2_m2": float(areas @ np.sum(bending**2, axis=(1, 2))),
    }


def compare(candidate, reference, queries):
    result = {
        "complete_tensor_relative": float(
            field_difference(candidate["volume"], reference["volume"], queries["volume_weights_m3"])
        ),
        "membrane_relative": float(
            field_difference(candidate["membrane"], reference["membrane"], queries["column_areas_m2"])
        ),
        "bending_relative": float(
            field_difference(candidate["bending"], reference["bending"], queries["column_areas_m2"])
        ),
        "peaks": [],
    }
    for first, second in zip(candidate["peaks"], reference["peaks"]):
        assert first["cutoff_m"] == second["cutoff_m"]
        row = {"cutoff_m": first["cutoff_m"]}
        for bank in ["all", "dedicated"]:
            for name in ["tensile", "vm"]:
                key = f"{bank}_{name}"
                row[key + "_relative"] = float(abs(first[key + "_pa"] / second[key + "_pa"] - 1))
                row[key + "_location_distance_m"] = float(
                    np.linalg.norm(np.array(first[key + "_location_m"]) - second[key + "_location_m"])
                )
        result["peaks"].append(row)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sampling", type=Path, required=True)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--parents", type=Path, nargs=3, required=True)
    cli.add_argument("--kernel-check", type=Path, required=True)
    cli.add_argument(
        "--particular-reuse", type=Path, help="Completed particular fields from a retained failed readout."
    )
    cli.add_argument("--bodies", nargs="+", choices=["coarse", "medium", "fine"], default=["coarse", "medium", "fine"])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert json.loads((args.sampling / "result.json").read_text())["passed"]
    assert json.loads(args.kernel_check.read_text())["passed"]
    start = json.loads((args.pressure / "start.json").read_text())
    model, load = EdgeScales(**start["candidate"]), start["problem"]["load_n"]
    targets = {
        "reference_field_change": 0.02,
        "reference_peak_change": 0.05,
        "candidate_field_error": 0.05,
        "candidate_peak_error": 0.10,
    }
    parents = []
    for parent in args.parents:
        info = json.loads((parent / "start.json").read_text())
        mesh = Path(info["mesh"]["path"])
        assert file_id(mesh) == info["mesh"]
        parents.append((parent, mesh))
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [
                file_id(args.sampling / "queries.npz"),
                file_id(args.sampling / "result.json"),
                file_id(args.kernel_check),
                file_id(args.pressure / "start.json"),
                file_id(args.pressure / "grid36.npz"),
                file_id(args.pressure / "grid48.npz"),
            ]
            + [
                file_id(path)
                for parent, mesh in parents
                for path in [mesh, parent / "normal-source-table.npz", parent / "start.json"]
            ],
            "particular_reuse": [
                file_id(args.particular_reuse / name)
                for name in ["start.json", "particular.npz", "particular-result.json"]
            ]
            if args.particular_reuse
            else None,
            "targets": targets,
            "bodies": args.bodies,
            "problem": start["problem"],
            "candidate": start["candidate"],
            "scope": "Complete original-domain volume integration and graded material-column membrane/bending. "
            "Fine-body pressure is held fixed through body refinement. Original source-table and pressure-grid changes "
            "are separate. Explicit positive-depth peak cutoffs do not remove any volume or column contribution. "
            "The ideal surface maximum remains unbounded. No native accuracy/cost claim or friction qualification.",
        },
    )
    with np.load(args.sampling / "queries.npz") as saved:
        queries = {key: saved[key] for key in saved.files}
    all_points = np.concatenate([queries[key] for key in ["volume_points_m", "column_points_m", "peak_points_m"]])
    grid_count = int(np.prod(queries["peak_grid_shape"]))
    boundary_vertices = queries["peak_points_m"][grid_count:]
    boundary_tree = cKDTree(boundary_vertices)
    boundary_offset = len(queries["volume_points_m"]) + len(queries["column_points_m"]) + grid_count
    queries["peak_points_m"] = queries["peak_points_m"][:grid_count]
    points = all_points[:boundary_offset]
    pressures = {}
    for count in [36, 48]:
        with np.load(args.pressure / f"grid{count}.npz") as saved:
            pressures[count] = {key: saved[key] for key in saved.files}
    if args.particular_reuse:
        old = json.loads((args.particular_reuse / "start.json").read_text())
        assert old["inputs"][0] == file_id(args.sampling / "queries.npz")
        assert old["problem"] == start["problem"] and old["candidate"] == start["candidate"]
        for receipt in old["inputs"]:
            assert file_id(receipt["path"]) == receipt
        with np.load(args.particular_reuse / "particular.npz") as saved:
            tensors = {key.removesuffix("_pa"): saved[key] for key in saved.files}
        assert all(len(value) == len(all_points) for value in tensors.values())
        numerical = json.loads((args.particular_reuse / "particular-result.json").read_text())
        numerical["reused_from"] = file_id(args.particular_reuse / "particular.npz")
    else:
        tensors, numerical = particular_fields(all_points, pressures, model, load, args.output)
        np.savez_compressed(args.output / "particular.npz", **{key + "_pa": value for key, value in tensors.items()})
    write_json(args.output / "particular-result.json", numerical)
    records, rows = {}, []
    for name, (parent, mesh_path) in zip(["coarse", "medium", "fine"], parents):
        if name not in args.bodies:
            continue
        begun = perf_counter()
        with np.load(mesh_path) as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
        corners = vertices[tets].reshape(-1, 3)
        retained_corners = 0.03 - corners[:, 2] + 64 * np.finfo(float).eps * 0.03 >= queries["peak_cutoffs_m"].min()
        corner_points = corners[retained_corners]
        distances, corner_indices = boundary_tree.query(corner_points)
        assert np.all(distances == 0)
        np.testing.assert_array_equal(boundary_vertices[corner_indices], corner_points)
        corner_mapping = boundary_offset + corner_indices
        write_json(args.output / "readout-progress.json", {"body": name, "queries": len(points), "stage": "locate"})
        readout = P2Readout(vertices, tets, points)
        affine = np.diag([1e-5, -0.5e-5, 1.1e-5])
        expected = (1e10 / 2.6) * (affine + affine.T) + (1e10 * 0.3 / (1.3 * 0.4)) * np.trace(affine) * np.eye(3)
        affine_value, _ = readout.field(readout.coordinates @ affine.T)
        affine_error = float(np.max(np.abs(affine_value - expected)) / np.linalg.norm(expected))
        assert affine_error < 1e-7
        del affine_value
        with np.load(parent / "normal-source-table.npz") as saved:
            np.testing.assert_array_equal(saved["coordinates_m"], readout.coordinates)
            displacement, source_points = saved["displacement_m_per_n"], saved["source_points_m"]
        axes = [np.unique(source_points[:, axis]) for axis in (0, 1)]
        coefficients = {}
        for count in [36, 48]:
            p = pressures[count]
            coefficients[count] = source_average(axes, p["points_m"], p["widths_m"]) @ p["sources5-force_n"]
        p = pressures[48]
        selected = np.array([i * 5 + j for i in [0, 2, 4] for j in [0, 2, 4]])
        coefficient3 = (
            source_average([axis[::2] for axis in axes], p["points_m"], p["widths_m"]) @ p["sources5-force_n"]
        )
        centroid = np.array([[0.4 * (1 - model.exponent) * model.width, 0]])
        point_weights = RegularGridInterpolator(axes, np.eye(25).reshape(5, 5, 25))(centroid)[0] * load
        variants = {
            "reference48": (np.einsum("nig,g->ni", displacement, coefficients[48]), tensors["reference48"]),
            "candidate-point": (np.einsum("nig,g->ni", displacement, point_weights), tensors["candidate"]),
        }
        if name == "fine":
            variants.update(
                {
                    "reference36": (np.einsum("nig,g->ni", displacement, coefficients[36]), tensors["reference36"]),
                    "reference48-source3": (
                        np.einsum("nig,g->ni", displacement[:, :, selected], coefficient3),
                        tensors["reference48"],
                    ),
                    "reference48-lower": (variants["reference48"][0], tensors["reference48-lower"]),
                }
            )
        values, summaries = {"corner_points_m": corner_points}, {}
        for label, (nodal, particular) in variants.items():
            regular, corner_regular = readout.field(nodal)
            tensor = regular + particular[: len(points)]
            corner_tensor = corner_regular[retained_corners] + particular[corner_mapping]
            summary = summarize(tensor, queries, points, corner_tensor, corner_points)
            summaries[label] = summary
            values[label + "-tensor_pa"] = tensor
            values[label + "-corner_tensor_pa"] = corner_tensor
            values[label + "-membrane_n_per_m"] = summary["membrane"]
            values[label + "-bending_n"] = summary["bending"]
            values[label + "-regular_displacement_m"] = nodal
            write_json(args.output / "readout-progress.json", {"body": name, "stage": label})
        np.savez_compressed(args.output / f"{name}-fields.npz", **values)
        records[name] = summaries
        rows.append(
            {
                "body": name,
                "seconds": perf_counter() - begun,
                "affine_readout_relative": affine_error,
                "common_queries": len(points),
                "own_unsmoothed_corner_queries": len(corner_points),
                "variants": {
                    label: {key: value for key, value in summary.items() if not isinstance(value, np.ndarray)}
                    for label, summary in summaries.items()
                },
            }
        )
        write_json(args.output / "field-progress.json", {"rows": rows})
        del values, readout, displacement, variants
    reference = records["fine"]["reference48"]
    errors = {name: compare(record["candidate-point"], reference, queries) for name, record in records.items()}
    changes = {
        name: compare(record["reference48"], reference, queries) for name, record in records.items() if name != "fine"
    }
    changes.update(
        {
            label: compare(records["fine"][label], reference, queries)
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
            "targets": targets,
            "numerical": numerical,
            "sampling": file_id(args.sampling / "result.json"),
            "volume_and_column_integration_convergence_checked": False,
            "native_scene_accuracy_qualified": False,
            "finite_surface_global_peak": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
