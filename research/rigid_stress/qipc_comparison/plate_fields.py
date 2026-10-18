"""Compare plate endpoint fields without mistaking missing or unequal-load fields for accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import (
    TetSampler,
    build_columns,
    build_surfaces,
    corotation,
    field_error,
    resultants,
    tangent,
    tensor_error,
    voigt_tensor,
)
from .compare_fields import area_error, face_assignment


def load(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def solid_rest(path, density):
    rest = load(path)
    vertices, tets = rest["vertices"], rest["tetrahedra"]
    volume = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
    rest["masses"] = np.bincount(tets.ravel(), weights=np.repeat(volume * density / 4, 4), minlength=len(vertices))
    return rest


def peaks(tensor, positions):
    flat = tensor.reshape(-1, 3, 3)
    deviator = flat - np.trace(flat, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    values = {
        "von_mises": np.sqrt(1.5 * np.einsum("qij,qij->q", deviator, deviator)),
        "maximum_principal": np.linalg.eigvalsh(flat)[:, -1],
    }
    return {
        key: {
            "raw_peak_pa": float(value.max()),
            "flat_index": int(value.argmax()),
            "material_location_m": positions.reshape(-1, 3)[value.argmax()].tolist(),
        }
        for key, value in values.items()
    }


def endpoint(run, start, level):
    end = (level + 1) * (start["args"]["ramp"] + start["args"]["hold"])
    history = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    rows = [row for row in history if abs(row["time_s"] - end) < 1e-10]
    return rows[0] if rows else None


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--candidates", type=Path, nargs="+", required=True)
    cli.add_argument("--labels", nargs="+", required=True)
    cli.add_argument("--kinds", choices=("solid", "native", "shell"), nargs="+", required=True)
    cli.add_argument("--surface-mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(args.candidates) == len(args.labels) == len(args.kinds)
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.reference / "start.json").read_text())
    rest = solid_rest(args.reference / "rest.npz", start["config"]["density"])
    reference_sampler = TetSampler.allocate(rest["vertices"], rest["tetrahedra"])
    domain = load(args.surface_mesh)
    comparisons = []
    for pair, (run, label, kind) in enumerate(zip(args.candidates, args.labels, args.kinds)):
        candidate_start = json.loads((run / "start.json").read_text())
        if kind == "solid":
            own = solid_rest(run / "rest.npz", candidate_start["config"]["density"])
        elif kind == "native":
            own = load(Path(candidate_start["elastic_mesh"]["path"]))
            assert (
                file_id(Path(candidate_start["elastic_mesh"]["path"]))["sha256"]
                == candidate_start["elastic_mesh"]["sha256"]
            )
        else:
            own = load(run / "rest.npz")
        sampler = reference_sampler if kind == "shell" else TetSampler.allocate(own["vertices"], own["tetrahedra"])
        columns = build_columns(
            reference_sampler, sampler, domain["outer_vertices"], domain["inner_vertices"], domain["mid_faces"]
        )
        surfaces = build_surfaces(
            reference_sampler, sampler, domain["outer_vertices"], domain["inner_vertices"], domain["mid_faces"], columns
        )
        shell_lookup = face_assignment(columns, own["vertices"], own["faces"]) if kind == "shell" else None
        if shell_lookup is not None:
            _, first = np.unique(columns.face_index, return_index=True)
            shell_surface_lookup = shell_lookup[first][surfaces.face_index]
        rows = []
        for level, force in enumerate(start["args"]["loads"]):
            expected_path = args.reference / "frames" / f"level{level}.npz"
            candidate_path = run / "frames" / f"level{level}.npz"
            row = {
                "load_level": level,
                "command_force_n": force,
                "reference_frame_present": expected_path.is_file(),
                "candidate_frame_present": candidate_path.is_file(),
                "qualified_accuracy_comparison": False,
            }
            rows.append(row)
            if not expected_path.is_file() or not candidate_path.is_file():
                row["status"] = "missing field; retain failure, no substitute or extrapolation"
                continue
            assert candidate_start["args"]["loads"][level] == force
            expected, actual = load(expected_path), load(candidate_path)
            reference_rotation = corotation(rest["vertices"], expected["positions_m"], rest["masses"])
            full_reference = reference_rotation.T @ expected["cauchy_pa"] @ reference_rotation
            reference_points = rest["vertices"][rest["tetrahedra"]].mean(axis=1)
            if kind == "solid":
                rotation = corotation(own["vertices"], actual["positions_m"], own["masses"])
                full_actual = rotation.T @ actual["cauchy_pa"] @ rotation
                actual_points = own["vertices"][own["tetrahedra"]].mean(axis=1)
                tensors = [full_actual[columns.native_elements], full_actual[surfaces.native_elements]]
            elif kind == "native":
                full_actual = voigt_tensor(actual["stress_voigt_link_pa"])
                actual_points = own["vertices"][own["tetrahedra"]]
                tensors = [
                    np.einsum("qi,qijk->qjk", metric.native_barycentric, full_actual[metric.native_elements])
                    for metric in (columns, surfaces)
                ]
            else:
                rotation = corotation(own["vertices"], actual["vertices_world_m"], own["masses_kg"])
                full_actual = rotation.T @ actual["cauchy_world_pa"] @ rotation
                triangles = own["vertices"][own["faces"]]
                normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                actual_points = (
                    triangles.mean(axis=1)[:, None] + normal[:, None] * np.array([-0.00025, 0, 0.00025])[None, :, None]
                )
                tensors = [
                    full_actual[lookup, 1]
                    + metric.z[:, None, None] * (full_actual[lookup, 2] - full_actual[lookup, 0]) / 0.0005
                    for metric, lookup in ((columns, shell_lookup), (surfaces, shell_surface_lookup))
                ]
            reference_tensors = [
                full_reference[columns.reference_elements],
                full_reference[surfaces.reference_elements],
            ]
            assert all(np.isfinite(value).all() for value in [full_reference, full_actual, *tensors])
            expected_tangent, expected_surface = [
                tangent(value, metric) for value, metric in zip(reference_tensors, (columns, surfaces))
            ]
            actual_tangent, actual_surface = [
                tangent(value, metric) for value, metric in zip(tensors, (columns, surfaces))
            ]
            reference_membrane, reference_bending = resultants(expected_tangent, columns)
            actual_membrane, actual_bending = resultants(actual_tangent, columns)
            reference_row, candidate_row = endpoint(args.reference, start, level), endpoint(run, candidate_start, level)
            equal_load = all(
                observed is not None
                and max(abs(observed[key] - force) for key in ("upper_reaction_n", "lower_reaction_n"))
                <= 0.01 * max(force, 1e-3)
                for observed in (reference_row, candidate_row)
            )
            row.update(
                reference_frame=file_id(expected_path),
                candidate_frame=file_id(candidate_path),
                reference_mechanics=reference_row,
                candidate_mechanics=candidate_row,
                both_actual_loads_within_one_percent=equal_load,
                status="diagnostic field contrast; equilibrium and reference uncertainty require separate admission",
                tangent_field=field_error(actual_tangent, expected_tangent, columns),
                membrane_n_per_m=area_error(actual_membrane, reference_membrane, columns.area, 5e-4),
                bending_n=area_error(actual_bending, reference_bending, columns.area, 2.5e-7),
                raw_reference_peaks=peaks(full_reference, reference_points),
                raw_candidate_peaks=peaks(full_actual, actual_points),
            )
            if kind != "shell":
                row["solid_only_full3D_on_columns"] = field_error(tensors[0], reference_tensors[0], columns)
            else:
                row["missing_shell_components"] = (
                    "No resolved solid thickness-normal stress/shear; tangential projected comparison only. Nominal thickness500um; source wall has variable normal projection."
                )
            for side, name in ((0, "inner_surface"), (1, "outer_surface")):
                mask = surfaces.side == side
                row[name] = tensor_error(actual_surface[mask], expected_surface[mask], surfaces.weights_m2[mask])
            output = args.output / f"pair{pair}-level{level}.npz"
            np.savez_compressed(
                output,
                points_m=columns.points,
                face_index=columns.face_index,
                weights_m3=columns.weights,
                area_m2=columns.area,
                reference_tangent_pa=expected_tangent,
                candidate_tangent_pa=actual_tangent,
                reference_membrane_n_per_m=reference_membrane,
                candidate_membrane_n_per_m=actual_membrane,
                reference_bending_n=reference_bending,
                candidate_bending_n=actual_bending,
                surface_points_m=surfaces.points,
                surface_side=surfaces.side,
                reference_surface_tangent_pa=expected_surface,
                candidate_surface_tangent_pa=actual_surface,
            )
            row["common_field_artifact"] = file_id(output)
        comparisons.append({"label": label, "kind": kind, "run": str(run), "levels": rows})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "reference": str(args.reference),
            "surface_domain": file_id(args.surface_mesh),
            "comparisons": comparisons,
            "reference_qualified": False,
            "scope": "Full-spatial endpoint contrasts on shared material columns and exact one-sided surfaces. Raw extrema retained separately. Unequal actual load or missing equilibrium is diagnostic only; no inferred accuracy ranking.",
        },
    )


if __name__ == "__main__":
    main()
