"""Compare all three candidates on one explicit material-column domain, retaining reference qualification limits."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import (
    CommonColumns,
    CommonSurfaces,
    corotation,
    field_error,
    resultants,
    tangent,
    time_integrated_error,
    tensor_error,
    voigt_tensor,
)


def face_assignment(columns, vertices, faces):
    centers = np.zeros((len(columns.area), 3))
    weight = np.zeros(len(columns.area))
    np.add.at(centers, columns.face_index, columns.weights[:, None] * columns.points)
    np.add.at(weight, columns.face_index, columns.weights)
    centers /= weight[:, None]
    triangles = vertices[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    inverse = np.linalg.inv(np.einsum("fai,fbi->fab", edges, edges))
    selected = []
    for point in centers:
        free = np.einsum("fij,fj->fi", inverse, np.einsum("fai,fi->fa", edges, point - triangles[:, 0]))
        bary = np.column_stack((1 - free.sum(axis=1), free))
        distance = np.linalg.norm(np.einsum("fi,fij->fj", bary, triangles) - point, axis=1)
        distance[bary.min(axis=1) < -1e-9] = np.inf
        index = distance.argmin()
        assert distance[index] < 1e-12, "Shell midpoint geometry is not the shared physical domain"
        selected.append(index)
    return np.array(selected)[columns.face_index]


def area_error(actual, reference, area, floor):
    denominator = np.einsum("f,fij,fij->", area, reference, reference)
    numerator = np.einsum("f,fij,fij->", area, actual - reference, actual - reference)
    rms = np.sqrt(denominator / area.sum())
    return {
        "relative_error": float(np.sqrt(numerator / denominator)) if rms > floor else None,
        "absolute_weighted_rms_error": float(np.sqrt(numerator / area.sum())),
        "reference_weighted_rms": float(rms),
        "relative_floor": floor,
    }


def index_frames(folder):
    frames = {}
    for path in sorted((folder / "frames").glob("step*.npz")):
        with np.load(path) as saved:
            time = saved["time_s"].item()
        tick = round(time / 0.01)
        assert abs(tick * 0.01 - time) < 1e-12 and tick not in frames
        frames[tick] = path
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dpr", type=Path, required=True)
    parser.add_argument("--scr", type=Path, required=True)
    parser.add_argument("--shell", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path)
    parser.add_argument("--shell-thickness", type=float, default=0.0005)
    parser.add_argument("--snapshot-times", type=float, nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot_times is not None:
        assert all(time > 0 and abs(round(time / 0.01) * 0.01 - time) < 1e-12 for time in args.snapshot_times)
    snapshot_ticks = None if args.snapshot_times is None else {round(time / 0.01) - 1 for time in args.snapshot_times}
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.columns) as saved:
        columns = CommonColumns(**{key: saved[key] for key in saved.files})
    surfaces = None
    if args.surfaces is not None:
        with np.load(args.surfaces) as saved:
            surfaces = CommonSurfaces(**{key: saved[key] for key in saved.files})
        np.testing.assert_allclose(surfaces.normal, columns.normal, rtol=0, atol=1e-15)
    with np.load(args.reference / "rest.npz") as saved:
        authored, masses = saved["authored_vertices_m"], saved["masses_kg"]
    with np.load(args.shell / "rest.npz") as saved:
        shell_vertices, shell_faces = saved["vertices_m"], saved["faces"]
        shell_masses = saved["vertex_masses_kg"] if "vertex_masses_kg" in saved.files else None
        shell_thickness = saved["face_thickness_m"] if "face_thickness_m" in saved.files else None
    shell_lookup = face_assignment(columns, shell_vertices, shell_faces)
    _, face_first = np.unique(columns.face_index, return_index=True)
    shell_faces_by_column = shell_lookup[face_first]
    shell_edges = shell_vertices[shell_faces[:, 1:]] - shell_vertices[shell_faces[:, :1]]
    shell_area = np.linalg.norm(np.cross(shell_edges[:, 0], shell_edges[:, 1]), axis=1) / 2
    if shell_masses is None:
        shell_masses = np.bincount(
            shell_faces.reshape(-1), weights=np.repeat(shell_area, 3), minlength=len(shell_vertices)
        )
    if shell_thickness is None:
        shell_thickness = np.full(len(shell_faces), args.shell_thickness)
    history = []
    snapshots = []
    candidate_indices = [index_frames(folder) for folder in (args.dpr, args.scr, args.shell)]
    unmatched = []
    files = sorted((args.reference / "frames").glob("step*.npz"))
    for path in files:
        with np.load(path) as reference:
            time = reference["time_s"].item()
            control_step = round(time / 0.01) - 1
            if abs((control_step + 1) * 0.01 - time) >= 1e-12:
                unmatched.append(
                    {"time_s": time, "reason": "reference frame is outside declared 10ms common observation grid"}
                )
                continue
            if not all(control_step + 1 in index for index in candidate_indices):
                unmatched.append({"time_s": time, "reason": "candidate frame missing at declared common time"})
                continue
            candidate_paths = [index[control_step + 1] for index in candidate_indices]
            rotation = corotation(authored, reference["positions_m"], masses)
            reference_tensor = reference["cauchy_pa"][columns.reference_elements]
            expected_full = rotation.T @ reference_tensor @ rotation
            expected = tangent(expected_full, columns)
            expected_surface = (
                tangent(rotation.T @ reference["cauchy_pa"][surfaces.reference_elements] @ rotation, surfaces)
                if surfaces is not None
                else None
            )
        expected_membrane, expected_bending = resultants(expected, columns)
        tensors = []
        full_tensors = []
        surface_tensors = []
        for candidate in candidate_paths[:2]:
            with np.load(candidate) as saved:
                assert abs(saved["time_s"].item() - time) < 1e-12
                full_corner = voigt_tensor(saved["stress_voigt_link_pa"])
                corner = full_corner[columns.native_elements]
                tensor = np.einsum("qi,qijk->qjk", columns.native_barycentric, corner)
                full_tensors.append(tensor)
                tensors.append(tangent(tensor, columns))
                if surfaces is not None:
                    surface_tensor = np.einsum(
                        "qi,qijk->qjk", surfaces.native_barycentric, full_corner[surfaces.native_elements]
                    )
                    surface_tensors.append(tangent(surface_tensor, surfaces))
        with np.load(candidate_paths[2]) as saved:
            assert abs(saved["time_s"].item() - time) < 1e-12
            shell_rotation = corotation(shell_vertices, saved["vertices_world_m"], shell_masses)
            fibers = saved["cauchy_world_pa"][shell_lookup]
            tensor = (
                fibers[:, 1]
                + columns.z[:, None, None] * (fibers[:, 2] - fibers[:, 0]) / shell_thickness[shell_lookup, None, None]
            )
            tensors.append(tangent(shell_rotation.T @ tensor @ shell_rotation, columns))
            if surfaces is not None:
                surface_fibers = saved["cauchy_world_pa"][shell_faces_by_column[surfaces.face_index]]
                surface_tensor = (
                    surface_fibers[:, 1]
                    + surfaces.z[:, None, None]
                    * (surface_fibers[:, 2] - surface_fibers[:, 0])
                    / shell_thickness[shell_faces_by_column[surfaces.face_index], None, None]
                )
                surface_tensors.append(tangent(shell_rotation.T @ surface_tensor @ shell_rotation, surfaces))
        entries = []
        finite = np.isfinite(expected).all() and all(np.isfinite(tensor).all() for tensor in tensors)
        if surfaces is not None:
            finite = (
                finite
                and np.isfinite(expected_surface).all()
                and all(np.isfinite(tensor).all() for tensor in surface_tensors)
            )
        if not finite:
            history.append({"time_s": time, "all_fields_finite": False, "errors": None})
            continue
        for method, tensor in zip(("dpr", "scr", "shell"), tensors):
            membrane, bending = resultants(tensor, columns)
            entries.append(
                {
                    "method": method,
                    "tangent_field": field_error(tensor, expected, columns),
                    "membrane_n_per_m": area_error(membrane, expected_membrane, columns.area, 5e-4),
                    "bending_n": area_error(bending, expected_bending, columns.area, 2.5e-7),
                }
            )
        if surfaces is not None:
            for entry, surface_tensor in zip(entries, surface_tensors):
                for side, name in ((0, "inner_surface_tangent"), (1, "outer_surface_tangent")):
                    selected = surfaces.side == side
                    entry[name] = tensor_error(
                        surface_tensor[selected], expected_surface[selected], surfaces.weights_m2[selected]
                    )
        for entry, full_tensor in zip(entries[:2], full_tensors):
            entry["solid_only_full_3d_on_common_columns"] = field_error(full_tensor, expected_full, columns)
        history.append({"time_s": time, "all_fields_finite": True, "errors": entries})
        export_snapshot = (
            path == files[-1] or control_step in (1, 19, 49, 99, 109, 119, 179, 249, 299, 419, 449, 539, 559)
            if snapshot_ticks is None
            else control_step in snapshot_ticks
        )
        if export_snapshot:
            payload = {
                "points_m": columns.points,
                "weights_m3": columns.weights,
                "face_index": columns.face_index,
                "area_m2": columns.area,
                "projected_thickness_m": columns.thickness,
                "time_s": time,
                "reference_tangent_pa": expected,
                "reference_full_3d_on_columns_pa": expected_full,
                "reference_membrane_n_per_m": expected_membrane,
                "reference_bending_n": expected_bending,
            }
            for method, tensor in zip(("dpr", "scr", "shell"), tensors):
                payload[f"{method}_tangent_pa"] = tensor
                payload[f"{method}_membrane_n_per_m"], payload[f"{method}_bending_n"] = resultants(tensor, columns)
            for method, tensor in zip(("dpr", "scr"), full_tensors):
                payload[f"{method}_full_3d_on_columns_pa"] = tensor
            if surfaces is not None:
                payload.update(
                    surface_points_m=surfaces.points,
                    surface_weights_m2=surfaces.weights_m2,
                    surface_side=surfaces.side,
                    reference_surface_tangent_pa=expected_surface,
                )
                for method, surface_tensor in zip(("dpr", "scr", "shell"), surface_tensors):
                    payload[f"{method}_surface_tangent_pa"] = surface_tensor
            destination = args.output / f"common-step{control_step:06d}.npz"
            np.savez_compressed(destination, **payload)
            snapshots.append(file_id(destination))
    reference_receipt = json.loads((args.reference / "result.json").read_text())
    candidate_receipts = [
        json.loads((folder / "result.json").read_text()) for folder in (args.dpr, args.scr, args.shell)
    ]
    complete = (
        reference_receipt["completed_requested_history"]
        and all(value["completed_requested_history"] for value in candidate_receipts)
        and not unmatched
        and all(row["all_fields_finite"] for row in history)
    )
    temporal = []
    if complete:
        for method in ("dpr", "scr", "shell"):
            entries = [next(entry for entry in row["errors"] if entry["method"] == method) for row in history]
            metrics = {}
            for key, floor in (
                ("tangent_field", 1.0),
                ("inner_surface_tangent", 1.0),
                ("outer_surface_tangent", 1.0),
                ("membrane_n_per_m", 5e-4),
                ("bending_n", 2.5e-7),
                ("solid_only_full_3d_on_common_columns", 1.0),
            ):
                if key == "solid_only_full_3d_on_common_columns" and method == "shell":
                    continue
                if key in ("inner_surface_tangent", "outer_surface_tangent") and surfaces is None:
                    continue
                reference_key = (
                    "reference_weighted_rms" if key in ("membrane_n_per_m", "bending_n") else "reference_tensor_rms_pa"
                )
                error_key = (
                    "absolute_weighted_rms_error"
                    if key in ("membrane_n_per_m", "bending_n")
                    else "absolute_tensor_rms_error_pa"
                )
                metrics[key] = time_integrated_error(
                    [row["time_s"] for row in history],
                    [entry[key][reference_key] for entry in entries],
                    [entry[key][error_key] for entry in entries],
                    floor,
                )
            temporal.append({"method": method, "metrics": metrics})
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-three-way-common-column-errors",
            "identity": runtime_identity(),
            "reference_receipt": file_id(args.reference / "result.json"),
            "columns": file_id(args.columns),
            "surfaces": file_id(args.surfaces) if args.surfaces is not None else None,
            "candidate_receipts": [file_id(folder / "result.json") for folder in (args.dpr, args.scr, args.shell)],
            "reference_qualified": False,
            "scope": "Diagnostic common-position/time comparisons. Reference spatial/dt/contact uncertainty and complete-grasp qualification remain required; these numbers do not establish a qualified accuracy ranking.",
            "stress_convention": "World physical QIPC/Shell Cauchy corotated by each link's mass-weighted best fit; native authored-frame linear stress. Shared undeformed tangent projector; finite-strain model differences remain separate.",
            "history": history,
            "complete_time_integrated_common_metrics": temporal,
            "complete_common_observation_coverage": bool(complete),
            "snapshots": snapshots,
            "requested_snapshot_times_s": args.snapshot_times,
            "snapshot_scope": "Snapshot selection affects derived field exports only. Every compatible recorded common observation still contributes to the history and complete-time metrics; original raw full fields and physical-step peak states stay at their source paths.",
            "unmatched_reference_frames": unmatched,
        },
    )


if __name__ == "__main__":
    main()
