"""Compare all three candidates on one explicit material-column domain, retaining reference qualification limits."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import CommonColumns, corotation, field_error, resultants, tangent, voigt_tensor


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dpr", type=Path, required=True)
    parser.add_argument("--scr", type=Path, required=True)
    parser.add_argument("--shell", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--shell-thickness", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.columns) as saved:
        columns = CommonColumns(**{key: saved[key] for key in saved.files})
    with np.load(args.reference / "rest.npz") as saved:
        authored, masses = saved["authored_vertices_m"], saved["masses_kg"]
    with np.load(args.shell / "rest.npz") as saved:
        shell_vertices, shell_faces = saved["vertices_m"], saved["faces"]
    shell_lookup = face_assignment(columns, shell_vertices, shell_faces)
    shell_edges = shell_vertices[shell_faces[:, 1:]] - shell_vertices[shell_faces[:, :1]]
    shell_area = np.linalg.norm(np.cross(shell_edges[:, 0], shell_edges[:, 1]), axis=1) / 2
    shell_masses = np.bincount(shell_faces.reshape(-1), weights=np.repeat(shell_area, 3), minlength=len(shell_vertices))
    history = []
    snapshots = []
    files = sorted((args.reference / "frames").glob("step*.npz"))
    for path in files:
        with np.load(path) as reference:
            time = reference["time_s"].item()
            control_step = round(time / 0.01) - 1
            assert abs((control_step + 1) * 0.01 - time) < 1e-12
            name = f"step{control_step:06d}.npz"
            candidate_paths = [args.dpr / "frames" / name, args.scr / "frames" / name, args.shell / "frames" / name]
            if not all(candidate.exists() for candidate in candidate_paths):
                continue
            rotation = corotation(authored, reference["positions_m"], masses)
            reference_tensor = reference["cauchy_pa"][columns.reference_elements]
            expected = tangent(rotation.T @ reference_tensor @ rotation, columns)
        expected_membrane, expected_bending = resultants(expected, columns)
        tensors = []
        for candidate in candidate_paths[:2]:
            with np.load(candidate) as saved:
                assert abs(saved["time_s"].item() - time) < 1e-12
                corner = voigt_tensor(saved["stress_voigt_link_pa"])[columns.native_elements]
                tensor = np.einsum("qi,qijk->qjk", columns.native_barycentric, corner)
                tensors.append(tangent(tensor, columns))
        with np.load(candidate_paths[2]) as saved:
            assert abs(saved["time_s"].item() - time) < 1e-12
            shell_rotation = corotation(shell_vertices, saved["vertices_world_m"], shell_masses)
            fibers = saved["cauchy_world_pa"][shell_lookup]
            tensor = fibers[:, 1] + columns.z[:, None, None] * (fibers[:, 2] - fibers[:, 0]) / args.shell_thickness
            tensors.append(tangent(shell_rotation.T @ tensor @ shell_rotation, columns))
        entries = []
        finite = np.isfinite(expected).all() and all(np.isfinite(tensor).all() for tensor in tensors)
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
        history.append({"time_s": time, "all_fields_finite": True, "errors": entries})
        if path == files[-1] or control_step in (1, 19, 49, 99):
            payload = {
                "points_m": columns.points,
                "weights_m3": columns.weights,
                "face_index": columns.face_index,
                "area_m2": columns.area,
                "projected_thickness_m": columns.thickness,
                "time_s": time,
                "reference_tangent_pa": expected,
                "reference_membrane_n_per_m": expected_membrane,
                "reference_bending_n": expected_bending,
            }
            for method, tensor in zip(("dpr", "scr", "shell"), tensors):
                payload[f"{method}_tangent_pa"] = tensor
                payload[f"{method}_membrane_n_per_m"], payload[f"{method}_bending_n"] = resultants(tensor, columns)
            destination = args.output / f"common-step{control_step:06d}.npz"
            np.savez_compressed(destination, **payload)
            snapshots.append(file_id(destination))
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-three-way-common-column-errors",
            "identity": runtime_identity(),
            "reference_receipt": file_id(args.reference / "result.json"),
            "columns": file_id(args.columns),
            "candidate_receipts": [file_id(folder / "result.json") for folder in (args.dpr, args.scr, args.shell)],
            "reference_qualified": False,
            "scope": "Diagnostic common-position/time comparisons. Reference spatial/dt/contact uncertainty and complete-grasp qualification remain required; these numbers do not establish a qualified accuracy ranking.",
            "stress_convention": "World physical QIPC/Shell Cauchy corotated by each link's mass-weighted best fit; native authored-frame linear stress. Shared undeformed tangent projector; finite-strain model differences remain separate.",
            "history": history,
            "snapshots": snapshots,
        },
    )


if __name__ == "__main__":
    main()
