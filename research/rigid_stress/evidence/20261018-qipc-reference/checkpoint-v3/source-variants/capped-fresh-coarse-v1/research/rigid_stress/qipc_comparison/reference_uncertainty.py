"""Measure paired complete-history QIPC uncertainty on exact shared material columns."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import (
    TetSampler,
    build_columns,
    build_surfaces,
    corotation,
    field_error,
    tangent,
    tensor_error,
    time_integrated_error,
)
from .compare_fields import area_error, resultants


def frame_index(folder):
    result = {}
    for path in sorted((folder / "frames").glob("step*.npz")):
        with np.load(path) as saved:
            time = saved["time_s"].item()
        tick = round(time / 0.01)
        if abs(tick * 0.01 - time) < 1e-12:
            assert tick not in result
            result[tick] = path
    return result


def load_rest(folder):
    with np.load(folder / "rest.npz") as saved:
        return {name: saved[name] for name in saved.files}


def wrench_difference(actual, reference, force_floor=1e-3):
    first, second = actual["contact_forces_n"].sum(axis=0), reference["contact_forces_n"].sum(axis=0)

    def moment(saved):
        return np.cross(saved["positions_m"], saved["contact_forces_n"]).sum(axis=0)

    return {
        "force_difference_n": (first - second).tolist(),
        "reference_force_n": second.tolist(),
        "force_relative_difference": (np.linalg.norm(first - second) / np.linalg.norm(second)).item()
        if np.linalg.norm(second) > force_floor
        else None,
        "moment_world_origin_difference_nm": (moment(actual) - moment(reference)).tolist(),
        "scope": "Net egg wrench, including cancellation of opposing pinch forces. This alone cannot qualify per-finger reactions.",
    }


def constitutive_snapshot(saved, rest):
    tets = rest["tetrahedra"]
    initial = rest["vertices_world_m"]
    inverse = np.linalg.inv((initial[tets[:, 1:]] - initial[tets[:, :1]]).transpose(0, 2, 1))
    current = saved["positions_m"]
    F = (current[tets[:, 1:]] - current[tets[:, :1]]).transpose(0, 2, 1) @ inverse
    left, singular, _ = np.linalg.svd(F)
    mu, lame = 1e10 / 2.6, 1e10 * 0.3 / (1.3 * 0.4)
    strain = np.log(singular)
    principal = 2 * mu * strain + lame * strain.sum(axis=1, keepdims=True)
    hencky = (left * principal[:, None]) @ left.transpose(0, 2, 1) / np.linalg.det(F)[:, None, None]
    small = singular - 1
    principal = 2 * mu * small + lame * small.sum(axis=1, keepdims=True)
    linear = (left * principal[:, None]) @ left.transpose(0, 2, 1)
    return {
        "maximum_principal_stretch_departure_from_one": np.abs(singular - 1).max().item(),
        "hencky_at_identical_F_vs_recorded_energy": tensor_error(hencky, saved["cauchy_pa"], rest["volumes_m3"]),
        "corotated_linear_at_identical_F_vs_recorded_energy": tensor_error(
            linear, saved["cauchy_pa"], rest["volumes_m3"]
        ),
        "scope": "Same-deformation constitutive diagnostic, volume weighted. It does not replace a different-energy dynamic history.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--surface-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rest = load_rest(args.reference)
    reference_index = frame_index(args.reference)
    reference_sampler = TetSampler.allocate(rest["authored_vertices_m"], rest["tetrahedra"])
    with np.load(args.surface_mesh) as saved:
        outer, inner, faces = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
    comparisons = []
    constitutive = []
    for tick, path in reference_index.items():
        with np.load(path) as saved:
            constitutive.append({"time_s": tick * 0.01, **constitutive_snapshot(saved, rest)})
    for candidate_number, folder in enumerate(args.candidates):
        candidate_rest = load_rest(folder)
        sampler = TetSampler.allocate(candidate_rest["authored_vertices_m"], candidate_rest["tetrahedra"])
        columns = build_columns(reference_sampler, sampler, outer, inner, faces)
        surfaces = build_surfaces(reference_sampler, sampler, outer, inner, faces, columns)
        np.savez_compressed(args.output / f"pair{candidate_number}-columns.npz", **asdict(columns))
        candidate_index = frame_index(folder)
        rows = []
        for tick in sorted(reference_index.keys() & candidate_index.keys()):
            with np.load(reference_index[tick]) as expected, np.load(candidate_index[tick]) as actual:
                tensors, boundaries = [], []
                for saved, own_rest, elements, surface_elements in (
                    (expected, rest, columns.reference_elements, surfaces.reference_elements),
                    (actual, candidate_rest, columns.native_elements, surfaces.native_elements),
                ):
                    rotation = corotation(own_rest["authored_vertices_m"], saved["positions_m"], own_rest["masses_kg"])
                    tensor = rotation.T @ saved["cauchy_pa"] @ rotation
                    tensors.append(tangent(tensor[elements], columns))
                    boundaries.append(tangent(tensor[surface_elements], surfaces))
                reference_nm, reference_mm = resultants(tensors[0], columns)
                actual_nm, actual_mm = resultants(tensors[1], columns)
                row = {
                    "time_s": tick * 0.01,
                    "common_tangent": field_error(tensors[1], tensors[0], columns),
                    "membrane": area_error(actual_nm, reference_nm, columns.area, 5e-4),
                    "bending": area_error(actual_mm, reference_mm, columns.area, 2.5e-7),
                    "net_egg_wrench": wrench_difference(actual, expected),
                    "mass_weighted_com_difference_m": (
                        candidate_rest["masses_kg"] @ actual["positions_m"] / candidate_rest["masses_kg"].sum()
                        - rest["masses_kg"] @ expected["positions_m"] / rest["masses_kg"].sum()
                    ).tolist(),
                }
                for side, name in ((0, "inner_surface"), (1, "outer_surface")):
                    selected = surfaces.side == side
                    row[name] = tensor_error(
                        boundaries[1][selected], boundaries[0][selected], surfaces.weights_m2[selected]
                    )
                rows.append(row)
        complete = (
            json.loads((args.reference / "result.json").read_text())["completed_requested_history"]
            and json.loads((folder / "result.json").read_text())["completed_requested_history"]
            and len(rows) == len(reference_index)
        )
        temporal = {}
        if complete:
            for key, floor in (
                ("common_tangent", 1.0),
                ("inner_surface", 1.0),
                ("outer_surface", 1.0),
                ("membrane", 5e-4),
                ("bending", 2.5e-7),
            ):
                reference_key = (
                    "reference_weighted_rms" if key in ("membrane", "bending") else "reference_tensor_rms_pa"
                )
                error_key = (
                    "absolute_weighted_rms_error" if key in ("membrane", "bending") else "absolute_tensor_rms_error_pa"
                )
                temporal[key] = time_integrated_error(
                    [row["time_s"] for row in rows],
                    [row[key][reference_key] for row in rows],
                    [row[key][error_key] for row in rows],
                    floor,
                )
        comparisons.append(
            {
                "candidate": file_id(folder / "result.json"),
                "candidate_folder": str(folder),
                "candidate_config": json.loads((folder / "result.json").read_text())["config"],
                "common_times": len(rows),
                "missing_candidate_ticks": sorted(reference_index.keys() - candidate_index.keys()),
                "material_columns": len(columns.area),
                "pair_quadrature_points": len(columns.points),
                "history": rows,
                "complete_time_integrated_common_metrics": temporal,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "paired-reference-history-uncertainty",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "surface_mesh": file_id(args.surface_mesh),
            "constitutive_at_identical_deformation": constitutive,
            "comparisons": comparisons,
            "reference_qualified": False,
            "scope": "The same material face-centroid columns are integrated on every union of both P1 partitions. Norms and resultants are exact along these columns up to geometric/roundoff tolerance; in-plane sampling remains finite. These are separate uncertainty contrasts, not an additive scalar error budget. Raw full-mesh peaks/solver caps remain in the source microstep journals. Complete finite histories and a small net wrench difference do not by themselves qualify per-finger reactions, contact regularization or continuum peaks.",
        },
    )


if __name__ == "__main__":
    main()
