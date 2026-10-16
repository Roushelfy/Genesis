"""Locate every last-three-level raw peak and measure compatibility of its prescribed incident boundary tractions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet

from .finite_traction_reference import physical_grip_patches, triangle_closest_points
from .summarize_fine_references import CASES, closed_source


def symmetric_traction_matrix(normals):
    """Map xx/yy/zz/xy/yz/xz symmetric stress components to outward-normal traction."""
    x, y, z = normals.T
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([x, zero, zero, y, zero, z], axis=1),
            np.stack([zero, y, zero, x, z, zero], axis=1),
            np.stack([zero, zero, z, zero, y, x], axis=1),
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = args.runtime_root
    args.output.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    normals_control = np.array([[1, 2, 3], [-2, 3, -4], [3, -4, 5]], dtype=float)
    normals_control /= np.linalg.norm(normals_control, axis=1)[:, None]
    stress_control = np.array([[2.0, 7, 13], [7, -3, -11], [13, -11, 5]])
    components_control = np.array([2.0, -3, 5, 7, -11, 13])
    np.testing.assert_allclose(
        symmetric_traction_matrix(normals_control) @ components_control,
        normals_control @ stress_control,
        rtol=1e-14,
        atol=1e-14,
    )
    for geometry in ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]:
        qualification = p / f"extended-finite-{geometry}-original-qualification-v1"
        closed_source(qualification)
        config_path = qualification / "results/configuration.json"
        retain(config_path)
        config = json.loads(config_path.read_text())
        runs = [Path(config["baseline_run"]), *(p / f"extended-finite-{geometry}-level{i}-v1" for i in [1, 2])]
        meshes = [Path(config["baseline_mesh"])]
        for run in runs[1:]:
            config_path = run / "results/configuration.json"
            retain(config_path)
            meshes.append(Path(json.loads(config_path.read_text())["mesh"]))
        for level, (run, mesh_path) in enumerate(zip(runs, meshes, strict=True)):
            closed_source(run)
            mesh_hash = retain(mesh_path)
            with np.load(mesh_path) as saved:
                mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T)
            boundary = mesh.boundary_facets()
            faces = mesh.facets[:, boundary].T
            triangles = mesh.p.T[faces]
            raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]
            centers = mesh.p.T[mesh.t.T].mean(axis=1)
            owners = mesh.f2t[0, boundary]
            normals *= np.sign(np.einsum("fi,fi->f", normals, triangles.mean(axis=1) - centers[owners]))[:, None]
            edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            pairs = np.flatnonzero((edges[order][1:] == edges[order][:-1]).all(axis=1))
            neighbors = np.repeat(np.arange(len(faces)), 3)[order]
            adjacency = sparse.csr_matrix(
                (np.ones(len(pairs)), (neighbors[pairs], neighbors[pairs + 1])), shape=(len(faces),) * 2
            )
            for case in CASES:
                path = run / f"results/{case}/snapshot.json"
                retain(path)
                snapshot = json.loads(path.read_text())
                assert snapshot["mesh_sha256"] == mesh_hash and snapshot["patch_radius_m"] == 0.003
                point = np.array(snapshot["reference_peak_position_m"])
                distances = np.linalg.norm(mesh.p.T - point, axis=1)
                vertex = np.argmin(distances)
                assert distances[vertex] < 1e-13
                incident = np.flatnonzero((faces == vertex).any(axis=1))
                _, distance_squared = triangle_closest_points(triangles, point)
                nearest_boundary = float(np.sqrt(distance_squared.min()))
                traction = np.zeros((len(incident), 3))
                normal_control = np.zeros_like(traction)
                selected_masks, amplitudes = [], []
                specs = physical_grip_patches(geometry, case, snapshot["patch_radius_m"])
                assert len(specs) == len(snapshot["patches"])
                for spec, patch in zip(specs, snapshot["patches"], strict=True):
                    np.testing.assert_array_equal(spec.centre_m, patch["physical_anchor_m"])
                    _, distance_squared = triangle_closest_points(triangles, spec.centre_m)
                    anchor = np.argmin(distance_squared)
                    selected = np.flatnonzero(distance_squared < spec.radius_m**2)
                    _, components = connected_components(adjacency[selected][:, selected], directed=False)
                    selected = selected[components == components[np.flatnonzero(selected == anchor)[0]]]
                    mask = np.isin(incident, selected)
                    selected_masks.append(mask)
                    relative_squared = np.sum((point - spec.centre_m) ** 2) / spec.radius_m**2
                    amplitude = spec.strength_n / patch["profile_integral_m2"]
                    amplitude *= np.exp(-relative_squared / (2 * 0.45**2)) * max(1 - relative_squared, 0) ** 2
                    amplitudes.append(float(amplitude))
                    values = np.broadcast_to(spec.direction / np.linalg.norm(spec.direction), traction.shape).copy()
                    if spec.is_local_normal_profile:
                        inward = -normals[incident]
                        tangent = np.array([0.0, 0, 1]) - inward[:, 2, None] * inward
                        tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-30)[:, None]
                        values = inward + spec.shear_fraction * spec.friction * tangent
                    traction += amplitude * mask[:, None] * values
                    normal_control -= amplitude * mask[:, None] * normals[incident]
                record = {
                    "geometry": geometry,
                    "case": case,
                    "level": level,
                    "tetrahedra": mesh.nelements,
                    "raw_global_peak_Pa": snapshot["reference_peak_Pa"],
                    "raw_global_peak_position_m": point.tolist(),
                    "nearest_material_boundary_distance_m": nearest_boundary,
                    "peak_vertex": int(vertex),
                    "incident_boundary_faces": len(incident),
                    "is_boundary_peak": bool(len(incident)),
                }
                arrays = {
                    "peak_m": point,
                    "boundary_face_indices": incident,
                    "outward_normals": normals[incident],
                    "prescribed_tractions_Pa": traction,
                    "normal_only_control_Pa": normal_control,
                    "selected_source_masks": np.array(selected_masks),
                    "source_amplitudes_Pa": np.array(amplitudes),
                }
                if len(incident):
                    matrix = symmetric_traction_matrix(normals[incident]).reshape(-1, 6)
                    fitted, _, rank, singular = np.linalg.lstsq(matrix, traction.reshape(-1), rcond=None)
                    residual = (matrix @ fitted).reshape(-1, 3) - traction
                    reciprocal = normals[incident] @ traction.T - traction @ normals[incident].T
                    normal_fit = np.linalg.lstsq(matrix, normal_control.reshape(-1), rcond=None)[0]
                    normal_error = np.linalg.norm(matrix @ normal_fit - normal_control.reshape(-1))
                    scale = np.linalg.norm(traction)
                    normal_scale = np.linalg.norm(normal_control)
                    assert normal_error <= 1e-10 * max(normal_scale, 1e-20)
                    record.update(
                        {
                            "traction_norm_Pa": float(scale),
                            "single_symmetric_tensor_fit_residual_Pa": float(np.linalg.norm(residual)),
                            "relative_fit_residual": float(np.linalg.norm(residual) / scale) if scale else None,
                            "maximum_reciprocal_symmetry_defect_Pa": float(abs(reciprocal).max()),
                            "normal_only_control_residual_Pa": float(normal_error),
                            "traction_matrix_rank": int(rank),
                        }
                    )
                    arrays.update(
                        fitted_symmetric_components_Pa=fitted,
                        fit_residual_Pa=residual,
                        reciprocal_defect_Pa=reciprocal,
                        matrix_singular_values=singular,
                    )
                np.savez_compressed(args.output / f"{geometry}-level{level}-{case}.npz", **arrays)
                rows.append(record)
                print(json.dumps(record), flush=True)
    assert len(rows) == 90
    summary = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "manufactured_symmetric_stress_control_passed": True,
        "all_normal_only_trace_controls_passed": True,
        "goal_completion": False,
        "limitations": "Diagnoses the exact recorded raw peaks and original boundary law on the latest three meshes. A single symmetric stress tensor requires n_i dot t_j = n_j dot t_i. Measured incompatibility concerns continuity of one stress tensor at that boundary vertex. It does not prove a stress singularity or continuum peak growth, and zero local defect does not exclude other singularities. Normal-only traces are algebraic controls only, not replacement physical loads. Original full fields, raw maxima, radii, loads and numerical budgets remain unchanged.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
