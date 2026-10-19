"""Independently integrate captured native tractions and solve their actual same-mesh elastic RHS."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import FreeElasticP2, quadratic_triangle_weights


def footprints(points, forces):
    """Summarize actual axial resultants and all nonzero-force support, without trimming."""
    rows = []
    for upper in (False, True):
        active = ((points[:, 2] > 0) == upper) & np.any(forces != 0, axis=1)
        selected, force = points[active], forces[active]
        normal = (-1 if upper else 1) * force[:, 2]
        radius2 = (selected[:, :2] ** 2).sum(axis=1)
        rows.append(
            {
                "upper": upper,
                "nonzero_samples": int(active.sum()),
                "force_vector_n": force.sum(axis=0).tolist(),
                "axial_force_n": float(normal.sum()),
                "minimum_signed_axial_sample_force_n": float(normal.min()),
                "all_nonzero_radius_max_m": float(np.sqrt(radius2.max())),
                "axial_force_rms_radius_m": float(np.sqrt(normal @ radius2 / normal.sum()))
                if np.all(normal >= 0) and normal.sum() > 0
                else None,
            }
        )
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--capture", type=Path, required=True)
    cli.add_argument("--additional-captures", type=Path, nargs="*", default=[])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.capture / "start.json").read_text())
    captures, cases = [args.capture, *args.additional_captures], []
    for folder in captures:
        captured = json.loads((folder / "result.json").read_text())
        assert captured["passed"]
        assert json.loads((folder / "start.json").read_text())["mesh"] == start["mesh"]
        cases.extend((folder, case) for case in captured["cases"])
    mesh = Path(start["mesh"]["path"])
    assert file_id(mesh) == start["mesh"]
    with np.load(mesh) as saved:
        vertices, tetrahedra, faces = saved["vertices"], saved["tetrahedra"], saved["mid_faces"]
    model = FreeElasticP2(
        vertices, tetrahedra, 1e10, 0.3, 2000, column_faces=faces, through_wall_ordering=True, cvxopt_factor=True
    )
    tree = cKDTree(model.vertices)
    rows = []
    for index, (folder, case) in enumerate(cases):
        assert case["capture_checks"]["passed"]
        assert all(file_id(Path(case[key]["path"])) == case[key] for key in ("quadrature", "replay", "original"))
        with np.load(case["quadrature"]["path"]) as saved:
            points, forces = saved["points_material_m"], saved["quadrature_forces_n"]
            native_nodes, source_sample_face = saved["p2_nodes"], saved["source_sample_face"]
        with np.load(case["replay"]["path"]) as saved:
            native_coordinates = saved["vertices_m"]
            native_force, native_rhs, native_stress = saved["forces_n"], saved["rhs_n"], saved["stress_pa"]
            original_points, original_forces = saved["original_contact_points_m"], saved["original_contact_forces_n"]
            np.testing.assert_array_equal(saved["tetrahedra"], tetrahedra)
        distance, permutation = tree.query(native_coordinates)
        assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(model.vertices)
        # Compute independent barycentric coordinates and P2 face functions from
        # positions. Do not reuse the exported native interpolation weights.
        _, first, inverse = np.unique(source_sample_face[:, 2], return_index=True, return_inverse=True)
        triangle = native_coordinates[native_nodes[first, :3]]
        edge = triangle[:, 1:] - triangle[:, :1]
        gram = np.einsum("fai,fbi->fab", edge, edge)
        dual = np.linalg.solve(gram, edge)
        local = np.einsum("pai,pi->pa", dual[inverse], points - triangle[inverse, 0])
        barycentric = np.column_stack((1 - local.sum(axis=1), local))
        assert barycentric.min() >= -1e-10 and barycentric.max() <= 1 + 1e-10
        np.testing.assert_allclose(np.einsum("pi,pij->pj", barycentric, triangle[inverse]), points, rtol=0, atol=1e-14)
        corners_and_mids = np.concatenate((triangle, (triangle + np.roll(triangle, -1, axis=1)) / 2), axis=1)
        distance, cpu_nodes = tree.query(corners_and_mids.reshape(-1, 3))
        assert distance.max() < 1e-14
        cpu_nodes = cpu_nodes.reshape(-1, 6)[inverse]
        weights = quadratic_triangle_weights(barycentric)
        integrated = np.zeros_like(model.vertices)
        np.add.at(integrated, cpu_nodes.reshape(-1), (weights[..., None] * forces[:, None]).reshape(-1, 3))
        mapped_force, mapped_rhs = np.zeros_like(integrated), np.zeros_like(integrated)
        mapped_force[permutation], mapped_rhs[permutation] = native_force, native_rhs
        force_error = float(np.linalg.norm(integrated - mapped_force) / np.linalg.norm(mapped_force))
        assert force_error <= 1e-10
        stress, displacement, correction, residual = model.solve(mapped_rhs)
        equation_error = float(np.linalg.norm(residual) / np.linalg.norm(mapped_rhs))
        gauge_correction = float(np.linalg.norm(correction) / np.linalg.norm(mapped_rhs))
        stress_error = float(np.linalg.norm(stress - native_stress) / np.linalg.norm(stress))
        maximum_component_error = float(np.max(np.abs(stress - native_stress)) / np.max(np.abs(stress)))
        passed = max(equation_error, gauge_correction, stress_error, maximum_component_error) <= 1e-8
        prefix = "cpu" if len(captures) == 1 else f"cpu-case{index}"
        path = args.output / f"{prefix}-level{case['level']}.npz"
        np.savez_compressed(
            path,
            vertices_m=model.vertices,
            tetrahedra=tetrahedra,
            stress_pa=stress,
            displacement_m=displacement,
            forces_n=mapped_rhs,
            independently_integrated_surface_force_n=integrated,
            load_n=case["load_n"],
        )
        rows.append(
            {
                "level": case["level"],
                "capture": file_id(folder / "result.json"),
                "load_n": case["load_n"],
                "original": case["original"],
                "quadrature": case["quadrature"],
                "replay": case["replay"],
                "independent_surface_integration_relative_error": force_error,
                "complete_equation_relative_error": equation_error,
                "rigid_projection_correction_relative": gauge_correction,
                "complete_corner_stress_relative_error": stress_error,
                "maximum_stress_component_relative_error": maximum_component_error,
                "native_rhs_minus_contact_force_relative": float(
                    np.linalg.norm(mapped_rhs - mapped_force) / np.linalg.norm(mapped_force)
                ),
                "physical_source_footprints": footprints(original_points, original_forces),
                "reconstructed_footprints": footprints(points, forces),
                "field": file_id(path),
                "passed": passed,
            }
        )
        write_json(args.output / "progress.json", {"cases": rows})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "capture": file_id(args.capture / "result.json"),
            "additional_captures": [file_id(folder / "result.json") for folder in args.additional_captures],
            "mesh": start["mesh"],
            "cases": rows,
            "passed": all(row["passed"] for row in rows),
            "whole_goal_complete": False,
            "scope": "Same-domain, same assembled linear-elastic RHS control. Independent geometric barycentric P2 integration of actual native quadrature must reproduce the native contact nodal load. Independent scikit-fem stiffness/CVXOPT solve then uses the exact saved native RHS, including its inertial/balance terms, and compares all corner stresses. This verifies integration and the elastic core; it does not independently validate those body terms, spatial continuum convergence, contact footprint accuracy or a new dynamic history. No change is made to native reconstruction or its radius.",
        },
    )
    assert all(row["passed"] for row in rows), "Independent reconstruction/core audit failed"


if __name__ == "__main__":
    main()
