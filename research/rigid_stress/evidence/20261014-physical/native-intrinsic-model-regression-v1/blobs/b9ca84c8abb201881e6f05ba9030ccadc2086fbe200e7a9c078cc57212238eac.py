"""Audit fixed-boundary egg thickness refinement, unchanged tractions and unsmoothed FEM fields."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .summarize_fine_references import CASES, closed_source, relative_change


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_layer_geometry(seed, axis_runs):
    meshes = [
        (2, seed),
        *((json.loads((run / "mesh.json").read_text())["shell_layers"], run / "mesh.npz") for run in axis_runs),
    ]
    assert meshes[0][0] == 2
    assert [row[0] for row in meshes] == sorted({row[0] for row in meshes})
    inputs = {str(path): sha256(path) for _, mesh in meshes for path in [mesh, mesh.with_suffix(".json")]}
    with np.load(seed) as saved:
        original_vertices = saved["vertices"].copy()
        original_boundary = saved["surface_triangles"].copy()
    count = len(original_vertices) // 3
    outer = original_boundary[np.all(original_boundary < count, axis=1)]
    outer = np.sort(outer, axis=1)
    outer = outer[np.lexsort(outer.T[::-1])]
    assert 2 * len(outer) == len(original_boundary)
    rows = []
    for layers, mesh in meshes:
        with np.load(mesh) as saved:
            vertices, tetrahedra, declared_boundary = [
                saved[key].copy() for key in ["vertices", "tetrahedra", "surface_triangles"]
            ]
        assert len(vertices) == (layers + 1) * count
        np.testing.assert_array_equal(vertices[:count], original_vertices[:count])
        np.testing.assert_array_equal(vertices[-count:], original_vertices[-count:])
        for layer in range(layers + 1):
            fraction = layer / layers
            expected = (1 - fraction) * original_vertices[:count] + fraction * original_vertices[-count:]
            np.testing.assert_allclose(vertices[layer * count : (layer + 1) * count], expected, rtol=0, atol=1e-15)
        local = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces, first, multiplicity = np.unique(
            np.sort(tetrahedra[:, local].reshape(-1, 3)), axis=0, return_index=True, return_counts=True
        )
        assert np.all((multiplicity == 1) | (multiplicity == 2))
        boundary, owners = faces[multiplicity == 1], first[multiplicity == 1] // 4
        declared_boundary = np.sort(declared_boundary, axis=1)
        declared_boundary = declared_boundary[np.lexsort(declared_boundary.T[::-1])]
        np.testing.assert_array_equal(boundary, declared_boundary)
        expected = np.concatenate((outer, outer + layers * count))
        expected = expected[np.lexsort(expected.T[::-1])]
        np.testing.assert_array_equal(boundary, expected)
        triangles = vertices[boundary]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        toward_exterior = triangles.mean(axis=1) - vertices[tetrahedra[owners]].mean(axis=1)
        signs = np.sign(np.einsum("fi,fi->f", normals, toward_exterior))
        assert np.all(signs != 0)
        signed_volume = np.einsum("fi,fi->f", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) * signs / 6
        surface_volume = math.fsum(signed_volume)
        surface_com = (
            np.array([math.fsum(signed_volume * triangles[:, :, axis].sum(axis=1) / 4) for axis in range(3)])
            / surface_volume
        )
        determinants = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
        assert np.all(determinants > 0)
        tetrahedron_volume = math.fsum(determinants / 6)
        np.testing.assert_allclose(tetrahedron_volume, surface_volume, rtol=1e-12, atol=1e-20)
        row = {
            "layers": layers,
            "tetrahedra": len(tetrahedra),
            "surface_triangles": len(boundary),
            "mesh_sha256": sha256(mesh),
            "oriented_surface_flux_volume_m3": surface_volume,
            "oriented_surface_flux_com_m": surface_com.tolist(),
            "tetrahedron_volume_m3": tetrahedron_volume,
            "mass_kg_at_unchanged_density2000": tetrahedron_volume * 2000,
            "minimum_positive_tetrahedron_volume_m3": float(determinants.min() / 6),
            "boundary_vertices_and_triangles_identical": True,
        }
        if rows:
            np.testing.assert_allclose(surface_volume, rows[0]["oriented_surface_flux_volume_m3"], rtol=1e-12)
            np.testing.assert_allclose(surface_com, rows[0]["oriented_surface_flux_com_m"], rtol=0, atol=1e-15)
        rows.append(row)
    assert inputs == {path: sha256(Path(path)) for path in inputs}
    return {
        "scope": "Independent mesh-only boundary connectivity and oriented surface-flux volume/COM audit",
        "rows": rows,
        "unchanged_input_sha256": inputs,
        "data_producers_naturally_closed": [
            {
                "run": str(run),
                "source_after_present": (run / "source-after.sha256").exists(),
                "returncode": (run / "returncode.txt").read_text().strip()
                if (run / "returncode.txt").exists()
                else None,
            }
            for run in axis_runs
        ],
        "stress_and_FEM_solve_completion_not_inferred": True,
        "goal_completion": False,
    }


def compare_tractions(first, second):
    with np.load(first) as original, np.load(second) as candidate:
        assert original.files == candidate.files
        positions, current = original["positions_m"], candidate["positions_m"]
        assert len(positions) == len(current)
        distances, indices = cKDTree(positions).query(current)
        assert len(np.unique(indices)) == len(indices)
        assert distances.max() <= 1e-13
        for key, tolerance in [("normals", 1e-11), ("area_weights_m2", 1e-22), ("tractions_Pa", 1e-8)]:
            np.testing.assert_allclose(candidate[key], original[key][indices], rtol=1e-11, atol=tolerance)
        return {
            "samples": len(current),
            "maximum_position_difference_m": float(distances.max()),
            "source_sha256": sha256(first),
            "candidate_sha256": sha256(second),
        }


def verify_fields(run, mesh, case):
    snapshot_path = run / f"results/{case}/snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["mesh_sha256"] == sha256(mesh)
    assert snapshot["geometry"] == "egg" and snapshot["reference_complete_budget_fraction"] <= 1
    field_path = run / f"results/{case}/reference_field.npz"
    with np.load(mesh) as saved:
        vertices, tetrahedra = saved["vertices"], saved["tetrahedra"]
    with np.load(field_path) as saved:
        tensor, saved_vm, displacement, rhs = [saved[key] for key in ["tensor_Pa", "vm_Pa", "displacement", "rhs_N"]]
        assert tensor.shape == (len(tetrahedra), 4, 3, 3)
        assert all(np.isfinite(value).all() for value in [tensor, saved_vm, displacement, rhs])
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        np.testing.assert_allclose(vm, saved_vm, rtol=1e-11, atol=1e-8)
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        peak, position = float(vm.max()), vertices[tetrahedra[cell, corner]]
        energy = float(0.5 * displacement @ rhs)
    np.testing.assert_allclose(peak, snapshot["reference_peak_Pa"], rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(position, snapshot["reference_peak_position_m"])
    np.testing.assert_allclose(energy, snapshot["reference_energy_J"], rtol=1e-10, atol=1e-24)
    return snapshot, {
        "field_sha256": sha256(field_path),
        "snapshot_sha256": sha256(snapshot_path),
        "raw_reconstructed_global_peak_Pa": peak,
        "raw_peak_position_m": position.tolist(),
        "reconstructed_energy_J": energy,
        "recorded_complete_FEM_budget_fraction": snapshot["reference_complete_budget_fraction"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-probes", type=Path, required=True)
    parser.add_argument("--axis-run", type=Path, nargs="+", required=True)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.axis_run) >= 2
    for run in [args.base_run, args.base_probes]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    seed_metadata = json.loads(args.seed.with_suffix(".json").read_text())
    assert seed_metadata["shell_layers"] == 2 and seed_metadata["geometry"]["name"] == "egg"
    assert sha256(args.seed) == seed_metadata["mesh_sha256"]
    geometry_audit = verify_layer_geometry(args.seed, args.axis_run)
    if args.geometry_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "summary.json").write_text(json.dumps(geometry_audit, indent=2) + "\n")
        print(json.dumps(geometry_audit), flush=True)
        return
    levels = [(2, args.base_run, args.base_probes, args.seed)]
    geometries = []
    for axis in args.axis_run:
        for run in [axis, axis / "fem", axis / "probes"]:
            closed_source(run)
            assert int((run / "returncode.txt").read_text()) == 0
        geometry = json.loads((axis / "mesh.json").read_text())
        assert geometry["seed_sha256"] == sha256(args.seed)
        assert geometry["outer_and_inner_boundary_vertices_byte_identical"]
        assert geometry["outer_and_inner_boundary_triangles_identical"]
        assert geometry["seed_prism_connectivity_independently_reconstructed"]
        assert geometry["mesh_sha256"] == sha256(axis / "mesh.npz")
        geometries.append(geometry)
        levels.append((geometry["shell_layers"], axis / "fem", axis / "probes", axis / "mesh.npz"))
    assert levels[0][0] == 2
    assert [row[0] for row in levels] == sorted({row[0] for row in levels})
    reports = [json.loads((probes / "results/summary.json").read_text()) for _, _, probes, _ in levels]
    assert len({report["probe_coordinate_sha256"] for report in reports}) == 1
    assert all(not report["outside_probe_indices"] for report in reports)
    rows, changes = [], []
    for case in CASES:
        previous_field, previous_row, baseline = None, None, None
        for layer, run, probes, mesh in levels:
            snapshot, verification = verify_fields(run, mesh, case)
            if baseline is None:
                baseline = snapshot
            for key in ["material", "pose", "omega_rad_s", "patch_radius_m", "reference_quadrature_level"]:
                assert snapshot[key] == baseline[key]
            np.testing.assert_allclose(snapshot["mass_kg"], baseline["mass_kg"], rtol=1e-12, atol=1e-18)
            np.testing.assert_allclose(snapshot["com_m"], baseline["com_m"], rtol=0, atol=1e-14)
            with (
                np.load(run / f"results/{case}/frozen_inputs.npz") as current,
                np.load(args.base_run / f"results/{case}/frozen_inputs.npz") as original,
            ):
                assert current.files == original.files
                for key in current.files:
                    if key in {"radii_m", "friction", "omega_rad_s"}:
                        np.testing.assert_array_equal(current[key], original[key])
                    else:
                        tolerance = 1e-11 if key == "collision_normals" else 1e-12 if key == "forces_N" else 1e-14
                        np.testing.assert_allclose(current[key], original[key], rtol=1e-12, atol=tolerance)
            assert len(snapshot["patches"]) == len(baseline["patches"])
            tractions = [
                compare_tractions(
                    args.base_run / f"results/{case}/reference_patch{index}.npz",
                    run / f"results/{case}/reference_patch{index}.npz",
                )
                for index in range(len(snapshot["patches"]))
            ]
            field_path = probes / f"results/{case}/reference_field.npz"
            with np.load(field_path) as saved:
                field = {
                    key: saved[key].copy() for key in ["points_m", "valid", "tensor_Pa", "vm_Pa", "displacement_m"]
                }
            assert field["valid"].all()
            row = {
                "case": case,
                "layers": layer,
                "mesh_sha256": sha256(mesh),
                "probe_field_sha256": sha256(field_path),
                "traction_identity_checks": tractions,
                "mass_kg": snapshot["mass_kg"],
                **verification,
            }
            rows.append(row)
            if previous_field is not None:
                np.testing.assert_array_equal(previous_field["points_m"], field["points_m"])
                change = {"case": case, "from_layers": previous_row["layers"], "to_layers": layer}
                for key in ["tensor_Pa", "vm_Pa", "displacement_m"]:
                    relative, absolute = relative_change(previous_field[key], field[key])
                    change[key] = {"relative_change": relative, "absolute_change": absolute}
                for key in ["raw_reconstructed_global_peak_Pa", "reconstructed_energy_J"]:
                    change[key] = abs(previous_row[key] / row[key] - 1)
                values = [change[key]["relative_change"] for key in ["tensor_Pa", "vm_Pa", "displacement_m"]]
                values += [change[key] for key in ["raw_reconstructed_global_peak_Pa", "reconstructed_energy_J"]]
                change["all_five_changes_below_initial3percent"] = all(v is not None and v <= 0.03 for v in values)
                changes.append(change)
            previous_field, previous_row = field, row
    result = {
        "scope": __doc__,
        "seed_sha256": sha256(args.seed),
        "geometry_records": geometries,
        "independent_boundary_and_surface_flux_audit": geometry_audit,
        "rows": rows,
        "adjacent_changes": changes,
        "fields_verified": len(rows),
        "initial_adjacent_change_diagnostic": 0.03,
        "physical_accuracy_accepted": False,
        "goal_completion": False,
        "limitations": "Only thickness-direction volume discretization varies on the original fixed polyhedral "
        "shell; all sampled boundary tractions and frozen physical inputs are checked. Raw global "
        "corner peaks and work identities are independently reconstructed. Equilibrium budgets "
        "are recorded from the independently assembled FEM driver, not reassembled here. A "
        "thickness tail does not establish CAD/surface/contact convergence, native accuracy, "
        "deformable-contact feedback or the full Franka application. All original gates remain.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fields_verified": len(rows), "adjacent_changes": changes}), flush=True)


if __name__ == "__main__":
    main()
