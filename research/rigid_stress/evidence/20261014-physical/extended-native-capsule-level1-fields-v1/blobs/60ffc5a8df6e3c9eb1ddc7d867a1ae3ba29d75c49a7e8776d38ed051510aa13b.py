"""Compare independent final fields on the unchanged egg polyhedron across paired surface refinements."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .audit_original_history_fields import wide_volume_bounds
from .audit_precise_spatial_state import assemble_state
from .compare_spatial_franka import common_probes, point_fields, qualify, weighted_change
from .qualify_articulated_precision import tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3 and args.step > 1
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = None
    rows, comparisons, inputs = [], [], {}
    previous_probe = None
    for index, run in enumerate(args.run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        static, config = restore_static(run)
        assert config.get("mode", "polyhedron") == "polyhedron"
        assert config["cold_closure"] and not config["freeze_contact"]
        assert config["body_refinement"] == config["contact_refinement"] == 0
        assert config["steps"] >= args.step
        physical = {key: config[key] for key in ["closure_run", "dt", "control_steps"]}
        physical.update(dhat_m=static.dhat, friction=static.friction)
        fingers = [static.rest[nodes].copy() for nodes in static.finger_nodes]
        if baseline is None:
            baseline = physical
            mass, com = static.body.total_mass, static.body.com.copy()
            fixed, components = static.fixed_initializer_points_m.copy(), static.fixed_initializer_components.copy()
            finger_positions, axes, finger_masses = fingers, static.axes.copy(), static.masses.copy()
            points, weights = common_probes(static.body)
            seed_mesh = static.body.mesh
        assert baseline == physical
        np.testing.assert_allclose(static.body.total_mass, mass, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(static.body.com, com, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(static.fixed_initializer_points_m, fixed)
        np.testing.assert_array_equal(static.fixed_initializer_components, components)
        np.testing.assert_array_equal(static.axes, axes)
        np.testing.assert_array_equal(static.masses, finger_masses)
        for actual, expected in zip(fingers, finger_positions, strict=True):
            np.testing.assert_array_equal(actual, expected)
        boundary = static.body.mesh.facets[:, static.body.mesh.boundary_facets()]
        midpoints = static.body.mesh.p[:, boundary].mean(axis=1).T
        seed_boundary = seed_mesh.facets[:, seed_mesh.boundary_facets()]
        triangles = seed_mesh.p[:, seed_boundary].transpose(2, 1, 0)
        edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        aa, ab, bb = [
            np.einsum("ni,ni->n", left, right) for left, right in [(edge_a, edge_a), (edge_a, edge_b), (edge_b, edge_b)]
        ]
        normal = np.cross(edge_a, edge_b)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        boundary_error = 0.0
        boundary_samples = np.concatenate((midpoints, static.body.mesh.p[:, np.unique(boundary)].T))
        for first in range(0, len(boundary_samples), 256):
            offset = boundary_samples[first : first + 256, None] - triangles[None, :, 0]
            da, db = np.einsum("pni,ni->pn", offset, edge_a), np.einsum("pni,ni->pn", offset, edge_b)
            v = (bb * da - ab * db) / (aa * bb - ab**2)
            z = (aa * db - ab * da) / (aa * bb - ab**2)
            supported = (v >= -1e-10) & (z >= -1e-10) & (v + z <= 1 + 1e-10)
            distance = abs(np.einsum("pni,ni->pn", offset, normal))
            boundary_error = max(boundary_error, float(np.where(supported, distance, np.inf).min(axis=1).max()))
        assert boundary_error <= 1e-12
        body = static.body
        evaluator = qualify(body, points)
        paths = [run / f"results/step{step:04d}.npz" for step in [args.step - 1, args.step]]
        for path in [*paths, run / "results/configuration.json", run / "results/reference_geometry.npz"]:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as saved:
            old, old_velocity = saved["displacement"].copy(), saved["velocity"].copy()
        with np.load(paths[1]) as saved:
            u, force = saved["displacement"].copy(), saved["contact_force_N"].copy()
        bulk = WidePiolaBulk(body)
        metrics, fields = assemble_state(static, config, u, old, old_velocity, None, bulk)
        metrics["original_saved_contact_force_difference_N"] = float(np.linalg.norm(fields["contact_force_N"] - force))
        vector = np.empty(body.basis.N, dtype=np.longdouble)
        vector[static.parts[0].permutation] = u[: body.basis.N]
        deformation = bulk.corner_deformation_gradients(vector)
        corner_tensor = tensor_from_f(deformation, body.lame)
        dev = corner_tensor - np.trace(corner_tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        corner_vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        determinant_lower, peak_upper = wide_volume_bounds(deformation, bulk.mu)
        matrix, _shift, objective = finite_frame(body, np.array(vector, dtype=float))
        tensor, vm, _raw_displacement = point_fields(body, vector, points)
        _, _, displacement = point_fields(body, objective, points)
        tensor = matrix.T @ tensor @ matrix
        _, energy = quadrature_fields(bulk, np.array(vector, dtype=float))
        cell, corner = np.unravel_index(np.argmax(corner_vm), corner_vm.shape)
        row = {
            "run": str(run),
            "tetrahedra": body.mesh.nelements,
            "step": args.step,
            "time_s": config["dt"] * args.step,
            "mass_kg": float(body.total_mass),
            "maximum_boundary_vertex_and_facet_centroid_distance_to_supported_seed_facet_m": boundary_error,
            **metrics,
            "raw_peak_lower_Pa": float(corner_vm.max()),
            "raw_peak_upper_Pa": peak_upper,
            "whole_volume_J_lower_bound": determinant_lower,
            "raw_peak_reference_position_m": body.corner_positions[cell, corner].tolist(),
            "nonlinear_energy_J": float(energy),
            "point_evaluator_qualification": evaluator,
        }
        probe = {"tensor": tensor, "vm": vm, "objective_displacement": displacement}
        if previous_probe is not None:
            comparison = {"coarse_tetrahedra": rows[-1]["tetrahedra"], "fine_tetrahedra": row["tetrahedra"]}
            assert comparison["coarse_tetrahedra"] < comparison["fine_tetrahedra"]
            for key in probe:
                comparison[key] = weighted_change(previous_probe[key][None], probe[key][None], weights)
            for key in ["raw_peak_lower_Pa", "nonlinear_energy_J"]:
                comparison[key] = abs(rows[-1][key] / row[key] - 1)
            comparisons.append(comparison)
        rows.append(row)
        previous_probe = probe
        np.savez_compressed(
            args.output / f"case{index:02d}.npz",
            vertices=body.mesh.p.T,
            tetrahedra=body.mesh.t.T,
            raw_corner_tensor_Pa=corner_tensor,
            raw_corner_vm_Pa=corner_vm,
            common_points_m=points,
            common_weights_m3=weights,
            objective_probe_tensor_Pa=tensor,
            probe_vm_Pa=vm,
            objective_probe_displacement_m=displacement,
            **fields,
        )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "common_physical_probes": len(points),
        "physical_configuration": baseline,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "One common physical time on an unchanged coarse polyhedral material domain. The paired meshes preserve the actual seed boundary, geometry/mass and original physical inputs; common physical probes and independently reconstructed force gates remain explicit. Surface and volume/contact resolution vary together on this branch, so this is not a separated bulk or contact-integration error. Raw maxima are unsmoothed whole-volume bounds. CAD geometry is a separate family and is excluded. These fields do not establish complete application/history, converged physical contact, model/feedback accuracy or verified warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"meshes": len(rows), "common_probes": len(points), "last_change": comparisons[-1]}), flush=True)


if __name__ == "__main__":
    main()
