"""Separate contact-integration changes on byte-identical egg bulk geometry and mass."""

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
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) == len(args.audit) >= 3 and args.step > 1
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = None
    rows, comparisons, inputs = [], [], {}
    previous_probe = None
    previous_level = -1
    for index, (run, audit_run) in enumerate(zip(args.run, args.audit, strict=True)):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        closed_source(audit_run)
        assert int((audit_run / "returncode.txt").read_text()) == 0
        static, config = restore_static(run)
        audit_path = audit_run / "results/summary.json"
        audit = json.loads(audit_path.read_text())
        assert Path(audit["run"]).resolve() == run.resolve()
        assert audit["states_checked"] == len(audit["rows"]) == config["steps"]
        assert all(
            row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
            for row in audit["rows"]
        )
        inputs[str(audit_path)] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        assert config["mode"] in ("cad", "polyhedron")
        assert config["cold_closure"] and not config["freeze_contact"]
        assert config["body_refinement"] == 0
        contact_level = config["contact_refinement"]
        assert contact_level > previous_level
        previous_level = contact_level
        assert config["steps"] >= args.step
        physical = {key: config[key] for key in ["closure_run", "dt", "control_steps", "geometry_run", "mode", "level"]}
        physical.update(dhat_m=static.dhat, friction=static.friction)
        fingers = [static.rest[nodes].copy() for nodes in static.finger_nodes]
        if baseline is None:
            baseline = physical
            mass, com = static.body.total_mass, static.body.com.copy()
            fixed, components = static.fixed_initializer_points_m.copy(), static.fixed_initializer_components.copy()
            finger_positions, axes, finger_masses = fingers, static.axes.copy(), static.masses.copy()
            points, weights = common_probes(static.body)
            reference_sha = hashlib.sha256((run / "results/reference_geometry.npz").read_bytes()).hexdigest()
        assert baseline == physical
        np.testing.assert_allclose(static.body.total_mass, mass, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(static.body.com, com, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(static.fixed_initializer_points_m, fixed)
        np.testing.assert_array_equal(static.fixed_initializer_components, components)
        np.testing.assert_array_equal(static.axes, axes)
        np.testing.assert_array_equal(static.masses, finger_masses)
        for actual, expected in zip(fingers, finger_positions, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert hashlib.sha256((run / "results/reference_geometry.npz").read_bytes()).hexdigest() == reference_sha
        mapping = static.contact_node_map
        np.testing.assert_allclose(np.asarray(mapping.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(mapping @ static.rest, static.contact_surface_rest, rtol=0, atol=1e-14)
        rest_triangles = static.contact_surface_rest[static.contact_surface_faces]
        face_areas = (
            np.linalg.norm(
                np.cross(rest_triangles[:, 1] - rest_triangles[:, 0], rest_triangles[:, 2] - rest_triangles[:, 0]),
                axis=1,
            )
            / 2
        )
        face_groups = static.contact_surface_groups[static.contact_surface_faces]
        assert np.all(face_groups == face_groups[:, :1])
        rest_areas = [float(face_areas[face_groups[:, 0] == group].sum()) for group in [0, 1, 2]]
        if index == 0:
            baseline_areas = rest_areas
        np.testing.assert_allclose(rest_areas, baseline_areas, rtol=1e-12, atol=1e-18)
        body = static.body
        evaluator = qualify(body, points)
        paths = [run / f"results/step{step:04d}.npz" for step in [args.step - 1, args.step]]
        for path in [
            *paths,
            run / "results/configuration.json",
            run / "results/reference_geometry.npz",
            run / "results/contact_geometry.npz",
            run / "results/contact_node_map.npz",
        ]:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as saved:
            old, old_velocity = saved["displacement"].copy(), saved["velocity"].copy()
        with np.load(paths[1]) as saved:
            u, force = saved["displacement"].copy(), saved["contact_force_N"].copy()
        bulk = WidePiolaBulk(body)
        metrics, fields = assemble_state(static, config, u, old, old_velocity, force, bulk)
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
        positions = fields["surface_positions_m"]
        normal, shear = fields["normal_surface_force_N"], fields["shear_surface_force_N"]
        source_wrenches = []
        for group in [1, 2]:
            selected = static.contact_surface_groups == group
            normal_force, shear_force = -normal[selected], -shear[selected]
            source_wrenches.append(
                {
                    "finger_source_group": group,
                    "opposite_finger_normal_resultant_N": normal_force.sum(axis=0).tolist(),
                    "opposite_finger_shear_resultant_N": shear_force.sum(axis=0).tolist(),
                    "opposite_finger_total_moment_about_fixed_body_com_Nm": np.asarray(
                        np.cross(positions[selected] - body.com, normal_force + shear_force).sum(axis=0), dtype=float
                    ).tolist(),
                }
            )
        total_surface = normal + shear
        force_error = float(np.linalg.norm(total_surface.sum(axis=0)))
        moment_error = float(np.linalg.norm(np.cross(positions - body.com, total_surface).sum(axis=0)))
        normal_moment_error = float(np.linalg.norm(np.cross(positions - body.com, normal).sum(axis=0)))
        assert force_error <= 1e-9 and normal_moment_error <= 1e-10
        body_samples = static.contact_surface_groups == 0
        sample_area = fields["surface_vertex_lumped_area_m2"][body_samples]
        normal_norm = np.linalg.norm(normal[body_samples], axis=1)
        shear_norm = np.linalg.norm(shear[body_samples], axis=1)
        row = {
            "run": str(run),
            "tetrahedra": body.mesh.nelements,
            "step": args.step,
            "time_s": config["dt"] * args.step,
            "mass_kg": float(body.total_mass),
            "contact_refinement": contact_level,
            "contact_surface_vertices": len(static.contact_surface_rest),
            "contact_surface_triangles": len(static.contact_surface_faces),
            "undeformed_group_surface_areas_m2": rest_areas,
            "minimum_P2_trace_interpolation_weight": float(mapping.data.min()),
            "per_finger_reaction_wrenches": source_wrenches,
            "total_contact_surface_internal_force_error_N": force_error,
            "total_contact_surface_internal_moment_error_Nm": moment_error,
            "normal_surface_internal_moment_error_Nm": normal_moment_error,
            "unsigned_body_normal_sample_force_N": float(normal_norm.sum()),
            "unsigned_body_shear_sample_force_N": float(shear_norm.sum()),
            "lumped_body_normal_force_density_L2_N_per_m": float(np.sqrt(np.sum(normal_norm**2 / sample_area))),
            "lumped_body_shear_force_density_L2_N_per_m": float(np.sqrt(np.sum(shear_norm**2 / sample_area))),
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
            comparison = {
                "coarse_contact_refinement": rows[-1]["contact_refinement"],
                "fine_contact_refinement": contact_level,
            }
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
        "limitations": "Only the body collision trace sampling changes, with exact P2 interpolation onto unchanged bulk unknowns. Reference geometry/initializer anchors are byte-identical, and original physical material/mass, finger meshes/axes/masses, controls, dt, friction and barrier range remain fixed. Each case independently resolves its original held initializer. Undeformed actual surface areas are checked by group; deformed surface and active lumped support can change. Opposite finger reactions identify source wrenches, not continuum contact islands. Force-density norms use discrete nodal forces divided by positive lumped sample areas, not a continuous pressure reconstruction. Normal-contact rotation invariance is checked separately; total frictional surface moment is measured without imposing an additional zero-moment acceptance gate on the existing finite-gap, lagged-tangent friction law. One final common-time comparison isolates this contact-integration axis conditionally on the fixed bulk/geometry and regularization; it does not establish bulk/geometry convergence, full histories/application, model/feedback accuracy or verified warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"contact_levels": len(rows), "common_probes": len(points), "last_change": comparisons[-1]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
