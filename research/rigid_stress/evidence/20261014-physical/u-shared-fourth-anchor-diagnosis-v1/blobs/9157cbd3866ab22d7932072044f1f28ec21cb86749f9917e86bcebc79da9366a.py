"""Compare independently audited fixed-pad U fields with shared physical initialization anchors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .qualify_articulated_precision import tensor_from_f
from .sample_load_fields import common_physical_probes, locate_probes
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--axis", choices=["coupled", "pad_at_fixed_body", "body_at_fixed_pad"], default="coupled")
    parser.add_argument("--history-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3 and len(args.history_audit) == len(args.run) and args.axis == "body_at_fixed_pad"
    args.output.mkdir(parents=True, exist_ok=False)
    probes = common_physical_probes("u")
    rows, observations, hashes, baseline = [], [], {}, None
    shared_configuration = None
    history_rows = []
    for index, run in enumerate(args.run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        directory = run / "solution"
        config = json.loads((directory / "configuration.json").read_text())
        report = json.loads((directory / "result.json").read_text())
        audit = args.history_audit[index]
        closed_source(audit)
        assert int((audit / "returncode.txt").read_text()) == 0
        audit_summary = audit / "results/summary.json"
        history = json.loads(audit_summary.read_text())
        assert Path(history["run"]).resolve() == run.resolve()
        assert history["stages_checked"] == 744
        assert history["all_retained_stages_independent_current_friction_and_original_force_gate"]
        assert history["all_retained_stages_full_corner_quadrature_and_raw_bounds"]
        assert history["requested_initialization_coordinates_independently_matched"]
        assert history["qualified_per_stencil_force_method_and_bulk_sources_verified"]
        for path, digest in history["input_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        anchor = Path(config["initialization_anchors"])
        anchor_sha = hashlib.sha256(anchor.read_bytes()).hexdigest()
        assert anchor_sha == history["common_initialization_anchor_sha256"]
        current_configuration = {key:value for key,value in config.items() if key not in ["mesh","output"]}
        if shared_configuration is None:
            shared_configuration = current_configuration
        assert current_configuration == shared_configuration
        hashes[str(audit_summary)] = hashlib.sha256(audit_summary.read_bytes()).hexdigest()
        hashes[str(anchor)] = anchor_sha
        history_rows.append({"audit":str(audit),"stages_checked":744,"shared_anchor_sha256":anchor_sha})
        physical = {
            key: config[key]
            for key in [
                "geometry",
                "pad_young",
                "force",
                "dhat",
                "friction",
                "initial_gap",
                "pad_thickness",
                "edge_lift",
                "force_steps",
                "gravity",
                "body_material",
                "pad_material",
                "body_rigid_coordinates",
                "updated_friction",
            ]
        }
        if baseline is None:
            baseline = physical
        assert physical == baseline and physical["geometry"] == "u"
        assert physical["body_material"] == physical["pad_material"] == "neo_hookean"
        material = report["material"]
        assert (material["body_young_Pa"], material["body_poisson"], material["body_density_kg_m3"]) == (
            1e10,
            0.3,
            2000,
        )
        assert (material["pad_young_Pa"], material["pad_poisson"], material["pad_density_kg_m3"]) == (
            config["pad_young"],
            0,
            1000,
        )
        assert report["complete_constrained_residual_N"] <= 1e-9
        assert all(row["complete_free_residual_n"] <= 1e-9 for row in report["accepted_steps"])
        mesh = Path(config["mesh"])
        assert hashlib.sha256(mesh.read_bytes()).hexdigest() == report["mesh_sha256"]
        with np.load(directory / "fields.npz") as saved:
            fields = {key: saved[key].copy() for key in saved.files}
        references = []
        for name, path, young, poisson, density in [
            ("body", mesh, 1e10, 0.3, 2000),
            ("left_shoe", directory / "left_shoe_mesh.npz", config["pad_young"], 0, 1000),
            ("right_shoe", directory / "right_shoe_mesh.npz", config["pad_young"], 0, 1000),
        ]:
            with np.load(path) as saved:
                reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density)
            references.append(reference)
        gradient = -fields["applied_force_N"] - fields["contact_force_N"]
        first, parts, body_data = 0, [], None
        for name, reference in zip(["body", "left_shoe", "right_shoe"], references):
            n = reference.basis.N
            section = slice(first, first + n)
            permutation = reference.component_dofs.reshape(-1)
            np.testing.assert_array_equal(
                fields["rest_positions"][first // 3 : (first + n) // 3], reference.coordinates
            )
            vector = np.empty(n)
            vector[permutation] = fields["displacement"][section]
            bulk = NeoHookeanBulk(reference)
            energy, force, _ = bulk.evaluate(vector, False)
            gradient[section] += force[permutation]
            rotation, translation, local = finite_frame(reference, vector)
            objective_energy, _, _ = bulk.evaluate(local, False)
            np.testing.assert_allclose(objective_energy, energy, rtol=1e-8, atol=1e-15)
            acceleration = np.empty(n)
            acceleration[reference.component_dofs] = config["gravity"]
            gravity = (reference.mass @ acceleration)[permutation]
            if name == "body":
                np.testing.assert_array_equal(permutation, fields["body_component_permutation"])
                np.testing.assert_allclose(
                    fields["applied_force_N"][section],
                    gravity,
                    rtol=1e-12,
                    atol=1e-15,
                )
                body_data = (reference, bulk, vector, rotation, translation, local, energy)
            else:
                actuation = (fields["applied_force_N"][section] - gravity).reshape(-1, 3)
                drive = -1 if name == "left_shoe" else 1
                np.testing.assert_allclose(
                    actuation.sum(axis=0), [drive * config["force"], 0, 0], rtol=1e-12, atol=1e-15
                )
                np.testing.assert_allclose(actuation[:, 1:], 0, rtol=0, atol=1e-15)
                active = abs(actuation[:, 0]) > 1e-15
                assert active.any()
                np.testing.assert_allclose(
                    actuation[active, 0], drive * config["force"] / active.sum(), rtol=1e-12, atol=1e-15
                )
            parts.append(
                {
                    "part": name,
                    "geometry_array_sha256": hashlib.sha256(
                        reference.mesh.p.tobytes() + reference.mesh.t.tobytes()
                    ).hexdigest(),
                    "tetrahedra": reference.mesh.nelements,
                    "discrete_mass_kg": float(reference.total_mass),
                    "nonlinear_energy_J": float(energy),
                    "whole_volume_J_lower_bound": bulk.determinant_lower_bound(vector),
                    "objective_maximum_nodal_displacement_m": float(
                        np.linalg.norm(local[reference.component_dofs], axis=1).max()
                    ),
                }
            )
            first += n
        assert first == len(gradient)
        mapping = sparse.load_npz(directory / "original_coordinate_map.npz")
        complete = float(np.linalg.norm(mapping.T @ gradient))
        n = references[0].basis.N
        body_error = float(np.linalg.norm(gradient[:n]))
        assert complete <= 1e-9 and body_error <= 1e-9
        difference = float(np.max(abs(gradient - fields["total_gradient_N"])))
        body, bulk, vector, rotation, translation, local, energy = body_data
        corners, vm, upper = bulk.stress_bounds(vector)
        np.testing.assert_allclose(corners, fields["body_tensor_Pa"], rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(vm, fields["body_vm_Pa"], rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(
            vm.max(), report["nonlinear_body_peak_bounds"]["sampled_corner_lower_Pa"], rtol=1e-12
        )
        np.testing.assert_allclose(
            upper.max(), report["nonlinear_body_peak_bounds"]["certified_global_upper_Pa"], rtol=1e-12
        )
        elements, valid, barycentric = locate_probes(body.mesh, probes)
        point_f = np.einsum("nc,ncij->nij", barycentric, bulk.corner_deformation_gradients(vector)[elements[valid]])
        tensor = np.full((len(probes), 3, 3), np.nan)
        tensor[valid] = rotation.T @ tensor_from_f(point_f, body.lame) @ rotation
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        probe_vm = np.sqrt(1.5 * np.sum(dev**2, axis=(-2, -1)))
        displacement = np.full((len(probes), 3), np.nan)
        displacement[valid] = (body.basis.probes(probes[valid].T) @ local).reshape(3, -1).T
        observation = {"valid": valid, "tensor_Pa": tensor, "vm_Pa": probe_vm, "objective_displacement_m": displacement}
        observations.append(observation)
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        contact = fields["contact_force_N"][:n].reshape(-1, 3)
        current = body.coordinates + fields["displacement"][:n].reshape(-1, 3)
        row = {
            "run": str(run),
            "body_tetrahedra": body.mesh.nelements,
            "body_mesh_sha256": report["mesh_sha256"],
            "pad_resolution": config["pad_resolution"],
            "independent_body_residual_N": body_error,
            "independent_original_mapped_residual_N": complete,
            "saved_gradient_reconstruction_max_difference_N": difference,
            "original_accepted_step_records": len(report["accepted_steps"]),
            "maximum_recorded_original_residual_N": max(
                r["complete_free_residual_n"] for r in report["accepted_steps"]
            ),
            "raw_global_peak_lower_Pa": float(vm.max()),
            "raw_global_peak_upper_Pa": float(upper.max()),
            "raw_peak_reference_position_m": body.corner_positions[cell, corner].tolist(),
            "body_nonlinear_energy_J": float(energy),
            "parts": parts,
            "contact_force_world_N": contact.sum(axis=0).tolist(),
            "contact_moment_about_reference_com_world_Nm": np.cross(current - body.com, contact).sum(axis=0).tolist(),
            "rotation_world_from_material": rotation.tolist(),
            "translation_world_m": translation.tolist(),
            "outside_probe_indices": np.flatnonzero(~valid).tolist(),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"level{index:02d}.npz",
            points_m=probes,
            **observation,
            reconstructed_total_gradient_N=gradient,
            objective_corner_tensor_Pa=rotation.T @ corners @ rotation,
        )
        for path in [
            mesh,
            *(
                directory / name
                for name in [
                    "fields.npz",
                    "configuration.json",
                    "result.json",
                    "coordinate_map.npz",
                    "original_coordinate_map.npz",
                    "left_shoe_mesh.npz",
                    "right_shoe_mesh.npz",
                ]
            ),
        ]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    common = np.logical_and.reduce([row["valid"] for row in observations])
    assert common.all() and len(common) == 75
    changes = []
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        if args.axis == "pad_at_fixed_body":
            assert previous["body_mesh_sha256"] == current["body_mesh_sha256"]
            assert previous["pad_resolution"] < current["pad_resolution"]
        elif args.axis == "body_at_fixed_pad":
            assert previous["body_tetrahedra"] < current["body_tetrahedra"]
            assert previous["pad_resolution"] == current["pad_resolution"]
            assert all(
                a["geometry_array_sha256"] == b["geometry_array_sha256"]
                for a, b in zip(previous["parts"][1:], current["parts"][1:])
            )
        else:
            assert previous["body_tetrahedra"] < current["body_tetrahedra"]
            assert previous["pad_resolution"] < current["pad_resolution"]
        change = {
            "previous_body_tetrahedra": rows[index - 1]["body_tetrahedra"],
            "body_tetrahedra": rows[index]["body_tetrahedra"],
            "previous_pad_resolution": rows[index - 1]["pad_resolution"],
            "pad_resolution": rows[index]["pad_resolution"],
            "common_material_points": int(common.sum()),
        }
        for key in ["tensor_Pa", "vm_Pa", "objective_displacement_m"]:
            relative, absolute = relative_change(observations[index - 1][key][common], observations[index][key][common])
            change[key] = {"relative_change": relative, "absolute_change": absolute}
        for key in ["raw_global_peak_lower_Pa", "body_nonlinear_energy_J"]:
            change[key] = abs(rows[index - 1][key] / rows[index][key] - 1)
        changes.append(change)
    result = {
        "scope": __doc__,
        "declared_spatial_axis": args.axis,
        "physical_configuration": baseline,
        "shared_initialization_configuration":shared_configuration,
        "independent_all_stage_audits":history_rows,
        "rows": rows,
        "adjacent_changes": changes,
        "common_material_points": int(common.sum()),
        "probe_sha256": hashlib.sha256(probes.tobytes()).hexdigest(),
        "input_sha256": hashes,
        "goal_completion": False,
        "limitations": "All declared histories have independent current-stencil friction/normal forces, original1e-9N constrained balances and complete corner/quadrature fields at every one of744 original loading stages. The same six physical initialization coordinates, pad5 geometry, source definitions,12/604/64/64 stages and physical parameters remain fixed. All75 original material probes remain, with no subset removal, and full global unaveraged lower/upper peaks and locations stay visible. Proper finite frames and direct nonlinear deformation gradients use the original comparison method. Body geometry/mass and volume/contact-trace discretization remain coupled; this removes the earlier initialization-coordinate/count confound without establishing a pure bulk axis, contact pressure, regularization/history tail, continuum feedback/model accuracy or verified warmed native cost. The original differently initialized cases are not replaced. Three useful levels alone do not prove convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"levels": len(rows), "common_material_points": int(common.sum()), "adjacent_changes": changes}))


if __name__ == "__main__":
    main()
