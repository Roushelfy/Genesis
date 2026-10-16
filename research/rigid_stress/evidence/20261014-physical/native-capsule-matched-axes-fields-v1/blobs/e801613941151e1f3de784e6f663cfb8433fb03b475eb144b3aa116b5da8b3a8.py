"""Qualify recording-only reference changes and independently reconstruct every retained contact stage."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .contact_reference import IndependentContactSystem
from .neohookean_reference import NeoHookeanBulk
from .qualify_articulated_precision import tensor_from_f
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def contact_forces(system, displacement, previous, centre):
    normal, positions = system.normal_collisions(displacement)
    tangential = ipctk.TangentialCollisions()
    tangential.build(system.mesh, positions, normal, system.barrier, system.friction)
    delta = system.mesh.vertices((displacement - previous).reshape(-1, 3))
    ids = system.mesh.to_full_vertex_id()
    groups = system.groups[ids]
    sources = {source: {kind: np.zeros_like(positions) for kind in ["normal", "shear"]} for source in [1, 2]}
    for kind, collisions in [("normal", normal), ("shear", tangential)]:
        for index in range(len(collisions)):
            collision = collisions[index]
            nodes = np.asarray(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
            nodes = nodes[nodes >= 0]
            involved = set(groups[nodes])
            assert involved in ({0, 1}, {0, 2})
            if kind == "normal":
                force = -system.barrier.gradient(collision, positions[nodes].reshape(-1)).reshape(-1, 3)
            else:
                force = -system.friction_potential.gradient(
                    collision, collision.dof(delta, system.mesh.edges, system.mesh.faces)
                ).reshape(-1, 3)
            np.add.at(sources[max(involved)][kind], nodes, force)
            del collision
    normal_force = sum(source["normal"] for source in sources.values())
    shear_force = sum(source["shear"] for source in sources.values())
    whole_normal = -system.barrier.gradient(normal, system.mesh, positions).reshape(-1, 3)
    whole_shear = -system.friction_potential.gradient(tangential, system.mesh, delta).reshape(-1, 3)
    np.testing.assert_allclose(normal_force, whole_normal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shear_force, whole_shear, rtol=1e-12, atol=1e-12)
    rows = []
    for source, force in sources.items():
        body = groups == 0
        total = force["normal"] + force["shear"]
        rows.append(
            {
                "source": source,
                "normal_force_on_body_N": force["normal"][body].sum(axis=0).tolist(),
                "shear_force_on_body_N": force["shear"][body].sum(axis=0).tolist(),
                "total_moment_on_body_about_current_com_Nm": np.cross(positions[body] - centre, total[body])
                .sum(axis=0)
                .tolist(),
            }
        )
    assert not ipctk.has_intersections(system.mesh, positions)
    return (
        system.mesh.to_full_dof(normal_force.reshape(-1)),
        system.mesh.to_full_dof(shear_force.reshape(-1)),
        len(normal),
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.baseline]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    directories = [run / "solution" for run in [args.run, args.baseline]]
    configs = [json.loads((directory / "configuration.json").read_text()) for directory in directories]
    assert {k: v for k, v in configs[0].items() if k != "output"} == {
        k: v for k, v in configs[1].items() if k != "output"
    }
    config = configs[0]
    assert config["body_material"] == config["pad_material"] == "neo_hookean"
    assert config["updated_friction"] and not config["resume_failed_step"]
    reports = [json.loads((directory / "result.json").read_text()) for directory in directories]
    accepted = [report["accepted_steps"] for report in reports]
    assert [(row["phase"], row["step"]) for row in accepted[0]] == [(row["phase"], row["step"]) for row in accepted[1]]
    assert all(row["complete_free_residual_n"] <= 1e-9 for records in accepted for row in records)
    for report in reports:
        assert max(report["complete_constrained_residual_N"], report["complete_body_residual_N"]) <= 1e-9
    with np.load(directories[0] / "fields.npz") as candidate, np.load(directories[1] / "fields.npz") as original:
        assert candidate.files == original.files
        array_checks = {}
        for key in original.files:
            assert candidate[key].dtype == original[key].dtype
            assert candidate[key].shape == original[key].shape
            if key in ["rest_positions", "surface_triangles", "body_component_permutation"]:
                np.testing.assert_array_equal(candidate[key], original[key])
            elif key in ["body_tensor_Pa", "body_vm_Pa"]:
                np.testing.assert_allclose(candidate[key], original[key], rtol=1e-4, atol=1e-3)
            elif key in ["displacement", "contact_force_N"]:
                np.testing.assert_allclose(candidate[key], original[key], rtol=1e-8, atol=1e-12)
            elif key == "applied_force_N":
                np.testing.assert_allclose(candidate[key], original[key], rtol=1e-12, atol=1e-15)
            elif key == "total_gradient_N":
                np.testing.assert_allclose(candidate[key], original[key], rtol=1e-8, atol=1e-9)
            else:
                raise AssertionError(f"Unspecified original array criterion: {key}")
            array_checks[key] = {
                "bit_identical": bool(np.array_equal(candidate[key], original[key])),
                "maximum_absolute_difference": float(np.max(abs(candidate[key] - original[key]))),
            }
        qualified_arrays = list(original.files)
        final = {key: candidate[key].copy() for key in candidate.files}
        baseline_final = {key: original[key].copy() for key in original.files}
    references = []
    for name, path, young, poisson, density in [
        ("body", Path(config["mesh"]), 1e10, 0.3, 2000),
        ("left_shoe", directories[0] / "left_shoe_mesh.npz", config["pad_young"], 0, 1000),
        ("right_shoe", directories[0] / "right_shoe_mesh.npz", config["pad_young"], 0, 1000),
    ]:
        with np.load(path) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density)
        references.append((name, reference))
    system = IndependentContactSystem(references, config["dhat"], 1e6, config["friction"], 1e-8)
    bulks = [NeoHookeanBulk(reference) for _, reference in references]
    body, bulk = references[0][1], bulks[0]
    n = body.basis.N
    final_vectors = []
    for saved in [final, baseline_final]:
        vector = np.empty(n)
        vector[system.parts[0].permutation] = saved["displacement"][:n]
        final_vectors.append(vector)
    final_quadrature = [
        tensor_from_f(bulk.deformation_gradients(vector).transpose(2, 3, 0, 1), body.lame) for vector in final_vectors
    ]
    np.testing.assert_allclose(final_quadrature[0], final_quadrature[1], rtol=1e-4, atol=1e-3)
    quadrature_difference = float(np.linalg.norm(final_quadrature[0] - final_quadrature[1]))
    quadrature_reference = float(np.linalg.norm(final_quadrature[1]))
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for directory in directories:
        for name in ["configuration.json", "result.json", "fields.npz", "left_shoe_mesh.npz", "right_shoe_mesh.npz"]:
            retain(directory / name)
    retain(Path(config["mesh"]))
    history = args.run / "accepted-contact-history"
    manifest = json.loads((history / "manifest.json").read_text())
    retain(history / "manifest.json")
    assert retain(history / "geometry.npz") == manifest["geometry_sha256"]
    assert retain(history / "stages.jsonl") == manifest["stages_sha256"]
    with np.load(history / "geometry.npz") as saved:
        for key, value in [
            ("rest_positions", system.rest),
            ("surface_triangles", system.faces),
            ("groups", system.groups),
        ]:
            np.testing.assert_array_equal(saved[key], value)
    records = [json.loads(line) for line in (history / "stages.jsonl").read_text().splitlines()]
    assert len(records) == manifest["accepted_stages"] == len(reports[0]["accepted_steps"])
    assert [r["ordinal"] for r in records] == list(range(1, len(records) + 1))
    assert [r["accepted_record_index"] for r in records] == list(range(len(records)))
    mappings = {}
    for call in manifest["calls"]:
        assert [r["step"] for r in records if r["call"] == call["call"]] == list(range(1, call["load_steps"] + 1))
        path = history / call["map_file"]
        assert retain(path) == call["map_sha256"] and call["tolerance_N"] == 1e-9
        mappings[call["call"]] = sparse.load_npz(path)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, previous_state, old_call = [], None, None
    for record in records:
        ordinal, call_id = record["ordinal"], record["call"]
        call = manifest["calls"][call_id]
        assert record["phase"] == call["phase"] and record["load_steps"] == call["load_steps"]
        accepted = reports[0]["accepted_steps"][record["accepted_record_index"]]
        assert accepted["phase"] == record["phase"] and accepted["step"] == record["step"]
        assert accepted["complete_free_residual_n"] == record["recorded_complete_residual_N"] <= 1e-9
        path = history / record["file"]
        assert retain(path) == record["sha256"]
        with np.load(path) as saved:
            displacement, previous, stored_contact, applied, stored_gradient = [
                saved[key].copy()
                for key in [
                    "displacement",
                    "previous_displacement",
                    "contact_force_N",
                    "applied_force_N",
                    "total_gradient_N",
                ]
            ]
        if old_call == call_id:
            np.testing.assert_array_equal(previous, previous_state)
            assert record["step"] == rows[-1]["step"] + 1
        elif previous_state is not None:
            expected = previous_state.copy()
            if record["phase"] == "free_body_force_control" and config["centre_gauge"]:
                expected[0::3] -= reports[0]["initial_common_x_shift_m"]
            np.testing.assert_array_equal(previous, expected)
        vector = np.empty(n)
        vector[system.parts[0].permutation] = displacement[:n]
        centre = body.com + (body.rigid_modes[:, :3].T @ (body.mass @ vector)) / body.total_mass
        normal, shear, stencils, source_rows = contact_forces(system, displacement, previous, centre)
        np.testing.assert_allclose(normal + shear, stored_contact, rtol=1e-12, atol=1e-12)
        gradient = -applied - normal - shear
        for part, part_bulk in zip(system.parts, bulks):
            count = part.reference.basis.N
            section = slice(3 * part.first_node, 3 * part.first_node + count)
            local = np.empty(count)
            local[part.permutation] = displacement[section]
            _, internal, _ = part_bulk.evaluate(local, False)
            gradient[section] += internal[part.permutation]
        np.testing.assert_allclose(gradient, stored_gradient, rtol=1e-10, atol=1e-12)
        residual = float(np.linalg.norm(mappings[call_id].T @ gradient))
        assert residual <= call["tolerance_N"]
        tensor, vm, upper = bulk.stress_bounds(vector)
        quadrature = tensor_from_f(bulk.deformation_gradients(vector).transpose(2, 3, 0, 1), body.lame)
        row = {
            "ordinal": ordinal,
            "phase": record["phase"],
            "step": record["step"],
            "normal_stencils": stencils,
            "independent_complete_constrained_residual_N": residual,
            "contact_force_reconstruction_max_difference_N": float(np.max(abs(normal + shear - stored_contact))),
            "raw_global_peak_lower_Pa": float(vm.max()),
            "raw_global_peak_upper_Pa": float(upper.max()),
            "sources": source_rows,
            "is_physical_time_sample": False,
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"stage{ordinal:06d}.npz",
            normal_contact_force_N=normal,
            shear_contact_force_N=shear,
            reconstructed_total_gradient_N=gradient,
            body_corner_tensor_Pa=tensor,
            body_corner_vm_Pa=vm,
            body_element_peak_upper_Pa=upper,
            body_quadrature_tensor_Pa=quadrature,
        )
        previous_state, old_call = displacement, call_id
        if ordinal % 25 == 0:
            print(json.dumps({"stages_checked": ordinal, "complete_residual_N": residual}), flush=True)
    np.testing.assert_array_equal(previous_state, final["displacement"])
    np.testing.assert_allclose(tensor, final["body_tensor_Pa"], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(vm, final["body_vm_Pa"], rtol=1e-10, atol=1e-6)
    original_map = directories[0] / "original_coordinate_map.npz"
    retain(original_map)
    final_body_residual = float(np.linalg.norm(gradient[:n]))
    final_original_residual = float(np.linalg.norm(sparse.load_npz(original_map).T @ gradient))
    assert max(final_body_residual, final_original_residual) <= 1e-9
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "baseline": str(args.baseline),
        "final_saved_arrays_qualified": qualified_arrays,
        "final_saved_array_comparisons": array_checks,
        "original_accepted_stage_keys_identical": True,
        "original_accepted_residual_records_identical": reports[0]["accepted_steps"] == reports[1]["accepted_steps"],
        "original_iteration_records_identical": reports[0]["iterations"] == reports[1]["iterations"],
        "final_complete_quadrature_tensor_absolute_difference_Pa": quadrature_difference,
        "final_complete_quadrature_tensor_relative_difference": quadrature_difference / quadrature_reference,
        "stress_field_qualification_criteria": {
            "rtol": 1e-4,
            "atol_Pa": 1e-3,
            "source": "PHYSICAL_VALIDATION_PROTOCOL_20261014.md original independent-field gates",
        },
        "independent_final_body_residual_N": final_body_residual,
        "independent_final_original_mapped_residual_N": final_original_residual,
        "stages_checked": len(rows),
        "all_retained_stages_independent_current_friction_and_original_force_gate": True,
        "all_retained_stages_full_corner_quadrature_and_raw_bounds": True,
        "rows": rows,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "The recording path is qualified against all original final arrays and complete corner/quadrature fields at the previously declared independent-field tolerances. Residual/iteration bit differences remain explicit; no bit identity is inferred from unchanged equations. Original full intermediate fields do not exist for a matched history comparison. Independent current normal/friction forces, constrained bulk balance and complete raw body fields are reconstructed at every new retained stage at the original1e-9N gate. These are original quasi-static initialization/load increments, not physical timesteps; endpoint intersection tests do not certify unrecorded paths between numerical Newton iterates. Physical shoe groups are source identities, not continuum islands. One coarse history does not establish spatial, regularization, load-increment or pressure convergence, native load-model accuracy, full Franka matching or verified warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stages_checked": len(rows), "recording_qualified": True}), flush=True)


if __name__ == "__main__":
    main()
