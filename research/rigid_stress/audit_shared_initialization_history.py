"""Independently audit every stage of a declared shared-initialization U contact reference."""

import argparse
import ast
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
    parser.add_argument("--qualified-common-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.qualified_common_audit]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    directory = args.run / "solution"
    config = json.loads((directory / "configuration.json").read_text())
    assert config["geometry"] == "u" and config["pad_resolution"] == 5
    assert config["body_material"] == config["pad_material"] == "neo_hookean"
    assert config["body_rigid_coordinates"] and config["centre_gauge"]
    assert config["updated_friction"] and not config["resume_failed_step"]
    assert config["preloaded_state"] is None and config["initialization_anchors"] is not None
    assert config["contact_initialization_steps"] == 604 and config["force_steps"] == 64
    qualified = json.loads((args.qualified_common_audit / "results/summary.json").read_text())
    assert qualified["stages_checked"] == 744
    assert qualified["all_retained_stages_independent_current_friction_and_original_force_gate"]
    assert qualified["requested_initialization_coordinates_independently_matched"]
    qualified_run = Path(qualified["run"])
    closed_source(qualified_run)
    assert int((qualified_run / "returncode.txt").read_text()) == 0
    qualified_auditor = (
        args.qualified_common_audit / "source/research/rigid_stress/audit_recorded_nonlinear_contact_history.py"
    )

    def function_tree(path):
        return next(
            ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "contact_forces"
        )

    assert function_tree(Path(__file__)) == function_tree(qualified_auditor)
    for name in ["contact_reference", "neohookean_reference", "skfem_reference", "qualify_articulated_precision"]:
        relative = Path("research/rigid_stress") / (name + ".py")
        assert (
            Path(__file__).with_name(name + ".py").read_bytes()
            == (args.qualified_common_audit / "source" / relative).read_bytes()
        )
    original_config = json.loads((qualified_run / "solution/configuration.json").read_text())
    excluded = {"mesh", "output"}
    assert {k: v for k, v in config.items() if k not in excluded} == {
        k: v for k, v in original_config.items() if k not in excluded
    }
    for name in [
        "contact_reference",
        "neohookean_reference",
        "skfem_reference",
        "cached_reference_newton",
        "grasp_contact_reference",
        "reference_contact_history",
    ]:
        relative = Path("research/rigid_stress") / (name + ".py")
        assert (args.run / "source" / relative).read_bytes() == (qualified_run / "source" / relative).read_bytes()
    report = json.loads((directory / "result.json").read_text())
    assert max(report["complete_constrained_residual_N"], report["complete_body_residual_N"]) <= 1e-9
    assert all(row["complete_free_residual_n"] <= 1e-9 for row in report["accepted_steps"])
    with np.load(directory / "fields.npz") as saved:
        final = {key: saved[key].copy() for key in saved.files}
    references = []
    for name, path, young, poisson, density in [
        ("body", Path(config["mesh"]), 1e10, 0.3, 2000),
        ("left_shoe", directory / "left_shoe_mesh.npz", config["pad_young"], 0, 1000),
        ("right_shoe", directory / "right_shoe_mesh.npz", config["pad_young"], 0, 1000),
    ]:
        with np.load(path) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density)
        references.append((name, reference))
    system = IndependentContactSystem(references, config["dhat"], 1e6, config["friction"], 1e-8)
    bulks = [NeoHookeanBulk(reference) for _, reference in references]
    body, bulk = references[0][1], bulks[0]
    n = body.basis.N
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for name in ["configuration.json", "result.json", "fields.npz", "left_shoe_mesh.npz", "right_shoe_mesh.npz"]:
        retain(directory / name)
    retain(args.qualified_common_audit / "results/summary.json")
    retain(qualified_auditor)
    retain(qualified_run / "solution/configuration.json")
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
    assert len(records) == manifest["accepted_stages"] == len(report["accepted_steps"])
    assert [r["ordinal"] for r in records] == list(range(1, len(records) + 1))
    assert [r["accepted_record_index"] for r in records] == list(range(len(records)))
    mappings = {}
    assert [(call["phase"], call["load_steps"]) for call in manifest["calls"]] == [
        ("collision_free_approach", 12),
        ("contact_initialization", 604),
        ("held_body_force_control", 64),
        ("free_body_force_control", 64),
    ]
    for call in manifest["calls"]:
        assert call["dhat_m"] == config["dhat"] and call["friction"] == config["friction"]
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
        accepted = report["accepted_steps"][record["accepted_record_index"]]
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
                expected[0::3] -= report["initial_common_x_shift_m"]
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
        peak_cell, peak_corner = np.unravel_index(np.argmax(vm), vm.shape)
        row = {
            "ordinal": ordinal,
            "phase": record["phase"],
            "step": record["step"],
            "normal_stencils": stencils,
            "independent_complete_constrained_residual_N": residual,
            "contact_force_reconstruction_max_difference_N": float(np.max(abs(normal + shear - stored_contact))),
            "raw_global_peak_lower_Pa": float(vm.max()),
            "raw_global_peak_upper_Pa": float(upper.max()),
            "raw_peak_reference_position_m": body.corner_positions[peak_cell, peak_corner].tolist(),
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
    anchor_path = Path(config["initialization_anchors"])
    retain(anchor_path)
    with np.load(anchor_path) as anchors:
        requested_points, requested_components = anchors["positions_m"], anchors["components"]
    original_map = directory / "original_coordinate_map.npz"
    retain(original_map)
    saved_map = sparse.load_npz(original_map).tocsc()
    columns = np.flatnonzero(np.diff(saved_map.indptr) == 1)
    indices = saved_map.indices[saved_map.indptr[columns]]
    omitted = np.setdiff1d(np.arange(n), indices[indices < n])
    assert len(omitted) == 6
    actual_points, actual_components = system.rest[omitted // 3], omitted % 3
    distances = np.linalg.norm(requested_points[:, None] - actual_points[None], axis=-1)
    distances[requested_components[:, None] != actual_components[None]] = np.inf
    assert np.all(distances.min(axis=1) <= 1e-12)
    final_body_residual = float(np.linalg.norm(gradient[:n]))
    final_original_residual = float(np.linalg.norm(sparse.load_npz(original_map).T @ gradient))
    assert max(final_body_residual, final_original_residual) <= 1e-9
    for index, part in enumerate(system.parts):
        reference = part.reference
        count = reference.basis.N
        section = slice(3 * part.first_node, 3 * part.first_node + count)
        acceleration = np.empty(count)
        acceleration[reference.component_dofs] = config["gravity"]
        gravity = (reference.mass @ acceleration)[part.permutation]
        actuation = (applied[section] - gravity).reshape(-1, 3)
        if index == 0:
            np.testing.assert_allclose(actuation, 0, rtol=0, atol=1e-15)
        else:
            drive = -1 if index == 1 else 1
            np.testing.assert_allclose(actuation.sum(axis=0), [drive * config["force"], 0, 0], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(actuation[:, 1:], 0, rtol=0, atol=1e-15)
            active = abs(actuation[:, 0]) > 1e-15
            assert active.any()
            np.testing.assert_allclose(
                actuation[active, 0], drive * config["force"] / active.sum(), rtol=1e-12, atol=1e-15
            )
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "qualified_common_initialization_audit": str(args.qualified_common_audit),
        "study": "Fixed-pad U body/geometry sequence with shared physical initialization anchors and original stage counts",
        "body_tetrahedra": body.mesh.nelements,
        "body_mass_kg": body.total_mass,
        "same_qualified_physical_controls_and_producer_sources_verified": True,
        "qualified_per_stencil_force_method_and_bulk_sources_verified": True,
        "independent_final_consistent_mass_gravity_and_uniform_shoe_actuation_verified": True,
        "same_solution_equivalence_claimed": False,
        "common_initialization_anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "requested_initialization_coordinates_independently_matched": True,
        "independent_final_body_residual_N": final_body_residual,
        "independent_final_original_mapped_residual_N": final_original_residual,
        "stages_checked": len(rows),
        "all_retained_stages_independent_current_friction_and_original_force_gate": True,
        "all_retained_stages_full_corner_quadrature_and_raw_bounds": True,
        "rows": rows,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "Independent all-stage current-friction forces, original1e-9N constrained balances, complete corner/quadrature tensors and raw unaveraged global bounds. Same pad5 geometry, qualified producer sources, shared physical initialization coordinates, physical controls and12/604/64/64 stages are explicitly held. Body geometry/mass approximation and volume/contact-trace resolution still change together. This does not establish a separated bulk axis, contact pressure, regularization/history convergence, continuum model/feedback accuracy or verified warmed native cost. The old differently initialized cases remain valid results of their declared setup and are not replaced.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stages_checked": len(rows), "independent_current_history_audit_passed": True}), flush=True)


if __name__ == "__main__":
    main()
