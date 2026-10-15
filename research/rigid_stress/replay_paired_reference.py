"""Qualify a reference linear-solver change against every original saved array and complete stress field."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_precise_spatial_state import assemble_state
from .franka_rigid_mode_coordinates import coordinate_map
from .precise_stencil_spatial import PreciseStencilSpatial
from .qualify_articulated_precision import peak, tensor_from_f
from .resume_interrupted_spatial_reference import compare_state_arrays
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    static, config = restore_static(args.run)
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert len(paths) == config["steps"] and paths
    args.output.mkdir(parents=True, exist_ok=False)
    identities = []
    for name in ["closure_state.npz", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
        source = args.run / "results" / name
        shutil.copy2(source, args.output / name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256((args.output / name).read_bytes()).hexdigest() == digest
        identities.append({"source": str(source), "sha256": digest})
    config["output"] = str(args.output)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "precision_provenance.json").write_text(
        json.dumps(
            {
                "source_run": str(args.run),
                "initial_state_and_meshes": identities,
                "all_original_states_resolved_from_byte_identical_initializer": True,
            },
            indent=2,
        )
        + "\n"
    )
    with np.load(args.output / "closure_state.npz") as saved:
        u = np.asarray(saved["displacement"], dtype=np.longdouble)
    velocity = np.zeros_like(u)
    system = PreciseStencilSpatial(static, config["dt"])
    system.q_target[:] = 0.018
    mapping = coordinate_map(system)
    gravity = system.gravity_force([0, 0, -9.81])
    bulk = WidePiolaBulk(static.body)
    rows = []
    for step, original in enumerate(paths, 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + config["dt"] * old_velocity
        u, _gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=gravity,
            initial_force=gravity,
            phase=f"qualified_ordering_dynamic{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / config["dt"]
        acceleration = (velocity - old_velocity) / config["dt"]
        destination = args.output / original.name
        np.savez_compressed(
            destination,
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=force,
            applied_force_N=gravity,
            body_component_permutation=system.parts[0].permutation,
        )
        metrics, _fields = assemble_state(static, config, u, previous, old_velocity, force, bulk)
        with np.load(original) as baseline, np.load(destination) as candidate:
            changes = compare_state_arrays(baseline, candidate)
            fields = []
            for saved in [baseline, candidate]:
                vector = np.empty(static.body.basis.N, dtype=np.longdouble)
                vector[static.parts[0].permutation] = saved["displacement"][: static.body.basis.N]
                corner = tensor_from_f(bulk.corner_deformation_gradients(vector), static.body.lame)
                quadrature = quadrature_fields(bulk, np.array(vector, dtype=float))[0]
                fields.append((corner, quadrature))
        corner_change = float(np.linalg.norm(fields[1][0] - fields[0][0]) / np.linalg.norm(fields[0][0]))
        quad_change = float(np.linalg.norm(fields[1][1] - fields[0][1]) / np.linalg.norm(fields[0][1]))
        peak_change = abs(peak(fields[1][0]) / peak(fields[0][0]) - 1)
        assert max(corner_change, quad_change, peak_change) <= 1e-6
        row = {
            "step": step,
            "time_s": step * config["dt"],
            **metrics,
            "all_saved_array_changes": changes,
            "all_corner_tensor_relative_change": corner_change,
            "all_quadrature_tensor_relative_change": quad_change,
            "raw_global_peak_relative_change": peak_change,
            "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "states_checked": len(rows),
        "rows": rows,
        "all_original_arrays_and_complete_fields_qualified": True,
        "declared_existing_budgets": {
            "state_relative": 1e-5,
            "state_absolute": 1e-12,
            "all_corner_and_quadrature_tensor_relative": 1e-6,
            "physical_body_and_mapped_force_N": 1e-9,
            "Newton_internal_target_N": 1e-10,
        },
        "goal_completion": False,
        "limitations": "A changed sparse ordering is qualified only on this complete saved prefix, with a byte-identical original initializer and unchanged physical equations/tolerances. Numerical equivalence and reference solve acceleration do not establish contact/geometry convergence, full application/history, model/feedback accuracy or warmed production cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
