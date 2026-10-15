"""Continue a closed paired-geometry reference from its unchanged initializer and audited physical prefix."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .audit_original_history_fields import wide_volume_bounds
from .audit_precise_spatial_state import assemble_state
from .franka_rigid_mode_coordinates import coordinate_map
from .precise_stencil_spatial import PreciseStencilSpatial
from .qualify_articulated_precision import tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.source_run, *args.qualification_run]:
        closed_source(run)
    for run in args.qualification_run:
        assert int((run / "returncode.txt").read_text()) == 0
    static, config = restore_static(args.source_run)
    assert config["mode"] in ["polyhedron", "cad"] and config["cold_closure"]
    original_config = args.source_run / "results/configuration.json"
    prefix = sorted((args.source_run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert [path.name for path in prefix] == [f"step{step:04d}.npz" for step in range(1, len(prefix) + 1)]
    assert len(prefix) < config["steps"]
    args.output.mkdir(parents=True, exist_ok=False)
    identities = []
    for name in ["closure_state.npz", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
        path = args.source_run / "results" / name
        shutil.copy2(path, args.output / name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256((args.output / name).read_bytes()).hexdigest() == digest
        identities.append({"source": str(path), "sha256": digest})
    with np.load(args.output / "closure_state.npz") as saved:
        u = np.asarray(saved["displacement"], dtype=np.longdouble)
    velocity = np.zeros_like(u)
    bulk = WidePiolaBulk(static.body)
    rows = []
    for step, path in enumerate(prefix, 1):
        with np.load(path) as saved:
            current, current_velocity, force = [
                saved[key].copy() for key in ["displacement", "velocity", "contact_force_N"]
            ]
            np.testing.assert_array_equal(current_velocity, (current - u) / config["dt"])
            np.testing.assert_array_equal(saved["acceleration"], (current_velocity - velocity) / config["dt"])
        metrics, _fields = assemble_state(static, config, current, u, velocity, force, bulk)
        vector = np.empty(static.body.basis.N, dtype=np.longdouble)
        vector[static.parts[0].permutation] = current[: static.body.basis.N]
        deformation = bulk.corner_deformation_gradients(vector)
        tensor = tensor_from_f(deformation, static.body.lame)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        raw_peak = float(np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1))).max())
        determinant_lower, upper = wide_volume_bounds(deformation, bulk.mu)
        assert determinant_lower > 0 and raw_peak <= upper
        shutil.copy2(path, args.output / path.name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256((args.output / path.name).read_bytes()).hexdigest() == digest
        rows.append(
            {
                "step": step,
                "time_s": step * config["dt"],
                **metrics,
                "copied_sha256": digest,
                "raw_global_peak_lower_Pa": raw_peak,
                "raw_global_peak_upper_Pa": upper,
            }
        )
        u, velocity = current, current_velocity
    config["output"] = str(args.output)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "source_returncode": int((args.source_run / "returncode.txt").read_text()),
        "original_configuration_sha256": hashlib.sha256(original_config.read_bytes()).hexdigest(),
        "qualification_runs": [str(run) for run in args.qualification_run],
        "initial_state_and_meshes": identities,
        "copied_prefix": rows,
        "original_state_arrays_and_fields_unchanged": True,
    }
    (args.output / "precision_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    system = PreciseStencilSpatial(static, config["dt"])
    system.q_target[:] = 0.018
    mapping = coordinate_map(system)
    gravity = system.gravity_force([0, 0, -9.81])
    for step in range(len(prefix) + 1, config["steps"] + 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + config["dt"] * old_velocity
        u, gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=gravity,
            initial_force=gravity,
            phase=f"generic_stencil_dynamic{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / config["dt"]
        acceleration = (velocity - old_velocity) / config["dt"]
        complete, body = (
            float(np.linalg.norm(mapping.T @ gradient)),
            float(np.linalg.norm(gradient[: static.body.basis.N])),
        )
        assert complete <= 1e-9 and body <= 1e-9
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=force,
            applied_force_N=gravity,
            body_component_permutation=system.parts[0].permutation,
        )
        rows.append(
            {
                "step": step,
                "time_s": step * config["dt"],
                "complete_dynamic_residual_N": complete,
                "complete_body_dynamic_residual_N": body,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "steps": rows,
        "original_byte_identical_prefix_states": len(prefix),
        "goal_completion": False,
        "limitations": "Only centered reference collision-force evaluation is generalized to supported point/edge/face stencils and the package's original unmollified friction magnitude. Original geometry/material/mass, closure, control, dt, gravity, physical1e-9N gates and stricter Newton1e-10 target remain fixed. All retained prefix states are independently force-audited and copied byte-identically before continuation. The initializer is retained, not independently reconstructed at every original closure target. Complete new histories still require a separate independent audit; this does not establish physical convergence, full application, remaining models/feedback or verified warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
