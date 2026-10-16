"""Resolve the complete original two-jaw dynamic control using precision-aware contact arithmetic."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .franka_rigid_mode_coordinates import coordinate_map
from .precise_stencil_spatial import PreciseStencilSpatial
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.qualification]:
        closed_source(run)
    assert int((args.qualification / "returncode.txt").read_text()) == 0
    qualification = json.loads((args.qualification / "results/summary.json").read_text())
    assert Path(qualification["source_run"]).resolve() == args.run.resolve()
    assert qualification["complete_mapped_dynamic_residual_N"] <= 1e-9
    assert qualification["complete_body_dynamic_residual_N"] <= 1e-9
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.run)
    provenance = []
    for name in ["closure_state.npz", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
        source = args.run / "results" / name
        destination = args.output / name
        shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        provenance.append({"source": str(source), "sha256": digest})
    config["output"] = str(args.output)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "precision_provenance.json").write_text(
        json.dumps(
            {
                "source_run": str(args.run),
                "qualification": str(args.qualification),
                "initial_state_and_meshes": provenance,
                "all_dynamic_steps_resolved_from_original_zero_velocity": True,
            },
            indent=2,
        )
        + "\n"
    )
    system = PreciseStencilSpatial(static, config["dt"])
    system.q_target[:] = 0.018
    with np.load(args.output / "closure_state.npz") as saved:
        u = np.asarray(saved["displacement"], dtype=np.longdouble)
    velocity = np.zeros_like(u)
    mapping = coordinate_map(system)
    external = system.gravity_force([0, 0, -9.81])
    rows = []
    for step in range(1, config["steps"] + 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + config["dt"] * old_velocity
        u, gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=external,
            initial_force=external,
            phase=f"precise_dynamic_step{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / config["dt"]
        acceleration = (velocity - old_velocity) / config["dt"]
        complete, body = (
            float(np.linalg.norm(mapping.T @ gradient)),
            float(np.linalg.norm(gradient[: system.body.basis.N])),
        )
        assert complete <= 1e-9 and body <= 1e-9
        row = {
            "step": step,
            "time_s": step * config["dt"],
            "complete_dynamic_residual_N": complete,
            "complete_body_dynamic_residual_N": body,
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=force,
            applied_force_N=external,
            body_component_permutation=system.parts[0].permutation,
        )
        print(json.dumps(row), flush=True)
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "qualification": str(args.qualification),
        "body_tetrahedra": system.body.mesh.nelements,
        "steps": rows,
        "goal_completion": False,
        "limitations": "All original dynamic steps are resolved from the byte-identical original held closure and zero velocity. Original physical inputs, dt and1e-9 external gates stay fixed; the numerical solve targets1e-10. Independent complete state/force/stress audits and old-state numerical comparisons remain necessary. This is the original fixed-arm/two-jaw control, not a full nine-joint application or physical contact/mesh/geometry tail acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
