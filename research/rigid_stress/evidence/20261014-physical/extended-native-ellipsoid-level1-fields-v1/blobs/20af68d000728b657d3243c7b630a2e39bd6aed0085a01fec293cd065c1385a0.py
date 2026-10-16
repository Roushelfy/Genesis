"""Independently verify every precision-aware spatial state and its unaveraged global field."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_original_history_fields import wide_volume_bounds
from .audit_precise_spatial_state import assemble_state
from .qualify_articulated_precision import tensor_from_f
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
    provenance = json.loads((args.run / "results/precision_provenance.json").read_text())
    source = Path(provenance["source_run"])
    static, config = restore_static(source)
    for item in provenance["initial_state_and_meshes"]:
        p = Path(item["source"])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
        assert hashlib.sha256((args.run / "results" / p.name).read_bytes()).hexdigest() == item["sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    bulk = WidePiolaBulk(static.body)
    n = static.body.basis.N
    with np.load(args.run / "results/closure_state.npz") as saved:
        previous = np.asarray(saved["displacement"], dtype=np.longdouble)
    old_velocity = np.zeros_like(previous)
    rows = []
    for step in range(1, config["steps"] + 1):
        with np.load(args.run / f"results/step{step:04d}.npz") as saved:
            u, velocity, acceleration, expected = [
                saved[k].copy() for k in ["displacement", "velocity", "acceleration", "contact_force_N"]
            ]
            np.testing.assert_array_equal(saved["applied_force_N"], static.gravity_force([0, 0, -9.81]))
            np.testing.assert_array_equal(saved["body_component_permutation"], static.parts[0].permutation)
        np.testing.assert_array_equal(velocity, (u - previous) / config["dt"])
        np.testing.assert_array_equal(acceleration, (velocity - old_velocity) / config["dt"])
        metrics, fields = assemble_state(static, config, u, previous, old_velocity, expected, bulk)
        vector = np.empty(n, dtype=np.longdouble)
        vector[static.parts[0].permutation] = u[:n]
        f = bulk.corner_deformation_gradients(vector)
        tensor = tensor_from_f(f, static.body.lame)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        j_lower, upper = wide_volume_bounds(f, bulk.mu)
        assert float(vm.max()) <= upper and j_lower > 0
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        row = {
            "step": step,
            "time_s": step * config["dt"],
            **metrics,
            "raw_global_peak_lower_Pa": float(vm.max()),
            "raw_global_peak_upper_Pa": upper,
            "whole_volume_J_lower_bound": j_lower,
            "raw_peak_reference_position_m": static.body.corner_positions[cell, corner].tolist(),
        }
        rows.append(row)
        np.savez_compressed(args.output / f"step{step:04d}.npz", **fields)
        previous, old_velocity = u, velocity
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "states_checked": len(rows),
        "rows": rows,
        "goal_completion": False,
        "limitations": "Every original dynamic step is independently reconstructed, including exact previous-state/velocity integration, unsmoothed centered normal/friction stencils, mass/armature and capped jaw PD. Raw global wide-coordinate field bounds remain visible. The original fixed-arm/two-jaw control and saved held initializer remain explicit. Full-arm application matching and physical spatial/contact/geometry convergence are separate incomplete requirements.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_checked": len(rows)}))


if __name__ == "__main__":
    main()
