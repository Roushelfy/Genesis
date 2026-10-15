"""Qualify rigid finger trace geometry, affine motion and force/moment pullback on an actual saved body mesh."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .refine_rigid_finger_trace import refine_rigid_finger_trace
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], {}
    for name in ["configuration.json", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
        path = args.run / "results" / name
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = np.random.default_rng(2026101507)
    for level in [0, 1, 2]:
        system, _ = restore_static(args.run)
        old = system.contact_node_map.copy()
        old_faces = system.contact_surface_faces.copy()
        old_groups = system.contact_surface_groups.copy()
        refine_rigid_finger_trace(system, level)
        mapping, rest = system.contact_node_map, system.contact_surface_rest
        body = system.contact_surface_groups == 0
        for key in ["data", "indices", "indptr"]:
            np.testing.assert_array_equal(getattr(mapping[body], key), getattr(old[old_groups == 0], key))
        if level == 0:
            np.testing.assert_array_equal(system.contact_surface_faces, old_faces)
            np.testing.assert_array_equal(system.contact_surface_groups, old_groups)
        affine_errors, force_errors, moment_errors, work_errors = [], [], [], []
        for _ in range(4):
            rotation = Rotation.random(random_state=rng).as_matrix()
            translation = rng.normal(size=3) * 0.1
            positions = system.rest @ rotation.T + translation
            surface = mapping @ positions
            affine_errors.append(float(abs(surface - (rest @ rotation.T + translation)).max()))
            forces = rng.normal(size=surface.shape) * 1e-3
            pulled = mapping.T @ forces
            force_errors.append(float(abs(pulled.sum(axis=0) - forces.sum(axis=0)).max()))
            moment_errors.append(
                float(abs(np.cross(positions, pulled).sum(axis=0) - np.cross(surface, forces).sum(axis=0)).max())
            )
            delta = rng.normal(size=positions.shape) * 1e-6
            work_errors.append(float(abs(np.sum(forces * (mapping @ delta)) - np.sum(pulled * delta))))
        assert max(affine_errors) <= 1e-14
        assert max(force_errors) <= 1e-10 and max(moment_errors) <= 1e-10 and max(work_errors) <= 1e-16
        rows.append(
            {
                "finger_refinement": level,
                "contact_vertices": len(rest),
                "contact_triangles": len(system.contact_surface_faces),
                "body_trace_rows_identical": True,
                "undeformed_group_areas_preserved_by_constructor": True,
                "maximum_affine_motion_error_m": max(affine_errors),
                "maximum_force_pullback_error_N": max(force_errors),
                "maximum_moment_pullback_error_Nm": max(moment_errors),
                "maximum_virtual_work_error_J": max(work_errors),
            }
        )
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "rows": rows,
        "input_sha256": inputs,
        "qualified": True,
        "goal_completion": False,
        "limitations": "This construction qualification preserves actual body rows, rigid finger surfaces, affine motion and wrench/virtual-work pullback. It does not solve new contact equilibria or establish stress/contact convergence. A zero-level complete-state/field regression and independent equilibrium audits remain required before accepting new resolved references.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
