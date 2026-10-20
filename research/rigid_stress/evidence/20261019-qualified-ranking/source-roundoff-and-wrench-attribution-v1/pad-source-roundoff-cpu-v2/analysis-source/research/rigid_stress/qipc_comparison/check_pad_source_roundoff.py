"""Quantify actual DPR37/SCR38 input reconciliation and its independent elastic response."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import FreeElasticP2, quadratic_triangle_weights
from .source_roundoff import reconcile_source_roundoff


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--frames", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stress_target = 1e-6
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "mesh": file_id(args.mesh), "frames": [file_id(p) for p in args.frames],
        "stress_perturbation_absolute_target_pa": stress_target,
        "force_budget": "16*source_epsilon*max(actual_command_force,unsigned_source_force_sum)",
        "primary_timing": False})
    with np.load(args.mesh) as saved:
        vertices, tets, outer, faces = (saved[k] for k in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    model = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000, column_faces=faces,
                          through_wall_ordering=True, cvxopt_factor=True)
    _, _, face_nodes = model.surface(faces)
    centers = outer[faces].mean(axis=1)
    shape = quadratic_triangle_weights(np.full((1, 3), 1 / 3))[0]
    rows = []
    for path in args.frames:
        with np.load(path) as saved:
            points, forces, normals = (saved[k] for k in ("source_points_m", "source_forces_n", "source_normals_material"))
            friction, command = saved["friction"].item(), saved["load_n"].item()
        fixed = reconcile_source_roundoff(points, forces, normals, friction, command)
        assert fixed.corrected_sources > 0 and fixed.force_change_bound_n > 0
        delta = fixed.forces - forces
        distance, indices = cKDTree(centers).query(points)
        assert distance.max() < 1e-14 and len(np.unique(indices)) == len(indices)
        nodal = np.zeros_like(model.vertices)
        np.add.at(nodal, face_nodes[indices].reshape(-1), (shape[None, :, None] * delta[:, None]).reshape(-1, 3))
        stress, displacement, correction, residual = model.solve(nodal)
        maximum_stress = np.abs(np.linalg.eigvalsh(stress)).max().item()
        assert maximum_stress <= stress_target
        # A substantial pure shear with zero normal load remains rejected even
        # in the presence of a large independent command scale.
        invalid = forces.copy()
        invalid[0] = 1e-8 * np.cross(normals[0], np.eye(3)[np.argmin(np.abs(normals[0]))])
        try:
            reconcile_source_roundoff(points, invalid, normals, friction, command)
        except ValueError as error:
            assert "exceeds physical roundoff budget" in str(error)
        else:
            raise AssertionError("Finite incompatible shear was accepted")
        scale_errors = []
        for scale in (1e-3, 1.0, 1e3):
            scaled = reconcile_source_roundoff(points, scale * forces, normals, friction, scale * command)
            np.testing.assert_allclose(scaled.forces, scale * fixed.forces, rtol=1e-12, atol=0)
            scale_errors.append(abs(scaled.force_change_bound_n / (scale * fixed.force_change_bound_n) - 1))
        c, s = np.cos(0.71), np.sin(0.71)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        rotated = reconcile_source_roundoff(points @ rotation.T, forces @ rotation.T, normals @ rotation.T,
                                             friction, command)
        np.testing.assert_allclose(rotated.forces, fixed.forces @ rotation.T, rtol=1e-12, atol=0)
        unchanged = reconcile_source_roundoff(points, fixed.forces, normals, friction, command)
        assert unchanged.corrected_sources == 0 and np.array_equal(unchanged.forces, fixed.forces)
        output = args.output / ("correction-" + path.parent.parent.name + "-" + path.stem + ".npz")
        np.savez_compressed(output, source_points_m=points, original_forces_n=forces, corrected_forces_n=fixed.forces,
                            p2_coordinates_m=model.vertices, nodal_correction_n=nodal, stress_correction_pa=stress,
                            displacement_correction_m=displacement, mass_correction_n=correction,
                            equation_residual_n=residual)
        rows.append({"frame": file_id(path), "corrected_sources": fixed.corrected_sources,
            "source_count": len(forces), "command_force_n": command,
            "unsigned_original_force_n": np.linalg.norm(forces, axis=1).sum().item(),
            "physical_force_budget_n": fixed.physical_force_budget_n,
            "force_change_bound_n": fixed.force_change_bound_n,
            "moment_change_bound_nm": fixed.moment_change_bound_nm,
            "actual_net_force_change_n": delta.sum(axis=0).tolist(),
            "actual_moment_change_nm": np.cross(points, delta).sum(axis=0).tolist(),
            "maximum_absolute_principal_stress_perturbation_pa": maximum_stress,
            "force_scale_invariance_errors": scale_errors, "finite_incompatible_load_rejected": True,
            "frame_rotation_conformance_passed": True,
            "already_admitted_sources_bitwise_preserved": True, "correction": file_id(output)})
        write_json(args.output / "progress.json", {"cases": rows})
    write_json(args.output / "result.json", {"passed": True, "cases": rows,
        "stress_perturbation_absolute_target_pa": stress_target,
        "original_trajectories_and_packets_unchanged": True, "whole_goal_complete": False,
        "scope": "Actual captured failing inputs, not invented tiny loads. Existing local sixteen-epsilon cone rejection identifies candidate rows; only an aggregate correction below a physical command/unsigned-force roundoff budget is allowed. Finite incompatible loads remain rejected, scales are checked and already-admitted inputs stay bitwise identical. The stress effect is the exact independent linear P2 response of the correction through the known physical outer-centroid transfer with free-link mass balance; it is not a bound for arbitrary continuum traction placement or a field admitted by the original failed Gaussian/SCR fit. Actual every-step path and finite lagged-law inconsistencies still require validation."})


if __name__ == "__main__":
    main()
