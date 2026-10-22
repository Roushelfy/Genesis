"""Compare actual native contact snapshots with independent P2/Hertz fields."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .check_native_hertz_recovery import discrete_particular, material_points, matrix
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.control / "mesh.npz") as data:
        vertices, tets = data["vertices"], data["tetrahedra"]
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    rows, bindings = [], [file_id(args.control / "mesh.npz")]
    for path in sorted(args.control.glob("frame*.npz"), key=lambda p: int(p.stem[5:])):
        with np.load(path) as values:
            data = {key: values[key] for key in values}
        bindings.append(file_id(path))
        loads = data["regular_force_n"]
        batch = len(loads)
        permutation = cpu.reference.native_dof_permutation(data["material_nodes_m"]).reshape(-1, 3)[:, 0] // 3
        forces, axes, peak = data["law_force_n"], data["law_axes_m"], data["law_pressure_pa"]
        bases = data["hertz_frames"].copy()
        for patch in np.flatnonzero(forces):
            x, y = data["law_direction"][patch]
            bases[patch] = bases[patch] @ np.array([[x, -y, 0], [y, x, 0], [0, 0, 1]])
        actual_probes, actual_corners = matrix(data["probe_tensor_pa"]), matrix(data["corner_tensor_pa"])
        frame_expected, corner_expected = np.zeros_like(actual_probes), []
        for env in range(batch):
            active = [patch for patch in range(env, len(forces), batch) if forces[patch] > 0]
            applied = np.empty_like(loads[env])
            applied[permutation] = loads[env]
            omega = data["omega_per_s"][env]
            centrifugal = np.cross(omega, np.cross(cpu.reference.coordinates - cpu.reference.com, omega))
            applied += (cpu.reference.mass @ centrifugal.reshape(-1)).reshape(-1, 3)
            regular, _, _, _ = cpu.solve(applied)
            corners = regular.copy()
            query = (
                np.concatenate([data["probe_position_m"][patch] for patch in active]) if active else np.empty((0, 3))
            )
            elements = (
                np.concatenate([data["probe_elements"][patch] for patch in active])
                if active
                else np.empty(0, dtype=int)
            )
            bary = (
                np.concatenate([data["probe_barycentric"][patch] for patch in active]) if active else np.empty((0, 4))
            )
            expected = np.einsum("nc,ncij->nij", bary, regular[elements])
            for index, patch in enumerate(active):
                center, basis = data["hertz_centers_m"][patch], bases[patch]
                local = material_points(query, center, basis)
                local[index * 89 : (index + 1) * 89] = data["probe_offsets"] * np.r_[axes[patch], axes[patch, 0]]
                particular = discrete_particular(local, axes[patch], peak[patch], forces[patch], True)
                expected += basis @ particular @ basis.T
                local = material_points(cpu.reference.corner_positions.reshape(-1, 3), center, basis)
                particular = discrete_particular(local, axes[patch], peak[patch], forces[patch], True)
                corners += (basis @ particular @ basis.T).reshape(corners.shape)
                radius = (3 * forces[patch] * 0.003 / (4 * (1e10 / 0.91))) ** (1 / 3)
                np.testing.assert_allclose(axes[patch], radius, rtol=1e-7, atol=1e-15)
                np.testing.assert_allclose(
                    2 * np.pi * np.prod(axes[patch]) * peak[patch] / 3, forces[patch], rtol=1e-12
                )
            actual = np.concatenate([actual_probes[patch] for patch in active]) if active else np.empty((0, 3, 3))
            field_error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1))
            corner_error = float(np.linalg.norm(actual_corners[env] - corners) / max(np.linalg.norm(corners), 1))
            assert max(field_error, corner_error) < 1e-7, (path.name, env, field_error, corner_error)
            tensors = np.concatenate([actual_corners[env].reshape(-1, 3, 3), actual])
            tensile, vm = stress_measures(tensors)
            np.testing.assert_allclose(
                [data["vm_pa"][env], data["principal_pa"][env]], [vm.max(), tensile.max()], rtol=1e-7, atol=1e-7
            )
            for index, patch in enumerate(active):
                frame_expected[patch] = expected[index * 89 : (index + 1) * 89]
            corner_expected.append(corners)
            rows.append(
                {
                    "frame": path.stem,
                    "env": env,
                    "active_regions": len(active),
                    "probe_relative": field_error,
                    "corner_relative": corner_error,
                    "native_vm_pa": float(data["vm_pa"][env]),
                    "native_principal_pa": float(data["principal_pa"][env]),
                }
            )
        np.savez_compressed(
            args.output / (path.stem + "-reference.npz"),
            probe_tensor_pa=frame_expected,
            corner_tensor_pa=np.stack(corner_expected),
        )
        write_json(args.output / "progress.json", {"rows": rows})
    assert rows and any(row["active_regions"] for row in rows)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "bindings": bindings,
            "same_load_independent_p2_fields": True,
            "analytical_sphere_halfspace_support_checked": True,
            "general_curved_material_friction_or_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
