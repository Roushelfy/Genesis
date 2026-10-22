"""Check actual friction snapshots with independent supplied-load P2 elasticity."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .check_native_hertz_recovery import discrete_particular, material_points, matrix
from .hertz_tangential import cerruti_stress, elliptic_tangential_stress
from .p2_elastic import FreeElasticP2


def particular(points, axes, peak, force, friction, coefficients):
    result = discrete_particular(points, axes, peak, force, True)
    for index, coefficient in enumerate(coefficients):
        if not np.any(coefficient):
            continue
        radius = (index + 1) / 16
        semiaxes = radius * axes
        tangent = friction * coefficient
        far = np.linalg.norm(points, axis=1) >= 8 * semiaxes[0]
        result[~far] += elliptic_tangential_stress(points[~far], semiaxes, tangent, peak=radius * peak, order=16)
        sources = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * semiaxes / np.sqrt(5)
        result[far] += cerruti_stress(points[far], sources, np.broadcast_to(force * radius**3 * tangent / 4, (4, 2)))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.control / "mesh.npz") as data:
        vertices, tets = data["vertices"], data["tetrahedra"]
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    paths = sorted(args.control.glob("full*.npz"), key=lambda p: int(p.stem[4:]))
    assert paths
    rows, bindings = [], [file_id(args.control / "mesh.npz")]
    for path in [paths[0], paths[len(paths) // 2], paths[-1]]:
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
        corners_expected, probes_expected = [], np.zeros_like(actual_probes)
        for env in range(batch):
            active = [patch for patch in range(env, len(forces), batch) if forces[patch] > 0]
            applied = np.empty_like(loads[env])
            applied[permutation] = loads[env]
            omega = data["omega_per_s"][env]
            centrifugal = np.cross(omega, np.cross(cpu.reference.coordinates - cpu.reference.com, omega))
            applied += (cpu.reference.mass @ centrifugal.reshape(-1)).reshape(-1, 3)
            regular, _, _, _ = cpu.solve(applied)
            corners = regular.copy()
            query = np.concatenate([data["probe_position_m"][patch] for patch in active])
            elements = np.concatenate([data["probe_elements"][patch] for patch in active])
            bary = np.concatenate([data["probe_barycentric"][patch] for patch in active])
            expected = np.einsum("nc,ncij->nij", bary, regular[elements])
            for index, patch in enumerate(active):
                center, basis = data["hertz_centers_m"][patch], bases[patch]
                local = material_points(query, center, basis)
                local[index * 89 : (index + 1) * 89] = data["probe_offsets"] * np.r_[axes[patch], axes[patch, 0]]
                field_args = (
                    axes[patch],
                    peak[patch],
                    forces[patch],
                    data["friction_input_friction"][patch],
                    data["friction_coefficients"][patch],
                )
                expected += basis @ particular(local, *field_args) @ basis.T
                local = material_points(cpu.reference.corner_positions.reshape(-1, 3), center, basis)
                corners += (basis @ particular(local, *field_args) @ basis.T).reshape(corners.shape)
            actual = np.concatenate([actual_probes[patch] for patch in active])
            field_error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1))
            corner_error = float(np.linalg.norm(actual_corners[env] - corners) / max(np.linalg.norm(corners), 1))
            assert max(field_error, corner_error) < 1e-7, (path.name, env, field_error, corner_error)
            for index, patch in enumerate(active):
                probes_expected[patch] = expected[index * 89 : (index + 1) * 89]
            corners_expected.append(corners)
            rows.append(
                {"frame": path.stem, "env": env, "probe_relative": field_error, "corner_relative": corner_error}
            )
        np.savez_compressed(
            args.output / (path.stem + "-reference.npz"),
            probe_tensor_pa=probes_expected,
            corner_tensor_pa=np.stack(corners_expected),
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "bindings": bindings,
            "scope": "Independent P2 derivatives, consistent-mass relief and stress under the captured native regular RHS plus declared normal/radial tangential particulars. Does not independently establish pressure, friction history, point residual boundary approximation, moving-footprint physical accuracy or full Franka accuracy.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
