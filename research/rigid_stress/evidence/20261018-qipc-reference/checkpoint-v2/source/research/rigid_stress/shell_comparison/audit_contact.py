"""Audit deformed contact geometry, the measured impulse ledger and exact PR elastic energy."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..finite_traction_reference import triangle_closest_points
from .common import file_identity, write_json


def geometry_audit(receipt):
    metadata = json.loads(receipt.read_text())
    with np.load(receipt.parent / "trajectory.npz") as saved:
        sheets = saved["sheet_positions"]
        spheres = saved["sphere_positions"]
        impulses = saved["impulses"] if "impulses" in saved and saved["impulses"].size else None
        if "faces" in saved:
            faces = saved["faces"]
            topology = "Recorded public get_faces output"
        else:
            mesh = trimesh.load(receipt.parent / "sheet.obj", process=False)
            distance, order = cKDTree(sheets[0, 0]).query(mesh.vertices)
            spacing = cKDTree(mesh.vertices).query(mesh.vertices, k=2)[0][:, 1].min()
            assert len(np.unique(order)) == len(order) == len(mesh.vertices)
            assert distance.max() < 0.01 * spacing
            faces = order[mesh.faces]
            topology = (
                "OBJ connectivity matched uniquely to the first recorded pose; no fracture/topology changes enabled"
            )
        sample_steps = [step for step in range(len(spheres)) if step % 10 == 0 or step == len(spheres) - 1]
        assert len(sample_steps) == len(sheets)
        rows = []
        for frame, step in enumerate(sample_steps):
            for env in range(spheres.shape[1]):
                triangles = sheets[frame, env, faces].astype(np.float64)
                center = spheres[step, env].astype(np.float64)
                try:
                    points, distance_squared = triangle_closest_points(triangles, center)
                except np.linalg.LinAlgError as error:
                    rows.append({"step": step, "env": env, "geometry_failure": str(error)})
                    continue
                face = distance_squared.argmin()
                normal = np.cross(triangles[face, 1] - triangles[face, 0], triangles[face, 2] - triangles[face, 0])
                normal /= np.linalg.norm(normal)
                distance = np.sqrt(distance_squared[face])
                vertex_distance = np.linalg.norm(sheets[frame, env] - center, axis=1)
                candidate_count = int((vertex_distance < 0.0105 + 0.002 * np.log(10)).sum())
                actual_count = (
                    int((np.linalg.norm(impulses[step, :, env], axis=1) > 0).sum()) if impulses is not None else None
                )
                rows.append(
                    {
                        "step": step,
                        "env": env,
                        "closest_face": int(face),
                        "finite_thickness_overlap_m": float(0.0105 - distance),
                        "oriented_closest_side_distance_m": float((center - points[face]) @ normal),
                        "vertex_candidates_at_recorded_pose": candidate_count,
                        "measured_nonzero_impulse_vertices": actual_count,
                        "sphere_center_m": center.tolist(),
                        "closest_point_m": points[face].tolist(),
                    }
                )
    return {
        "source": file_identity(receipt),
        "topology": topology,
        "samples": rows,
        "timing": "Geometry uses post-control-step public poses; impulses belong to the coupling solve before shell integration. No continuous-collision or exact simultaneity claim.",
        "height_marker_correction": "Historical fell_through fields in raw receipts are only a below-initial-plane height marker. Geometric evidence here determines passage/penetration.",
    }


def impulse_ledger(receipt, output):
    metadata = json.loads(receipt.read_text())
    dt, mass = metadata["args"]["dt"], metadata["sphere_mass_kg"]
    assert metadata["args"]["substeps"] == 1 and metadata["args"]["motion"] == "drop"
    with np.load(receipt.parent / "trajectory.npz") as saved:
        # Accumulate the logged FP32 samples in FP64 to expose their numerical floor.
        velocity = saved["sphere_velocities"].astype(np.float64)
        position = saved["sphere_positions"].astype(np.float64)
        angular = saved["sphere_angular_velocities"].astype(np.float64)
        impulses = saved["impulses"].astype(np.float64)
        points = saved["contact_points"].astype(np.float64)
        inertia = saved["sphere_inertia_kg_m2"].reshape(-1, 3, 3)[0].astype(np.float64)
        shell_energy = saved["shell_energy_j"].astype(np.float64)
        initial = saved["initial_total_energy_j"].astype(np.float64)
    isotropic = np.trace(inertia) / 3
    assert np.linalg.norm(inertia - isotropic * np.eye(3)) <= 1e-12
    impulse = impulses.sum(axis=1)
    moment_impulse = np.cross(points, impulses).sum(axis=1)
    gravity_impulse = mass * np.array([0.0, 0.0, -9.81]) * dt
    delta_p = mass * np.diff(velocity, axis=0) - gravity_impulse
    same = delta_p + impulse[1:]
    lag = delta_p + impulse[:-1]
    angular_momentum = isotropic * angular + np.cross(position, mass * velocity)
    angular_residual = (
        np.diff(angular_momentum, axis=0) - np.cross(position[:-1], gravity_impulse) + moment_impulse[:-1]
    )
    sphere_energy = (
        0.5 * mass * (velocity**2).sum(axis=2)
        + 0.5 * isotropic * (angular**2).sum(axis=2)
        + mass * 9.81 * position[..., 2]
    )
    total_energy = sphere_energy + shell_energy.sum(axis=2)
    np.savez_compressed(
        output / "impulse-energy-ledger.npz",
        impulse_n_s=impulse,
        moment_impulse_about_world_origin_nm_s=moment_impulse,
        same_step_linear_residual_n_s=same,
        lagged_linear_residual_n_s=lag,
        lagged_angular_residual_nm_s=angular_residual,
        shell_energy_j=shell_energy,
        total_energy_j=total_energy,
        initial_total_energy_j=initial,
    )
    return {
        "source": file_identity(receipt),
        "maximum_same_step_linear_residual_n_s": float(np.linalg.norm(same, axis=2).max()),
        "maximum_lagged_linear_residual_n_s": float(np.linalg.norm(lag, axis=2).max()),
        "maximum_contact_impulse_n_s": float(np.linalg.norm(impulse, axis=2).max()),
        "maximum_lagged_world_origin_angular_residual_nm_s": float(np.linalg.norm(angular_residual, axis=2).max()),
        "maximum_contact_moment_impulse_nm_s": float(np.linalg.norm(moment_impulse, axis=2).max()),
        "maximum_energy_increase_over_initial_j": float((total_energy - initial).max()),
        "maximum_total_energy_j": float(total_energy.max()),
        "initial_total_energy_j": initial.tolist(),
        "energy_scope": "Exact PR SVK membrane plus hinge energy, shell/sphere kinetic and gravitational potential. Fixed boundaries have zero velocity; damping/friction remain active. No energy assigned to pending coupling-force buffers.",
        "moment_convention": "About the world origin, using logged shell coupling points and public sphere center/velocities; report residual without assuming the splitting conserves angular momentum.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.runs / "diagnostic-controls-v2/group0/drop-n2/result.json",
        args.runs / "diagnostic-controls-v2/group0/drop-n16/result.json",
        args.runs / "contact-ledger-v1/group0/drop-n16-ledger/result.json",
    ]
    geometry = [geometry_audit(path) for path in paths]
    ledger = impulse_ledger(paths[-1], args.output)
    write_json(
        args.output / "result.json",
        {"kind": "deformed-contact-and-conservation-audit", "geometry": geometry, "ledger": ledger},
    )


if __name__ == "__main__":
    main()
