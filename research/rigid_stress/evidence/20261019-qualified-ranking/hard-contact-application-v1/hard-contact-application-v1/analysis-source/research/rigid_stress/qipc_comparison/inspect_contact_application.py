"""Separate captured rigid application points, collision witnesses and physical pressure centers."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .study_edge_wrench_allocation import ground_pressure_geometry
from .study_far_edge_contact import projected_surface


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--pressure-geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame, args.geometry, args.pressure_geometry]],
            "scope": "Decompose application-to-witness and witness-to-material/pressure-center offsets in an existing same-frame capture. Normal separation times two is an inferred midpoint depth, cross-checked against the geometry when available. Preserve actual source and region rejection; no load or acceptance change.",
        },
    )
    with np.load(args.frame) as f:
        frame = {key: f[key] for key in f.files}
    with np.load(args.geometry) as f:
        geometry = {key: f[key] for key in f.files}
    pressure = {
        row["contact"]: row
        for row in json.loads(args.pressure_geometry.read_text())["cases"]
        if row["kind"] == "pressure"
    }
    pressure[0] = ground_pressure_geometry(frame)
    nodes = frame["material_nodes_m"][frame["material_surface_nodes"].T]
    rows = []
    active = frame["contact_valid"] & (np.linalg.norm(frame["contact_force_n"], axis=2) > 0)
    for env, contact in np.argwhere(active):
        point, witness, normal, force = [
            frame[key][env, contact]
            for key in ["contact_position_m", "source_witness_m", "contact_normal", "contact_force_n"]
        ]
        normal /= np.linalg.norm(normal)
        axis = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        axis /= np.linalg.norm(axis)
        plane = np.stack([axis, np.cross(normal, axis)])
        centers, normals, anchors = projected_surface(
            nodes, frame["material_face_normal"], witness[None], plane, -normal, witness, return_geometry=True
        )
        material, material_normal = centers[0], normals[0]
        delta = point - witness
        quaternion = frame["frame_quaternion"][env]
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        origin = frame["frame_position_m"][env]
        pair_geometry = geometry["initial"][env, contact]
        material_geom = int(frame["source_geom"][env, contact])
        initial_surface = rotation.inv().apply(pair_geometry[0, 6:9] - origin)
        row = {
            "environment": int(env),
            "contact": int(contact),
            "pair": int(frame["source_pair"][env, contact]),
            "material_geom": material_geom,
            "application_m": point.tolist(),
            "witness_m": witness.tolist(),
            "material_projection_m": material.tolist(),
            "anchor": int(anchors[0]),
            "source_status": int(frame["source_status"][env, contact]),
            "region_status": int(frame["region_status"][env, contact]),
            "application_minus_witness_m": delta.tolist(),
            "normal_midpoint_offset_m": float(delta @ normal),
            "transverse_midpoint_offset_m": float(np.linalg.norm(delta - (delta @ normal) * normal)),
            "inferred_midpoint_penetration_m": float(2 * delta @ normal),
            "witness_material_distance_m": float(np.linalg.norm(witness - material)),
            "initial_collision_surface_m": initial_surface.tolist(),
            "witness_initial_surface_distance_m": float(np.linalg.norm(witness - initial_surface)),
            "material_collision_normal_angle_deg": float(
                np.degrees(np.arccos(np.clip(-material_normal @ normal, -1, 1)))
            ),
            "application_to_witness_moment_nm": np.cross(witness - point, force).tolist(),
        }
        if env == 0 and contact in pressure:
            target = np.array(pressure[contact]["center"])
            basis = np.array(pressure[contact]["basis"])
            shift = target - witness
            row.update(
                {
                    "pressure_center_m": target.tolist(),
                    "witness_to_pressure_local_m": (shift @ basis).tolist(),
                    "witness_to_pressure_moment_nm": np.cross(shift, force).tolist(),
                    "application_to_pressure_moment_nm": np.cross(target - point, force).tolist(),
                    "pressure_width_m": pressure[contact]["width"],
                    "pressure_half_length_m": pressure[contact]["along"],
                }
            )
        rows.append(row)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "maximum_capture_pose_position_difference_m": float(
                np.max(np.linalg.norm(frame["position_m"] - frame["frame_position_m"], axis=1))
            ),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
