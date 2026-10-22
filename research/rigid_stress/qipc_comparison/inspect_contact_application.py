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
    cli.add_argument("--pressure-geometry", type=Path)
    cli.add_argument("--contacts", type=Path)
    cli.add_argument("--baseline-frame", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [
                file_id(p)
                for p in [args.frame, args.geometry, args.pressure_geometry, args.contacts, args.baseline_frame]
                if p is not None
            ],
            "scope": "Decompose application-to-witness and witness-to-material/pressure-center offsets in a captured stress-input frame. Check raw collider association when supplied. The auxiliary initial boundary query uses post-step geometry and is labelled separately. Normal separation times two is an inferred midpoint depth, checked against the raw penetration when supplied. Preserve actual source and region rejection; no load or acceptance change.",
        },
    )
    with np.load(args.frame) as f:
        frame = {key: f[key] for key in f.files}
    with np.load(args.geometry) as f:
        geometry = {key: f[key] for key in f.files}
    raw = None
    if args.contacts is not None:
        with np.load(args.contacts) as f:
            raw = {key: f[key] for key in f.files}
    replay_keys = [
        "contact_force_n",
        "contact_position_m",
        "contact_normal",
        "source_witness_m",
        "frame_position_m",
        "frame_quaternion",
        "region_status",
        "step_valid",
    ]
    if args.baseline_frame is not None:
        with np.load(args.baseline_frame) as f:
            for key in replay_keys:
                np.testing.assert_array_equal(frame[key], f[key], err_msg=key)
    pressure = (
        {}
        if args.pressure_geometry is None
        else {
            row["contact"]: row
            for row in json.loads(args.pressure_geometry.read_text())["cases"]
            if row["kind"] == "pressure"
        }
    )
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
            "poststep_query_in_input_frame_m": initial_surface.tolist(),
            "witness_poststep_query_distance_m": float(np.linalg.norm(witness - initial_surface)),
            "material_collision_normal_angle_deg": float(
                np.degrees(np.arccos(np.clip(-material_normal @ normal, -1, 1)))
            ),
            "application_to_witness_moment_nm": np.cross(witness - point, force).tolist(),
        }
        if raw is not None:
            index = raw["sort_idx"][env, contact]
            first = material_geom == raw["geom_a"][env, index]
            assert first or material_geom == raw["geom_b"][env, index]
            sign = -1 if first else 1
            actual_witness = raw["witness_a" if first else "witness_b"][env, index]
            np.testing.assert_allclose(rotation.apply(point) + origin, raw["position"][env, index], atol=1e-13, rtol=0)
            np.testing.assert_allclose(rotation.apply(witness) + origin, actual_witness, atol=1e-13, rtol=0)
            np.testing.assert_allclose(rotation.apply(force), sign * raw["force"][env, index], atol=1e-13, rtol=0)
            np.testing.assert_allclose(rotation.apply(normal), -sign * raw["normal"][env, index], atol=1e-13, rtol=0)
            midpoint = (raw["witness_a"][env, index] + raw["witness_b"][env, index]) / 2
            row.update(
                {
                    "raw_contact": int(index),
                    "raw_witness_method": int(raw["witness_method"][env, index]),
                    "raw_penetration_m": float(raw["penetration"][env, index]),
                    "penetration_inference_error_m": float(2 * delta @ normal - raw["penetration"][env, index]),
                    "raw_midpoint_error_m": float(np.linalg.norm(raw["position"][env, index] - midpoint)),
                    "raw_friction": float(raw["friction"][env, index]),
                    "raw_sol_params": raw["sol_params"][env, index].tolist(),
                }
            )
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
            "raw_contact_association_checked": raw is not None,
            "baseline_frame_exact_keys": replay_keys if args.baseline_frame is not None else [],
        },
    )


if __name__ == "__main__":
    main()
