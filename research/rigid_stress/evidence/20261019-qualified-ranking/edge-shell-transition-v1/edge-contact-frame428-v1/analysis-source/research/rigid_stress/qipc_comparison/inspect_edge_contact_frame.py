"""Compare a captured sharp region's material and opposing-face traction frames."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--environment", type=int, default=0)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.hard, args.frame, args.boundaries]],
        },
    )
    with np.load(args.hard) as saved:
        hard = {key: saved[key] for key in saved.files}
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.boundaries) as saved:
        boundary = {key: saved[key] for key in saved.files}
    env = args.environment
    rotation = Rotation.from_quat(frame["frame_quaternion"][env][[1, 2, 3, 0]])
    origin = frame["frame_position_m"][env]
    batch = hard["profile"].shape[0]
    rows = []
    for region in np.flatnonzero(hard["source_count"][env] > 0):
        assert hard["profile"][env, region] == 2
        patch = region * batch + env
        material_basis = hard["basis"][patch]
        face_basis = hard["geometry_basis"][env, region] @ np.diag([1, -1, -1])
        np.testing.assert_allclose(face_basis.T @ face_basis, np.eye(3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(face_basis), 1, rtol=0, atol=1e-12)
        normal = rotation.inv().apply(boundary["normal"][env, region])
        np.testing.assert_allclose(face_basis[:, 2], -normal, rtol=0, atol=1e-12)
        tangent = rotation.inv().apply(boundary["tangent"][env, region])
        plane = np.column_stack([tangent, np.cross(normal, tangent)])
        root = rotation.inv().apply(frame["region_center_m"][env, region] - origin)
        minimum = hard["minimum_m"][env, region]
        face_origin = minimum + normal * ((root - minimum) @ normal)
        force = hard["allocation_original"][env, region]
        moment = hard["original_moment_nm"][env, region]
        local_force = force @ face_basis
        local_moment = (moment - np.cross(face_origin, force)) @ face_basis
        assert local_force[2] > 0
        centroid = np.array([-local_moment[1], local_moment[0]]) / local_force[2]
        required_point = face_origin + face_basis[:, :2] @ centroid
        original_point = hard["center_m"][patch]
        count = boundary["count"][env, region]
        bounds = boundary["planes"][env, region, :count]
        margin = bounds[:, 2] - bounds[:, :2] @ ((required_point - root) @ plane)
        friction = hard["allocation_friction"][env, region]
        material_force = force @ material_basis
        original_moment = (moment - np.cross(original_point, force)) @ face_basis
        material_height = (original_point - face_origin) @ face_basis[:, 2]
        mapped_moment = original_moment + np.cross((original_point - face_origin) @ face_basis, local_force)
        np.testing.assert_allclose(mapped_moment, local_moment, rtol=1e-10, atol=1e-17)
        contacts = np.flatnonzero(
            (hard["leader"][env] == region)
            & frame["contact_valid"][env]
            & (np.linalg.norm(frame["contact_force_n"][env], axis=1) > 0)
        )
        position = (frame["contact_position_m"][env, contacts] - face_origin) @ face_basis
        forces = frame["contact_force_n"][env, contacts] @ face_basis
        np.testing.assert_allclose(forces.sum(axis=0), local_force, rtol=1e-12, atol=1e-15)
        rows.append(
            {
                "region": int(region),
                "contacts": contacts.tolist(),
                "opposing_face_force_n": local_force.tolist(),
                "opposing_face_moment_nm": local_moment.tolist(),
                "opposing_face_shear_normal_ratio": float(np.linalg.norm(local_force[:2]) / local_force[2]),
                "opposing_face_normalized_cone_margin": float(
                    (friction * local_force[2] - np.linalg.norm(local_force[:2])) / np.linalg.norm(force)
                ),
                "material_shear_normal_ratio": float(np.linalg.norm(material_force[:2]) / material_force[2]),
                "friction": float(friction),
                "normal_rotation_degrees": float(
                    np.degrees(np.arccos(np.clip(face_basis[:, 2] @ material_basis[:, 2], -1, 1)))
                ),
                "material_pressure_center_height_from_opposing_face_m": float(material_height),
                "collision_positions_from_opposing_edge_m": position.tolist(),
                "collision_forces_in_opposing_frame_n": forces.tolist(),
                "required_pressure_centroid_from_opposing_edge_m": centroid.tolist(),
                "required_centroid_face_margins_m": margin.tolist(),
                "required_centroid_inside_opposing_face": bool(np.all(margin >= 0)),
                "frame_change_preserves_force_and_moment": True,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Same captured force and moment expressed in the undeformed material frame and contact-time rigid opposing-face frame. A nonnegative normal-pressure distribution on one planar face requires its centroid inside that face. Changing frame alone preserves the full wrench. Deformation, contact on adjacent box faces, pressure physics and any nearby-load correction require separate checks. No native gates or fields change.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
