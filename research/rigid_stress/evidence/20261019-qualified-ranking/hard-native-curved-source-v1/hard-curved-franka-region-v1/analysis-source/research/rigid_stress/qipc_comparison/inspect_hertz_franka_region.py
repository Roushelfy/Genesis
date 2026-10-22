"""Attribute captured finite-box contact-region rejection without rerunning dynamics."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.geometry) as saved:
        geometry = {key: saved[key] for key in saved.files}
    is_loaded = frame["contact_valid"] & (np.linalg.norm(frame["contact_force_n"], axis=2) > 0)
    rows = []
    for i_b, i_c in zip(*np.nonzero(is_loaded & (frame["region_status"] != 0))):
        initial = geometry["initial"][i_b, i_c]
        i_g = int(initial[1, 0])
        row = {
            "environment": int(i_b),
            "contact": int(i_c),
            "opponent_geom": i_g,
            "opponent_type": int(initial[1, 1]),
            "region_status": int(frame["region_status"][i_b, i_c]),
            "geometry_kind": int(geometry["region_geometry_kind"][i_b, i_c]),
            "normal_residual": float(geometry["region_residual"][i_b, i_c]),
            "curvature_eigenvalues_per_m": np.linalg.eigvalsh(frame["region_curvature_per_m"][i_b, i_c]).tolist(),
            "force_n": frame["contact_force_n"][i_b, i_c].tolist(),
        }
        if row["opponent_type"] == 5:
            quaternion = geometry["geom_quaternion"][i_b, i_g]
            rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
            center = rotation.apply(
                frame["region_center_m"][i_b, i_c] - geometry["geom_position"][i_b, i_g], inverse=True
            )
            normal = rotation.apply(-geometry["region_normal"][i_b, i_c], inverse=True)
            i_axis = int(np.argmax(np.abs(normal)))
            half = geometry["geom_data"][i_g, :3] / 2
            margin = half - np.abs(center)
            is_tangential = np.arange(3) != i_axis
            row.update(
                selected_box_axis=i_axis,
                box_half_extents_m=half.tolist(),
                region_center_in_box_m=center.tolist(),
                box_outward_normal_local=normal.tolist(),
                tangent_face_margins_m=margin[is_tangential].tolist(),
                finite_face_rejected=bool(
                    np.any(margin[is_tangential] <= 128 * np.finfo(float).eps * np.linalg.norm(half))
                ),
            )
        rows.append(row)
    assert rows
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "frame": file_id(args.frame),
            "geometry": file_id(args.geometry),
            "rejected_regions": rows,
            "scope": "Read-only attribution of the captured native local normal root and finite opponent face gate. This inspection retains the original failed stress step and supplies no accepted throughput.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
