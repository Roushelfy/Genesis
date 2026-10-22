"""Attribute a captured native Franka refusal to its actual geometry checks."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.frame) as saved:
        data = {key: saved[key] for key in saved.files}
    rows = []
    batch = len(data["input_status"])
    for patch in np.flatnonzero(data["law_force_n"] > 0):
        anchor = int(data["hertz_anchors"][patch])
        center, basis = data["hertz_centers_m"][patch], data["hertz_frames"][patch]
        row = {
            "patch": int(patch),
            "environment": int(patch % batch),
            "anchor": anchor,
            "normal_force_n": float(data["law_force_n"][patch]),
            "axes_m": data["law_axes_m"][patch].tolist(),
            "orthonormal_error": float(np.linalg.norm(basis.T @ basis - np.eye(3))),
            "determinant_error": float(abs(np.linalg.det(basis) - 1)),
            "minimum_material_depth_m": float(((data["material_nodes_m"] - center) @ basis[:, 2]).min()),
        }
        if anchor >= 0:
            triangle = data["material_nodes_m"][data["material_surface_nodes"][:3, anchor]]
            normal = data["material_face_normal"][anchor]
            edges = triangle[1:] - triangle[0]
            local = np.linalg.solve(edges @ edges.T, edges @ (center - triangle[0]))
            inward_alignment = float(-normal @ basis[:, 2])
            row.update(
                center_plane_distance_m=float(abs((center - triangle[0]) @ normal)),
                face_barycentric=np.r_[1 - local.sum(), local].tolist(),
                inward_normal_alignment=inward_alignment,
                normal_angle_degrees=float(np.degrees(np.arccos(np.clip(inward_alignment, -1, 1)))),
                original_normal_gate_passed=inward_alignment > 1 - 1e-10,
            )
        rows.append(row)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "frame": file_id(args.frame),
            "input_status": data["input_status"].tolist(),
            "recovery_status": data["recovery_status"].tolist(),
            "native_step_valid": data["step_valid"].tolist(),
            "active_regions": rows,
            "scope": "Read-only geometry attribution of the original refused snapshot. Original1e-10 frame-normal gate and source identity are retained. Inspection completion supplies no admitted stress update or physical accuracy claim.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
