"""Inspect retained radius refusals using the exact collision witnesses and geom frames."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--failure", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=False), "failure": file_id(args.failure)},
    )
    with np.load(args.failure) as saved:
        data = {name: saved[name] for name in saved.files}
    failed = data["valid"] & (np.linalg.norm(data["force"], axis=-1) > 1e-30) & (data["status"] != 0)
    kinds = {
        int(gs.GEOM_TYPE.BOX): "box",
        int(gs.GEOM_TYPE.SPHERE): "sphere",
        int(gs.GEOM_TYPE.PLANE): "plane",
        int(gs.GEOM_TYPE.MESH): "mesh",
        int(gs.GEOM_TYPE.CYLINDER): "cylinder",
        int(gs.GEOM_TYPE.CAPSULE): "capsule",
    }
    rows = []
    for env, slot in np.argwhere(failed):
        source = data["contact_sort_idx"][env, slot]
        row = {
            "environment": int(env),
            "slot": int(slot),
            "source": int(source),
            "status": int(data["status"][env, slot]),
            "radius_status": int(data["radius_status"][env, slot]),
            "source_force_n": data["source_force"][env, source].tolist(),
            "source_normal": data["source_normal"][env, source].tolist(),
            "geoms": [],
        }
        for side in ["a", "b"]:
            index = data["source_geom_" + side][env, source]
            point = data["witness_point_" + side][env, source]
            quaternion = data["geoms_quat"][env, index]
            local = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).inv().apply(point - data["geoms_pos"][env, index])
            kind = data["geoms_type"][index]
            geom = {
                "side": side,
                "index": int(index),
                "kind": kinds.get(int(kind), str(kind)),
                "witness_vertex": int(data["witness_vertex_" + side][env, source]),
                "witness_local_m": local.tolist(),
                "parameters": data["geoms_data"][index].tolist(),
            }
            if kind == gs.GEOM_TYPE.BOX:
                half = 0.5 * data["geoms_data"][index][:3]
                projected = np.clip(local, -half, half)
                tolerance = 128 * np.finfo(float).eps * np.linalg.norm(half)
                if np.linalg.norm(projected - local) <= tolerance:
                    gaps = half - np.abs(local)
                    axis = gaps.argmin()
                    projected[axis] = np.copysign(half[axis], local[axis])
                geom["projected_box_point_m"] = projected.tolist()
                geom["boundary_coordinate_count"] = int(np.count_nonzero(np.abs(np.abs(projected) - half) <= tolerance))
                geom["signed_distance_to_box_faces_m"] = (half - np.abs(local)).tolist()
            row["geoms"].append(geom)
        rows.append(row)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "failed_contacts": rows,
            "scope": "Diagnostic geometry only. No contact, radius, force or numerical threshold is changed.",
        },
    )


if __name__ == "__main__":
    main()
