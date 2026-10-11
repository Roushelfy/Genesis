"""Independent FP64 geometry audit at the actual rigid/stress call boundary."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
CAPTURE = ROOT / "online-source-geometry-v1/results"
OUT = ROOT / "source-geometry-audit-v1"
OUT.mkdir(exist_ok=True)
with np.load(CAPTURE / "before-recovery-591.npz") as saved:
    geometry = {name: saved[name] for name in saved.files}
with np.load(CAPTURE / "failure/snapshot-591.npz") as saved:
    failure = {name: saved[name] for name in saved.files}
descriptions = json.loads((CAPTURE / "geometry-description.json").read_text())
env, slot = 13, 2
contact = int(geometry["contact_order"][env, slot])
normal = geometry["contact_normal"][env, contact]
record = {"scope": __doc__, "tick": 591, "env": env, "raw_slot": slot, "collider_index": contact,
          "world_normal": normal.tolist(), "geometries": []}
reference_rotation = Rotation.from_quat(failure["frame_quaternion"][env, [1, 2, 3, 0]])
reference_position = failure["frame_position"][env]
for side in ("a", "b"):
    geom = int(geometry[f"contact_geom_{side}"][env, contact])
    start, end = int(geometry["geom_vertex_start"][geom]), int(geometry["geom_vertex_end"][geom])
    rotation = Rotation.from_quat(geometry["geom_quaternion"][env, geom, [1, 2, 3, 0]])
    position = geometry["geom_position"][env, geom]
    local = geometry["vertices_local"][start:end]
    world = rotation.apply(local) + position
    witness = geometry[f"witness_{side}"][env, contact]
    dots, dot = world @ normal, witness @ normal
    gap_low, gap_high = float(dot - dots.min()), float(dots.max() - dot)
    item = {"side": side, "geom": geom, "link": int(geometry["geom_link"][geom]),
            "vertices": len(world), "witness_world": witness.tolist(),
            "min_support_gap_m": min(gap_low, gap_high), "low_support_gap_m": gap_low,
            "high_support_gap_m": gap_high,
            "nearest_vertex_distance_m": float(np.linalg.norm(world - witness, axis=1).min())}
    if int(geometry["geom_link"][geom]) == 12:
        authored = reference_rotation.inv().apply(world - reference_position)
        saved_vertices = np.load(CAPTURE / "source-vertices.npz")["positions"]
        distances = np.linalg.norm(authored[:, None] - saved_vertices[None], axis=2)
        item["frame_and_source_vertex_max_difference_m"] = float(distances.min(axis=1).max())
        assert item["frame_and_source_vertex_max_difference_m"] < 1e-13
        actual_witness = reference_rotation.inv().apply(witness - reference_position)
        item["saved_reference_witness_difference_m"] = float(np.linalg.norm(actual_witness - failure["witness"][env, slot]))
        assert item["saved_reference_witness_difference_m"] < 1e-13
        triangles = np.asarray(descriptions[geom]["faces"], dtype=int)
        face_points = world[triangles]
        faces_normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
        faces_normals /= np.linalg.norm(faces_normals, axis=1)[:, None]
        item["closest_facet_axis_normal_difference"] = float(np.minimum(
            np.linalg.norm(faces_normals - normal, axis=1), np.linalg.norm(faces_normals + normal, axis=1)).min())
    record["geometries"].append(item)
for folder in ("online-surface-fan-v4", "online-surface-fan-gjk-v1", "online-surface-fan-mpr-accuracy-v1"):
    with np.load(ROOT / folder / "results/failure/snapshot-591.npz") as saved:
        record[folder] = {"raw_status": int(saved["raw_status"][env, slot]),
                          "producer_method": int(saved["method"][env, slot]),
                          "normalized_force_error": float(saved["fit_force_error"][env, slot]),
                          "normal": saved["normal"][env, slot].tolist(),
                          "force": saved["force"][env, slot].tolist()}
(OUT / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
