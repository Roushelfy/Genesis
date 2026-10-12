"""Offline report of original immutable cloud weights and feasible LP load concentration."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent
out = {}
for label in ("b2048-623102", "b4096-623103"):
    run = root / f"native-body-failure-audit-v3/{label}"
    with np.load(run / "results/cloud.npz") as saved:
        cloud = {key: saved[key].copy() for key in saved.files}
    profiles, faces = cloud["profiles"], cloud["faces"]
    weights = profiles / profiles.sum()
    print(json.dumps({"case": label, "weights_min_max": [float(weights.min()), float(weights.max())]}), flush=True)
    forces = None
    if (run / "results/inner-primal.npz").exists():
        with np.load(run / "results/inner-primal.npz") as primal:
            forces = primal["force"].copy()
    rows = []
    for face in np.unique(faces):
        selected = faces == face
        row = {"face": int(face), "count": int(selected.sum()), "weight": float(weights[selected].sum()),
               "normal": cloud["normals"][selected][0].tolist()}
        if forces is not None:
            row["lp_force_norm_sum"] = float(np.linalg.norm(forces[selected], axis=1).sum())
        rows.append(row)
    output = {"case": label, "weights_min_max": [float(weights.min()), float(weights.max())], "faces": rows}
    if forces is not None:
        target_force, target_moment, radius = cloud["force"], cloud["moment"], float(cloud["radius"])
        recovered_moment = np.cross(cloud["positions"] - cloud["anchor"], forces).sum(axis=0)
        output["force_error"] = float(np.linalg.norm(forces.sum(axis=0) - target_force) / np.linalg.norm(target_force))
        output["moment_error_original_point"] = float(np.linalg.norm(recovered_moment - target_moment - np.cross(cloud["point"] - cloud["anchor"], forces.sum(axis=0) - target_force)) / radius / np.linalg.norm(target_force))
        output["lp_objective"] = float(np.sum((forces - weights[:, None] * target_force) ** 2 / weights[:, None]) / 2)
    out[label] = output
    print(json.dumps(output, indent=2), flush=True)
(root / "failure-cloud-weight-analysis-v1.json").write_text(json.dumps(out, indent=2) + "\n")
