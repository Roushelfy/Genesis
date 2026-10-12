"""Quantify CPU/GPU native sample-count differences without deleting quadrature points."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

root = Path(__file__).parent
destination = root / "source-prior-cloud-comparison-v1/results"
destination.mkdir(parents=True, exist_ok=False)
reports = []
for case in ("step602", "policy876", "balanced_compression"):
    clouds = {}
    for backend in ("cpu", "cuda"):
        with np.load(root / f"source-prior-fields-v3/{backend}/results/{case}-cloud-0.npz") as data:
            clouds[backend] = {name: data[name].copy() for name in data.files}
    rows = []
    for source in (0, 1):
        left = {name: value[clouds["cpu"]["source"] == source] for name, value in clouds["cpu"].items() if name in ("positions", "profiles", "faces", "force")}
        right = {name: value[clouds["cuda"]["source"] == source] for name, value in clouds["cuda"].items() if name in left}
        unmatched = np.zeros(len(right["profiles"]), dtype=bool)
        face_l1 = 0.0
        for face in np.union1d(left["faces"], right["faces"]):
            cpu_indices, gpu_indices = left["faces"] == face, right["faces"] == face
            face_l1 += abs(left["profiles"][cpu_indices].sum() - right["profiles"][gpu_indices].sum())
            if cpu_indices.any():
                distances = cKDTree(left["positions"][cpu_indices]).query(right["positions"][gpu_indices])[0]
                unmatched[gpu_indices] = distances > 1e-12
            else:
                unmatched[gpu_indices] = True
        total = float(right["profiles"].sum())
        rows.append({"source": source, "cpu_points": len(left["profiles"]), "gpu_points": len(right["profiles"]),
                     "cpu_profile_integral": float(left["profiles"].sum()), "gpu_profile_integral": total,
                     "per_face_integral_l1_relative_difference": float(face_l1 / total),
                     "gpu_unmatched_point_count": int(unmatched.sum()),
                     "gpu_unmatched_profile_fraction": float(right["profiles"][unmatched].sum() / total),
                     "gpu_unmatched_sample_force_magnitude_N": float(np.linalg.norm(right["force"][unmatched], axis=1).sum())})
    reports.append({"case": case, "sources": rows,
                    "interpretation": "Weighted differences are measured,not discarded. This does not alone establish the precise floating-point branch causing extra points or justify a production cutoff."})
(destination / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2), flush=True)
