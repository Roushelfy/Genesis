"""Diagnose reference-point and force/moment error scales without acceptance changes."""

import json
from pathlib import Path

import numpy as np

from research.rigid_stress.mechanics import P2Shell

root = Path(__file__).parent
run = root / "native-selected-wrench-oracle-v1/b2048-623102"
with np.load(run / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
    model = P2Shell(asset["vertices"], asset["tetrahedra"], asset["surface_triangles"], 1e10, .3, 2000, 2, factor_backend="none")
rows = []
for path in sorted(run.glob("fields-*.npz")):
    with np.load(path) as saved:
        valid = saved["contact_valid"]
        force = np.where(valid[:, :, None], saved["contact_force"], 0)
        point = np.where(valid[:, :, None], saved["contact_position"], 0)
        free = np.where(valid[:, :, None], saved["contact_free_moment"], 0)
        magnitude = np.linalg.norm(force, axis=2)
        scale = np.maximum(1e-30, magnitude.sum(axis=1))
        radius_scale = np.maximum(1e-30, (magnitude * saved["contact_radius"] + np.linalg.norm(free, axis=2)).sum(axis=1))
        original_force = force.sum(axis=1)
        original_moment = (np.cross(point, force) + free).sum(axis=1)
        actual_force = saved["force"].sum(axis=1)
        actual_moment = np.cross(model.xyz[None], saved["force"]).sum(axis=1)
        force_error = np.linalg.norm(actual_force - original_force, axis=1)
        moment_error = np.linalg.norm(actual_moment - original_moment, axis=1)
        reference = (point * magnitude[:, :, None]).sum(axis=1) / scale[:, None]
        centered_error = np.linalg.norm(actual_moment - original_moment - np.cross(reference, actual_force - original_force), axis=1)
        for index in np.flatnonzero((force_error / scale > 1e-8) | (moment_error / radius_scale > 1e-8)):
            rows.append({"sample": path.name, "env": int(saved["envs"][index]), "contacts": int(valid[index].sum()),
                         "force_error_N": float(force_error[index]), "moment_error_Nm": float(moment_error[index]),
                         "force_ratio": float(force_error[index] / scale[index]), "moment_radius_ratio": float(moment_error[index] / radius_scale[index]),
                         "centered_moment_radius_ratio": float(centered_error[index] / radius_scale[index]),
                         "nodal_force": actual_force[index].tolist(), "original_force": original_force[index].tolist(),
                         "nodal_moment": actual_moment[index].tolist(), "original_moment": original_moment[index].tolist(),
                         "norm_sum_force_N": float(scale[index]), "moment_scale_Nm": float(radius_scale[index]), "reference": reference[index].tolist()})
(root / "native-wrench-capture-diagnostic-v1.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"failing_fields": len(rows), "first": rows[:8]}, indent=2), flush=True)
