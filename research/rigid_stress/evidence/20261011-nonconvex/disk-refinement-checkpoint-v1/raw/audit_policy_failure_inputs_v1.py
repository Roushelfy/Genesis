"""Inspect the preserved first policy failure without altering its original numerical inputs."""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).parent
source = root / "policy-failure-capture-v2/failure/before-rigid-integration.npz"
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=False)
with np.load(source) as saved:
    shapes = {key: list(saved[key].shape) for key in saved.files}
    rows = {}
    for prefix in ("contacts", "raw_sources", "fit_contacts", "fit_sources", "tractions", "packing", "certificates"):
        selected = {}
        for key in saved.files:
            if key.startswith(prefix + ".") and len(saved[key].shape) in (1, 2, 3) and saved[key].shape[0] == 32:
                value = saved[key][24]
                if value.size <= 128:
                    selected[key] = value.tolist()
                elif prefix in ("contacts", "raw_sources", "certificates"):
                    selected[key] = value[:8].tolist()
        rows[prefix] = selected
    payload = {
        key: saved[key][24, :8].copy()
        for key in saved.files
        if key.startswith(("contacts.", "raw_sources.", "certificates."))
        and len(saved[key].shape) >= 2
        and saved[key].shape[0] == 32
    }
    payload["failed_env"] = 24
    payload["failed_tick"] = 876
    np.savez(destination / "original-environment-24.npz", **payload)
(destination / "summary.json").write_text(json.dumps({"shapes": shapes, "env24": rows}, indent=2) + "\n")
print(json.dumps(rows), flush=True)
