"""Publish small byte-exact load/source inputs from closed native failed captures; no load changes."""
import hashlib
import json
from pathlib import Path

import numpy as np

directory = Path(__file__).parent
asset = Path("tests/rigid/assets/nonconvex_shared_edge_roundoff.npz")
assert not asset.exists()
with np.load(directory / "native-snapshot-repeat-capture-v1/b2048-623102/failure/before-after-recovery.npz") as saved:
    valid = saved["before.fit_contacts.valid"][0]
    values = {}
    for name, source in (("witness", "fit_sources.witness"), ("point", "fit_contacts.position"), ("force", "tractions.priors.force"), ("radius", "fit_contacts.radius"), ("friction", "fit_contacts.friction"), ("uncertainty", "fit_sources.uncertainty"), ("old_anchor_face", "fit_contacts.anchor_face"), ("original_anchor", "fit_contacts.center")):
        values[name] = np.stack([saved[f"{prefix}.{source}"][0, valid] for prefix in ("before", "after")])
    values["frame_position"] = np.stack([saved[f"{prefix}.fit_contacts.frame_position"][0] for prefix in ("before", "after")])
    values["free_moment"] = np.zeros_like(values["force"])
    values["environment"] = saved["environment"]
np.savez_compressed(asset, **values)
private = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement") / asset
assert not private.exists()
private.write_bytes(asset.read_bytes())
assert asset.read_bytes() == private.read_bytes()
print(json.dumps({"asset": str(asset), "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(), "variants": len(values["witness"]), "sources": values["witness"].shape[1], "old_anchor_faces": values["old_anchor_face"].tolist()}), flush=True)
