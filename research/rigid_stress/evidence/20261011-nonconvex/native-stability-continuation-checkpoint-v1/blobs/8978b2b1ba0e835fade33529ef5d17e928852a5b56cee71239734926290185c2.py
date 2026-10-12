"""Publish unchanged captured inputs and independently derived CPU dual as an offline oracle initializer."""
import hashlib
import json
from pathlib import Path

import numpy as np

directory = Path(__file__).parent
qualification = directory / "native-group-failure-audit-v5"
assert (qualification / "returncode.txt").read_text().strip() == "0"
summary = json.loads((qualification / "summary.json").read_text())
assert summary["independent_full_field"]["complete_residual_budget_fraction"] <= 1
with np.load(directory / "native-certified-failure-capture-v1/b4096-623103/failure/before-rigid-integration.npz") as saved:
    valid = saved["fit_contacts.valid"][0]
    values = {name:saved[f"fit_contacts.{key}"][0,valid].copy() for name,key in (("anchor","center"),("point","position"),("radius","radius"),("friction","friction"),("anchor_face","anchor_face"))}
    values["force"] = saved["tractions.priors.force"][0,valid].copy()
    values["moment"] = saved["tractions.priors.anchor_moment"][0,valid].copy()
    values["oracle_dual"] = np.array(summary["independent_dual"]["primal_SOCP"]["true_cone_dual_root"]["dual"])
asset = Path("tests/rigid/assets/nonconvex_varied4096_slow_fan.npz")
assert not asset.exists()
np.savez_compressed(asset,**values)
print(json.dumps({"asset":str(asset),"sha256":hashlib.sha256(asset.read_bytes()).hexdigest(),"sources":int(valid.sum()),"oracle_initializer_source":"Independent CPU primal SOCP followed by independently verified true-cone dual root. No native multipliers/forces initialize the oracle."}),flush=True)
