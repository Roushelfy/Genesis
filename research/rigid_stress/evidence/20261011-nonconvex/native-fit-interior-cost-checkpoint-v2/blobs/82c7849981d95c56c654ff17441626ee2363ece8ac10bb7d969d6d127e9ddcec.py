"""Publish first-accept work counters alongside the independently checked complete raw applicability run."""

import hashlib
import json
from pathlib import Path

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
first = json.loads((root / "native-first-warm-accept-cost-v1.json").read_text())
previous = Path("research/rigid_stress/evidence/20261011-nonconvex/native-fit-interior-cost-checkpoint-v1/manifest.json")
manifest = json.loads(previous.read_text())
assert first["first_accept_positive_sample_fraction"] < 0.14
assert first["interior_extra_evaluations_work_fraction"] < 0.04
extras = [root / name for name in ("archive_native_fit_interior_cost_v2.py", "run_archive_native_fit_interior_cost_v2.sh",
    "audit_native_first_warm_accept_cost_v1.py", "run_audit_native_first_warm_accept_cost_v1.sh",
    "native-first-warm-accept-cost-v1.json", "native-first-warm-accept-cost-v1.log",
    "native-first-warm-accept-cost-v1-allocation.log", "publish_native_checkpoint_v1.py")]
summary = {**manifest["summary"], "first_warm_accept": first,
           "referenced_verified_v1_manifest_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
           "decision": "No unconditional Gram prepass or deferred full-scatter verification/replay implementation is selected: interior cases average1.43 evaluations, additional interior evaluations are3.8% of total sample-evaluation work, first-warm-accepted cases cover13.8% of samples and5.66% of sample-evaluations; a deferred verification route replays86.2% of sample work. Current validation/normalization and physical verification remain required. These work proxies and measured25.03ms fit/10.31ms scatter costs give no reasonable expectation of material gain for those unconditional routes. They are not latency bounds and do not rule out unknown selective schemes."}
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-fit-interior-cost-checkpoint-v2"),
        [root / "native-fit-interior-v1/b2048-623102"], extras, summary,
        "Complete unchanged-source76537c2f applicability/cost audit with first-warm-accept supplement. Work counters are not per-environment wall times or rigorous speedup bounds. A proposed optimization is rejected by its required new scans/normalization/verification/replay costs relative to the measured available work, not by relaxing physical or numerical acceptance. Full raw frame/environment counters and actual outcomes retained. No direct numerical solver or new history shortcut is implemented; future unknown optimizations are not ruled out.")
