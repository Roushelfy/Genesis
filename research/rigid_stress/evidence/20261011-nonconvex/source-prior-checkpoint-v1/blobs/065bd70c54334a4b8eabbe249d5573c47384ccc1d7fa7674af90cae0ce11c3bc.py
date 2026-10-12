"""Independently audit the captured B1024 policy807 continuous separating ray and preserve its pair inputs."""

import json
from pathlib import Path

import numpy as np

from tests.rigid.test_stress_compatibility import _continuous_disk_polar_bound

root = Path(__file__).parent
output = root / "native-policy-independent-audit-v1/results"
output.mkdir(parents=True, exist_ok=False)
metadata = json.loads((root / "native-policy-failure-v2/failure/metadata.json").read_text())
with np.load(root / "native-policy-failure-v2/failure/before-rigid-integration.npz") as saved:
    data = {name: saved[name].copy() for name in saved.files}
raw_status = data["contacts.status"][0]
bad_raw = int(np.flatnonzero(raw_status == 36)[0])
indices = data["packing.indices"][0]
bad_fit = int(np.flatnonzero(indices == bad_raw)[0])
first, last = int(data["tractions.support.start"][0, bad_fit]), int(data["tractions.support.end"][0, bad_fit])
assert first <= bad_fit < last and int(data["tractions.status"][0, bad_fit]) == 6
support = list(range(first, last))
ordered = [i for i in support if i != bad_fit]
ordered.insert(1, bad_fit)
if len(ordered) == 1:
    ordered *= 2
anchors, radii, friction = (data["fit_contacts." + name][0, ordered] for name in ("center", "radius", "friction"))
ray = data["tractions.certificate"][0, bad_fit]
force = data["fit_contacts.force"][0, bad_fit]
moment = data["fit_sources.anchor_moment"][0, bad_fit]
radius = float(data["fit_contacts.radius"][0, bad_fit])
scale = max(np.linalg.norm(force), np.linalg.norm(moment) / radius)
target = np.concatenate((force / scale, moment / (radius * scale)))
assert np.linalg.norm(ray) > 0 and target @ ray > 1e-8
with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as mesh:
    bound = _continuous_disk_polar_bound(mesh, anchors, radii, friction, ray)
assert bound < 0
pair = {}
for name in ("position", "center", "anchor_face", "force", "normal", "friction", "radius"):
    pair["contacts." + name] = data["fit_contacts." + name][0, support]
for name in ("geom", "pair", "status", "free_moment", "anchor_moment", "barycentric"):
    pair["raw_sources." + name] = data["fit_sources." + name][0, support]
np.savez_compressed(output / "pair-inputs.npz", **pair)
report = {"scope": __doc__, "step_call": metadata["step_call"], "workload_tick_before_step": metadata["workload_tick_before_step"], "failed_environments": metadata["failed_environments"], "unsupported_predicate_environments": metadata["unsupported_predicate_environments"], "raw_contact": bad_raw, "fit_contact": bad_fit, "group_size": last-first, "strict_continuous_polar_upper_bound": float(bound), "native_continuous_polar_upper_bound": float(data["wrench_search.bound"][0,bad_fit]), "target_margin": float(target @ ray), "native_search_evaluations": int(data["wrench_search.evaluations"][0,bad_fit]), "native_verification_nodes": int(data["wrench_search.visited"][0,bad_fit]), "original_radius_m": radius, "actual_friction": friction.tolist(), "stage": "The original benchmark traceback points into its timed repetition;cad0c46c capture fails at call1408 =600 warmup+808 first-repeat steps. Earlier checkpoint wording 'warmup aborted' is incorrect;the underlying abort/invalid-throughput evidence is unchanged."}
(output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
