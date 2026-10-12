"""Offline provenance/frame and repeated-recovery chart audit; preserve every original force."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

directory = Path(__file__).parent
output = directory / "snapshot-producer-audit-v1"
with np.load(directory / "native-snapshot-repeat-capture-v1/b2048-623102/failure/before-after-recovery.npz") as asset:
    saved = {key: asset[key].copy() for key in asset.files}
changes = []
selected = {}
for key in saved:
    if not key.startswith("before.") or "after." + key[7:] not in saved:
        continue
    before, after = saved[key], saved["after." + key[7:]]
    if not np.array_equal(before, after, equal_nan=True):
        finite = np.isfinite(before) & np.isfinite(after)
        changes.append({"field": key[7:], "shape": before.shape, "changed_count": int((~np.isclose(before, after, rtol=0, atol=0, equal_nan=True)).sum()), "maximum_finite_difference": float(abs(before[finite].astype(float) - after[finite].astype(float)).max(initial=0))})
    if key[7:] in ("state.peak", "contacts.force", "contacts.normal", "contacts.center", "contacts.position", "tractions.dual", "tractions.scale", "tractions.force_error", "tractions.moment_error", "tractions.fan_refined", "tractions.sample_count", "tractions.evaluations", "tractions.status", "tractions.circle.refined", "tractions.priors.weight_sum", "tractions.priors.group_scale", "fit_contacts.valid", "raw_sources.witness", "fit_contacts.center", "fit_contacts.position"):
        selected[key[7:]] = {"before": before.tolist(), "after": after.tolist()}
snapshot = {"environment": int(saved["environment"]), "changes": changes, "selected": selected}
(output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
print(json.dumps({"environment": snapshot["environment"], "changes": changes, "selected": {key:value for key,value in selected.items() if key.startswith("tractions.") or key == "state.peak"}}), flush=True)
capture = directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz"
with np.load(capture) as asset:
    original = {key: asset[key].copy() for key in asset.files}
keys = [key for key in original if key.startswith(("rigid.geom_info.", "rigid.geoms.", "rigid.vert_info.", "rigid.face_info."))]
shapes = {key:original[key].shape for key in keys}
(output / "producer-schema.json").write_text(json.dumps(shapes, indent=2) + "\n")
rows = []
rotation = Rotation.from_quat(original["contacts.frame_quaternion"][0, [1, 2, 3, 0]])
frame_position = original["contacts.frame_position"][0]
for index in np.flatnonzero(original["contacts.valid"][0]):
    slot = int(original["collider.contact_sort_idx"][0, index])
    pair = {key: original[f"collider.contact_data.{key}"][0, slot].tolist() for key in ("geom_a", "geom_b", "link_a", "link_b", "pos", "normal", "force", "penetration", "pair_idx")}
    pair["witness_a"] = original["collider.contact_data.witness.point_a"][0, slot].tolist()
    pair["witness_b"] = original["collider.contact_data.witness.point_b"][0, slot].tolist()
    geom = int(original["raw_sources.geom"][0, index])
    for name in ("pos", "quat"):
        pair["egg_geom_" + name] = original[f"rigid.geoms.{name}"][0, geom].tolist()
    other = int(pair["geom_b"] if pair["geom_a"] == geom else pair["geom_a"])
    for name in ("pos", "quat"):
        pair["other_geom_" + name] = original[f"rigid.geoms.{name}"][0, other].tolist()
    side = "a" if pair["geom_a"] == geom else "b"
    sign = -1 if side == "a" else 1
    local_witness = rotation.inv().apply(np.array(pair[f"witness_{side}"]) - frame_position)
    local_force = rotation.inv().apply(sign * np.array(pair["force"]))
    local_normal = rotation.inv().apply(-sign * np.array(pair["normal"]))
    np.testing.assert_allclose(local_witness, original["raw_sources.witness"][0, index], rtol=0, atol=2e-16)
    np.testing.assert_allclose(local_force, original["contacts.force"][0, index], rtol=0, atol=2e-16)
    np.testing.assert_allclose(local_normal, original["contacts.normal"][0, index], rtol=0, atol=2e-15)
    pair.update(raw_contact=int(index), slot=slot, stress_geom=geom, stress_side=side,
                frame_verified=True, reference_witness=local_witness.tolist(),
                reference_force=local_force.tolist(), reference_normal=local_normal.tolist())
    rows.append(pair)
summary = {"scope": __doc__, "environment": int(original["environment"]), "contacts": rows, "geometry_schema": shapes}
(output / "producer.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
