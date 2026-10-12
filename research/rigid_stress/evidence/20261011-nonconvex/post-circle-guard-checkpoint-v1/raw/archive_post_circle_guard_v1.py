"""Archive closed post-circle qualification, numerical fallback and scalar baseline; four-scope timings remain open."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/post-circle-guard-checkpoint-v1")
assert not destination.exists()
folders = [
    "circle-cooperative-regression-v1",
    "verified-circle-scalar-b32-profile-v1",
    "cooperative-fallback-regression-v1",
    "cooperative-fallback-regression-v2",
]
files = []
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
assert "52 passed" in (root / "circle-cooperative-regression-v1/cpu/tests.log").read_text()
failed = (root / "circle-cooperative-regression-v1/cuda/tests.log").read_text()
assert "1 failed" in failed and "51 passed" in failed and "cpu_oracle[517-2]" in failed
for relative in ("cooperative-fallback-regression-v1/targeted/cuda", "cooperative-fallback-regression-v2/combined/cpu"):
    assert "52 passed" in (root / relative / "tests.log").read_text()
profile = json.loads((root / "verified-circle-scalar-b32-profile-v1/profile.json").read_text())
assert profile["source_revision"] == "dfb3c56b9d7cc2a78abee30eda2c2ca1ae52e316"
assert profile["load_model"].endswith("verified_single_face_disks")
check = profile["recovery_snapshot_check"]
assert check["original_point_max_difference_m"] == 0
assert max(check["normalized_force_error_max"], check["normalized_moment_error_max"]) < 1e-8
assert (np.array(profile["complete_residual_N"]) <= np.maximum(1e-11, 1e-8 * np.array(profile["rhs_norm_N"]))).all()
for name in (
    "run_circle_cooperative_regression_v1.sh",
    "run_verified_circle_scalar_profile_v1.sh",
    "traction_original_point_candidate_v10.py",
    "traction_sample_cooperative_candidate_v2.py",
    "traction_scalar_fallback_candidate_v1.py",
    "traction_cooperative_fallback_candidate_v1.py",
    "probe_original_point_candidate_v10.py",
    "probe_cooperative_fallback_candidate_v1.py",
    "run_cooperative_fallback_regression_v1.sh",
    "run_cooperative_fallback_regression_v2.sh",
    "cooperative-fallback-launch-v1-note.txt",
    "run_verified_circle_matched_scopes_v1.sh",
    "archive_post_circle_guard_v1.py",
    "run_archive_post_circle_guard_v1.sh",
):
    files.append(root / name)
for pattern in (
    "circle-cooperative-regression-*allocation.log",
    "verified-circle-scalar-profile-*allocation.log",
    "cooperative-fallback-*allocation.log",
):
    files.extend(root.glob(pattern))
assert all(original.is_file() for original in files)
destination.mkdir(parents=True)
entries, external = [], []
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000 or (original.suffix == ".npz" and original.stat().st_size >= 1_000_000):
        external.append({"original": str(original), "bytes": original.stat().st_size, "sha256": digest})
    else:
        artifact = destination / "raw" / original.relative_to(root)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, artifact)
        entries.append(
            {
                "original": str(original),
                "artifact": str(artifact.relative_to(destination)),
                "bytes": original.stat().st_size,
                "sha256": digest,
            }
        )
manifest = {
    "scope": __doc__,
    "source_base_revision": "dfb3c56b9d7cc2a78abee30eda2c2ca1ae52e316",
    "unguarded_cooperative": {"cpu_pass": 52, "cuda_pass": 51, "cuda_fail": 1, "failed_fixture": "517-2"},
    "masked_scalar_fallback": {
        "cpu_pass": 52,
        "cuda_pass": 52,
        "cpu_seconds": 411.66,
        "cuda_seconds": 571.51,
        "note": "The CUDA directory label targeted is historical; its frozen launcher and collected-case log establish a full 52-case run. V2 passes the intended scope explicitly. Failed fast-path evaluations are retained in the retry counter.",
    },
    "scalar_baseline": {
        "intrinsic_ms": profile["stage_wall_ms"]["intrinsic_contact_pipeline"],
        "gpu_uuid": profile["gpu_uuid"],
        "snapshot_check": check,
        "native_buffer_bytes": profile["native_buffer_bytes"],
    },
    "open_measurement": "verified-circle-matched-b32-v1 on one GPU: profile/rigid/recovery/live/policy, scalar vs experimental cooperative plus masked native scalar retry. No timing results or running logs are published here.",
    "note": "Production remains the verified-circle scalar engine. Neither the new four-scope result nor native cooperative selection, large-batch acceptance, or plateau is complete.",
    "files": entries,
    "external_files": external,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert Path(entry["original"]).stat().st_size == entry["bytes"]
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(
    json.dumps(
        {
            "artifacts": len(entries),
            "bytes": sum(entry["bytes"] for entry in entries),
            "external_artifacts": len(external),
            "external_bytes": sum(entry["bytes"] for entry in external),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ),
    flush=True,
)
