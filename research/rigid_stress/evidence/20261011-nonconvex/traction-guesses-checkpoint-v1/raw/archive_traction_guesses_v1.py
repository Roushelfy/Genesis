"""Archive independent live dual-guess acceptance and complete mixed-Hessian rejection evidence."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/traction-guesses-checkpoint-v1")
assert not destination.exists()
folders = ["workspace-seed-online-b16-v1", "workspace-seed-online-b32-v1", "float-hessian-regression-v1"]
files = []
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
summaries = []
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
for envs in (16, 32):
    folder = root / f"workspace-seed-online-b{envs}-v1/results"
    saved = json.loads((folder / "summary.json").read_text())
    assert saved["completed_steps"] == 1280 and len(saved["oracle_captures"]) == 9
    assert saved["source_sha256"] == saved["source_sha256_after"]
    assert max(row["max_force_error"] for row in saved["records"]) <= 1e-8
    assert max(row["max_moment_error"] for row in saved["records"]) <= 1e-8
    maximum_residual_fraction = 0.0
    for capture in saved["oracle_captures"]:
        with np.load(folder / f"snapshot-{capture['tick']}.npz") as arrays:
            assert len(arrays["full_residual"]) == envs
            fraction = np.max(arrays["full_residual"] / arrays["full_residual_budget"])
            assert fraction <= 1.0
            maximum_residual_fraction = max(maximum_residual_fraction, fraction)
            assert (arrays["status"][arrays["valid"]] == 0).all()
    summaries.append({
        "envs": envs, "seed": saved["arguments"]["seed"], "steps": saved["completed_steps"],
        "cpu_full_fields": 9 * envs, "saved_state_resets": saved["records"][-1]["reset_environments"],
        "maximum_normalized_force_error": max(row["max_force_error"] for row in saved["records"]),
        "maximum_normalized_moment_error": max(row["max_moment_error"] for row in saved["records"]),
        "maximum_total_fit_evaluations": max(row["max_evaluations"] for row in saved["records"]),
        "maximum_complete_residual_budget_fraction": maximum_residual_fraction,
    })
mixed_log = (root / "float-hessian-regression-v1/cuda/tests.log").read_text()
plain_log = re.sub(r"\x1b\[[0-9;]*m", "", mixed_log)
assert "2 failed" in plain_log and "53 passed" in plain_log and "Failed tests" in plain_log
mixed_summary = plain_log[plain_log.index("Failed tests"):]
for name in (
    "probe_workspace_seed_online_v1.py", "run_workspace_seed_online_v1.sh",
    "traction_workspace_seed_candidate_v1.py", "traction_float_hessian_candidate_v1.py", "probe_float_hessian_v1.py",
    "run_float_hessian_regression_v1.sh", "archive_traction_guesses_v1.py", "run_archive_traction_guesses_v1.sh",
):
    files.append(root / name)
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
        entries.append({"original": str(original), "artifact": str(artifact.relative_to(destination)), "bytes": original.stat().st_size, "sha256": digest})
manifest = {
    "scope": __doc__, "source_base_revision": "a509a6e8639026e5031e97d1cafbbd740a2dd54d",
    "native_cooperative_publication": "c560c5e4", "workspace_dual_guess_full_field_validation": summaries,
    "cpu_full_fields": sum(row["cpu_full_fields"] for row in summaries),
    "saved_state_resets": sum(row["saved_state_resets"] for row in summaries),
    "mixed_hessian_rejection_summary": mixed_summary,
    "note": "All production paths remain Quadrants. Live CPU oracle work is intrusive and provides no throughput rate. Warm dual guesses are numerical initial guesses, not assumed contact identity or cached acceptance. The mixed Hessian candidate cannot be selected while this unchanged-budget fixture fails. The aggregate model and native diagnostic cloud remain separate unselected alternatives; overall nonconvex/larger-batch goal is unfinished.",
    "files": entries, "external_files": external,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cpu_full_fields": manifest["cpu_full_fields"], "saved_state_resets": manifest["saved_state_resets"], "mixed_hessian_summary": mixed_summary}, indent=2), flush=True)
