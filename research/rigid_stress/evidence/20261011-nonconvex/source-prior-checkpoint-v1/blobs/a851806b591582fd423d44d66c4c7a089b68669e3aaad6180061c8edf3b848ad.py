"""Publish an unselected source-prior physical model, independent policy proof and sample-cloud audit."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/source-prior-checkpoint-v1")
assert not destination.exists()
folders = ["source-prior-fields-v1", "source-prior-fields-v3", "source-prior-fields-v4", "native-policy-failure-v1", "native-policy-failure-v2", "native-policy-independent-audit-v1", "source-prior-cloud-comparison-v1"]
files, skip = [], {"cache", "tmp", "__pycache__", ".pytest_cache"}
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    closures = [path for path in (root / folder).rglob("source.sha256") if not any(part in skip for part in path.relative_to(root / folder).parts)]
    assert closures, folder
    for before in closures:
        assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
fields = {}
max_wrench, max_fraction, max_peak_difference = 0.0, 0.0, 0.0
for epoch, expected in (("v3", ["step602", "policy876", "balanced_compression"]), ("v4", ["copied_policy807"])):
    records = {}
    for backend in ("cpu", "cuda"):
        run = root / f"source-prior-fields-{epoch}/{backend}"
        assert (run / "returncode.txt").read_text().strip() == "0"
        records[backend] = json.loads((run / "results/summary.json").read_text())
        assert [row["case"] for row in records[backend]] == expected
        for row in records[backend]:
            max_wrench = max(max_wrench, max(row["force_error"] + row["moment_error"]))
            max_fraction = max(max_fraction, max(row["complete_residual_budget_fraction"]))
            assert max(row["cone_excess_N"]) <= 1e-12
            np.testing.assert_allclose(row["native_peak_Pa"], row["oracle_peak_Pa"], rtol=1e-4, atol=1e-3)
            max_peak_difference = max(max_peak_difference, float(np.max(np.abs(np.array(row["native_peak_Pa"]) - np.array(row["oracle_peak_Pa"])))))
            if row["case"] == "balanced_compression":
                assert max(abs(value) for value in row["aggregate_force_N"] + row["aggregate_anchor_moment_Nm"]) == 0
                assert min(min(values) for values in row["recovered_source_force_norms_N"]) > 0.049
                assert min(row["native_peak_Pa"]) > 12000
    for a, b in zip(records["cpu"], records["cuda"]):
        assert a["actual_friction"] == b["actual_friction"]
        np.testing.assert_allclose(a["native_peak_Pa"], b["native_peak_Pa"], rtol=1e-4, atol=1e-3)
    fields[epoch] = records
assert max_wrench <= 1e-8 and max_fraction <= 1
audit = json.loads((root / "native-policy-independent-audit-v1/results/summary.json").read_text())
assert audit["strict_continuous_polar_upper_bound"] < 0 and audit["native_continuous_polar_upper_bound"] < 0
assert audit["group_size"] == 3 and len(audit["failed_environments"]) == 32
assert audit["unsupported_predicate_environments"] == []
assert audit["step_call"] == 1408 and audit["workload_tick_before_step"] == 807
assert "Ndarray shouldn't be passed in via `qd.template()`" in (root / "native-policy-failure-v1/diagnostic.log").read_text()
clouds = json.loads((root / "source-prior-cloud-comparison-v1/results/summary.json").read_text())
for report in clouds:
    for row in report["sources"]:
        assert row["per_face_integral_l1_relative_difference"] < 1e-14
        assert row["gpu_unmatched_profile_fraction"] < 1e-14
for name in ("generate_source_prior_candidate_v1.py", "run_generate_source_prior_v1.sh", "source_prior_candidate_v1.py", "source_prior_candidate_v2.py", "source_prior_candidate_v3.py", "probe_source_prior_fields_v1.py", "probe_source_prior_fields_v2.py", "probe_source_prior_fields_v3.py", "run_source_prior_fields_v1.sh", "run_source_prior_fields_v2.sh", "run_source_prior_fields_v3.sh", "run_source_prior_fields_v4.sh", "source-prior-fields-v2-cpu-allocation.log", "source-prior-fields-v2-cuda-allocation.log", "probe_native_policy_failure_v1.py", "probe_native_policy_failure_v2.py", "run_native_policy_failure_v1.sh", "run_native_policy_failure_v2.sh", "audit_native_policy_failure_v2.py", "run_audit_native_policy_failure_v2.sh", "compare_source_prior_clouds_v1.py", "run_compare_source_prior_clouds_v1.sh", "archive_source_prior_v1.py", "run_archive_source_prior_v1.sh"):
    files.append(root / name)
destination.mkdir(parents=True)
entries, external, blobs = [], [], set()
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000:
        external.append({"original": str(original), "bytes": original.stat().st_size, "sha256": digest})
    else:
        artifact = destination / "blobs" / (digest + original.suffix)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            shutil.copy2(original, artifact)
        blobs.add(artifact)
        entries.append({"original": str(original), "run_path": str(original.relative_to(root)), "artifact": str(artifact.relative_to(destination)), "bytes": original.stat().st_size, "sha256": digest})
manifest = {"scope": __doc__, "source_base_revision": "cad0c46c", "selection": "Unselected offline model;no production module is patched.", "physical_law": "Retain immutable individual source forces Fi,profile integrals Pi and every unchanged physical support/R/mu. w_iq=alpha_iq/sum(Pi),g_iq=alpha_iq/Pi*Fi. Minimize sum||f-g||^2/(2w) subject to the complete aggregate original wrench and every local actual Coulomb cone. Original load magnitude sets conditioning/error scale even when the total wrench cancels. This explicitly changes individual-component conservation into aggregate conservation;it is not silently substituted into existing acceptance.", "full_field_comparisons": 24, "fields": fields, "max_normalized_wrench_error": max_wrench, "max_complete_residual_budget_fraction": max_fraction, "max_peak_difference_Pa": max_peak_difference, "policy_independent_audit": audit, "sample_cloud_audit": clouds, "limitations": "Fit oracle shares native geometry/profile samples but no fitted tractions or dual;CPU elasticity assembly/tensor/vonMises/peak/fullgauge residual are independent. CPU/GPU extra points carry at most3.67e-16 profile mass in the three audited clouds;no production cutoff is introduced and the exact floating-point branch remains unestablished. No source-prior live/policy/lifecycle/performance qualification,production selection or whole-goal completion is claimed. Native policy capture is untimed and preserves physical rejection;it does not produce valid throughput. The first capture export fails on an incorrect generic template annotation;the corrected Tensor annotation captures successfully. Initial v2 field launchers fail before Python due to a missing generator filename;the corrected v3/v4 epochs close successfully. Earlier checkpoint wording 'policy warmup aborted' is corrected:the traceback is in the timed repetition,with cad0c46c reproduced at1408 total calls/tick807. Original archived manifests/hashes are preserved.", "files": entries, "external_files": external}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "unique_blobs": len(blobs), "unique_blob_bytes": sum(path.stat().st_size for path in blobs), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "max_normalized_wrench_error": max_wrench, "max_complete_residual_budget_fraction": max_fraction, "max_peak_difference_Pa": max_peak_difference, "policy_independent_audit": audit}, indent=2), flush=True)
