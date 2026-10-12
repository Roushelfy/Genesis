"""Native explicit source-prior model qualification, failed attempts and first copied B1024 policy rollout."""

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-source-priors-checkpoint-v1")
assert not destination.exists()
runs = [root / f"native-source-priors-{stage}/{backend}" for stage in
        ("regression-v1", "regression-v2", "regression-v3", "combined-v1") for backend in ("cpu", "cuda")]
runs += [root / f"native-source-priors-debug-v{i}/cpu" for i in (1, 2, 3)]
runs += [root / "native-source-priors-policy-v1", root / "validate-native-source-policy-v1"]
for run in runs[:-1]:
    after = run / ("source-after-manual-verified.sha256" if "regression-v2" in str(run) else "source-after.sha256")
    assert (run / "source.sha256").read_bytes() == after.read_bytes()
assert "62 passed" in (root / "native-source-priors-regression-v3/cuda/tests.log").read_text()
assert "61 passed" in (root / "native-source-priors-regression-v3/cpu/tests.log").read_text()
assert "ACTUAL: array([36, 36]" in (root / "native-source-priors-debug-v3/cpu/tests.log").read_text()
fields = []
for backend in ("cpu", "cuda"):
    run = root / f"native-source-priors-combined-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert "6 passed" in (run / "tests.log").read_text()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "6" and suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            properties = {item.attrib["name"]: ast.literal_eval(item.attrib["value"])
                          for item in case.find("properties").findall("property")}
            assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
            assert max(properties["complete_residual_budget_fraction"]) <= 1
            assert max(properties["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **properties})
assert len(fields) == 8 and sum(len(item["native_peak_Pa"]) for item in fields) == 24
policy = json.loads((root / "native-source-priors-policy-v1/validated-summary.json").read_text())
summary = {"full_field_samples": 24,
           "force_or_moment_error_max": max(max(item["force_errors"] + item["moment_errors"]) for item in fields),
           "complete_residual_budget_fraction_max": max(max(item["complete_residual_budget_fraction"]) for item in fields),
           "peak_difference_Pa_max": max(max(item["peak_difference_Pa"]) for item in fields),
           "fields": fields, "policy": policy,
           "qualification": "62 CUDA full-suite cases pass before frontier merge. CPU61 pass/1 test classification mismatch:strict continuous-wrench36 instead of strict continuous-force35. The corrected diagnostic test accepts both physical certificates without modifying recovery acceptance; combined native frontier+source-prior CPU6/CUDA6 allpass,including complete independent fields and lifecycle. All56 legacy cases pass on CPU/CUDA in the prior checkpoint. Failed kernel-default,disabled-zero-dimension and missing-test-argument attempts are retained;v2was canceled after diagnosing the invalid ndarray shape."}
files = []
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
files.append(root / "native-source-priors-regression-v2/INTERRUPTED.md")
files += [root / name for name in ("run_native_source_priors_regression_v1.sh", "run_native_source_priors_regression_v2.sh",
                                  "run_native_source_priors_regression_v3.sh", "run_native_source_priors_combined_v1.sh",
                                  "run_native_source_priors_debug_v1.sh", "run_native_source_priors_debug_v2.sh",
                                  "run_native_source_priors_debug_v3.sh", "run_native_source_priors_policy_v1.sh",
                                  "validate_native_source_policy_v1.py", "run_validate_native_source_policy_v1.sh",
                                  "archive_native_source_priors_v1.py", "run_archive_native_source_priors_v1.sh")]
destination.mkdir(parents=True)
entries, blobs = [], set()
for original in sorted(set(files)):
    content = original.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    compressed = original.suffix in {".json", ".log"} and len(content) >= 200000
    stored = gzip.compress(content, mtime=0) if compressed else content
    artifact = destination / "blobs" / (digest + original.suffix + (".gz" if compressed else ""))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        artifact.write_bytes(stored)
    blobs.add(artifact)
    entries.append({"original": str(original), "run_path": str(original.relative_to(root)),
                    "artifact": str(artifact.relative_to(destination)), "encoding": "gzip" if compressed else "identity",
                    "bytes": len(content), "sha256": digest, "stored_bytes": len(stored),
                    "stored_sha256": hashlib.sha256(stored).hexdigest()})
manifest = {"scope": __doc__, "summary": summary,
            "limitations": "Explicit alternative body aggregate conservation,retaining all local force priors;old componentwise default preserved. Policy includes one copied32-condition bank only,before the sampling frontier merge. Combined model+frontier is numerically qualified; broaderlive/policyseeds,B2048/B4096/four-scopes,grasp outcomes,maximum throughput and plateau remain ongoing/unverified. No old-model policy speedup can be computed from an aborted run. Only low-shell same-mesh consistency is claimed.",
            "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    stored = (destination / entry["artifact"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    restored = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
    assert hashlib.sha256(restored).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "unique_blobs": len(blobs),
                  "unique_blob_bytes": sum(path.stat().st_size for path in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": summary}, indent=2), flush=True)
