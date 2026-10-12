"""Rejected first-triangle loop-bound candidate with same-GPU profiling and unchanged-budget CPU/CUDA regressions."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-segment-bounds-rejected-v1")
assert not destination.exists()
names = [f"native-segment-bounds-regression-v1/{backend}" for backend in ("cpu", "cuda")]
names += [f"native-segment-bounds-v1/b2048-varied-profile-{mode}-623102" for mode in ("off", "on")]
runs = [root / name for name in names]
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0", run
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), run
fields, tests = [], {}
for backend in ("cpu", "cuda"):
    suite = ElementTree.parse(root / f"native-segment-bounds-regression-v1/{backend}/report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "23" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    tests[backend] = {"passed": 23, "seconds": float(suite.attrib["time"])}
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
assert len(fields) == 16 and sum(len(row["native_peak_Pa"]) for row in fields) == 48
profiles = {}
for mode in ("off", "on"):
    profile = json.loads((root / f"native-segment-bounds-v1/b2048-varied-profile-{mode}-623102/result.json").read_text())
    assert profile["first_triangle_segment_bounds"] == (mode == "on")
    snapshot = profile["recovery_snapshot_check"]
    assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
    assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
    profiles[mode] = {"stage_wall_ms": profile["stage_wall_ms"], "snapshot": snapshot, "gpu_uuid": profile["gpu_uuid"],
                      "native_buffer_bytes": profile["native_buffer_bytes"], "source_revision": profile["source_revision"]}
assert profiles["off"]["gpu_uuid"] == profiles["on"]["gpu_uuid"]
summary = {"selection": "rejected", "reason": "Same-GPU stage profiling shows no contact-pipeline benefit. Preserve the simpler original walk and skip further live timing for this candidate.",
           "tests": tests, "independent_source_prior_fields": 48, "fields": fields, "profiles": profiles,
           "contact_wall_ratio_off_over_on": profiles["off"]["stage_wall_ms"]["intrinsic_contact_pipeline"] / profiles["on"]["stage_wall_ms"]["intrinsic_contact_pipeline"],
           "force_or_moment_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "complete_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
files = []
for run in runs:
    for current, dirs, entries in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in entries)
    for line in (run / "source.sha256").read_text().splitlines():
        path = Path(line.split("  ", 1)[1])
        if path.is_absolute() and path.parent == root:
            files.append(path)
files.extend(root / name for name in ("segment-bounds-profile-pair-v1-allocation.log", "segment-bounds-tests-v1-cpu-allocation.log", "segment-bounds-tests-v1-cuda-allocation.log", "run_segment_bounds_profile_pair_v1.sh", "archive_native_segment_bounds_v1.py", "run_archive_native_segment_bounds_v1.sh"))
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
    entries.append({"original": str(original), "run_path": str(original.relative_to(root)), "artifact": str(artifact.relative_to(destination)),
                    "encoding": "gzip" if compressed else "identity", "bytes": len(content), "sha256": digest,
                    "stored_bytes": len(stored), "stored_sha256": hashlib.sha256(stored).hexdigest()})
path = destination / "manifest.json"
path.write_text(json.dumps({"scope": __doc__, "summary": summary, "files": entries,
                           "limitations": "The candidate is qualified numerically and rejected from production by profiling cost evidence. CPU/CUDA both include U/cavity gaps/holes and footprint/full-field checks, plus the 48 recorded source-prior fields. Live/policy ablation is unmeasured because isolated profiling gives no reasonable gain expectation. This is one candidate decision. The active goal's remaining optimization and broader measurement requirements remain open."}, indent=2) + "\n")
for row in entries:
    assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
    content = (destination / row["artifact"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
    restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
    assert hashlib.sha256(restored).hexdigest() == row["sha256"]
print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs), "stored_bytes": sum(path.stat().st_size for path in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tests": tests,
                  "full_fields": 48, "maximum_wrench_error": summary["force_or_moment_error_max"],
                  "maximum_full_residual_budget_fraction": summary["complete_residual_budget_fraction_max"],
                  "maximum_peak_difference_Pa": summary["peak_difference_Pa_max"], "contact_wall_ratio_off_over_on": summary["contact_wall_ratio_off_over_on"],
                  "profiles": profiles}, indent=2), flush=True)
