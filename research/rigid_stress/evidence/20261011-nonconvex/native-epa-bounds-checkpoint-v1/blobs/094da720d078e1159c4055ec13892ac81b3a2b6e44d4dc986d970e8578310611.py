"""Qualified native EPA distance-bound comparison, producer/SAT proof, unchanged-budget stress and valid B4096 live."""
import ast
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-epa-bounds-checkpoint-v1")
assert not destination.exists()
names = [f"captured-gjk-patch-probe-v{i}" for i in range(4,11)]
names += ["captured-convex-sat-audit-v1","publish-epa-shallow-fixture-v1","epa-shallow-regression-v1/cpu",
          "native-epa-rollout-v1/b4096-varied-live-frontier-623103"]
names += [f"epa-shallow-regression-v2/{backend}" for backend in ("cpu","cuda")]
names += [f"native-epa-regression-v{i}/{backend}" for i in (1,2) for backend in ("cpu","cuda")]
names += [f"epa-fp32-baseline-v1/{backend}" for backend in ("cpu","cuda")]
runs = [root / name for name in names]
for run in runs:
    assert (run / "returncode.txt").exists(),run
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(),run
fields = []
test_summary = {}
for backend in ("cpu","cuda"):
    run = root / f"native-epa-regression-v2/{backend}"
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "18" and suite.attrib["errors"] == "0"
    expected_failures = "1" if backend == "cpu" else "0"
    assert suite.attrib["failures"] == expected_failures
    failed = [case.attrib["name"] for case in suite.findall("testcase") if case.find("failure") is not None]
    if backend == "cpu":
        assert len(failed) == 1 and "convex_collision_across_geom_scales" in failed[0]
    test_summary[backend] = {"passed":18-int(expected_failures),"failed":int(expected_failures),
                             "failures":failed,"seconds":float(suite.attrib["time"])}
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            assert case.find("failure") is None
            values = {item.attrib["name"]:ast.literal_eval(item.attrib["value"])
                      for item in case.find("properties").findall("property")}
            assert max(values["force_errors"]+values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend":backend,"case":case.attrib["name"],**values})
assert len(fields) == 16 and sum(len(row["native_peak_Pa"]) for row in fields) == 48
baseline_cpu = (root / "epa-fp32-baseline-v1/cpu/tests.log").read_text()
patched_cpu = (root / "native-epa-regression-v2/cpu/tests.log").read_text()
pattern = r"Max absolute difference among violations: ([0-9.eE+-]+)"
old_difference,new_difference = (float(re.search(pattern,log).group(1)) for log in (baseline_cpu,patched_cpu))
assert old_difference == new_difference == 1.16215267e-6
assert (root / "epa-fp32-baseline-v1/cuda/returncode.txt").read_text().strip() == "0"
proof = json.loads((root / "captured-gjk-patch-probe-v9/summary.json").read_text())
selected = next(row for row in proof["records"] if row["patch"])["producer_inactive_slot_trace"]["multicontact_gjk_state"]
wrong = next(row for row in selected if row["pair"][0] == 14)
assert wrong["nearest_face_values"][13][0] == -2
lower,upper = wrong["bound_exit_trace_B"][1:]
assert 0 < lower-upper < 1e-18
sat = json.loads((root / "captured-convex-sat-audit-v1/summary.json").read_text())
fixed = json.loads((root / "captured-gjk-patch-probe-v10/summary.json").read_text())
for record in fixed["records"]:
    for expected in sat["contacts"]:
        rows = [row for row in record["selected_contacts"] if row["geom_a"] == expected["geom_a"]]
        assert rows and all(np.array(row["normal"]) @ expected["minimum_translation_A_direction"] > .99 for row in rows)
        np.testing.assert_allclose(max(row["penetration"] for row in rows),expected["minimum_translation_A_m"],rtol=1e-7,atol=1e-11)
live_run = root / "native-epa-rollout-v1/b4096-varied-live-frontier-623103"
assert (live_run / "returncode.txt").read_text().strip() == "0"
live = json.loads((live_run / "result.json").read_text())
handling = json.loads((live_run / "certified-failure-responses.json").read_text())
assert live["envs"] == 4096 and live["scope"] == "live" and live["seed"] == 623103 and live["varied"]
assert live["conditions"] is None and live["intrinsic_load_partition"] == "body_source_prior"
assert live["dt"] == .01 and live["substeps"] == 1 and live["precision"] == "64"
assert len(live["repeats"]) == 3
for row in live["repeats"]:
    assert row["steps"] == 1280 and row["resets"] == 4643
    assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
    assert len(row["failed_steps_per_environment"]) == 4096 and max(row["failed_steps_per_environment"]) == 0
assert handling["events"] == [] and handling["failed_environment_steps"] == 0
budgets = [float(np.max(np.array(row["residual_N"])/np.maximum(1e-11,1e-8*np.array(row["rhs_N"])))) for row in live["warmup_quality"]]
assert max(budgets) <= 1
rate = 4096*1280*3/sum(row["seconds"] for row in live["repeats"])
summary = {"tests":test_summary,"independent_full_fields":48,"fields":fields,
           "force_or_moment_error_max":max(max(row["force_errors"]+row["moment_errors"]) for row in fields),
           "complete_residual_budget_fraction_max":max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max":max(max(row["peak_difference_Pa"]) for row in fields),
           "unchanged_CPU_FP32_scale_failure": {"old_difference_m":old_difference,"patched_difference_m":new_difference,"acceptance_m":1e-7,
               "CUDA_baseline_and_patched":"both pass","status":"pre-existing CPU failure;no tolerance/assertion changes"},
           "independent_SAT":sat,"EPA_bound_trace":proof,"repaired_captured_pose":fixed,
           "B4096_live":{"effective_env_steps_per_second":rate,"batch_steps_per_second":rate/4096,
               "repeat_rates":[row["transitions_per_second"] for row in live["repeats"]],"invalid_environment_steps":0,
               "resets_per_repetition":4643,"source_revision":live["source_revision"],"gpu_uuid":live["gpu_uuid"],
               "native_buffer_bytes":live["native_buffer_bytes"],"device_free_bytes":live["device_free_bytes"],
               "torch_peak_allocated_bytes":live["torch_peak_allocated_bytes"],"warmup_complete_residual_budget_fraction_max":max(budgets)}}
files = []
for run in runs:
    for current,dirs,entries in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache","tmp","__pycache__",".pytest_cache"}]
        files.extend(Path(current)/name for name in entries)
    for line in (run/"source.sha256").read_text().splitlines():
        path = Path(line.split("  ",1)[1])
        if path.is_absolute() and path.parent == root:
            files.append(path)
for pattern in ("allocation-gjkpatchprobev*.log","allocation-convexSATAuditV1.log","allocation-epaShallow*.log",
                "allocation-epaOldFP32*.log","allocation-nativeEpa*.log"):
    files.extend(root.glob(pattern))
files.extend((root/"archive_native_epa_bounds_v1.py",root/"run_archive_native_epa_bounds_v1.sh"))
destination.mkdir(parents=True)
entries,blobs = [],set()
for original in sorted(set(files)):
    content = original.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    compressed = original.suffix in {".json",".log"} and len(content)>=200000
    stored = gzip.compress(content,mtime=0) if compressed else content
    artifact = destination/"blobs"/(digest+original.suffix+(".gz" if compressed else ""))
    artifact.parent.mkdir(parents=True,exist_ok=True)
    if not artifact.exists(): artifact.write_bytes(stored)
    blobs.add(artifact)
    entries.append({"original":str(original),"run_path":str(original.relative_to(root)),"artifact":str(artifact.relative_to(destination)),
                    "encoding":"gzip" if compressed else "identity","bytes":len(content),"sha256":digest,
                    "stored_bytes":len(stored),"stored_sha256":hashlib.sha256(stored).hexdigest()})
manifest = {"scope":__doc__,"summary":summary,
            "limitations":"Goal remains active. Low FP64 contact/stress checks pass and the original varied B4096 workload now has zero failures across complete repeats. A pre-existing CPU FP32 finger/lamp depth failure remains unchanged and is disclosed; the corresponding CUDA check passes before and after. This generic rigid collision correction applies with stress enabled or disabled, changes producer contact trajectories, and requires new current-source four-scope performance baselines;the old failure-reset rate is not a matched optimization baseline. Broader seeds,grasp outcomes,per-environment tail work,current profiling and applicable optimization/plateau remain pending. No physical mesh/integration convergence or RL training throughput claim.","files":entries}
path = destination/"manifest.json"
path.write_text(json.dumps(manifest,indent=2)+"\n")
for row in entries:
    assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
    content = (destination/row["artifact"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
    restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
    assert hashlib.sha256(restored).hexdigest() == row["sha256"]
print(json.dumps({"entries":len(entries),"unique_blobs":len(blobs),"stored_bytes":sum(path.stat().st_size for path in blobs),
                  "manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"tests":test_summary,
                  "full_fields":48,"maximum_wrench_error":summary["force_or_moment_error_max"],
                  "maximum_full_residual_budget_fraction":summary["complete_residual_budget_fraction_max"],
                  "maximum_peak_difference_Pa":summary["peak_difference_Pa_max"],"B4096_live":summary["B4096_live"]},indent=2),flush=True)
