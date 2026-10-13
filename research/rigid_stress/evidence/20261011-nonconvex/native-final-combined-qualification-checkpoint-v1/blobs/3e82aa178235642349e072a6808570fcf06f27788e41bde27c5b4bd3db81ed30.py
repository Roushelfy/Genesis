"""Publish closed CPU/CUDA combined qualification and its complete matched live pair."""
import ast
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from publish_native_bounded_checkpoint_v1 import publish_bounded

root = Path(__file__).parent
pairs_path = root / "native-final-combined-live-qualified-pairs-v5.json"
pairs = json.loads(pairs_path.read_text())
runs, reports, fields = [], {}, []
for backend in ("cpu", "cuda"):
    run = root / f"native-final-combined-regression-v2/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "35" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[backend] = dict(suite.attrib)
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            props = {p.attrib["name"]: ast.literal_eval(p.attrib["value"]) for p in case.find("properties")}
            assert max(props["force_errors"] + props["moment_errors"]) <= 1e-8
            assert max(props["complete_residual_budget_fraction"]) <= 1
            assert max(props["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **props})
    runs.append(run)
for route in ("baseline", "candidate"):
    run = root / f"native-final-combined-live-v1/b16384-{route}-991909"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    runs.append(run)
# The live and regression snapshots must identify the same candidate engine.
for path in (runs[0] / "source/genesis/engine/solvers/rigid/stress").glob("*.py"):
    for run in (runs[1], runs[3]):
        assert path.read_bytes() == (run / "source/genesis/engine/solvers/rigid/stress" / path.name).read_bytes()
references = {}
for name in ("native-final-combined-attempt-checkpoint-v2", "native-final-isolation-budget-checkpoint-v1",
             "native-final-combined-dynamic-checkpoint-v1", "native-final-combined-wrench-oracle-checkpoint-v1"):
    manifest = Path("research/rigid_stress/evidence/20261011-nonconvex") / name / "manifest.json"
    references[name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
summary = {**pairs, "qualification_complete_for_this_combination": True, "reports": reports,
    "fields": fields, "independent_fixture_fields": sum(len(row["native_peak_Pa"]) for row in fields),
    "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
    "complete_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
    "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
    "referenced_manifest_sha256": references,
    "remaining": "Final-source matched scopes/scaling and final optimization/cost plateau documentation remain required. This combination qualification does not mark the goal complete."}
extras = [pairs_path, root / "native-final-combined-live-check-pairs-v5.log"]
extras += [root / name for name in ("archive_native_final_combined_qualification_v1.py",
    "run_archive_native_final_combined_qualification_v1.sh", "check_native_final_memory_pairs_v5.py",
    "run_check_native_final_memory_pairs_v5.sh", "publish_native_checkpoint_v1.py")]
extras += list(root.glob("native-final-combined-regression-v2*-allocation.log"))
publish_bounded(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-final-combined-qualification-checkpoint-v1"),
    runs, extras, summary,
    "Clean four-candidate combination.35CPU/35CUDA cases and96 independent fixture full fields retain final wrench1e-8, complete residual including6gauge max(1e-11N,1e-8||rhs||), stress RTOL1e-4/ATOL1e-3Pa. Max/full native checkpoint, partial reset, release, source failure isolation and exact cached/direct positions included. Same-source matched live B16384/991909 bank,600warmup/three1280-step repeats, all environments and outcomes retained. The older failed repeat-precision assertions and their corrected budget-aware isolation proof are explicitly referenced, not removed. Captured dynamic464 and original-wrench/inertia-relief160 additional fields separately pass CPU FP64.200ms assigned-device samples include setup and are not exact peaks. No physical convergence, full RL training or goal completion claim.")
