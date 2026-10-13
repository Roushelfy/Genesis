"""Preserve the first clean combination, including its failed lifecycle assertion."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

from publish_native_bounded_checkpoint_v1 import publish_bounded

root = Path(__file__).parent
pairs_path = root / "native-final-combined-live-qualified-pairs-v5.json"
pairs = json.loads(pairs_path.read_text())
runs, regression, fields = [], {}, []
for backend in ("cpu", "cuda"):
    for kind in ("smoke", "regression"):
        run = root / f"native-final-combined-{kind}-v1/{backend}"
        expected_failure = backend == "cuda" and kind == "regression"
        assert (run / "returncode.txt").read_text().strip() == str(int(expected_failure))
        assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
        assert int(suite.attrib["failures"]) == int(expected_failure)
        assert int(suite.attrib["tests"]) == (33 if kind == "regression" else 3)
        regression[f"{backend}-{kind}"] = dict(suite.attrib)
        if kind == "regression":
            for case in suite.findall("testcase"):
                if "wrench_cones_full_fields" in case.attrib["name"]:
                    properties = {p.attrib["name"]: ast.literal_eval(p.attrib["value"])
                                  for p in case.find("properties").findall("property")}
                    assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
                    assert max(properties["complete_residual_budget_fraction"]) <= 1
                    fields.append({"backend": backend, "case": case.attrib["name"], **properties})
        runs.append(run)
for route in ("baseline", "candidate"):
    runs.append(root / f"native-final-combined-live-v1/b16384-{route}-991909")
supplement = root / "native-final-combined-supplement-v1"
assert (supplement / "tests/rigid/test_stress_shared_lifecycle.py").read_bytes() == (
    root / "native-final-combined-regression-v1/cuda/source" /
    "tests/rigid/test_stress_shared_lifecycle.py").read_bytes() if (
        root / "native-final-combined-regression-v1/cuda/source/tests/rigid/test_stress_shared_lifecycle.py").exists() else True
runs.append(supplement)
extras = [pairs_path, root / "native-final-combined-live-check-pairs-v5.log"]
extras += [root / name for name in ("archive_native_final_combined_attempt_v1.py",
    "run_archive_native_final_combined_attempt_v1.sh", "check_native_final_memory_pairs_v5.py",
    "run_check_native_final_memory_pairs_v5.sh", "publish_native_checkpoint_v1.py")]
extras += list(root.glob("native-final-combined-*-v1-allocation.log"))
summary = {**pairs, "qualification_complete": False, "regression": regression,
    "recorded_independent_full_fields": sum(len(row["native_peak_Pa"]) for row in fields),
    "fields": fields, "failure": "CUDA shared lifecycle compared warm release output with the first cold output at force RTOL1e-12/ATOL1e-15 N; maximum violating difference3.48909721e-11 N. Investigation pending. Final physical wrench/residual/stress gates remain unchanged.",
    "source_note": "The regression-v1 hash list omitted the unchanged shared-lifecycle test. Its exact recorded Git-base blob and hash are supplied by native-final-combined-supplement-v1. Later runner versions include this test explicitly."}
publish_bounded(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-final-combined-attempt-checkpoint-v1"),
    runs, extras, summary,
    "First clean four-candidate combination. CPU33 passes and CUDA32 passes/1 lifecycle comparison fails. This is retained exploratory evidence, not complete numerical/lifecycle qualification or completion. Matched changing live B16384 pair600warmup/three1280steps retains all outcomes and environments. Same final accuracy budgets and FP64/Q10/s8/body_source_prior/max/dt.01/sub1. Independent CPU FP64 fields remain in the matrix; failed assertion and all logs are preserved.200ms assigned-device memory samples include setup and are not exact peaks. No physical convergence or full RL training claim.")
