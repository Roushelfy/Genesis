"""Preserve isolation diagnostics, unchanged-budget oracle checks and test corrections."""
import ast
from pathlib import Path
from xml.etree import ElementTree

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
names = ("native-final-combined-lifecycle-v1", "native-final-combined-lifecycle-v2",
         "native-final-combined-lifecycle-fp64-v1", "native-final-combined-lifecycle-v3",
         "native-final-combined-native-lifecycle-v1")
runs, reports = [], {}
for name, failures in zip(names, (1, 1, 0, 0, 2)):
    run = root / name / "cuda"
    assert (run / "returncode.txt").read_text().strip() == str(int(failures > 0))
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "2" and int(suite.attrib["failures"]) == failures
    assert suite.attrib["errors"] == suite.attrib["skipped"] == "0"
    reports[name] = dict(suite.attrib)
    runs.append(run)
case = ElementTree.parse(runs[3] / "report.xml").getroot().find("testsuite").findall("testcase")[1]
props = {p.attrib["name"]: ast.literal_eval(p.attrib["value"]) for p in case.find("properties")}
assert max(props["unaffected_environment_force_errors"]) <= 1e-8
assert max(props["original_wrench_errors"]) <= 1e-8
assert max(props["complete_residual_budget_fractions"]) <= 1
summary = {"reports": reports, "budget_checks": props,
    "scope": "Two CUDA shared lifecycle cases with independent equal-history controls, complete CPU FP64 stiffness/gauge residuals and original-point aggregate wrench checks. Complete35-case combined qualification and final-source scopes are separate.",
    "diagnosis": "Strict comparison already fails on two independent initial solves before any release:2.1169926922581794e-11 N at block12. Forced FP64 Hessian passes the strict comparison. Final physical budgets remain wrench1e-8, complete residual max(1e-11N,1e-8||rhs||), stress RTOL1e-4/ATOL1e-3Pa. The final test compares controls within those budgets, preserves exact release/reset observation checks, and adds independent original wrench and complete K/gauge checks.",
    "native_test_correction": "The first new native checkpoint test incorrectly expected pre-restore displacement to survive restore notifications, which explicitly invalidate derived displacement/observations. Both failed source/logs are preserved. A corrected baseline/restore/partial-reset sequence is included in the separate35-case matrix."}
extras = [root / "archive_native_final_isolation_budget_v1.py", root / "run_archive_native_final_isolation_budget_v1.sh"]
extras += list(root.glob("native-final-combined-lifecycle*-allocation.log"))
extras += list(root.glob("native-final-combined-native-lifecycle-v1*-allocation.log"))
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-final-isolation-budget-checkpoint-v1"),
    runs, extras, summary, "All failed/passed diagnostic sources and reports retained. Unchanged final numerical budgets. This scope does not establish completion of the full goal,35-case matrix or final-source throughput plateau. No physical convergence or full RL training claim.")
