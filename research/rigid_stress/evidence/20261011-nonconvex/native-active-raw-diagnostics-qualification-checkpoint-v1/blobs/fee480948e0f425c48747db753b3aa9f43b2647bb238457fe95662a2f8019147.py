"""Publish closed candidate pairs and CPU/CUDA independent field qualifications."""

import argparse
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

from publish_native_bounded_checkpoint_v1 import publish_bounded

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("family", choices=("valid-source-capture", "coalesced-positions", "fp32-warm-jacobian", "active-raw-diagnostics"))
args = parser.parse_args()
root = Path(__file__).parent
pairs_path = root / f"native-{args.family}-qualified-pairs-v3.json"
pairs = json.loads(pairs_path.read_text())
assert pairs["family"] == args.family
runs, fields, regression, cached_differences = [], [], {}, []
for row in pairs["pairs"]:
    for mode in ("off", "on"):
        runs.append(root / f"native-{args.family}-v1/b{row['batch']}-varied-live-{mode}-{row['seed']}")
for backend in ("cpu", "cuda"):
    run = root / f"native-{args.family}-regression-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "31" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    current = []
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" not in case.attrib["name"]:
            continue
        properties = {item.attrib["name"]: ast.literal_eval(item.attrib["value"])
                      for item in case.find("properties").findall("property")}
        assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
        assert max(properties["complete_residual_budget_fraction"]) <= 1
        assert max(properties["peak_difference_Pa"]) <= 1e-3
        if "cached_position_max_difference_m" in properties:
            assert backend == "cuda" and properties["cached_position_max_difference_m"] <= 1e-15
            cached_differences.append(properties["cached_position_max_difference_m"])
        current.append({"backend": backend, "case": case.attrib["name"], **properties})
    assert len(current) == 16 and sum(len(row["native_peak_Pa"]) for row in current) == 48
    fields.extend(current)
    regression[backend] = {"passed": 31, "seconds": float(suite.attrib["time"]), "independent_full_fields": 48}
    runs.append(run)
if args.family == "fp32-warm-jacobian":
    for version in (1, 2, 3):
        smoke = root / f"native-fp32-warm-jacobian-smoke-v{version}/cuda"
        assert (smoke / "returncode.txt").read_text().strip() == ("0" if version == 3 else "1")
        assert (smoke / "source.sha256").read_bytes() == (smoke / "source-after.sha256").read_bytes()
        runs.append(smoke)
summary = {**pairs, "regression": regression, "recorded_independent_full_fields": 96, "fields": fields,
           "cached_vs_direct_position_case_count": len(cached_differences),
           "cached_vs_direct_position_difference_m_max": max(cached_differences, default=0),
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "global_peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
extras = [pairs_path, root / f"native-{args.family}-check-pairs-v3.log"]
extras.extend(root / name for name in ("archive_native_final_candidate_v1.py", "run_archive_native_final_candidate_v1.sh",
              "check_native_final_memory_pairs_v3.py", "run_check_native_final_memory_pairs_v3.sh", "publish_native_checkpoint_v1.py"))
extras.extend(root.glob(f"native-{args.family}-*allocation.log"))
publish_bounded(root, Path(f"research/rigid_stress/evidence/20261011-nonconvex/native-{args.family}-qualification-checkpoint-v1"),
                runs, extras, summary,
                "Private candidate source closures/patches, same-GPU bank-matched changing-contact off/on pairs,600warmup/three1280-step repeats,FP64/Q10/s8/body_source_prior/max,dt.01/sub1,RTX PRO6000 Blackwell. Both routes include every environment, observation/contact-radius updates/reset/metrics/history/fallback. All per-environment rigid/contact/grasp/descent counters and positions agree; zero invalid/overflow/failed lift outcomes. Independent CPU FP64 contact wrench/local cones/full signed P2 nodal load/K and six-gauge residual/full tensor/von Mises/global peak checks retain final1e-8 wrench,max(1e-11N,1e-8||rhs||) residual,RTOL1e-4/ATOL1e-3Pa.200ms assigned-device telemetry includes setup, not exact transient peaks. Candidate effects are separate comparisons and not additive. Clean production selection/combined qualification and final plateau remain separate; no physical convergence or full RL training claim.")
