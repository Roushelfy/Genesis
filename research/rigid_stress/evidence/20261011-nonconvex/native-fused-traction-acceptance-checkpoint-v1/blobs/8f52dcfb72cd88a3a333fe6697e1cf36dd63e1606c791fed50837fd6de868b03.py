"""Fused original-wrench/cone validation and signed P2 scatter, GPU live and independent field qualification."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
names = [f"native-fused-traction-acceptance-regression-v1/{backend}" for backend in ("cpu", "cuda")]
names += [f"native-fused-componentwise-regression-v2/{backend}" for backend in ("cpu", "cuda")]
names += [f"native-fused-traction-acceptance-v1/b2048-varied-{scope}-{mode}-623102" for scope in ("profile", "live") for mode in ("off", "on")]
runs = [root / name for name in names]
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
tests, fields = {}, []
for backend in ("cpu", "cuda"):
    suites = [ElementTree.parse(root / name / backend / "report.xml").getroot().find("testsuite")
              for name in ("native-fused-traction-acceptance-regression-v1", "native-fused-componentwise-regression-v2")]
    for suite, count in zip(suites, (23, 6)):
        assert suite.attrib["tests"] == str(count)
        assert suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    tests[backend] = {"passed": 23, "seconds": float(suites[0].attrib["time"]),
                      "additional_cached_componentwise_cases": 6, "componentwise_seconds": float(suites[1].attrib["time"])}
    for case in suites[0].findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
assert len(fields) == 16 and sum(len(row["native_peak_Pa"]) for row in fields) == 48
profiles, live = {}, {}
for mode in ("off", "on"):
    profile = json.loads((root / f"native-fused-traction-acceptance-v1/b2048-varied-profile-{mode}-623102/result.json").read_text())
    result = json.loads((root / f"native-fused-traction-acceptance-v1/b2048-varied-live-{mode}-623102/result.json").read_text())
    assert profile["fused_traction_acceptance"] == result["fused_traction_acceptance"] == (mode == "on")
    snapshot = profile["recovery_snapshot_check"]
    assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
    assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
    assert result["envs"] == 2048 and result["scope"] == "live" and result["seed"] == 623102
    assert result["dt"] == .01 and result["substeps"] == 1 and result["precision"] == "64"
    assert result["intrinsic_load_partition"] == "body_source_prior" and len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["steps"] == 1280 and row["resets"] == 2323
        assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
        assert max(row["failed_steps_per_environment"]) == 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
    assert max(budgets) <= 1
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
    profiles[mode] = {"stage_wall_ms": profile["stage_wall_ms"], "snapshot": snapshot,
                      "gpu_uuid": profile["gpu_uuid"], "native_buffer_bytes": profile["native_buffer_bytes"]}
    live[mode] = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                  "repeats": [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in result["repeats"]],
                  "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"],
                  "device_free_bytes": result["device_free_bytes"], "warmup_full_residual_budget_fraction_max": max(budgets)}
assert profiles["off"]["gpu_uuid"] == profiles["on"]["gpu_uuid"]
assert live["off"]["gpu_uuid"] == live["on"]["gpu_uuid"]
failed = root / "native-fused-componentwise-regression-v1/cpu"
assert (failed / "returncode.txt").read_text().strip() == "1"
assert "cannot create weak reference to 'numpy.float64'" in (failed / "tests.log").read_text()
assert (failed / "source.sha256").read_bytes() == (failed / "source-after.sha256").read_bytes()
runs.append(failed)
summary = {"tests": tests, "recorded_independent_full_fields": 48, "additional_cached_componentwise_field_checks": 36,
           "fields": fields, "profiles": profiles, "live": live,
           "live_speedup": live["on"]["effective_env_steps_per_second"] / live["off"]["effective_env_steps_per_second"],
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
extras = list(root.glob("fused-*-allocation.log"))
extras += [root / name for name in ("archive_native_fused_traction_acceptance_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_fused_traction_acceptance_v1.sh", "run_fused_traction_acceptance_profile_pair_v1.sh", "run_fused_traction_acceptance_live_pair_v1.sh")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-fused-traction-acceptance-checkpoint-v1"), runs, extras, summary,
        "Fused final projection preserves original-point wrench and local cones, caching only six resultant components plus cone validity, and scatters signed P2 basis contributions without a force-cloud buffer. Final1e-8 wrench acceptance is unchanged. Componentwise source contributions use separate atomics rather than first summing components; independent FP64 field checks qualify the rounding difference. CPU23/CUDA23 plus six extra componentwise cases on each backend pass. The v1 test adapter's NumPy-scalar template warning causes one CPU failure and is preserved; v2 passes a native Python float required by Quadrants. Its pending v1 CUDA job was cancelled before execution. Matched live/profile pairs include selected EPA/seed/sample-major/port pruning, without exact capacity or device frontier count; subsequent combinations require qualification. Goal, broad four-scope/seeds/batches, tail costs, dynamic profiles and plateau remain active. No physical convergence or RL training claim.")
