"""Publish matched sample-memory/tail candidates and their unchanged-budget qualifications."""

import ast
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
families = ("compact-cached-positions", "parallel-refined-fan", "cooperative-refined-fan", "sample-contiguous-current")
versions = (1, 2, 1, 1)
runs, pairs = [], {}
for family, version in zip(families, versions):
    check = root / f"native-{family}-qualified-pairs-v1.json"
    data = json.loads(check.read_text())
    assert data["family"] == family
    pairs[family] = data["pairs"]
    for pair in data["pairs"]:
        assert pair["identical_per_environment_rigid_contact_grasp_counters"]
        assert pair["rigid_position_difference_m_max"] <= 1e-12
        for mode in ("off", "on"):
            run = root / f"native-{family}-v{version}/b{pair['batch']}-varied-live-{mode}-{pair['seed']}"
            assert (run / "returncode.txt").read_text().strip() == "0"
            assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
            runs.append(run)
qualifications, fields = {}, []
for family, version, backend, count in (
    ("parallel-refined-fan", 2, "cpu", 23),
    ("parallel-refined-fan", 2, "cuda", 23),
    ("cooperative-refined-fan", 1, "cuda", 23),
    ("sample-contiguous-current", 1, "cuda", 31),
):
    run = root / f"native-{family}-regression-v{version}/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == str(count)
    assert suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    current_fields = []
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            properties = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
            assert max(properties["complete_residual_budget_fraction"]) <= 1
            assert max(properties["peak_difference_Pa"]) <= 1e-3
            current_fields.append({"family": family, "backend": backend, "case": case.attrib["name"], **properties})
    expected_cases = 16 if count == 31 else 8
    assert len(current_fields) == expected_cases
    field_count = sum(len(row["native_peak_Pa"]) for row in current_fields)
    assert field_count == expected_cases * 3
    qualifications[f"{family}/{backend}"] = {"passed": count, "seconds": float(suite.attrib["time"]), "recorded_independent_full_fields": field_count}
    fields.extend(current_fields)
    runs.append(run)
cache_manifest = Path("research/rigid_stress/evidence/20261011-nonconvex/native-cached-fan-positions-checkpoint-v1/manifest.json")
cache_sha = hashlib.sha256(cache_manifest.read_bytes()).hexdigest()
assert cache_sha == "2ec2cfb3e63237e2fe765e7adc808158c9539c83ee39e43f9928e0e919ba4b19"
cache_qualification = json.loads(cache_manifest.read_text())["summary"]
assert cache_qualification["recorded_independent_full_fields"] == 96
failed = [root / "native-parallel-refined-fan-v1/b2048-varied-live-off-623102",
          root / "native-parallel-refined-fan-regression-v1/cpu",
          root / "native-parallel-refined-fan-regression-v1/cuda",
          root / "native-parallel-refined-fan-smoke-v1/cuda"]
for run in failed:
    assert (run / "returncode.txt").read_text().strip() == "1"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
runs.extend(failed)
for run in (root / "native-parallel-refined-fan-smoke-v2/cpu", root / "native-parallel-refined-fan-smoke-v2/cuda",
            root / "native-cooperative-refined-fan-smoke-v1/cuda"):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    runs.append(run)
summary = {
    "pairs": pairs, "qualifications": qualifications, "independent_fields": fields,
    "recorded_independent_full_fields": sum(len(row["native_peak_Pa"]) for row in fields),
    "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
    "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
    "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
    "prior_counted_position_qualification_manifest_sha256": cache_sha,
    "prior_counted_position_qualification": {key: cache_qualification[key] for key in ("regression", "recorded_independent_full_fields", "cached_vs_direct_position_difference_m_max", "wrench_error_max", "full_residual_budget_fraction_max", "peak_difference_Pa_max")},
    "selection": {"compact-cached-positions": "Qualified end-to-end gain atB2048 and separated-seedB8192; combined production cleanup/qualification pending.",
                  "parallel-refined-fan": "Qualified1.017149258x live gain; integration with counted positions and production cleanup/qualification pending.",
                  "cooperative-refined-fan": "Not selected:1.002938634x is sub-percent and within ordinary repeat variation.",
                  "sample-contiguous-current": "Not selected:1.003031333x is sub-percent and within ordinary repeat variation."},
    "failed_parallel_first_version": "A temporary underscore loop variable was overwritten by a returned vector. CUDA10 failed13 passed; CPU recorded failures followed by compiler segmentation fault; benchmark warmup and targeted GPU smoke failed. All raw records and source closures are preserved; removing the unnecessary loop fixes the implementation.",
}
extras = [root / name for name in (
    "archive_native_contact_tail_storage_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_contact_tail_storage_v1.sh",
    "check_native_pairs_v1.py", "run_check_native_pairs_v1.sh", "native-check-pairs-v1-allocation.log",
    "run_compact_cached_positions_pair_v1.sh", "run_parallel_refined_fan_pair_v2.sh", "run_cooperative_refined_fan_pair_v1.sh", "run_sample_contiguous_current_pair_v1.sh",
    "native-compact-cached-positions-pair-v1-b2048-allocation.log", "native-compact-cached-positions-pair-v1-b8192-allocation.log",
    "native-parallel-refined-fan-pair-v2-allocation.log", "native-cooperative-refined-fan-pair-v1-allocation.log", "native-sample-contiguous-current-pair-v1-allocation.log",
)]
extras.extend(root / f"native-{family}-qualified-pairs-v1.json" for family in families)
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-contact-tail-storage-checkpoint-v1"), runs, extras, summary,
        "Low full hollow shell FP64/Q10/s8,body_source_prior,mesh64/GJK patch-on,dt0.01/substep1 and max output. Within each pair only storage/scheduling changes; every physical sample, contact, radius, friction and final wrench/full gauge residual/stress budget remains unchanged. Each route600 warmup plus three1280-step changing live repeats with timed observations/contact updates/reset/metrics/history/fallback; all environment failures and grasp outcomes are verified. Seed734001/B8192 is separated from623102/B2048. Complete counted positions are qualified by the separately published31 CPU/31 CUDA matrix and96 independent fields. Refined fan parallel sampling has23 CPU/23 CUDA and48 fields; cooperative fan refit has23 CUDA/24 fields; scalar profile/index layout has31 CUDA/48 fields. The latter two tiny gains are unselected. Fan/math order may change floating-point results while unchanged final budgets pass. Assigned-device memory is sampled200ms including setup, not an exact transient maximum. GreaterB live probes, combined production implementation/qualification, final matched four scopes/seeds and a plateau remain pending. No individual environment GPU wall-time attribution, physical convergence or RL training claim.")
