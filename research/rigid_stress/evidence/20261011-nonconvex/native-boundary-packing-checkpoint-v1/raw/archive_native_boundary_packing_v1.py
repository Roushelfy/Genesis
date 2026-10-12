"""Publish qualified native boundary fallback, bounded GPU packing and copied B1024 scopes."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-boundary-packing-checkpoint-v1")
assert not destination.exists()
folders = ["native-boundary-certificate-regression-v2", "native-boundary-certificate-benchmark-v2", "sparse-contact-pack-regression-v2", "sparse-contact-pack-benchmark-v2", "native-sparse-pack-regression-v1", "native-sparse-packing-cases-v1", "native-sparse-pack-live-v1", "copied-native-four-scopes-b1024-v1"]
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
files = []
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    closures = [path for path in (root / folder).rglob("source.sha256") if not any(part in skip for part in path.relative_to(root / folder).parts)]
    assert closures, folder
    for before in closures:
        assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
for backend in ("cpu", "cuda"):
    assert "56 passed" in (root / f"native-boundary-certificate-regression-v2/{backend}/tests.log").read_text()
    assert "10 passed" in (root / f"native-sparse-pack-regression-v1/{backend}/tests.log").read_text()
assert "10 passed" in (root / "sparse-contact-pack-regression-v2/cuda/tests.log").read_text()


def load(path):
    return json.loads((root / path).read_text())


def valid_rate(result):
    assert len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
        if row["failed_steps_per_environment"] is None:
            assert result["scope"] == "rigid"
        else:
            assert max(row["failed_steps_per_environment"]) == 0
        np.testing.assert_allclose(row["transitions_per_second"], result["envs"] * row["steps"] / row["seconds"], rtol=1e-12)
    return result["envs"] * sum(row["steps"] for row in result["repeats"]) / sum(row["seconds"] for row in result["repeats"])


def matched(folder, variants, nested="live"):
    results = [load(f"{folder}/{nested + '/' if nested else ''}{variant}/result.json") for variant in variants]
    left, right = results
    for name in ("gpu_uuid", "dt", "substeps", "envs", "conditions_sha256", "condition_count", "seed", "load_model", "precision", "dofs", "tetrahedra", "quadrature", "output_mode", "method_selected", "effective_contact_block_size", "intrinsic_support", "egg_position_m", "cooperative_tractions", "warm_start_tractions"):
        assert left[name] == right[name], (folder, name)
    peak_difference, residual_fraction = 0.0, 0.0
    for a, b in zip(left["warmup_quality"], right["warmup_quality"]):
        for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts", "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
            assert a[name] == b[name], (folder, name)
        np.testing.assert_allclose(b["maximum_stress_Pa"], a["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
        peak_difference = max(peak_difference, float(np.max(np.abs(np.array(a["maximum_stress_Pa"]) - np.array(b["maximum_stress_Pa"])))))
        for row in (a, b):
            fraction = float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
            assert fraction <= 1
            residual_fraction = max(residual_fraction, fraction)
    rates = {name: valid_rate(result) for name, result in zip(variants, results)}
    for result in results:
        for row in result["repeats"]:
            assert row["steps"] == 1280 and row["resets"] == 37
    return {"matched_live_rates_env_steps_per_second": rates, "speedup": rates[variants[1]] / rates[variants[0]], "maximum_logged_peak_difference_Pa": peak_difference, "maximum_complete_residual_budget_fraction": residual_fraction, "gpu_uuid": left["gpu_uuid"], "native_buffer_bytes": [result["native_buffer_bytes"] for result in results], "repeats": {name: [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in result["repeats"]] for name, result in zip(variants, results)}}


boundary = matched("native-boundary-certificate-benchmark-v2", ("sampled", "boundary"))
candidate = matched("sparse-contact-pack-benchmark-v2", ("native", "sparse_pack"))
native = matched("native-sparse-pack-live-v1", ("linear_scan", "native_hybrid"), nested="")
assert native["speedup"] > 1
profiles = {}
for folder, variants in [("native-boundary-certificate-benchmark-v2", ("sampled", "boundary")), ("sparse-contact-pack-benchmark-v2", ("native", "sparse_pack"))]:
    profiles[folder] = {}
    for variant in variants:
        result = load(f"{folder}/profile/{variant}/result.json")
        check = result["recovery_snapshot_check"]
        assert check["original_point_max_difference_m"] == 0
        assert max(check["normalized_force_error_max"], check["normalized_moment_error_max"]) <= 1e-8
        assert check["nodal_force_max_difference_N"] <= 1e-12
        fraction = float(np.max(np.array(result["complete_residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(result["rhs_norm_N"]))))
        assert fraction <= 1
        profiles[folder][variant] = {"stage_wall_ms": result["stage_wall_ms"], "pressure_subdivision_wall_ms": result["pressure_subdivision_wall_ms"], "recovery_snapshot_check": check, "complete_residual_budget_fraction": fraction}
copied = {}
for scope in ("rigid", "recovery", "live"):
    assert (root / f"copied-native-four-scopes-b1024-v1/{scope}/returncode.txt").read_text().strip() == "0"
    result = load(f"copied-native-four-scopes-b1024-v1/{scope}/result.json")
    assert result["envs"] == 1024 and result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
    rate = valid_rate(result)
    copied[scope] = {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / 1024, "repeats": [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in result["repeats"]], "native_buffer_bytes": result.get("native_buffer_bytes"), "torch_peak_allocated_bytes": result.get("torch_peak_allocated_bytes"), "device_free_bytes": result["device_free_bytes"], "gpu_uuid": result["gpu_uuid"], "recovery_snapshot_check": result.get("recovery_snapshot_check"), "source_revision": result["source_revision"]}
assert (root / "copied-native-four-scopes-b1024-v1/policy/returncode.txt").read_text().strip() == "1"
policy_log = (root / "copied-native-four-scopes-b1024-v1/policy/benchmark.log").read_text()
assert "env=6, contact=2, status=34" in policy_log and "24 additional preserved environment failures" in policy_log
assert not (root / "copied-native-four-scopes-b1024-v1/policy/result.json").exists()
copied["policy"] = {"valid_throughput": None, "outcome": "Warmup aborted on sampled-wrench incompatibility, continuum unresolved;32 copied environments (6 modulo32) preserve the contact failure. No policy inference or training throughput is claimed."}
packing_cases = {backend: load(f"native-sparse-packing-cases-v1/{backend}/results/summary.json") for backend in ("cpu", "cuda")}
for records in packing_cases.values():
    assert [row["case"] for row in records] == ["sparse", "mixed", "dense", "released", "restored"]
for name in ("archive_native_boundary_packing_v1.py", "run_archive_native_boundary_packing_v1.sh", "run_native_boundary_certificate_regression_v2.sh", "run_native_boundary_certificate_benchmark_v2.sh", "probe_sampled_wrench_search_v2.py", "run_sparse_contact_pack_regression_v2.sh", "run_sparse_contact_pack_benchmark_v2.sh", "probe_sparse_contact_pack_v2.py", "sparse_contact_pack_candidate_v2.py", "run_native_sparse_pack_regression_v1.sh", "run_native_sparse_packing_cases_v1.sh", "probe_sparse_packing_cases_v1.py", "run_native_sparse_pack_live_v1.sh", "parallel_contact_clear_candidate_v1.py", "probe_parallel_contact_clear_v1.py", "run_copied_native_four_scopes_b1024_v1.sh"):
    files.append(root / name)
destination.mkdir(parents=True)
entries, external = [], []
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000 or (original.suffix == ".npz" and original.stat().st_size >= 1_000_000):
        external.append({"original": str(original), "bytes": original.stat().st_size, "sha256": digest})
    else:
        artifact = destination / "raw" / original.relative_to(root)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, artifact)
        entries.append({"original": str(original), "artifact": str(artifact.relative_to(destination)), "bytes": original.stat().st_size, "sha256": digest})
manifest = {"scope": __doc__, "source_base_revision": "cd76762e", "native_tests": {"boundary_fallback_cpu": "56 passed475.98s", "boundary_fallback_cuda": "56 passed668.20s", "packing_cpu": "10 passed119.77s", "packing_cuda": "10 passed137.72s", "packing_candidate_cuda": "10 passed113.54s"}, "boundary_fallback": boundary, "packing_candidate": candidate, "native_packing": native, "profiles": profiles, "native_packing_cases": packing_cases, "copied_b1024_scopes": copied, "limitations": "Diagnostic strengthening does not recover a physically incompatible individual component. Healthy-scene matched timing retains the same footprint/FP64/error budgets. Sparse packing preserves exact contact order and all contacts;lists above32 use the original linear scan,CPU keeps that scan. B1024 uses32 copied conditions and sourcecd76762e before these new changes;it is supplemental scaling evidence,not a diverse large-batch qualification or optimum. Recovery-only replays one frozen contact snapshot. Rigid-only minimum10s runs more steps/reset cycles than live,so these scope rates are not an exactly matched trajectory slowdown comparison. Policy warmup failure is retained;no valid policy throughput exists. No whole-goal completion,physical convergence or optimization plateau is claimed.", "files": entries, "external_files": external}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "boundary_fallback": boundary, "packing_candidate": candidate, "native_packing": native, "copied_b1024_scopes": copied}, indent=2), flush=True)
