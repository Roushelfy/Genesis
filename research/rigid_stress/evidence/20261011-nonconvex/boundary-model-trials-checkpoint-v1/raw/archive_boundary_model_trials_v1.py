"""Publish closed diagnostic/model/packing trials, including failed or unselected epochs."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/boundary-model-trials-checkpoint-v1")
assert not destination.exists()
folders = ["boundary-penalty-native-samples-v1", "boundary-native-continuous-v1", "native-boundary-certificate-regression-v1", "native-boundary-certificate-benchmark-v1", "sparse-contact-pack-regression-v1", "sparse-contact-pack-benchmark-v1", "sparse-packing-cases-v1", "aggregate-wrench-fields-v1"]
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
files = []
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
cpu_log = (root / "native-boundary-certificate-regression-v1/cpu/tests.log").read_text()
assert "55 passed" in cpu_log and "1 failed" in cpu_log and "-3.179773792205687e-05" in cpu_log
assert "56 passed" in (root / "native-boundary-certificate-regression-v1/cuda/tests.log").read_text()
assert "10 passed" in (root / "sparse-contact-pack-regression-v1/cuda/tests.log").read_text()
for scope in ("profile", "live"):
    assert "QuadrantsRuntimeTypeError" in (root / f"native-boundary-certificate-benchmark-v1/{scope}/sampled/benchmark.log").read_text()

left, right = [json.loads((root / f"sparse-contact-pack-benchmark-v1/live/{name}/result.json").read_text()) for name in ("native", "sparse_pack")]
for name in ("gpu_uuid", "dt", "substeps", "envs", "conditions_sha256", "condition_count", "seed", "load_model", "precision", "dofs", "tetrahedra", "quadrature", "output_mode", "method_selected", "native_buffer_bytes", "effective_contact_block_size", "intrinsic_support", "egg_position_m", "cooperative_tractions", "warm_start_tractions"):
    assert left[name] == right[name], name
peak_difference, residual_fraction = 0.0, 0.0
for a, b in zip(left["warmup_quality"], right["warmup_quality"]):
    for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts", "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
        assert a[name] == b[name], name
    np.testing.assert_allclose(b["maximum_stress_Pa"], a["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
    peak_difference = max(peak_difference, float(np.max(np.abs(np.array(a["maximum_stress_Pa"]) - np.array(b["maximum_stress_Pa"])))))
    for row in (a, b):
        fraction = float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
        assert fraction <= 1
        residual_fraction = max(residual_fraction, fraction)
rates = {}
for name, result in zip(("native", "sparse_pack"), (left, right)):
    assert len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["steps"] == 1280 and row["resets"] == 37 and row["invalid_environment_steps"] == 0
        assert row["scatter_overflow_calls"] == 0 and max(row["failed_steps_per_environment"]) == 0
    rates[name] = 32 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
packing = {"matched_live_rates_env_steps_per_second": rates, "speedup": rates["sparse_pack"] / rates["native"], "maximum_logged_peak_difference_Pa": peak_difference, "maximum_complete_residual_budget_fraction": residual_fraction, "gpu_uuid": left["gpu_uuid"], "native_buffer_bytes": left["native_buffer_bytes"], "cases": {backend: json.loads((root / f"sparse-packing-cases-v1/{backend}/results/summary.json").read_text()) for backend in ("cpu", "cuda")}, "selection": "Quadratic sparse packing is unselected because dense GPU cost reaches25.42 ms. The hybrid removes quadratic tails and passes CPU/GPU ordering/clearing cases; its full-trajectory qualification is a separate epoch."}
aggregate = {backend: json.loads((root / f"aggregate-wrench-fields-v1/{backend}/results/summary.json").read_text()) for backend in ("cpu", "cuda")}
for backend, records in aggregate.items():
    assert [row["tick"] for row in records] == [602, 876]
    for row in records:
        assert max(row["force_error"] + row["moment_error"]) <= 1e-8
        assert max(row["cone_excess_N"]) <= 1e-12 and max(row["complete_residual_budget_fraction"]) <= 1
        np.testing.assert_allclose(row["native_peak_Pa"], row["oracle_peak_Pa"], rtol=1e-4, atol=1e-3)
for a, b in zip(aggregate["cpu"], aggregate["cuda"]):
    assert a["actual_friction"] == b["actual_friction"]
    np.testing.assert_allclose(a["native_peak_Pa"], b["native_peak_Pa"], rtol=1e-4, atol=1e-3)
diagnostics = {backend: json.loads((root / f"boundary-penalty-native-samples-v1/{backend}/results/summary.json").read_text()) for backend in ("cpu", "cuda")}
for records in diagnostics.values():
    assert records[0]["status"] == [0, 3] and records[-1]["status"] == [0, 6]
    assert records[-1]["search_bound"][1] < 0 and records[-1]["certificate_margin"][1] > 1e-3
audit = json.loads((root / "boundary-native-continuous-v1/results/summary.json").read_text())
assert audit["classification"] == "continuous_incompatible"
for name in ("probe_boundary_penalty_native_samples_v1.py", "run_boundary_penalty_native_samples_v1.sh", "probe_boundary_native_continuous_v1.py", "run_boundary_native_continuous_v1.sh", "boundary_wrench_penalty_candidate_v1.py", "run_native_boundary_certificate_regression_v1.sh", "run_native_boundary_certificate_benchmark_v1.sh", "probe_sampled_wrench_search_v1.py", "sparse_contact_pack_candidate_v1.py", "sparse_contact_pack_candidate_v2.py", "probe_sparse_contact_pack_v1.py", "run_sparse_contact_pack_regression_v1.sh", "run_sparse_contact_pack_benchmark_v1.sh", "probe_sparse_packing_cases_v1.py", "run_sparse_packing_cases_v1.sh", "probe_aggregate_wrench_fields_v1.py", "run_aggregate_wrench_fields_v1.sh", "archive_boundary_model_trials_v1.py", "run_archive_boundary_model_trials_v1.sh"):
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
manifest = {"scope": __doc__, "source_base_revision": "cd76762e", "direct_boundary_regression": {"cpu": "55 passed,1 failed: different but valid weaker ray violates the old fixed fixture gap assertion; replace-only design is not selected.", "cuda": "56 passed", "benchmark": "Legacy baseline has a typed-workspace class error. No matched diagnostic cost or speedup is claimed; frozen failing logs are retained."}, "native_sample_diagnostics": diagnostics, "independent_continuous_audit": audit, "sparse_packing": packing, "aggregate_pair_experiment": aggregate, "aggregate_limitations": "Unselected, explicitly different conservation object: complete pair wrench instead of every point component.12 independently validated full fields across CPU/GPU,two saved fixtures and three actual friction settings. Fit oracle shares the selected native geometry/profile cloud but no fitted native forces or duals; elasticity assembly/recovery is independent. CPU/GPU sample counts differ near face degeneracies and remain an adoption concern; both oracle/peak comparisons pass. No live or policy model, lifecycle matrix, complete large-batch result or production model selection is claimed.", "files": entries, "external_files": external}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sparse_packing": packing, "aggregate_pair_experiment": aggregate}, indent=2), flush=True)
