"""Publish frozen intrinsic-profile checks, actual sliding fixtures and failed development runs."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--source-revision", required=True)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
assert not args.destination.exists()
runs = (
    "footprint-v1", "footprint-regression-v1-cpu", "footprint-regression-v1-cuda",
    "live-mapping-v1", "footprint-regression-v2-cpu", "footprint-regression-v2-cuda",
    "live-mapping-v2", "live-mapping-v3", "fit-diagnosis-v1",
    "fit-diagnosis-fan-v1", "live-mapping-fan-v1", "footprint-regression-v3-cpu",
    "fit-diagnosis-fan-v2", "live-mapping-fan-v2",
    "footprint-regression-v4-cpu", "footprint-regression-v4-cuda",
    "sliding-footprint-v1-cpu", "sliding-footprint-v1-cuda",
    "live-mapping-fan-v3", "footprint-regression-v5-cpu", "footprint-regression-v5-cuda",
)
checks = (
    ("footprint-v1/cpu.log", "2 passed", False),
    ("footprint-v1/cuda.log", "2 passed", False),
    ("footprint-regression-v1-cpu/tests.log", "22 passed", False),
    ("footprint-regression-v1-cuda/tests.log", "22 passed", False),
    ("footprint-regression-v2-cpu/tests.log", "22 passed", False),
    ("footprint-regression-v2-cuda/tests.log", "22 passed", False),
    ("footprint-regression-v4-cpu/tests.log", "5 failed, 19 passed", True),
    ("sliding-footprint-v1-cpu/tests.log", "2 failed", True),
    ("sliding-footprint-v1-cuda/tests.log", "2 failed", True),
    ("footprint-regression-v5-cpu/tests.log", "26 passed", False),
    ("footprint-regression-v5-cuda/tests.log", "26 passed", False),
)
for name, expected, has_failure in checks:
    output = re.sub(r"\x1b\[[0-9;]*m", "", (args.raw/name).read_text())
    assert expected in output, name
    assert bool(re.search(r"\b\d+ failed\b", output)) == has_failure, name

sources_by_run = {}
closures = {}
for run in runs:
    before = args.raw/run/"source.sha256"
    after = args.raw/run/"source-after.sha256"
    if not after.exists():
        after = args.raw/run/"source-after-cancellation.sha256"
    assert before.read_bytes() == after.read_bytes(), run
    sources = dict(line.split("  ",1)[::-1] for line in before.read_text().splitlines())
    for name, expected in sources.items():
        assert not Path(name).is_absolute()
        assert digest(args.raw/run/"source"/name) == expected, (run,name)
    sources_by_run[run] = sources
    closures[run] = after.name
final_sources = sources_by_run["footprint-regression-v5-cpu"]
assert final_sources == sources_by_run["footprint-regression-v5-cuda"]
for name, expected in final_sources.items():
    assert digest(args.repo/name) == expected, name
probe_sources = sources_by_run["live-mapping-fan-v3"]
for name, expected in probe_sources.items():
    assert digest(args.repo/name) == expected, name
    if name in final_sources:
        assert final_sources[name] == expected, name

baseline = json.loads((args.raw/"fit-diagnosis-v1/results.json").read_text())
assert len(baseline["contacts"]) == 2
for contact in baseline["contacts"]:
    outer = next(lp for lp in contact["lp"] if lp["sides"] == 128 and lp["outer"])
    assert outer["status"] == 2
    assert all(native["status"] == 3 for native in contact["native"])
fan = json.loads((args.raw/"fit-diagnosis-fan-v2/results.json").read_text())
assert len(fan["contacts"]) == 2
for contact in fan["contacts"]:
    assert all(lp["status"] == 0 for lp in contact["lp"] if not lp["outer"])
    assert all(native["status"] == 0 for native in contact["native"])
    assert all(max(native["force_error"],native["moment_error"]) <= 1e-9 for native in contact["native"])

snapshots = {}
for run in ("live-mapping-fan-v2", "live-mapping-fan-v3"):
    results = json.loads((args.raw/run/"results.json").read_text())
    assert results["native_contact_shape"] == [750,4]
    assert results["native_sample_count"] == 8200
    assert results["reset_environments"] == 2
    assert results["mapped_source_vertices"] == results["source_vertex_count"] == 42
    assert len(results["captures"]) == 7
    for record in results["captures"]:
        assert record["source_failures"] == record["footprint_failures"] == record["fit_failures"] == record["invalid_envs"] == 0
        assert max(record["max_wrench_errors"][:2]) <= 1e-8
        assert record["max_wrench_errors"][2] <= 1e-12
        assert record["max_wrench_errors"][3] >= -1e-12
    assert sum(record["loaded_contacts"] for record in results["captures"]) == 117
    snapshots[run] = {key:results[key] for key in ("scope","arguments","native_contact_shape","native_sample_count","reset_environments","captures")}

artifacts = []
args.destination.mkdir(parents=True)


def copy_artifact(original, relative):
    target = args.destination/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original,target)
    expected = digest(original)
    assert digest(target) == expected
    artifacts.append({"path":str(relative),"original":str(original),"sha256":expected})


for run in runs:
    for original in sorted((args.raw/run).iterdir()):
        if original.is_file():
            copy_artifact(original,Path(run)/original.name)
    for name in sources_by_run[run]:
        copy_artifact(args.raw/run/"source"/name,Path(run)/"source"/name)
prefixes = (
    "run_footprint", "run_live_mapping", "run_fit_diagnosis", "run_sliding_footprint",
    "run_force_feasibility", "run_continuum_inspection", "diagnose_live_mapping",
    "footprint-regression", "sliding-footprint", "fan-v1-cancellation", "archive_footprint",
)
for original in sorted(args.raw.iterdir()):
    if original.is_file() and original.name.startswith(prefixes):
        copy_artifact(original,Path(original.name))
copy_artifact(args.raw/"footprint-archive-support/check_nonconvex_contact.sh",Path("check_nonconvex_contact.sh"))
layout = []
for batch in (1024,2048):
    cells = 8200*750*batch
    layout.append({"batch":batch,"profile_GiB":cells*8/2**30,"sample_geometry_GiB":cells*100/2**30,"combined_GiB":cells*108/2**30})
manifest = {
    "source_revision":args.source_revision,
    "requirement_revision":"1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "native_nonconvex_goal_complete":False,
    "scope":"Native intrinsic profile/local integration and standalone vector recovery, including intrusive snapshots on the unchanged old-law live trajectory. No new-law live or policy throughput acceptance.",
    "final_checks":{"cpu":"26 passed","cuda":"26 passed","snapshots":snapshots},
    "checks":[{"log":name,"result":expected} for name,expected,_ in checks],
    "source_closures":closures,
    "contact_repair":{
        "vertex_identity":"Known same-geom collision vertices bind only to a unique reference P1/P2 boundary node within the declared 1e-8 m correspondence bound; runtime witness equality is checked at floating-point roundoff.",
        "profile":"Certified intrinsic port paths with verified covered planar segments; positive area Gaussian/taper profile; no orientation mask.",
        "local_integration":"Q10 on the witnessed face's three anchor/edge triangles; other faces keep Q10. Original radius, application point, force, free moment and friction are unchanged.",
        "feasibility":"Original sampled cones are infeasible under a circumscribed 128-sided LP; a looser internal 5e-9 threshold and 1024 evaluations do not fix them. The finite anchor partition gives constructive inner-cone LP witnesses and native convergence in 20/19 evaluations.",
        "lp_boundary_resolution":"The repaired LP additionally includes the exact target force ray wherever it lies in the true local cone; this avoids classifying near-sliding force as infeasible solely because of angular polygon resolution.",
        "acceptance":"Wrench 1e-8; complete residual including all six gauge rows max(1e-11 N,1e-8 ||rhs||); stress rtol=1e-4,atol=1e-3 Pa.",
    },
    "development_failures":[
        "live-mapping-v2: noncontiguous public getter tensors rejected by Quadrants interop; corrected with contiguous observation/geometry copies.",
        "fit-diagnosis-fan-v1: typed tensor extent is runtime, so qd.static cannot evaluate its comparison; corrected using compile-time tensor dimension.",
        "live-mapping-fan-v1 and footprint-regression-v3: cancelled after that authoritative compilation failure; separately closed unchanged source hashes, no acceptance claimed.",
        "footprint-regression-v4: zero-length scalar ndarray placeholder is invalid; CPU 5 failed/19 passed, CUDA cancelled; corrected to scalar inactive placeholders.",
        "sliding-footprint-v1: CPU oracle used in-place multiplication across an expanding broadcast dimension; corrected without changing its equations or numerical budgets.",
    ],
    "dense_layout_estimates":{"scope":"Declared storage size, not measured GPU peak memory; excludes ports, solver and all other state. This is a proof layout, not the selected live layout.","sample_geometry_bytes_per_cell":100,"rows":layout},
    "remaining":[
        "General primitive/SDF/decomposition and aggregate source association",
        "General local integration/distance refinement and explicit continuum incompatibility diagnostics",
        "Compact contact/candidate layout with shared geometry and small local sample storage",
        "Native live scene lifecycle, full trajectories, resets and state restoration under the new law",
        "New matched four-scope GPU baseline, profiling, batch exploration and measured optimization plateau",
    ],
    "artifacts":artifacts,
}
manifest_path = args.destination/"manifest.json"
manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
for artifact in artifacts:
    assert digest(args.destination/artifact["path"]) == artifact["sha256"]
    assert digest(Path(artifact["original"])) == artifact["sha256"]
print(json.dumps({"artifacts_verified":len(artifacts),"manifest_sha256":digest(manifest_path)}),flush=True)
