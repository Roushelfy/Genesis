"""Publish two complete live cohorts and qualify prototype/clean CUDA against the unchanged CPU route."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from publish_native_bounded_checkpoint_v1 import publish_bounded

root = Path(__file__).parent
pairs_path = root / "native-sampler-warp-qualified-pairs-all-v6.json"
pairs = json.loads(pairs_path.read_text())
assert {(row["batch"], row["seed"]) for row in pairs["pairs"]} == {(8192, 623105), (16384, 734001)}
proof = json.loads((root / "native-sampler-warp-clean-proof-v1.json").read_text())
revision = "76537c2fb64cd469e83ba9921b0724b5c82a5f36"
relative = Path("genesis/engine/solvers/rigid/stress")
cpu = root / "native-final-combined-regression-v2/cpu"
prototype = root / "native-sampler-warp-regression-v1/cuda"
clean = root / "native-sampler-warp-clean-regression-v1/cuda"
reports, fields, runs = {}, [], []
for route, run in (("cpu_inherited", cpu), ("cuda_prototype", prototype), ("cuda_clean", clean)):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "35"
    assert suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[route] = dict(suite.attrib)
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            properties = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties")}
            assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
            assert max(properties["complete_residual_budget_fraction"]) <= 1
            assert max(properties["peak_difference_Pa"]) <= 1e-3
            fields.append({"route": route, "case": case.attrib["name"], **properties})
    runs.append(run)
for path in (clean / "source" / relative).glob("*.py"):
    assert path.read_bytes() == subprocess.check_output(["git", "show", f"{revision}:{relative / path.name}"])
    if path.name != "surface_frontier.py":
        assert path.read_bytes() == (prototype / "source" / relative / path.name).read_bytes()
        assert path.read_bytes() == (cpu / "source" / relative / path.name).read_bytes()
assert hashlib.sha256((prototype / "source" / relative / "surface_frontier.py").read_bytes()).hexdigest() == proof["prototype_source_sha256"]
assert hashlib.sha256((clean / "source" / relative / "surface_frontier.py").read_bytes()).hexdigest() == proof["clean_source_sha256"]
for pair in pairs["pairs"]:
    for route in ("off", "on"):
        run = root / f"native-sampler-warp-v1/b{pair['batch']}-varied-live-{route}-{pair['seed']}"
        assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        for path in (prototype / "source" / relative).glob("*.py"):
            assert path.read_bytes() == (run / "source" / relative / path.name).read_bytes()
        runs.append(run)
smoke = root / "native-sampler-warp-smoke-v1/cuda"
assert (smoke / "returncode.txt").read_text().strip() == "0"
assert (smoke / "source.sha256").read_bytes() == (smoke / "source-after.sha256").read_bytes()
smoke_suite = ElementTree.parse(smoke / "report.xml").getroot().find("testsuite")
assert smoke_suite.attrib["tests"] == "3"
assert smoke_suite.attrib["errors"] == smoke_suite.attrib["failures"] == smoke_suite.attrib["skipped"] == "0"
runs.append(smoke)
with np.load(clean / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
    edges = np.concatenate([asset["tetrahedra"][:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    p1_nodes = len(asset["vertices"])
    mesh = {"linear_geometry_nodes": p1_nodes, "P2_nodes": p1_nodes + len(np.unique(np.sort(edges, axis=1), axis=0)),
            "tetrahedra": len(asset["tetrahedra"]), "exterior_faces": len(asset["surface_triangles"])}
summary = {**pairs, "numerical_source_revision": revision, "reports": reports, "fields": fields, "mesh": mesh,
           "prototype_hard_growth_smoke": dict(smoke_suite.attrib),
           "cleanup_proof": proof, "independent_fixture_fields_by_route": {
               route: sum(len(row["native_peak_Pa"]) for row in fields if row["route"] == route) for route in reports},
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "global_peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
           "cpu_route_note": "CPU35 is inherited from the byte-identical CPU numerical/dispatch path. Surface frontier allocation and execution are CUDA-only. Prototype35 and clean35 are fresh CUDA runs.",
           "remaining": "Final source dynamic/original-wrench CPU audits, four-scope curves, copied-cohort plateau and final report remain required."}
extras = [pairs_path, root / "native-sampler-warp-check-pairs-all-v6.log", root / "native-sampler-warp-clean-proof-v1.json"]
extras += [root / name for name in ("archive_native_sampler_warp_qualification_v1.py", "run_archive_native_sampler_warp_qualification_v1.sh",
    "check_native_final_memory_pairs_v6.py", "run_check_native_final_memory_pairs_v6.sh", "clean_native_sampler_warp_v1.py",
    "run_clean_native_sampler_warp_v1.sh", "run_sampler_warp_pair_v1.sh", "run_sampler_warp_wide_pair_v1.sh", "publish_native_checkpoint_v1.py")]
extras += list(root.glob("native-sampler-warp*-allocation.log"))
publish_bounded(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-sampler-warp-qualification-checkpoint-v1"), runs, extras, summary,
    "Warp aggregation preserves every eligible Q10/s8 surface sample, original-point wrench, signed P2 scatter and local friction constraints. Each CUDA warp stays on one source, reduces complete counts/FP64 maximum distance and reserves one full sample interval. Radius, geometry/profile math and exact growth/replay remain unchanged. Prototype35CUDA and clean35CUDA pass; unchanged CPU route inherits35CPU. Recorded fixture fields retain wrench1e-8, full residual including6gauge max(1e-11N,1e-8||rhs||), stressRTOL1e-4/ATOL1e-3Pa. Two same-assigned-GPU off/on cohorts, B8192/623105 and B16384/734001, use identical banks,600warmup and three1280-step varied live repeats. Every environment/outcome/reset/fallback is included.200ms memory samples include setup and are not exact transient peaks. Policy/fullRL/physical convergence and final goal completion are not claimed.")
