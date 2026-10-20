"""Retain accessible-boundary controls, their failing baselines and exact sources."""

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

RUN = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking")
HELPER_SOURCE = RUN / "meta/publish-resolved-pad-and-inputs-v1.py"
exec(compile(HELPER_SOURCE.read_text().split("\nresolved = ", 1)[0], str(HELPER_SOURCE), "exec"))

campaigns = ["native-pair-accessible-baseline-v1", "native-pair-accessible-tests-v1", "native-pair-accessible-tests-v2"]
jobs = [read(RUN / name / "submission.json")["job"] for name in campaigns]
accounting = subprocess.check_output(
    ["sacct", "--array", "-j", ",".join(jobs), "-n", "-P", "--format=JobID,State,ExitCode,Elapsed,End,NodeList"], text=True
)
observed = {line.split("|")[0]: line.split("|")[1:3] for line in accounting.splitlines() if "|" in line}
files, external = collect(campaigns)
files.append(HELPER_SOURCE)
records = []
for stage, campaign, job in zip((0, 1, 2), campaigns, jobs):
    root = RUN / campaign
    engine, test = root / "engine-source/pair_geometry.py", root / "test-source/test_stress_pair_geometry.py"
    files.extend([engine, test])
    source = read(root / "source.json")
    engine_record = next(row for row in source["files"] if row["path"].endswith("/stress/pair_geometry.py"))
    assert engine_record["sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert source["patch_sha256"] == hashlib.sha256((root / "source.patch").read_bytes()).hexdigest()
    for group, backend in enumerate(("cpu",) if stage == 0 else ("gpu", "cpu")):
        prefix = root / f"group{group}"
        state = ["COMPLETED", "0:0"] if stage == 2 else ["FAILED", "1:0"]
        assert observed[f"{job}_{group}"] == state
        result = read(prefix / "tests/result.json")
        assert result["passed"] == (stage == 2)
        assert (prefix / "returncode.txt").read_text().strip() == ("0" if stage == 2 else "1")
        start = read(prefix / "tests/start.json")
        assert start["tests"][0]["sha256"] == hashlib.sha256(test.read_bytes()).hexdigest()
        cases = list(ET.parse(prefix / "tests/junit.xml").iter("testcase"))
        assert len(cases) == (1 if stage == 0 else 5)
        assert all(case.find("error") is None and case.find("skipped") is None for case in cases)
        failures = [case for case in cases if case.find("failure") is not None]
        assert len(failures) == (0 if stage == 2 else 1)
        if failures:
            assert failures[0].attrib["name"] == "test_pair_accessible_boundary_selection[u]"
        if stage == 0:
            assert "16 / 16" in failures[0].find("failure").attrib["message"]
        files.append(prefix / "tests/junit.xml")
        records.append({
            "campaign": campaign, "backend": backend, "scheduler_state": state,
            "source_revision": source["revision"], "engine_source": identity(engine), "test_source": identity(test),
            "cases": [{"name": case.attrib["name"], "passed": case.find("failure") is None} for case in cases],
        })

working = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/automatic-pad-geometry")
for relative, directory in (("genesis/engine/solvers/rigid/stress/pair_geometry.py", "engine-source"), ("tests/rigid/test_stress_pair_geometry.py", "test-source")):
    assert (working / relative).read_bytes() == (RUN / campaigns[-1] / directory / Path(relative).name).read_bytes()
receipt = {
    "passed": True, "records": records, "scheduler_accounting": accounting,
    "baseline_missed_selected_samples": 16,
    "final_geometry_cases_per_backend": 5,
    "primitive_segment_cases_per_backend": 18,
    "accessible_representations": ["two_boxes", "two_boxes_reversed", "one_mesh"],
    "blocked_representations": ["three_boxes", "one_mesh"],
    "selected_material_samples_per_environment": 8,
    "environments": 2, "common_pose_change_and_partial_reset_checked": True,
    "geometry_absolute_tolerance_m": 1e-12,
    "pressure_model_admitted": False, "throughput_measured": False, "whole_goal_complete": False,
}
path = RUN / "meta/native-pair-accessible-terminal-v1.json"
with path.open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
files.append(path)
scope = (
    "Accessible opposing-boundary queries with material-side, local mesh stationarity and two-link occlusion checks. "
    "The published nearest-only baseline misses all16 selected samples in a two-environment U-shaped control. "
    "The first candidate-search revision retains a false-accessibility failure in the blocked mesh case. "
    "The final source passes five cases on each of CPU/GPU, including three equivalent accessible representations, "
    "analytic/mesh transverse blockers, common rigid transforms, isolation, resets and18 primitive segment controls per backend. "
    "The original source-splitting force/moment checks also pass. "
    "These are geometry component results, not pressure/friction-history validation or production throughput. "
    "Composite interior union distance, unsupported shapes and pressure/lifecycle integration remain open."
)
packet = freeze("native-pair-accessible-v1", files, sorted(external), scope)
(DEST / packet["packet"] / "README.md").write_text(
    "# Accessible opposing-boundary component\n\n" + scope + "\n\n"
    "Each campaign retains its actual base revision and patch, executed engine/test files, pytest start hashes, "
    "individual JUnit outcomes and terminal scheduler accounting. The manifest preserves original and encoded "
    "payload sizes and SHA-256. Failed observations stay unchanged; numerical outputs and caches stay on data or node scratch.\n"
)
print(json.dumps(packet, indent=2))
