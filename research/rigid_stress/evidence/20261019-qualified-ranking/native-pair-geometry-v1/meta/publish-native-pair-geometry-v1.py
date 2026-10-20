"""Publish the terminal pair-geometry checks and retain both setup failures."""

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

RUN = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking")
HELPER_SOURCE = RUN / "meta/publish-resolved-pad-and-inputs-v1.py"
exec(compile(HELPER_SOURCE.read_text().split("\nresolved = ", 1)[0], str(HELPER_SOURCE), "exec"))

campaigns = [f"native-pair-geometry-tests-v{version}" for version in (1, 2, 3)]
jobs = [read(RUN / name / "submission.json")["job"] for name in campaigns]
accounting = subprocess.check_output(
    ["sacct", "--array", "-j", ",".join(jobs), "-n", "-P", "--format=JobID,State,ExitCode,Elapsed,End,NodeList"],
    text=True,
)
observed = {line.split("|")[0]: line.split("|")[1:3] for line in accounting.splitlines() if "|" in line}
checks = []
files, external = collect(campaigns)
files.append(HELPER_SOURCE)
for version, campaign, job in zip((1, 2, 3), campaigns, jobs):
    root = RUN / campaign
    engine = root / "engine-source/pair_geometry.py"
    test = root / "test-source/test_stress_pair_geometry.py"
    files.extend([engine, test])
    inventory = read(root / "source.json")
    assert inventory["revision"] == "e0eaeb7a94d9ff54c56948beeb71a09f25382106"
    record = next(row for row in inventory["files"] if row["path"].endswith("/stress/pair_geometry.py"))
    assert record["sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert inventory["patch_sha256"] == hashlib.sha256((root / "source.patch").read_bytes()).hexdigest()
    for group, backend in enumerate(("gpu", "cpu")):
        prefix = root / f"group{group}"
        expected = ["COMPLETED", "0:0"] if version == 3 else ["FAILED", "1:0"]
        assert observed[f"{job}_{group}"] == expected
        result = read(prefix / "tests/result.json")
        assert result["passed"] == (version == 3)
        assert (prefix / "returncode.txt").read_text().strip() == ("0" if version == 3 else "1")
        start = read(prefix / "tests/start.json")
        assert start["tests"][0]["sha256"] == hashlib.sha256(test.read_bytes()).hexdigest()
        cases = list(ET.parse(prefix / "tests/junit.xml").iter("testcase"))
        assert len(cases) == 4
        failures = [case.find("failure") for case in cases if case.find("failure") is not None]
        assert all(case.find("error") is None and case.find("skipped") is None for case in cases)
        assert len(failures) == (4 if version == 1 else 2 if version == 2 else 0)
        if version == 1:
            assert all("preserve_mesh_precision" in failure.attrib["message"] for failure in failures)
        elif version == 2:
            assert all("set_vel" in failure.attrib["message"] for failure in failures)
        files.append(prefix / "tests/junit.xml")
        checks.append(
            {
                "campaign": campaign,
                "backend": backend,
                "scheduler_state": expected,
                "cases": [{"name": case.attrib["name"], "passed": case.find("failure") is None} for case in cases],
                "engine_source": identity(engine),
                "test_source": identity(test),
            }
        )

source = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/automatic-pad-geometry")
for relative, directory in (
    ("genesis/engine/solvers/rigid/stress/pair_geometry.py", "engine-source"),
    ("tests/rigid/test_stress_pair_geometry.py", "test-source"),
):
    assert (source / relative).read_bytes() == (RUN / campaigns[-1] / directory / Path(relative).name).read_bytes()

receipt = {
    "passed": True,
    "terminal_checks": checks,
    "scheduler_accounting": accounting,
    "geometry_and_velocity_absolute_tolerance": 1e-12,
    "split_source_force_absolute_tolerance_n": 1e-14,
    "split_source_moment_absolute_tolerance_nm": 1e-16,
    "checks_are_component_controls": True,
    "automatic_pressure_model_admitted": False,
    "friction_history_validated": False,
    "production_throughput_measured": False,
    "whole_goal_complete": False,
}
receipt_path = RUN / "meta/native-pair-geometry-terminal-v1.json"
with receipt_path.open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
files.append(receipt_path)
scope = (
    "CPU/GPU component controls for current opposing geometry and link-pair source grouping. "
    "All four final cases pass on each backend: analytic primitive and mesh queries with live pose changes; "
    "U-shaped/cavity boundary distance and material occlusion; selected-environment translation, velocity and reset; "
    "two-geoms-per-link grouping with unchanged split-source force/moment and invalid-source isolation. "
    "Earlier Mesh-option and velocity-setter test setup failures are retained with their exact source. "
    "Queries describe nearest opposing boundaries and do not establish complete reachable support, a pressure law, "
    "friction history, production lifecycle behavior, or throughput. The extended goal remains active."
)
packet = freeze("native-pair-geometry-v1", files, sorted(external), scope)
target = DEST / packet["packet"]
(target / "README.md").write_text(
    "# Native current-geometry component checks\n\n"
    + scope
    + "\n\nThe manifest binds each retained payload to its data-side source, size and SHA-256. "
    "Compressed files retain both original and encoded identities. "
    "The captured patch applies to public commit `e0eaeb7a94d9ff54c56948beeb71a09f25382106`; "
    "each campaign also includes the exact new engine and test files. "
    "Pytest start receipts bind the executed tests, and JUnit records preserve individual outcomes. "
    "Runtime outputs and private caches remain outside the source checkout.\n"
)
print(json.dumps(packet, indent=2))
