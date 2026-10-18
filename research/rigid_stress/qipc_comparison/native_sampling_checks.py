"""Run the selected native sampling/lifecycle checks inside a captured Slurm worker."""

import argparse
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import genesis

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "SLURM_JOB_ID" in os.environ
    root = Path.cwd()
    assert Path(genesis.__file__).resolve() == root / "genesis/__init__.py"
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted((root / "tests").rglob("*.py"))
    paths.extend((root / "pyproject.toml", root / "conftest.py"))
    paths = [path for path in paths if path.is_file()]
    before = [file_id(path) for path in paths]
    engine_paths = sorted((root / "genesis").rglob("*.py"))
    engine_before = [file_id(path) for path in engine_paths]
    for path in paths:
        destination = args.output / "test-source" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lifecycle = "tests/rigid/test_stress.py::test_native_rigid_lifecycle_and_partial_reset"
    suffixes = (
        ("manual-False-full-4", "manual-False-max-1", "every_step-True-full-4")
        if args.backend == "cpu"
        else ("manual-False-full-1", "manual-True-full-4", "manual-False-max-4", "every_step-True-full-4")
    )
    tests = [f"{lifecycle}[{suffix}]" for suffix in suffixes]
    tests.append("tests/rigid/test_stress.py::test_native_full_field_unbatched_shared_operator[manual]")
    execution_root, environment = root, os.environ.copy()
    negative_import = None
    if args.negative_control:
        assert args.backend == "cuda"
        execution_root = Path(os.environ["TMPDIR"]) / "sampling-negative-source"
        package = execution_root / "genesis"
        shutil.copytree(
            root / "genesis", package, symlinks=True, ignore=shutil.ignore_patterns("assets", "__pycache__")
        )
        (package / "assets").symlink_to(root / "genesis/assets", target_is_directory=True)
        target = package / "engine/solvers/rigid/rigid_solver.py"
        original = target.read_text()
        condition = "if self.stress_recovery is None or not self.stress_recovery.is_update_requested:"
        assert original.count(condition) == 1
        altered = original.replace(condition, "if self.stress_recovery is None:")
        target.write_text(altered)
        (args.output / "negative-control.patch").write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    altered.splitlines(True),
                    fromfile="original",
                    tofile="unconditional-fused-recovery",
                )
            )
        )
        environment["PYTHONPATH"] = str(execution_root) + os.pathsep + environment["PYTHONPATH"]
        probe = subprocess.run(
            [sys.executable, "-c", "import genesis; print(genesis.__file__)"],
            cwd=execution_root,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        negative_import = probe.stdout.strip()
        assert negative_import == str(package / "__init__.py"), negative_import
        tests = [str(root / f"{lifecycle}[manual-True-full-4]")]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "--backend",
        args.backend,
        "--tb=short",
        "-q",
        "--junitxml",
        str(args.output / "report.xml"),
        "-o",
        f"cache_dir={args.output / 'pytest-cache'}",
        "--basetemp",
        str(args.output / "pytest-tmp"),
        *tests,
    ]
    if args.negative_control:
        command.extend(("--import-mode=importlib", "--maxfail=1"))
    with (args.output / "tests.log").open("w") as log:
        result = subprocess.run(
            command, cwd=execution_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    after = [file_id(path) for path in paths]
    assert before == after, "Executed test source changed during this worker"
    assert engine_before == [file_id(path) for path in engine_paths], "Engine source changed during this worker"
    expected_failure = (
        args.negative_control and result.returncode == 1 and "previous_peak" in (args.output / "tests.log").read_text()
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "native-manual-stress-sampling-focused-checks",
            "identity": runtime_identity(),
            "genesis_import": genesis.__file__,
            "negative_control": args.negative_control,
            "negative_import": negative_import,
            "expected_snapshot_failure_observed": expected_failure,
            "command": command,
            "test_sources": before,
            "engine_sources": engine_before,
            "source_unchanged": True,
            "returncode": result.returncode,
            "log": file_id(args.output / "tests.log"),
            "scope": "Existing lifecycle and shared-operator capabilities, extended with manual requests. Numerical contact/field oracles, substep maxima, retained snapshots, partial reset/restore and ordinary rigid trajectories remain checked. Separate profiling must verify skipped recovery work and practical cost.",
        },
    )
    raise SystemExit(0 if expected_failure else (1 if args.negative_control else result.returncode))


if __name__ == "__main__":
    main()
