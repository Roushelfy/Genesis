"""Run the selected native sampling/lifecycle checks inside a captured Slurm worker."""

import argparse
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
    with (args.output / "tests.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    after = [file_id(path) for path in paths]
    assert before == after, "Executed test source changed during this worker"
    assert engine_before == [file_id(path) for path in engine_paths], "Engine source changed during this worker"
    write_json(
        args.output / "result.json",
        {
            "kind": "native-manual-stress-sampling-focused-checks",
            "identity": runtime_identity(),
            "genesis_import": genesis.__file__,
            "command": command,
            "test_sources": before,
            "engine_sources": engine_before,
            "source_unchanged": True,
            "returncode": result.returncode,
            "log": file_id(args.output / "tests.log"),
            "scope": "Existing lifecycle and shared-operator capabilities, extended with manual requests. Numerical contact/field oracles, substep maxima, retained snapshots, partial reset/restore and ordinary rigid trajectories remain checked. Separate profiling must verify skipped recovery work and practical cost.",
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
