"""Run existing witnessed-source and nonconvex ambiguity regressions."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["tests/rigid/test_stress_sources.py", "tests/rigid/test_stress_nonconvex.py"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "--backend",
        args.backend,
        "-q",
        "-o",
        f"cache_dir={Path(os.environ['TMPDIR']) / 'source-bounds-pytest-cache'}",
        "--basetemp",
        str(Path(os.environ["TMPDIR"]) / "source-bounds-pytest-temp"),
        "--junitxml",
        str(args.output / "tests.xml"),
        files[0],
        files[1] + "::test_native_source_provenance_ambiguity_and_offset_wrench",
    ]
    with (args.output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=args.source, stdout=log, stderr=subprocess.STDOUT)
    write_json(
        args.output / "result.json",
        {
            "passed": result.returncode == 0,
            "command": command,
            "test_sources": [file_id(args.source / name) for name in files],
            "returncode": result.returncode,
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
