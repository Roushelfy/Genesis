"""Run focused material-face ownership tests on the captured source."""

import argparse
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
    test = "tests/rigid/test_stress_nonconvex.py"
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
        f"cache_dir={args.output / 'cache'}",
        "--basetemp",
        str(args.output / "tmp"),
        "--junitxml",
        str(args.output / "tests.xml"),
        test + "::test_material_boundary_adjacency_uses_vertex_identity",
        test + "::test_material_boundary_rejects_interior_and_duplicate_faces",
    ]
    with (args.output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=args.source, stdout=log, stderr=subprocess.STDOUT)
    write_json(
        args.output / "result.json",
        {
            "passed": result.returncode == 0,
            "command": command,
            "test_source": file_id(args.source / test),
            "returncode": result.returncode,
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
