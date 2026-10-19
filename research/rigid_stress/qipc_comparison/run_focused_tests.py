"""Run selected observable regressions with a verified engine import and durable receipts."""

import argparse
from pathlib import Path

import pytest

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tests", type=Path, nargs="+", required=True)
    cli.add_argument("--select", required=True)
    cli.add_argument("--engine-root", type=Path, required=True)
    cli.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert Path(gs.__file__).resolve().is_relative_to(args.engine_root.resolve())
    command = [
        *map(str, args.tests),
        "-k",
        args.select,
        "--backend",
        args.backend,
        "-n",
        "0",
        "-v",
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        f"--junitxml={args.output / 'junit.xml'}",
    ]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "engine_import": gs.__file__,
            "tests": [file_id(path) for path in args.tests],
            "pytest_arguments": command,
        },
    )
    code = pytest.main(command)
    write_json(args.output / "result.json", {"exit_code": int(code), "passed": code == 0})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
