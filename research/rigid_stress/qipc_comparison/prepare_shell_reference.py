"""Install the pinned optional independent FEM reference into a task-local directory."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--only-binary=:all:",
            "--target",
            str(args.output / "site"),
            "--cache-dir",
            str(args.output / "cache"),
            "--report",
            str(args.output / "install.json"),
            "pyfe3d==0.10.0",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
