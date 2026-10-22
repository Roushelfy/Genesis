"""Run the public preset/full-field contract against the explicitly selected native checkout."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-node", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.source) + os.pathsep + environment.get("PYTHONPATH", "")
    import_command = [sys.executable, "-c", "import genesis; print(genesis.__file__)"]
    imported = subprocess.check_output(import_command, cwd=args.source, env=environment, text=True).strip()
    assert imported.startswith(str(args.source) + "/"), imported
    rows = []
    for backend in ("cpu", "cuda"):
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "--backend",
            backend,
            "-q",
            "-o",
            f"cache_dir={args.output / (backend + '-cache')}",
            "--basetemp",
            str(args.output / (backend + "-tmp")),
            "--junitxml",
            str(args.output / (backend + ".xml")),
            *(args.test_node or ["tests/rigid/test_stress.py::test_native_full_field_unbatched_shared_operator"]),
        ]
        with (args.output / (backend + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        rows.append({"backend": backend, "command": command, "returncode": result.returncode})
        write_json(
            args.output / "result.json",
            {
                "passed": all(row["returncode"] == 0 for row in rows),
                "kind": "public-preset-and-full-field-contract",
                "genesis_import": imported,
                "checks": rows,
            },
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
