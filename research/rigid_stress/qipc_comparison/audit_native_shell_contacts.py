"""Compare complete native shell loads and fields across fixed source variants."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--cases", nargs="+", default=["flat", "warped", "egg"])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", dict(identity=runtime_identity(require_gpu_inventory=False)))
    rows = []
    for name in args.cases:
        before = args.before / (name + "-fields.npz")
        after = args.after / (name + "-fields.npz")
        with np.load(before) as left, np.load(after) as right:
            errors = {}
            for key in ["actual_load", "actual_field"]:
                assert left[key].shape == right[key].shape
                errors[key] = float(np.linalg.norm(left[key] - right[key]) / max(np.linalg.norm(left[key]), 1e-30))
            rows.append(dict(name=name, before=file_id(before), after=file_id(after), errors=errors))
        write_json(args.output / "result.json", dict(passed=False, complete=False, rows=rows))
        assert errors["actual_load"] < 1e-10 and errors["actual_field"] < 1e-8, rows[-1]
    write_json(args.output / "result.json", dict(passed=True, complete=True, rows=rows))


if __name__ == "__main__":
    main()
