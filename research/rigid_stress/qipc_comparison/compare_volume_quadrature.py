"""Compare pressure-quadrature loads and complete fields on the preserved failed frame."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .check_native_atlas import relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--candidate", choices=("angle16", "angle32", "angle64"), default="angle16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {path.parent.name: path.parent for path in args.campaign.glob("group*/*/load-field.npz")}
    assert set(runs) == {"angle16", "angle32", "angle64", "angle128", "edge64", "edge128"}
    fields = {}
    for name, path in runs.items():
        report = json.loads((path / "result.json").read_text())
        assert report["complete"] and report["field_valid"] and report["scatter_status"] == 0
        with np.load(path / "load-field.npz") as data:
            fields[name] = dict(data)
    rows = []
    for name, data in sorted(fields.items()):
        errors = {key: relative(value, fields["angle128"][key]) for key, value in data.items()}
        admitted = (
            errors["force"] < 1e-6 and max(errors[key] for key in ("displacement", "stress_tensor", "peak")) < 1e-4
        )
        rows.append(dict(name=name, errors=errors, admitted=admitted, input=file_id(runs[name] / "load-field.npz")))
    result = dict(
        identity=runtime_identity(require_gpu_inventory=False), reference="angle128", rows=rows, complete=False
    )
    write_json(args.output / "result.json", result)
    result["candidate"] = args.candidate
    assert next(row["admitted"] for row in rows if row["name"] == args.candidate)
    assert next(row["admitted"] for row in rows if row["name"] == "edge128")
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
