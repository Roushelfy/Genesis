"""Compare complete matrix-capture values and bytes without conflating long-double storage padding with values."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.first, args.second]:
        closed_source(run)
    assert (args.first / "parent-source.sha256").read_bytes() == (args.second / "parent-source.sha256").read_bytes()
    rows, inputs = [], {}
    for name in ["first_dynamic_matrix.npz", "original_coordinate_map.npz", "first_dynamic_state.npz"]:
        paths = [run / "results" / name for run in [args.first, args.second]]
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        inputs.update(zip(map(str, paths), hashes))
        fields = []
        with np.load(paths[0]) as first, np.load(paths[1]) as second:
            assert first.files == second.files
            for key in first.files:
                a, b = first[key], second[key]
                assert a.dtype == b.dtype and a.shape == b.shape
                np.testing.assert_array_equal(a, b)
                value = {
                    "array": key,
                    "dtype": str(a.dtype),
                    "shape": a.shape,
                    "values_exact": True,
                    "storage_bytes_identical": a.tobytes() == b.tobytes(),
                }
                if not value["storage_bytes_identical"]:
                    assert a.dtype == np.longdouble and a.itemsize == 16 and np.finfo(a.dtype).nmant == 63
                    assert a.flags.c_contiguous and b.flags.c_contiguous
                    changed = np.flatnonzero(a.view(np.uint8).reshape(-1) != b.view(np.uint8).reshape(-1)) % 16
                    positions = np.unique(changed).tolist()
                    assert all(position >= 10 for position in positions)
                    value["different_storage_byte_positions_modulo16"] = positions
                    value["only_nonvalue_x87_long_double_padding_differs"] = True
                fields.append(value)
        rows.append(
            {
                "file": name,
                "file_bytes_identical": hashes[0] == hashes[1],
                "all_arrays_values_exact": True,
                "arrays": fields,
            }
        )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "first": str(args.first),
        "second": str(args.second),
        "actual_producer_codes": [int((run / "returncode.txt").read_text()) for run in [args.first, args.second]],
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Every array value, dtype and shape must match exactly. Actual file/storage byte differences remain explicit; only directly verified trailing nonvalue bytes of the80-bit x87 number in a16-byte NumPy long-double slot are labeled padding. This does not assert raw-file identity or accept a physical state.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
