"""Compare complete saved native friction scenes across source variants."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, bindings = [], []
    for original in sorted(args.reference.glob("full*.npz"), key=lambda p: int(p.stem[4:])):
        changed = args.candidate / original.name
        bindings.extend([file_id(original), file_id(changed)])
        with np.load(original) as left, np.load(changed) as right:
            for key in [
                "contact_valid",
                "contact_status",
                "source_geom",
                "source_pair",
                "source_status",
                "law_status",
                "step_valid",
                "binding_lost",
            ]:
                np.testing.assert_array_equal(left[key], right[key])
            assert left["step_valid"].all() and right["step_valid"].all()
            metrics = {}
            for key in [
                "corner_tensor_pa",
                "probe_tensor_pa",
                "vm_pa",
                "principal_pa",
                "friction_line",
                "friction_coefficients",
            ]:
                np.testing.assert_array_equal(np.isfinite(left[key]), np.isfinite(right[key]))
                first, second = np.nan_to_num(left[key]), np.nan_to_num(right[key])
                relative = float(
                    np.linalg.norm(first - second) / max(np.linalg.norm(first), np.linalg.norm(second), 1e-30)
                )
                assert relative < 1e-7, (original.name, key, relative)
                metrics[key] = relative
            np.testing.assert_allclose(left["position"], right["position"], rtol=0, atol=1e-12)
            rows.append({"frame": original.stem, "relative_differences": metrics})
    assert rows
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "bindings": bindings,
            "scope": "Complete saved native force-driven scenes under matching source/configuration, including source/validity, corners/probes, physical history, sampled peaks and trajectory. Numerical conformance only; no independent physical contact-law admission.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
