"""Compare retained nearby-load stress responses at two unchanged support samplings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []
    for group in (0, 2):
        results = []
        for index in (group, group + 1):
            path = args.campaign / f"group{index}/study/result.json"
            receipts.append(file_id(path))
            result = json.loads(path.read_text())
            start_path = path.parent / "start.json"
            receipts.append(file_id(start_path))
            start = json.loads(start_path.read_text())
            assert result["passed"] and len(result["rows"]) == 1
            assert start["quadrature"] == (25 if index == group else 81)
            assert start["budgets"] == [0., 0.001, 0.01, 0.1]
            results.append((result["rows"][0], start))
        low, high = results[0][0], results[1][0]
        assert results[0][1]["state"] == results[1][1]["state"]
        assert low["environment"] == high["environment"]
        for row in (low, high):
            assert all(not b["accepted"] for b in row["budgets"])
            assert row["budgets"][-1]["variant"] == "unbudgeted-projection"
            assert row["budgets"][-1]["physically_evaluable"]
            for b in row["budgets"]:
                if b["budget"] is not None:
                    assert all(s["status"] == "PrimalInfeasible" for s in b["sources"])
        fields = []
        for row in (low, high):
            receipt = row["budgets"][-1]["field"]
            assert file_id(receipt["path"]) == receipt
            receipts.append(receipt)
            with np.load(receipt["path"]) as saved:
                fields.append({k: saved[k] for k in
                    ("vertices", "tetrahedra", "stress_pa", "baseline_stress_pa", "delta_stress_pa")})
        np.testing.assert_array_equal(fields[0]["vertices"], fields[1]["vertices"])
        np.testing.assert_array_equal(fields[0]["tetrahedra"], fields[1]["tetrahedra"])
        vertices, tets = fields[1]["vertices"], fields[1]["tetrahedra"]
        volumes = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
        measurements = {}
        baseline_scale = field_norm(fields[1]["baseline_stress_pa"], volumes)
        for name in ("baseline_stress_pa", "stress_pa", "delta_stress_pa"):
            change = field_norm(fields[0][name] - fields[1][name], volumes)
            measurements[name] = {"rms_change_pa": float(change / np.sqrt(volumes.sum())),
                "relative_to_q81_own_field": float(change / field_norm(fields[1][name], volumes)),
                "relative_to_q81_baseline": float(change / baseline_scale)}
        rows.append({"environment": low["environment"], "complete_field_sampling": measurements,
            "q25_stress_relative_correction": low["budgets"][-1]["complete_stress_relative_change"],
            "q81_stress_relative_correction": high["budgets"][-1]["complete_stress_relative_change"],
            "original_small_budgets_all_unresolved": True})
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": True, "rows": rows, "receipts": receipts, "engineering_approximation_admitted": False,
        "scope": "Compare full weighted tensors on identical physical volume. This measures angular sampling of the declared local prior and admissible traction, not continuum accuracy. PrimalInfeasible is the sampled optimizer outcome; original continuous certificates are separate. No small predeclared budget is accepted, and the unbudgeted diagnostic is never an accepted observation."})


if __name__ == "__main__":
    main()
