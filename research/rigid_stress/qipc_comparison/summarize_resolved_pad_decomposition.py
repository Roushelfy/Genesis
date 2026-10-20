"""Bind signed transfer contrasts to admitted old/new native and reference fields."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.data_root
    new_path = root / "resolved-pad-ranking-v1/group0/ranking/result.json"
    old_path = root / "canonical-stvk-resilient-method-ranking-v1/group0/ranking/result.json"
    new, old = (json.loads(p.read_text()) for p in (new_path, old_path))
    assert all(d["passed"] and d["errors_admitted"] for d in (new, old))
    receipts = [file_id(new_path), file_id(old_path)]
    checked = {}
    for receipt in new["receipts"] + old["receipts"]:
        checked.setdefault(receipt["path"], receipt)
        assert checked[receipt["path"]] == receipt
    for path, receipt in checked.items():
        assert file_id(path) == receipt
    rows = []
    for level in (0, 1):
        measured = {}
        for index, q in enumerate((9, 25)):
            path = root / "resolved-pad-decomposition-v2" / f"group{2 * level + index}" / f"decomposition-level{level}-q{q}/result.json"
            receipts.append(file_id(path))
            result = json.loads(path.read_text())
            assert result["surface_quadrature_points_per_face"] == q and result["depth_quadrature_points"] == 4
            assert result["physical_volume_weighting"] and len(result["cases"]) == 1
            case = result["cases"][0]
            detail = next(d for d in new["details"] if d["level"] == level)
            assert [r["field"] for r in case["raw"]] == [
                detail["raw"][0]["field"], old["candidates"]["dpr"]["frames"][level], detail["raw"][1]["field"]]
            assert [r["mesh"] for r in case["raw"]] == [detail["raw"][0]["mesh"], detail["raw"][0]["mesh"], detail["raw"][1]["mesh"]]
            decomposition = case["three_field_decomposition"]
            assert decomposition["passed_tensor_identity"]
            measured[q] = decomposition["metrics"]
        for metric, fine in measured[25].items():
            new_error = next(r for r in new["rows"] if r["level"] == level and r["metric"] == metric)
            old_error = next(r for r in old["rows"] if r["level"] == level and r["metric"] == metric and r["method"] == "dpr")
            np.testing.assert_allclose([fine["total_relative_l2"], fine["second_relative_l2"]],
                                       [new_error["relative_error"], old_error["relative_error"]], rtol=1e-10, atol=1e-13)
            # Directly integrated signed product c=<new-old, old-reference>.
            # Bilinearity gives <-delta, new-reference>=-||delta||²-c.
            # This uses the signed product, not subtraction of scalar norms.
            first2 = fine["first_difference_rms"]**2
            cross = fine["twice_cross_mean_square"] / 2
            reversed_cross = -first2 - cross
            denominator = fine["first_difference_rms"] * fine["total_difference_rms"]
            np.testing.assert_allclose(
                first2 + fine["total_difference_rms"]**2 + 2 * reversed_cross,
                fine["second_difference_rms"]**2, rtol=1e-10, atol=1e-20)
            coarse = measured[9][metric]
            rows.append({"level": level, "metric": metric, **fine,
                "fixed_minus_resolved_vs_resolved_minus_reference_cross_correlation": float(reversed_cross / denominator),
                "fixed_error_identity_twice_cross_mean_square": float(2 * reversed_cross),
                "q9_q25_transfer_relative_l2_change": abs(coarse["first_relative_l2"] - fine["first_relative_l2"]),
                "q9_q25_signed_cross_correlation_change": abs(coarse["cross_correlation"] - fine["cross_correlation"])})
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": True, "rows": rows, "receipts": receipts,
        "scope": "The original direct identity is new-reference=(new-old)+(old-reference). The reversed identity old-reference=(old-new)+(new-reference) uses bilinearity of the directly integrated signed product. Remaining resolved-reference error combines source sampling, contact-pressure response, geometry/elastic discretization and reference effects; it is not an isolated elastic-discretization error. These contrasts do not change qualified error intervals or justify incorrect-boundary loads."})


if __name__ == "__main__":
    main()
