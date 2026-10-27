"""Quantify the old coarse-parent graph's remote load from saved independent book oracles.

Read existing arrays only. Scientific array analysis requires Slurm rtx-mid and
image 1_26; this audit advances no physics and assembles no elastic operator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def _identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific array analysis requires Slurm rtx-mid and image 1_26.")
    output = args.output.resolve()
    if not output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np

    rows, inputs = [], []
    for run in args.runs:
        run = run.resolve()
        cpu = run / "cpu"
        model_path, cases_path = cpu / "inputs/book/model.npz", cpu / "inputs/book/cases.json"
        cases = json.loads(cases_path.read_text())
        with np.load(model_path, allow_pickle=False) as data:
            quads = data["quads"]
            n_nodes = len(data["vertices"])
        selected_path, unmasked_path = cpu / "oracles/q48.npz", cpu / "oracles/unmasked-q48.npz"
        with np.load(selected_path, allow_pickle=False) as data:
            selected = data["raw"].reshape((n_nodes, 6, len(cases)))
        with np.load(unmasked_path, allow_pickle=False) as data:
            unmasked = data["raw"].reshape((n_nodes, 6, len(cases)))
        records_path = cpu / "oracles/q48.json"
        records = json.loads(records_path.read_text())
        for i_b, case in enumerate(cases):
            if case["category"] != "tip":
                continue
            parents = set(case["expected_parents"])
            excluded = set(range(len(quads))) - parents
            selected_nodes = set(quads[sorted(parents)].reshape(-1).tolist())
            far_nodes = sorted(set(quads[sorted(excluded)].reshape(-1).tolist()) - selected_nodes)
            record = next(row for row in records if row["environment"] == i_b and row["slot"] == 0)
            source = case["contacts"][0]
            magnitude = np.linalg.norm(source["force"])
            old_force = float(np.linalg.norm(unmasked[far_nodes, :3, i_b].sum(axis=0)))
            old_moment = float(np.linalg.norm(unmasked[far_nodes, 3:, i_b]))
            zero = not np.count_nonzero(selected[far_nodes, :, i_b])
            distances = [distance for witness in record["edge_witnesses"] for distance in witness["distance_m"]]
            radius = source["radius"]
            graph_wrong = zero and old_force > 0 and min(distances) > radius
            graph_wrong &= record["hit_parents"] == [0, 1] and set(record["selected_parents"]) == parents
            rows.append(
                {
                    "run": str(run),
                    "case": case["name"],
                    "passed": bool(graph_wrong),
                    "selected_parents": sorted(parents),
                    "excluded_parents": sorted(excluded),
                    "far_exclusive_nodes": far_nodes,
                    "radius_m": radius,
                    "physical_hinge_distance_m": distances,
                    "old_coarse_graph_far_resultant_N": old_force,
                    "old_coarse_graph_far_force_fraction": float(old_force / magnitude),
                    "old_coarse_graph_far_nodal_moment_L2_Nm": old_moment,
                    "selected_component_far_raw_exact_zero": zero,
                }
            )
        inputs.extend(_identity(path) for path in (model_path, cases_path, selected_path, unmasked_path, records_path))
    result = {
        "schema": "genesis.planar-footprint-readback.v1",
        "complete": True,
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "scope": "Saved independent CPU physical-edge and wrong-parent raw-load evidence only; no elastic-field qualification or new physics execution.",
        "worker": _identity(Path(__file__).resolve()),
        "inputs": inputs,
        "rows": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise ArithmeticError("The saved book coarse-parent counterexample is incomplete or inconsistent.")
    print(json.dumps({"passed": True, "rows": len(rows), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
