"""Derive checks from saved controlled trajectories; no new simulation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import file_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for campaign in ("controls-v1", "diagnostic-controls-v2", "press-controls-v3"):
        for receipt in sorted((args.runs / campaign).glob("group*/*/result.json")):
            metadata = json.loads(receipt.read_text())
            row = {"run": str(receipt.parent.relative_to(args.runs)), "receipt": file_identity(receipt)}
            if metadata["kind"] == "field-conversion":
                with np.load(receipt.parent / "fields.npz") as data:
                    stress = data["bend_tensors"].reshape(-1, 3, 2, 3, 3)
                    interior = data["interior_faces"]
                    row.update(
                        top_signed_xx_mean_pa=float(stress[interior, 2, :, 0, 0].mean()),
                        bottom_signed_xx_mean_pa=float(stress[interior, 0, :, 0, 0].mean()),
                        expected_cylindrical_top_sign="compressive for the declared upward normal",
                    )
            else:
                with np.load(receipt.parent / "trajectory.npz") as data:
                    if "caps" in data and data["caps"].size:
                        row.update(
                            cap_exhausted_substeps=int(data["caps"].sum()),
                            mean_iterations_per_control_env=float(data["iterations"].mean()),
                            max_iterations_per_control_env=int(data["iterations"].max()),
                            maximum_true_l2_relative_residual=float(data["residuals"][..., 0].max()),
                            maximum_true_block_jacobi_relative_residual=float(data["residuals"][..., 1].max()),
                            maximum_recursive_squared_budget_fraction=float(data["residuals"][..., 2].max()),
                            active_vertex_impulse_samples=int((np.linalg.norm(data["impulses"], axis=-1) > 0).sum()),
                            total_impulse_n_s=data["impulses"].sum(axis=(0, 1, 2)).tolist(),
                        )
                    row.update(peak_pa=float(data["peak_pa"].max()), fell_through=metadata["fell_through"])
            rows.append(row)
    write_json(args.output / "result.json", {"kind": "controlled-output-audit", "records": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
