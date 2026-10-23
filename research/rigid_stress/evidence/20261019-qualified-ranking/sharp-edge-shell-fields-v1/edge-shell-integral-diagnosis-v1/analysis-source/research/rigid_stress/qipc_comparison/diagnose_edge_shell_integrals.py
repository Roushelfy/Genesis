"""Locate integration sensitivity in particular and regular stress contributions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_edge_shell_fields import resultants


def energy_parts(candidate, reference, candidate_particular, reference_particular, weights):
    regular = candidate - candidate_particular - reference + reference_particular
    particular = candidate_particular - reference_particular
    norm = float(weights @ np.sum(reference**2, axis=(1, 2)))
    pieces = {
        "regular": float(weights @ np.sum(regular**2, axis=(1, 2))),
        "particular": float(weights @ np.sum(particular**2, axis=(1, 2))),
        "cross": float(2 * weights @ np.sum(regular * particular, axis=(1, 2))),
    }
    total = float(weights @ np.sum((candidate - reference) ** 2, axis=(1, 2)))
    np.testing.assert_allclose(sum(pieces.values()), total, rtol=1e-10, atol=1e-30)
    return {"reference_squared_norm": norm, "error_squared_norm": total, "error_parts": pieces}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--studies", type=Path, nargs=2, required=True)
    cli.add_argument("--samplings", type=Path, nargs=2, required=True)
    cli.add_argument("--particulars", type=Path, nargs=2, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        file_id(path)
        for study, sampling, particular in zip(args.studies, args.samplings, args.particulars)
        for path in [
            study / "start.json",
            study / "result.json",
            study / "coarse-fields.npz",
            study / "fine-fields.npz",
            sampling / "queries.npz",
            particular / "particular.npz",
        ]
    ]
    write_json(
        args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False), "inputs": inputs}
    )
    rows = []
    for study, sampling, particular in zip(args.studies, args.samplings, args.particulars):
        start = json.loads((study / "start.json").read_text())
        for receipt in start["inputs"] + start["particular_reuse"]:
            assert file_id(receipt["path"]) == receipt
        assert file_id(sampling / "queries.npz") in start["inputs"]
        assert file_id(particular / "particular.npz") in start["particular_reuse"]
        recorded = json.loads((study / "result.json").read_text())
        with np.load(sampling / "queries.npz") as saved:
            queries = {key: saved[key] for key in saved.files if key.startswith(("volume_", "column_"))}
        nv, nc = len(queries["volume_points_m"]), len(queries["column_points_m"])
        with np.load(particular / "particular.npz") as saved:
            cp, rp = saved["candidate_pa"], saved["reference48_pa"]
        cm, cb = resultants(cp[nv : nv + nc], queries)
        rm, rb = resultants(rp[nv : nv + nc], queries)
        fields = []
        for body, label in [("coarse", "candidate-point"), ("fine", "reference48")]:
            with np.load(study / f"{body}-fields.npz") as saved:
                fields.append([saved[label + suffix] for suffix in ["-tensor_pa", "-membrane_n_per_m", "-bending_n"]])
        candidate, reference = fields
        volume = energy_parts(candidate[0][:nv], reference[0][:nv], cp[:nv], rp[:nv], queries["volume_weights_m3"])
        membrane = energy_parts(candidate[1], reference[1], cm, rm, queries["column_areas_m2"])
        bending = energy_parts(candidate[2], reference[2], cb, rb, queries["column_areas_m2"])
        width, along = start["candidate"]["width"], start["candidate"]["along"]
        cuts = np.r_[0, width * np.array([0.25, 1, 4]), along * np.array([0.25, 1, 4]), np.inf]
        depth = 0.03 - queries["volume_points_m"][:, 2]
        depth_rows = []
        for left, right in zip(cuts[:-1], cuts[1:]):
            selected = (depth >= left) & (depth < right)
            pieces = energy_parts(
                candidate[0][:nv][selected],
                reference[0][:nv][selected],
                cp[:nv][selected],
                rp[:nv][selected],
                queries["volume_weights_m3"][selected],
            )
            depth_rows.append(
                {
                    "depth_low_m": float(left),
                    "depth_high_m": float(right) if np.isfinite(right) else None,
                    "queries": int(selected.sum()),
                    **pieces,
                }
            )
        local = np.linalg.norm(queries["column_outer_m"] - [0, 0, 0.03], axis=1) < 8 * along
        regions = {}
        for label, selected in [("within_8_half_lengths", local), ("outside_8_half_lengths", ~local)]:
            regions[label] = {
                name: energy_parts(
                    c[selected], r[selected], pc[selected], pr[selected], queries["column_areas_m2"][selected]
                )
                for name, c, r, pc, pr in [
                    ("membrane", candidate[1], reference[1], cm, rm),
                    ("bending", candidate[2], reference[2], cb, rb),
                ]
            }
        for name, value in [
            ("complete_tensor_relative", volume),
            ("membrane_relative", membrane),
            ("bending_relative", bending),
        ]:
            np.testing.assert_allclose(
                (value["error_squared_norm"] / value["reference_squared_norm"]) ** 0.5,
                recorded["candidate_errors"]["coarse"][name],
                rtol=1e-12,
                atol=1e-14,
            )
        rows.append(
            {
                "study": str(study),
                "volume": volume,
                "membrane": membrane,
                "bending": bending,
                "volume_depth_bands": depth_rows,
                "column_regions": regions,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "scope": "Exact saved-array energy decomposition. Regular, particular and signed cross terms sum to the original errors. "
            "Disjoint depth/column regions diagnose integration sensitivity; this adds no quadrature or physical qualification.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
