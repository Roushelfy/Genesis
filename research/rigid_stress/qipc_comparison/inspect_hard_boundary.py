"""Summarize raw source-separated boundary loads and their nearest curved faces."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.archive) as saved:
        geometry = saved["geometry"]
        rows = []
        for source, center in enumerate(saved["centers_m"]):
            nearest = np.argsort(geometry[source, :, 1])[:8]
            orders = []
            for order, force in zip(saved["orders"], saved["force_n"]):
                actual_force = force[source].sum(axis=0)
                actual_moment = np.cross(saved["vertices_m"], force[source]).sum(axis=0)
                expected_force = saved["expected_force_n"][source]
                orders.append(
                    {
                        "order": int(order),
                        "force_error_n": (actual_force - expected_force).tolist(),
                        "moment_error_nm": (actual_moment - np.cross(center, expected_force)).tolist(),
                    }
                )
            rows.append(
                {
                    "source": source,
                    "original_slot": int(saved["selected"][source]),
                    "anchor": int(saved["anchors"][source]),
                    "center_m": center.tolist(),
                    "nearest_faces": nearest.tolist(),
                    "geometry": geometry[source, nearest].tolist(),
                    "orders": orders,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "input": file_id(args.archive),
            "rows": rows,
            "geometry_columns": [
                "affine_seed_distance_m",
                "curved_closest_distance_m",
                "projection_distance_m",
                "affine_bary0",
                "affine_bary1",
                "affine_bary2",
                "closest_bary0",
                "closest_bary1",
                "closest_bary2",
                "projected_bary0",
                "projected_bary1",
                "projected_bary2",
                "source_normal_dot",
                "projection_regular",
            ],
            "scope": "Raw loads before production rejection; nearest-face geometry diagnosis, not physical accuracy.",
        },
    )


if __name__ == "__main__":
    main()
