"""Inspect retained actual source forces at a rejected pad stress observation."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.frame) as saved:
        force = saved["source_forces_n"]
        normal = saved["source_normals_material"]
        status = saved["contact_status"]
        friction = saved["friction"].item()
        field_finite = bool(np.isfinite(saved["stress_pa"]).all())
        elastic_force_finite = bool(np.isfinite(saved["forces_n"]).all())
        rhs_finite = bool(np.isfinite(saved["rhs_n"]).all())
        residual_finite = bool(np.isfinite(saved["equation_residual_n"]).all())
        points = saved["source_points_m"]
    normal = normal / np.linalg.norm(normal, axis=1)[:, None]
    magnitude = np.linalg.norm(force, axis=1)
    compression = np.einsum("ni,ni->n", force, normal)
    tangent = np.linalg.norm(force - compression[:, None] * normal, axis=1)
    allowance = 16 * np.finfo(force.dtype).eps * magnitude
    cone_excess = tangent - friction * compression
    admitted = (compression >= -allowance) & (cone_excess <= allowance)
    rows = []
    for index in np.flatnonzero(status != 0):
        rows.append(
            {
                "source_index": int(index),
                "status": int(status[index]),
                "position_material_m": points[index].tolist(),
                "force_material_n": force[index].tolist(),
                "force_norm_n": float(magnitude[index]),
                "compression_n": float(compression[index]),
                "tangent_norm_n": float(tangent[index]),
                "cone_excess_n": float(cone_excess[index]),
                "local_roundoff_allowance_n": float(allowance[index]),
                "recomputed_input_admitted": bool(admitted[index]),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "frame": file_id(args.frame),
            "source_count": len(force),
            "friction": friction,
            "unsigned_source_force_sum_n": float(magnitude.sum()),
            "net_source_force_n": force.sum(axis=0).tolist(),
            "maximum_cone_excess_n": float(np.max(cone_excess)),
            "rejected_sources": rows,
            "field_finite": field_finite,
            "scattered_force_finite": elastic_force_finite,
            "rhs_finite": rhs_finite,
            "residual_finite": residual_finite,
            "scope": "Read-only inspection of actual captured sources, with the existing per-source16-epsilon friction-cone test reproduced in NumPy. No input force, normal, physical law, acceptance tolerance or solver is changed. This inspection cannot admit a rejected trajectory.",
        },
    )


if __name__ == "__main__":
    main()
