"""Inspect retained actual source forces at a rejected pad stress observation."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def surface_cone_certificate(mesh, points, force, friction, radius):
    with np.load(mesh) as saved:
        triangle = saved["outer_vertices"][saved["mid_faces"]]
    center = triangle.mean(axis=1)
    distance = np.linalg.norm(center[:, None] - points[None], axis=-1)
    assert np.max(np.min(distance, axis=0)) <= 1e-14, "Source positions do not match this declared mesh"
    # Every point within a geodesic radius is within that Euclidean radius.
    # A triangle's centroid plus its maximum corner radius encloses the whole
    # triangle, so this selects a conservative superset of all possible faces.
    cover = np.linalg.norm(triangle - center[:, None], axis=-1).max(axis=1)
    selected = np.any(distance <= radius + cover[:, None], axis=1)
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    # The origin is inside this convex full-egg outer surface. Orient inward.
    normal *= -np.sign(np.einsum("ni,ni->n", normal, center))[:, None]
    vertical = normal[selected, 2]
    margin = float(np.min(-vertical - friction * np.sqrt(np.maximum(1 - vertical**2, 0))))
    resultant = force.sum(axis=0)
    scale = float(np.linalg.norm(force, axis=1).sum())
    # If every admissible traction has f_z <= -margin * pressure, then
    # -F_z >= margin/sqrt(1+mu^2) * ||F_xy|| is a necessary condition.
    defect = float(resultant[2] + margin / np.sqrt(1 + friction**2) * np.linalg.norm(resultant[:2]))
    allowance = 32 * np.finfo(force.dtype).eps * scale
    return {
        "mesh": file_id(mesh),
        "radius_m": radius,
        "candidate_faces": int(selected.sum()),
        "minimum_downward_force_per_unit_compression": margin,
        "necessary_force_cone_condition_defect_n": defect,
        "roundoff_allowance_n": allowance,
        "infeasibility_proven_on_conservative_support": bool(margin > 0 and defect > allowance),
        "scope": "Independent necessary-condition certificate using inward normals of the declared P1 outer triangles and a conservative Euclidean superset of geodesic-radius support. It does not identify a native solver status, assume native fit convergence, or alter any source. Failure to prove infeasibility is not proof of feasibility.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--mesh", type=Path)
    cli.add_argument("--radius", type=float, default=0.002)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.frame) as saved:
        force = saved["source_forces_n"]
        normal = saved["source_normals_material"]
        status = saved["contact_status"] if "contact_status" in saved else None
        friction = saved["friction"].item()
        field_finite = bool(np.isfinite(saved["stress_pa"]).all()) if "stress_pa" in saved else None
        elastic_force_finite = bool(np.isfinite(saved["forces_n"]).all()) if "forces_n" in saved else None
        rhs_finite = bool(np.isfinite(saved["rhs_n"]).all()) if "rhs_n" in saved else None
        residual_finite = (
            bool(np.isfinite(saved["equation_residual_n"]).all()) if "equation_residual_n" in saved else None
        )
        points = saved["source_points_m"]
    normal = normal / np.linalg.norm(normal, axis=1)[:, None]
    magnitude = np.linalg.norm(force, axis=1)
    compression = np.einsum("ni,ni->n", force, normal)
    tangent = np.linalg.norm(force - compression[:, None] * normal, axis=1)
    allowance = 16 * np.finfo(force.dtype).eps * magnitude
    cone_excess = tangent - friction * compression
    admitted = (compression >= -allowance) & (cone_excess <= allowance)
    rows = []
    for index in np.flatnonzero(status != 0 if status is not None else ~admitted):
        rows.append(
            {
                "source_index": int(index),
                "status": int(status[index]) if status is not None else None,
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
            "native_status_observed": status is not None,
            "friction": friction,
            "unsigned_source_force_sum_n": float(magnitude.sum()),
            "net_source_force_n": force.sum(axis=0).tolist(),
            "maximum_cone_excess_n": float(np.max(cone_excess)),
            "rejected_sources": rows,
            "field_finite": field_finite,
            "scattered_force_finite": elastic_force_finite,
            "rhs_finite": rhs_finite,
            "residual_finite": residual_finite,
            "surface_force_cone_certificate": surface_cone_certificate(args.mesh, points, force, friction, args.radius)
            if args.mesh is not None
            else None,
            "scope": "Read-only inspection of actual captured sources, with the existing per-source16-epsilon friction-cone test reproduced in NumPy. No input force, normal, physical law, acceptance tolerance or solver is changed. This inspection cannot admit a rejected trajectory.",
        },
    )


if __name__ == "__main__":
    main()
