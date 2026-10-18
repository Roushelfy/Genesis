"""Independently inspect reported edge/triangle pairs in a rejected native checkpoint, without stepping a solver."""

import argparse
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def decimal_intersection(points):
    """Solve E0 + t(E1-E0) = T0 + u(T1-T0) + v(T2-T0), using exact binary inputs."""
    with localcontext() as context:
        context.prec = 80
        p = [[Decimal.from_float(float(x)) for x in row] for row in points]
        e0, e1, t0, t1, t2 = p
        matrix = [
            [e1[k] - e0[k], t0[k] - t1[k], t0[k] - t2[k], t0[k] - e0[k]] for k in range(3)
        ]
        for column in range(3):
            pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
            if matrix[pivot][column] == 0:
                return {"unique_plane_crossing": False, "reason": "Singular linear system; no coplanar verdict inferred"}
            matrix[pivot], matrix[column] = matrix[column], matrix[pivot]
            scale = matrix[column][column]
            matrix[column] = [value / scale for value in matrix[column]]
            for row in range(3):
                if row != column:
                    scale = matrix[row][column]
                    matrix[row] = [a - scale * b for a, b in zip(matrix[row], matrix[column])]
        t, u, v = [row[3] for row in matrix]
        bary = [1 - u - v, u, v]
        return {
            "unique_plane_crossing": True,
            "segment_parameter": str(t),
            "triangle_barycentric": [str(x) for x in bary],
            "segment_crosses_plane": 0 <= t <= 1,
            "plane_crossing_inside_triangle": min(bary) >= 0,
            "closed_segment_triangle_intersection": 0 <= t <= 1 and min(bary) >= 0,
            "arithmetic": "80-digit Decimal from exact stored float64 coordinates; no geometric tolerance or clipping",
        }


def native_predicate(points):
    """Float64 transcription of the pinned detector's narrow phase, not an independent certificate."""
    e0, e1, t0, t1, t2 = points
    normal = np.cross(t1 - t0, t2 - t0)
    da, db = np.array([e0 - t0, e1 - t0]) @ normal

    def point_in_triangle(point):
        u, v, w = t1 - t0, t2 - t0, point - t0
        uu, uv, vv, wu, wv = u @ u, u @ v, v @ v, w @ u, w @ v
        denominator = uv * uv - uu * vv
        if abs(denominator) < 1e-30:
            return False
        s, t = (uv * wv - vv * wu) / denominator, (uv * wu - uu * wv) / denominator
        return bool(s >= -1e-12 and t >= -1e-12 and s + t <= 1 + 1e-12)

    def segment_intersects(a, da_, b, db_):
        cross = np.cross(da_, db_)
        denominator = cross @ cross
        if denominator < 1e-30:
            return False
        u = np.cross(b - a, db_) @ cross / denominator
        v = np.cross(b - a, da_) @ cross / denominator
        return bool(-1e-12 <= u <= 1 + 1e-12 and -1e-12 <= v <= 1 + 1e-12)

    opposite = not ((da > 0 and db > 0) or (da < 0 and db < 0))
    coplanar = bool(abs(da) < 1e-14 and abs(db) < 1e-14)
    if not opposite:
        result = False
    elif coplanar:
        result = point_in_triangle(e0) or point_in_triangle(e1) or any(
            segment_intersects(e0, e1 - e0, a, b - a) for a, b in ((t0, t1), (t1, t2), (t2, t0))
        )
    else:
        t = da / (da - db)
        result = bool(0 <= t <= 1 and point_in_triangle(e0 + t * (e1 - e0)))
    return {
        "intersection": result,
        "coplanar_branch": coplanar,
        "unnormalized_plane_products_m3": [float(da), float(db)],
        "unit_normal_signed_endpoint_distances_m": [float(da / np.linalg.norm(normal)), float(db / np.linalg.norm(normal))],
        "scope": "Diagnostic float64 source transcription. GPU instruction ordering can differ; Decimal plane crossing supplies the independent pair verdict.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.failure / "steps.jsonl").read_text().splitlines()]
    rejected = [row for row in rows if row.get("edge_triangle_intersection_count", 0) > 0]
    assert len(rejected) == 1 and rejected[0] == rows[-1] and rows[-1]["invalid_step"]
    row = rows[-1]
    frame = args.failure / "frames" / f"step{row['step']:06d}.npz"
    dump = args.failure / "failure-native-checkpoint" / "dump" / f"frame_{row['step'] + 1}.npz"
    static = args.failure / "failure-contact-static.npz"
    with np.load(static, allow_pickle=False) as saved:
        topo = {key: saved[key] for key in saved.files}
    with np.load(dump, allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    with np.load(frame, allow_pickle=False) as saved:
        fields = {key: saved[key] for key in saved.files}
    positions = state["GlobalVertexManager/positions"].reshape(-1, 3)
    body = topo["body_id"].reshape(-1)
    edges, faces = topo["surf_edges"].reshape(-1, 2), topo["surf_faces"].reshape(-1, 3)
    assert len(positions) == len(body) == int(topo["n_verts"])
    n_rigid = state["RigidBodyDynamics/t"].size // 3
    fem = positions[body >= n_rigid]
    dumped_fem = state["FiniteElementMethod/x"].reshape(-1, 3)
    assert np.array_equal(dumped_fem, fields["positions_m"])
    active = positions[fields["active_global_ids"]]
    consistency = {
        "dump_FEM_equals_saved_field_exactly": True,
        "global_FEM_shape": list(fem.shape),
        "saved_FEM_shape": list(dumped_fem.shape),
        "global_FEM_equals_saved_field_exactly": bool(np.array_equal(fem, dumped_fem)),
        "global_FEM_maximum_component_difference_m": float(np.max(np.abs(fem - dumped_fem))) if fem.shape == dumped_fem.shape else None,
        "global_active_equals_saved_points_exactly": bool(np.array_equal(active, fields["active_global_positions_m"])),
        "global_active_maximum_component_difference_m": float(np.max(np.abs(active - fields["active_global_positions_m"]))),
        "scope": "Global positions and FEM public fields are distinct checkpoint arrays. Pair tests below use native global positions only, retaining every measured discrepancy rather than assuming state equality.",
    }
    pairs = []
    for index, (edge, face) in enumerate(row["edge_triangle_intersection_pairs"]):
        ids = np.concatenate((edges[edge], faces[face]))
        points = positions[ids]
        assert len(set(ids.tolist())) == 5
        matrix = np.column_stack((points[1] - points[0], points[2] - points[3], points[2] - points[4]))
        t, u, v = np.linalg.solve(matrix, points[2] - points[0])
        bary = np.array([1 - u - v, u, v])
        np.savez_compressed(args.output / f"pair{index}.npz", global_ids=ids, body_ids=body[ids], positions_m=points)
        pairs.append(
            {
                "edge": edge,
                "face": face,
                "global_vertex_ids": ids.tolist(),
                "body_ids": body[ids].tolist(),
                "owner_types": ["rigid" if x < n_rigid else "FEM" for x in body[ids]],
                "positions_m": points.tolist(),
                "float64_segment_parameter": float(t),
                "float64_triangle_barycentric": bary.tolist(),
                "float64_linear_system_residual_m": float(np.linalg.norm(matrix @ [t, u, v] - (points[2] - points[0]))),
                "decimal": decimal_intersection(points),
                "native_predicate_transcription": native_predicate(points),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "rejected-edge-triangle-independent-saved-geometry-audit",
            "identity": runtime_identity(),
            "inputs": [file_id(x) for x in (static, dump, frame, args.failure / "steps.jsonl")],
            "failure_time_s": row["time_s"],
            "failure_step": row["step"],
            "all_previous_recorded_ET_counts_zero": all(x.get("edge_triangle_intersection_count") == 0 for x in rows[:-1]),
            "saved_geometry_consistency": consistency,
            "number_of_rigid_bodies": n_rigid,
            "pairs": pairs,
            "scope": "Independent narrow-phase tests only for the actually reported pairs at the saved rejected state. No exhaustive contact search, CCD certification, solver continuation, repair, or physical reference qualification. Body ids use native static ownership; rigid names are not inferred from face/vertex order.",
        },
    )


if __name__ == "__main__":
    main()
