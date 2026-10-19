"""Independently check retained continuous force-cone rejection witnesses."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def triangle_distance(point, triangle):
    a, b, c = triangle
    edges = np.column_stack((b - a, c - a))
    uv = np.linalg.solve(edges.T @ edges, edges.T @ (point - a))
    distances = []
    if min(uv) >= 0 and uv.sum() <= 1:
        distances.append(np.linalg.norm(point - a - edges @ uv))
    for start, end in ((a, b), (b, c), (c, a)):
        delta = end - start
        fraction = np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0, 1)
        distances.append(np.linalg.norm(point - start - fraction * delta))
    return min(distances)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    triangle = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    controls = (
        ([0.2, 0.3, 0.4], 0.4),
        ([2, 0, 0], 1),
        ([0.7, 0.7, 0], np.sqrt(0.08)),
        ([-0.2, 0.4, 0.3], np.sqrt(0.13)),
    )
    for scale in (0.001, 1, 1000):
        for point, expected in controls:
            np.testing.assert_allclose(
                triangle_distance(scale * np.array(point), scale * triangle), scale * expected, rtol=1e-12, atol=1e-15
            )
    receipt = json.loads((args.run / "rejected-step.json").read_text())
    state = args.run / "rejected-state.npz"
    assert receipt["state"] == file_id(state) and receipt["passed"] is False
    with np.load(state) as saved:
        data = {key: saved[key] for key in saved.files}
    vertices = data["mesh.vertices"]
    triangles = vertices[data["mesh.surface_nodes"][:, :3]]
    opposite = vertices[data["mesh.surface_opposite_vertices"]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # The opposite tetrahedral vertex fixes the inward material side even on
    # a cavity or nonconvex face; no origin-inside or convexity assumption.
    normals *= np.where(np.einsum("fi,fi->f", normals, opposite - triangles[:, 0]) >= 0, 1, -1)[:, None]
    rows = []
    for env in np.flatnonzero(data["failure.status"]):
        contact = int(data["failure.contact"][env])
        status = int(data["failure.status"][env])
        assert status == 36, "This independent reader covers force-only continuous certificates"
        ray = data["certificates.ray"][contact, env]
        np.testing.assert_array_equal(ray[3:], np.zeros(3))
        np.testing.assert_allclose(np.linalg.norm(ray[:3]), 1, rtol=0, atol=1e-12)
        active = data["contacts.valid"][:, env]
        forces = data["contacts.force"][active, env]
        points = data["contacts.position"][active, env]
        free_moment = data["sources.free_moment"][active, env]
        target = forces.sum(axis=0)
        moment = (np.cross(points - data["failure.position"][env], forces) + free_moment).sum(axis=0)
        np.testing.assert_allclose(target, data["failure.target_force"][env], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(moment, data["failure.target_moment"][env], rtol=1e-12, atol=1e-15)
        assert active.sum() == data["failure.group_count"][env]
        centers = data["contacts.center"][active, env]
        radii = data["contacts.radius"][active, env]
        friction = data["contacts.friction"][active, env]
        supports, face_counts = [], []
        for center, radius, mu in zip(centers, radii, friction):
            assert radius > 0 and mu >= 0
            distance = np.array([triangle_distance(center, tri) for tri in triangles])
            allowance = (
                256
                * np.finfo(float).eps
                * (np.linalg.norm(center) + radius + np.max(np.linalg.norm(triangles, axis=-1)))
            )
            selected = distance <= radius + allowance
            assert selected.any()
            component = normals[selected] @ ray[:3]
            tangent = np.linalg.norm(ray[:3] - component[:, None] * normals[selected], axis=1)
            supports.extend((component + mu * tangent).tolist())
            face_counts.append(int(selected.sum()))
        maximum_support = max(supports)
        force_scale = np.linalg.norm(forces, axis=1).sum()
        target_projection = float(np.dot(target, ray[:3]))
        roundoff = 256 * np.finfo(float).eps * (1 + friction.max())
        proven = maximum_support < -roundoff and target_projection > (1e-8 + roundoff) * force_scale
        rows.append(
            {
                "environment": int(env),
                "native_status": status,
                "original_contact_count": int(active.sum()),
                "support_faces_per_source": face_counts,
                "maximum_cone_support": float(maximum_support),
                "target_force_projection_n": target_projection,
                "unsigned_source_force_sum_n": float(force_scale),
                "normalized_force_separation": target_projection / force_scale,
                "unit_roundoff_allowance": float(roundoff),
                "independent_infeasibility_proven": bool(proven),
                "native_certificate_margin": float(data["certificates.margin"][contact, env]),
            }
        )
    passed = bool(rows) and all(row["independent_infeasibility_proven"] for row in rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "state": file_id(state),
            "receipt": file_id(args.run / "rejected-step.json"),
            "rows": rows,
            "geometry_controls": 12,
            "all_recorded_failed_environments_checked": True,
            "scope": "Independent closest-triangle geometry and homogeneous Coulomb support bounds on a conservative Euclidean-ball superset of every original source footprint. All source friction/radius values and grouped force/moment are retained. A strict common polar ray with positive target projection proves force infeasibility beyond the unchanged1e-8 unsigned-source-force budget. No sampled fit, native status alone, dropped environment, force clipping or changed radius is used as proof. This checks the captured first rejection, not a complete failure-rate history.",
        },
    )
    if not passed:
        raise SystemExit("The independent conservative support did not certify every captured rejection")


if __name__ == "__main__":
    main()
