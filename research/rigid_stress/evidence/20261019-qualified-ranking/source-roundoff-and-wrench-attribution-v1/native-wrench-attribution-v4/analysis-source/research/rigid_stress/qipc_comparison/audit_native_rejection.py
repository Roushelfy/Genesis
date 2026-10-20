"""Independently bound retained continuous wrench-cone rejection witnesses."""

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


def bound_support(triangles, normals, centers, radii, friction, anchor, radius, ray):
    """Bound convex polar support using longest-edge binary subdivision."""
    maximum, visited, deepest = -np.inf, 0, 0
    for center, support_radius, mu in zip(centers, radii, friction):
        allowance = (
            512
            * np.finfo(float).eps
            * (np.linalg.norm(center) + support_radius + np.max(np.linalg.norm(triangles, axis=-1)))
        )
        for triangle, normal in zip(triangles, normals):
            stack = [(triangle, 0)]
            while stack:
                tri, depth = stack.pop()
                visited += 1
                deepest = max(deepest, depth)
                if visited > 200000:
                    return False, float(maximum), visited, deepest
                if triangle_distance(center, tri) > support_radius + allowance:
                    continue
                relative = (tri - anchor) / radius
                value = ray[:3] + np.cross(ray[3:], relative)
                component = value @ normal
                polar = component + mu * np.linalg.norm(value - component[:, None] * normal, axis=1)
                rounding = (
                    512
                    * np.finfo(float).eps
                    * (1 + mu)
                    * (
                        1
                        + np.linalg.norm(value, axis=1)
                        + np.linalg.norm(relative, axis=1)
                        + np.linalg.norm(ray[3:])
                        * (np.linalg.norm(anchor) + np.max(np.linalg.norm(tri, axis=1)))
                        / radius
                    )
                )
                upper = float(np.max(polar + rounding))
                if upper < 0:
                    maximum = max(maximum, upper)
                    continue
                if depth >= 24:
                    return False, upper, visited, deepest
                pairs = ((0, 1), (1, 2), (2, 0))
                lengths = [np.linalg.norm(tri[b] - tri[a]) for a, b in pairs]
                a, b = pairs[int(np.argmax(lengths))]
                other, midpoint = 3 - a - b, (tri[a] + tri[b]) / 2
                stack.extend(
                    (
                        (np.array([tri[a], midpoint, tri[other]]), depth + 1),
                        (np.array([midpoint, tri[b], tri[other]]), depth + 1),
                    )
                )
    return bool(np.isfinite(maximum) and maximum < 0), float(maximum), visited, deepest


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
    actual_state = file_id(state)
    assert all(actual_state[key] == receipt["state"][key] for key in ("bytes", "sha256")) and receipt["passed"] is False
    with np.load(state) as saved:
        data = {key: saved[key] for key in saved.files}
    observations = []
    for env in np.flatnonzero(data["failure.status"]):
        valid_ids = np.flatnonzero(data["contacts.valid"][:, env])
        force_ids = np.flatnonzero(np.linalg.norm(data["contacts.force"][:, env], axis=1))
        status_ids = np.flatnonzero(data["contacts.status"][:, env])
        ray_ids = np.flatnonzero(np.linalg.norm(data["certificates.ray"][:, env], axis=1))
        ids = np.unique(np.r_[valid_ids, force_ids, status_ids, ray_ids, data["failure.contact"][env]])
        observations.append(
            {
                "environment": int(env),
                "valid_dtype": str(data["contacts.valid"].dtype),
                "valid_indices": valid_ids.tolist(),
                "nonzero_force_indices": force_ids.tolist(),
                "status_indices": status_ids.tolist(),
                "certificate_indices": ray_ids.tolist(),
                "contacts": [
                    {
                        "index": int(i),
                        "force": data["contacts.force"][i, env].tolist(),
                        "position": data["contacts.position"][i, env].tolist(),
                        "center": data["contacts.center"][i, env].tolist(),
                        "certificate": data["certificates.ray"][i, env].tolist(),
                    }
                    for i in ids
                ],
            }
        )
    write_json(args.output / "observed.json", {"environments": observations})
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
        first_contact = int(data["failure.contact"][env])
        status = int(data["failure.status"][env])
        assert status in (35, 36), "This reader covers continuous force/wrench certificates"
        active = data["contacts.valid"][:, env].astype(bool)
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
        assert np.all(radii > 0) and np.all(friction >= 0)
        candidates = np.flatnonzero(active & (np.linalg.norm(data["certificates.ray"][:, env], axis=1) > 0))
        assert len(candidates), "No retained nonzero certificate on any physical group member"
        proofs = []
        for contact in candidates:
            ray = data["certificates.ray"][contact, env]
            ray = ray / np.linalg.norm(ray)
            anchor = data["contacts.center"][contact, env]
            radius = data["contacts.radius"][contact, env]
            application = data["contacts.position"][contact, env]
            anchor_moments = data["sources.anchor_moment"][active, env]
            target_moment = (np.cross(points - anchor, forces) + free_moment).sum(axis=0)
            force_scale = np.maximum(
                np.linalg.norm(forces, axis=1), np.linalg.norm(anchor_moments, axis=1) / radii
            ).sum()
            moment_scale = (
                np.linalg.norm(anchor_moments, axis=1)
                + (np.linalg.norm(centers - application, axis=1) + radii) * np.linalg.norm(forces, axis=1)
            ).sum()
            projected = float(target @ ray[:3] + target_moment @ ray[3:] / radius)
            force_coefficient = ray[:3] + np.cross(ray[3:], (application - anchor) / radius)
            budget = 1e-8 * (
                force_scale * np.linalg.norm(force_coefficient) + moment_scale / radius * np.linalg.norm(ray[3:])
            )
            rounding = 1024 * np.finfo(float).eps * (force_scale + moment_scale / radius)
            resolved, bound, nodes, depth = bound_support(
                triangles, normals, centers, radii, friction, anchor, radius, ray
            )
            proofs.append(
                {
                    "certificate_contact": int(contact),
                    "ray": ray.tolist(),
                    "maximum_continuous_polar_support": bound,
                    "visited_nodes": nodes,
                    "maximum_depth": depth,
                    "target_wrench_projection_n": projected,
                    "acceptance_projection_budget_n": float(budget),
                    "roundoff_projection_allowance_n": float(rounding),
                    "independent_infeasibility_proven": bool(resolved and projected > budget + rounding),
                }
            )
        rows.append(
            {
                "environment": int(env),
                "native_status": status,
                "first_preserved_contact": first_contact,
                "captured_valid_dtype": str(data["contacts.valid"].dtype),
                "original_contact_count": int(active.sum()),
                "independent_infeasibility_proven": any(proof["independent_infeasibility_proven"] for proof in proofs),
                "proofs": proofs,
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
            "scope": "Independent closest-triangle geometry, longest-edge binary subdivision and convex Coulomb polar bounds on a conservative Euclidean-ball superset of every original source footprint. All source friction/radius values and grouped force/moment are retained. A strict common polar field plus positive target wrench projection beyond independent1e-8 force/moment budgets proves infeasibility. A propagated raw status can have no certificate; all nonzero group-member certificates are checked against the complete physical group. No sampled fit, native status alone, dropped environment, force clipping or changed radius is proof. This checks the captured first rejection, not a complete failure-rate history.",
        },
    )
    if not passed:
        raise SystemExit("The independent conservative support did not certify every captured rejection")


if __name__ == "__main__":
    main()
