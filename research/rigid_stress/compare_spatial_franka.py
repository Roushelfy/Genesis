"""Compare solved Panda spatial refinements at fixed material probes and physical times."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .skfem_reference import SkfemElasticReference
from .stable_neohookean_reference import StableNeoHookeanBulk
from .summarize_fine_references import closed_source


def common_probes(reference):
    """Five fixed interior points per seed tetrahedron; weights sum to its volume."""
    # Distinct square-free radicals avoid rational barycentric planes of every
    # nested midpoint refinement. Centroids instead lie on child interfaces,
    # where discontinuous FEM stress has multiple traces.
    roots = np.sqrt([2, 3, 5, 7])
    barycentric = np.vstack((np.sqrt([11, 13, 17, 19]), *[np.roll(roots, i) for i in range(4)]))
    barycentric /= barycentric.sum(axis=1, keepdims=True)
    corners = reference.mesh.p[:, reference.mesh.t].transpose(2, 1, 0)
    points = np.einsum("pc,ecj->epj", barycentric, corners).reshape(-1, 3)
    volume = abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    weights = np.repeat(volume / len(barycentric), len(barycentric))
    return points, weights


def point_fields(reference, displacement, points):
    """Interpolate affine displacement gradients, then evaluate nonlinear Cauchy stress."""
    elements = reference.mesh.element_finder()(*points.T)
    local = reference.basis.mapping.invF(points.T[:, :, None], tind=elements)[:, :, 0]
    barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
    assert barycentric.min() > 1e-8
    # Thin tetrahedra amplify cancellation of finite rigid motion in a sum of
    # nodal gradients. Form physical barycentric gradients in extended precision
    # and subtract one nodal displacement before contraction (partition unity).
    corners = reference.mesh.p[:, reference.mesh.t[:, elements]].transpose(2, 1, 0).astype(np.longdouble)
    edges = corners[:, 1:] - corners[:, :1]
    cross = np.stack(
        (np.cross(edges[:, 1], edges[:, 2]), np.cross(edges[:, 2], edges[:, 0]), np.cross(edges[:, 0], edges[:, 1])),
        axis=1,
    )
    inverse = cross / np.sum(edges[:, 0] * cross[:, 0], axis=1)[:, None, None]
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    coordinates = reference.scalar_basis.elem.doflocs
    dof_barycentric = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
    nodal = displacement[reference.component_dofs[reference.scalar_basis.element_dofs[:, elements]]].astype(
        np.longdouble
    )
    h = np.zeros((len(points), 3, 3), dtype=np.longdouble)
    for index, node in enumerate(dof_barycentric):
        if np.max(node) == 1:
            vertex = int(np.argmax(node))
            derivative = (4 * barycentric[:, vertex] - 1)[:, None] * gradients[:, vertex]
        else:
            first, second = np.flatnonzero(node == 0.5)
            derivative = 4 * (
                barycentric[:, first, None] * gradients[:, second] + barycentric[:, second, None] * gradients[:, first]
            )
        h += (nodal[index] - nodal[0])[:, :, None] * derivative[:, None, :]
    h = np.asarray(h, dtype=float)
    trace = np.trace(h, axis1=-2, axis2=-1)
    delta_j = trace + 0.5 * (trace**2 - np.einsum("pij,pji->p", h, h)) + np.linalg.det(h)
    assert np.all(delta_j > -1)
    b_minus_i = h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2)
    lam, mu = reference.lame
    tensor = (mu * b_minus_i + lam * np.log1p(delta_j)[:, None, None] * np.eye(3)) / (1 + delta_j)[:, None, None]
    deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
    probes = reference.scalar_basis.probes(points.T)
    values = probes @ displacement[reference.component_dofs]
    return tensor, vm, values


def qualify(reference, points):
    """Independent quadratic polynomial and finite rotation test of point evaluation."""
    # Generate the manufactured field at exact affine-P2 vertex/edge locations,
    # rather than amplifying rounded coordinate subtraction in thin elements.
    coordinates = np.empty(reference.coordinates.shape, dtype=np.longdouble)
    vertices = reference.mesh.p.T.astype(np.longdouble)
    coordinates[reference.scalar_basis.dofs.nodal_dofs[0]] = vertices
    coordinates[reference.scalar_basis.dofs.edge_dofs[0]] = vertices[reference.mesh.edges.T].mean(axis=1)
    np.testing.assert_allclose(coordinates, reference.coordinates, rtol=0, atol=1e-15)
    x, y, z = coordinates.T
    u = np.zeros(reference.basis.N)
    u[reference.component_dofs] = np.column_stack(
        (0.002 * x + 0.01 * y * z, -0.001 * y + 0.02 * x * z, 0.0007 * z - 0.015 * x * y)
    )
    tensor, _, values = point_fields(reference, u, points)
    x, y, z = points.T
    expected_values = np.column_stack((0.002 * x + 0.01 * y * z, -0.001 * y + 0.02 * x * z, 0.0007 * z - 0.015 * x * y))
    h = np.zeros((len(points), 3, 3))
    h[:, 0] = np.column_stack((np.full(len(x), 0.002), 0.01 * z, 0.01 * y))
    h[:, 1] = np.column_stack((0.02 * z, np.full(len(x), -0.001), 0.02 * x))
    h[:, 2] = np.column_stack((-0.015 * y, -0.015 * x, np.full(len(x), 0.0007)))
    f = np.eye(3) + h
    j = np.linalg.det(f)
    lam, mu = reference.lame
    expected_tensor = (mu * (f @ f.swapaxes(-1, -2) - np.eye(3)) + lam * np.log(j)[:, None, None] * np.eye(3)) / j[
        :, None, None
    ]
    np.testing.assert_allclose(values, expected_values, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(tensor, expected_tensor, rtol=1e-8, atol=2e-5)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformed = np.zeros_like(u)
    extended_rotation = rotation.astype(np.longdouble)
    transformed[reference.component_dofs] = (
        coordinates @ (extended_rotation - np.eye(3, dtype=np.longdouble)).T
        + u[reference.component_dofs].astype(np.longdouble) @ extended_rotation.T
        + np.array([0.003, -0.002, 0.001], dtype=np.longdouble)
    )
    rotated_tensor, _, _ = point_fields(reference, transformed, points)
    local_tensor = rotation.T @ rotated_tensor @ rotation
    np.testing.assert_allclose(local_tensor, tensor, rtol=1e-8, atol=2e-4)
    return {
        "polynomial_displacement_max_error_m": float(abs(values - expected_values).max()),
        "polynomial_tensor_max_error_Pa": float(abs(tensor - expected_tensor).max()),
        "finite_rotation_tensor_max_error_Pa": float(abs(local_tensor - tensor).max()),
    }


def weighted_change(a, b, weights=None):
    difference = b - a
    if weights is None:
        numerator, denominator = np.linalg.norm(difference), np.linalg.norm(b)
    else:
        shape = (1, len(weights)) + (1,) * (a.ndim - 2)
        w = weights.reshape(shape)
        numerator, denominator = np.sqrt(np.sum(w * difference**2)), np.sqrt(np.sum(w * b**2))
    return {
        "absolute_l2_change": float(numerator),
        "relative_l2_change": float(numerator / denominator) if denominator else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument(
        "--axis", choices=("body_refinement", "contact_refinement", "local_levels"), default="body_refinement"
    )
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.sample_every > 0
    args.output.mkdir(parents=True, exist_ok=False)
    cases, frozen = [], None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        config.setdefault("contact_refinement", 0)
        config.setdefault("local_levels", 0)
        physical = {k: v for k, v in config.items() if k not in ("output", args.axis)}
        assert physical["cold_closure"]
        if frozen is None:
            frozen = physical
        assert physical == frozen
        report = json.loads((run / "results/summary.json").read_text())
        assert len(report["steps"]) == config["steps"]
        assert all(
            row["complete_dynamic_residual_N"] <= 1e-9 and row["complete_body_dynamic_residual_N"] <= 1e-9
            for row in report["steps"]
        )
        with np.load(run / "results/reference_geometry.npz") as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
            fixed = (saved["fixed_initializer_points_m"].copy(), saved["fixed_initializer_components"].copy())
        cases.append((reference.mesh.nelements, run, config, report, reference, fixed))
    cases.sort(key=lambda row: row[2][args.axis])
    seed = cases[0][4]
    points, weights = common_probes(seed)
    np.savez_compressed(args.output / "common_probes.npz", reference_points_m=points, volume_weights_m3=weights)
    qualifications = []
    for _, run, _, _, reference, fixed in cases:
        np.testing.assert_array_equal(fixed[0], cases[0][5][0])
        np.testing.assert_array_equal(fixed[1], cases[0][5][1])
        np.testing.assert_allclose(reference.total_mass, seed.total_mass, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(reference.com, seed.com, rtol=0, atol=1e-13)
        qualifications.append({"run": str(run), **qualify(reference, points)})
    if args.qualify_only:
        output = {
            "scope": "Quadratic polynomial and objective point-field qualification",
            "qualifications": qualifications,
            "common_material_points": len(points),
            "goal_completion": False,
        }
        (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
        return
    assert len(cases) >= 2
    closure = json.loads((Path(frozen["closure_run"]) / "results/configuration.json").read_text())
    template, _ = build_from_capture(Path(closure["capture"]), closure["dhat"])
    step_indices = sorted(set(range(args.sample_every, frozen["steps"] + 1, args.sample_every)) | {frozen["steps"]})
    histories, summaries = [], []
    for nelements, run, config, report, reference, _ in cases:
        bulk = StableNeoHookeanBulk(reference)
        count = reference.basis.N
        rest = np.vstack((reference.coordinates, template.rest[len(template.body.coordinates) :]))
        groups = [
            np.arange(len(reference.coordinates)),
            *[nodes - len(template.body.coordinates) + len(reference.coordinates) for nodes in template.finger_nodes],
        ]
        rows = []
        for step in step_indices:
            with np.load(run / f"results/step{step:04d}.npz") as saved:
                raw = saved["displacement"].copy()
                force = saved["contact_force_N"].reshape(-1, 3)
                vector = np.empty(count)
                permutation = saved["body_component_permutation"]
                np.testing.assert_array_equal(permutation, reference.component_dofs.reshape(-1))
                vector[permutation] = raw[:count]
                velocity = np.empty(count)
                velocity[permutation] = saved["velocity"][:count]
            rotation, translation, local = finite_frame(reference, vector)
            tensor, vm, _ = point_fields(reference, vector, points)
            objective_tensor = rotation.T @ tensor @ rotation
            _, _, objective_displacement = point_fields(reference, local, points)
            sampled_velocity = (reference.scalar_basis.probes(points.T) @ velocity[reference.component_dofs]) @ rotation
            positions = rest + raw.reshape(-1, 3)
            wrenches = np.array(
                [
                    np.r_[force[nodes].sum(axis=0), np.cross(positions[nodes], force[nodes]).sum(axis=0)]
                    for nodes in groups
                ]
            )
            row = {
                "tensor": objective_tensor,
                "vm": vm,
                "objective_displacement": objective_displacement,
                "velocity": sampled_velocity,
                "body_elastic_energy_J": bulk.evaluate(vector, False)[0],
                "force_N": wrenches[:, :3],
                "moment_Nm": wrenches[:, 3:],
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"case{len(histories):02d}-step{step:04d}.npz",
                reference_points_m=points,
                rotation_world_from_reference=rotation,
                translation_world_m=translation,
                contact_wrenches_world=wrenches,
                **row,
            )
        histories.append(rows)
        with np.load(run / "results/body_fields.npz") as saved:
            raw_vm, upper = saved["vm_Pa"], saved["upper_peak_Pa"]
            assert raw_vm.shape == (nelements, 4)
            np.testing.assert_allclose(raw_vm.max(), report["body_peak_lower_Pa"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(upper.max(), report["body_peak_upper_Pa"], rtol=0, atol=1e-8)
            element, corner = np.unravel_index(raw_vm.argmax(), raw_vm.shape)
        summaries.append(
            {
                "run": str(run),
                "body_tetrahedra": nelements,
                "contact_refinement": config["contact_refinement"],
                "local_levels": config["local_levels"],
                "fixed_seed_contact_boundary": (run / "results/frozen_contact_boundary.npz").exists(),
                "final_peak_lower_Pa": report["body_peak_lower_Pa"],
                "final_peak_upper_Pa": report["body_peak_upper_Pa"],
                "final_lower_peak_reference_position_m": reference.corner_positions[element, corner].tolist(),
                "max_full_dynamic_residual_N": max(r["complete_dynamic_residual_N"] for r in report["steps"]),
                "max_body_dynamic_residual_N": max(r["complete_body_dynamic_residual_N"] for r in report["steps"]),
            }
        )
    comparisons = []
    for index, (old, new) in enumerate(pairwise(histories)):
        changes = {
            key: weighted_change(
                np.stack([r[key] for r in old]),
                np.stack([r[key] for r in new]),
                weights if key in ("tensor", "vm", "objective_displacement", "velocity") else None,
            )
            for key in new[0]
        }
        comparisons.append(
            {
                "previous_run": str(cases[index][1]),
                "current_run": str(cases[index + 1][1]),
                "previous_tetrahedra": cases[index][0],
                "current_tetrahedra": cases[index + 1][0],
                "previous_contact_refinement": cases[index][2]["contact_refinement"],
                "current_contact_refinement": cases[index + 1][2]["contact_refinement"],
                "varying_discretizations": [
                    key
                    for key, before, after in [
                        ("body_refinement", cases[index][2]["body_refinement"], cases[index + 1][2]["body_refinement"]),
                        ("local_levels", cases[index][2]["local_levels"], cases[index + 1][2]["local_levels"]),
                        (
                            "contact_refinement",
                            cases[index][2]["contact_refinement"],
                            cases[index + 1][2]["contact_refinement"],
                        ),
                        (
                            "fixed_seed_contact_boundary",
                            summaries[index]["fixed_seed_contact_boundary"],
                            summaries[index + 1]["fixed_seed_contact_boundary"],
                        ),
                    ]
                    if before != after
                ],
                "common_time_changes": changes,
                "raw_lower_peak_relative_change": abs(
                    summaries[index + 1]["final_peak_lower_Pa"] - summaries[index]["final_peak_lower_Pa"]
                )
                / summaries[index + 1]["final_peak_lower_Pa"],
            }
        )
    output = {
        "scope": __doc__,
        "declared_discretization_axis": args.axis,
        "qualifications": qualifications,
        "cases": summaries,
        "comparisons": comparisons,
        "common_material_points": len(points),
        "sample_times_s": [i * frozen["dt"] for i in step_indices],
        "goal_completion": False,
        "limitations": "Fixed polyhedral geometry and physical held points. Configurations, explicit pairwise varying discretizations and fixed-contact-boundary markers define each comparison; body refinement changes contact triangles unless that boundary is frozen. A collision-only sweep keeps the volume fixed. Fixed material-coordinate interior probes carry seed-volume diagnostic weights, not an exact continuum norm. Tensors use each solution's proper finite body frame; contact wrenches use the shared fixed reference origin. Complete nonlinear global peak lower/upper bounds are retained separately. Two spatial levels or an equal-constructor control do not establish a physical tail. Contact quadrature/regularization/CAD geometry/full-arm support matching and feedback/model/accuracy-cost remain separate required studies.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
