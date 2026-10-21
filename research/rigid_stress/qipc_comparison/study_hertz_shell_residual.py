"""Test Hertz particular-stress subtraction on one fixed hollow polar-land egg.

The pressure is prescribed. This isolates finite-shell stress recovery from
contact-pressure prediction; all refinements retain the same faceted domain.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
from skfem import FacetBasis, asm
from skfem.quadrature import get_quadrature_tet

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .p2_elastic import FreeElasticP2
from .prepare_adaptive import split_controls, split_edges, topology
from .prepare_contact_reference import refine_contact_surface
from .prepare_inputs import make_mesh
from .prepare_land_reference import land_surface
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import elliptic_stress, field_difference, supplied_traction, tensor_at_points


def refine_polar_edges(outer, inner, faces, radius, target):
    for _ in range(16):
        edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
        endpoints = outer[edges]
        middle = endpoints.mean(axis=1)
        lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
        nearby = np.linalg.norm(middle[:, :2], axis=1) < radius + lengths / 2
        selected = nearby & (middle[:, 2] > 0.02) & (lengths > target)
        if not selected.any():
            topology(faces, len(outer))
            return outer, inner, faces
        outer, inner, faces = split_edges(outer, inner, faces, {tuple(edge) for edge in edges[selected]})
    raise RuntimeError("Declared polar refinement did not reach its edge size")


def particular(points, axes, peak, center, order):
    local = points - center
    local[:, 2] *= -1
    tensor = elliptic_stress(local, axes, peak=peak, order=order)
    tensor[:, 2, :2] *= -1
    tensor[:, :2, 2] *= -1
    return tensor


def pressure_load(model, axes, load, center, radial, angular):
    t, weights = np.polynomial.legendre.leggauss(radial)
    t, weights = (t + 1) / 2, weights / 2
    angle = 2 * np.pi * (np.arange(angular) + 0.5) / angular
    points = np.zeros((radial, angular, 3))
    points[:, :, :2] = (
        np.sqrt(1 - t[:, None] ** 2)[..., None] * (np.column_stack((np.cos(angle), np.sin(angle))) * axes)[None]
    )
    points += center
    force = np.zeros_like(model.vertices)
    locations = points.reshape(-1, 3)
    values = -np.repeat(3 * t * t * weights * load / angular, angular)
    for start in range(0, len(locations), 128):
        force[:, 2] += (
            model.reference.scalar_basis.probes(locations[start : start + 128].T).T @ values[start : start + 128]
        )
    return force


def remote_traction(model, axes, peak, center, quadrature, stress_order):
    mesh = model.reference.mesh
    facets = mesh.boundary_facets()
    # On the complete loaded plane, prescribed and particular tractions cancel
    # analytically, including the sub-face footprint. Every other boundary,
    # including the inner cavity, contributes minus the particular traction.
    coplanar = np.all(mesh.p[2, mesh.facets[:, facets]] == center[2], axis=0)
    boundary = FacetBasis(mesh, model.reference.element, facets=facets[~coplanar], intorder=quadrature)
    points = boundary.global_coordinates().transpose(1, 2, 0)
    tensor = particular(points.reshape(-1, 3), axes, peak, center, stress_order).reshape(*points.shape[:2], 3, 3)
    traction = np.einsum("fqij,jfq->ifq", tensor, boundary.normals)
    return asm(supplied_traction, boundary, traction=traction).reshape(-1, 3)


def column_resultants(field, projection, thickness, depths, weights):
    tangential = projection[:, None] @ field @ projection[:, None]
    membrane = np.einsum("q,fqij,f->fij", weights, tangential, thickness)
    bending = np.einsum("q,fqij,f->fij", weights * (0.5 - depths), tangential, thickness**2)
    return membrane, bending


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "field_relative": 0.05,
        "peak_relative": 0.10,
        "reference_field_change": 0.02,
        "reference_peak_change": 0.05,
    }
    write_json(
        args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False), "targets": targets}
    )
    load, young, poisson, density = 0.2, 1e10, 0.3, 2000
    axes, _, _, peak = ellipse(load, young / (1 - poisson**2), np.diag([50.0, 80.0]))
    center = np.array([0.0, 0.0, 0.03])
    split_controls()
    unit, base_faces = refine_contact_surface(2e-4, 0.004, 0.0006)
    base_outer, base_inner = land_surface(unit, 0.0003, 0.0006)
    specifications = [
        ("coarse", 2e-4, [0, 0.5, 1]),
        ("medium", 5e-5, [0, 0.125, 0.5, 1]),
        ("fine", 2.5e-5, [0, 0.03125, 0.125, 0.5, 1]),
    ]
    meshes = []
    for name, edge, depths in specifications:
        outer, inner, faces = refine_polar_edges(base_outer, base_inner, base_faces, 4 * axes[0], edge)
        vertices, tets, boundary, volumes = make_mesh(outer, inner, faces, len(depths) - 1, depths)
        path = args.output / f"{name}-mesh.npz"
        np.savez_compressed(
            path,
            vertices=vertices,
            tetrahedra=tets,
            surface_triangles=boundary,
            outer_vertices=outer,
            inner_vertices=inner,
            mid_faces=faces,
        )
        meshes.append((name, vertices, tets, boundary, volumes, path))
    np.testing.assert_allclose([row[4].sum() for row in meshes], meshes[0][4].sum(), rtol=1e-12)
    _, common_vertices, common_tets, _, common_volume, _ = meshes[1]
    q, weights = get_quadrature_tet(2)
    bary = np.column_stack((1 - q.sum(axis=0), q.T))
    points = np.einsum("qc,tci->tqi", bary, common_vertices[common_tets]).reshape(-1, 3)
    volume_weights = (common_volume[:, None] * (6 * weights)).reshape(-1)
    radius = np.linspace(-1.5, 1.5, 41)
    x, y = np.meshgrid(radius * axes[0], radius * axes[1], indexing="ij")
    probes = [np.column_stack((x.reshape(-1), y.reshape(-1), np.full(x.size, center[2])))]
    depths = np.r_[0, np.geomspace(0.005 * axes[0], 0.0005, 64)]
    for along in (0, 1):
        x, z = np.meshgrid(np.linspace(-2, 2, 61) * axes[along], depths, indexing="ij")
        p = np.tile(center, (x.size, 1))
        p[:, along], p[:, 2] = x.reshape(-1), center[2] - z.reshape(-1)
        probes.append(p)
    probes = np.concatenate(probes)
    gauss, weights = np.polynomial.legendre.leggauss(8)
    column_depths, column_weights = (gauss + 1) / 2, weights / 2
    outer_center, inner_center = base_outer[base_faces].mean(axis=1), base_inner[base_faces].mean(axis=1)
    columns = (
        outer_center[:, None] * (1 - column_depths[None, :, None])
        + inner_center[:, None] * column_depths[None, :, None]
    )
    triangles = (base_outer + base_inner)[base_faces] / 2
    area_vector = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(area_vector, axis=1) / 2
    normal = area_vector / (2 * areas[:, None])
    projection = np.eye(3) - normal[:, :, None] * normal[:, None, :]
    thickness = np.abs(np.einsum("fi,fi->f", outer_center - inner_center, normal))
    sets = {"volume": points, "probes": probes, "columns": columns.reshape(-1, 3)}
    particular_fields = {
        order: {key: particular(p, axes, peak, center, order) for key, p in sets.items()} for order in [16, 96]
    }
    records, saved_fields = [], []
    for name, vertices, tets, boundary, volumes, mesh_path in meshes:
        print(f"Preparing {name}: {len(tets)} tetrahedra.", flush=True)
        begun = perf_counter()
        model = FreeElasticP2(vertices, tets, young, poisson, density, column_faces=boundary, cvxopt_factor=True)
        construction = perf_counter() - begun
        force = pressure_load(model, axes, load, center, 24, 512)
        np.testing.assert_allclose(force.sum(axis=0), [0, 0, -load], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.cross(model.vertices - center, force).sum(axis=0), 0, atol=1e-15)
        order = 16 if name == "coarse" else 96
        remote = remote_traction(model, axes, peak, center, 12, 96)
        complete = force + remote
        force_error = np.linalg.norm(complete.sum(axis=0)) / load
        moment_error = np.linalg.norm(np.cross(model.vertices - center, complete).sum(axis=0)) / (load * axes[0])
        print(f"{name} particular boundary errors: {force_error:.3g}, {moment_error:.3g}.", flush=True)
        assert force_error < 1e-5 and moment_error < 1e-5, (name, force_error, moment_error)
        particular_corners = particular(
            model.reference.corner_positions.reshape(-1, 3), axes, peak, center, order
        ).reshape(len(tets), 4, 3, 3)
        arrays, metrics, fields = {}, {}, {}
        for method, applied in [("direct", force), ("corrected", -remote)]:
            begun = perf_counter()
            corners, displacement, balance, residual = model.solve(applied)
            solve_seconds = perf_counter() - begun
            relative_residual = np.linalg.norm(residual) / np.linalg.norm(applied)
            assert relative_residual < 1e-7, (name, method, relative_residual)
            current = {key: tensor_at_points(model, displacement, p) for key, p in sets.items()}
            if method == "corrected":
                current = {key: value + particular_fields[order][key] for key, value in current.items()}
                corners += particular_corners
            column_field = current["columns"].reshape(len(base_faces), 8, 3, 3)
            membrane, bending = column_resultants(column_field, projection, thickness, column_depths, column_weights)
            current.update(membrane=membrane, bending=bending)
            sampled = np.concatenate((current["probes"], corners.reshape(-1, 3, 3)))
            tension, vm = stress_measures(sampled)
            metrics[method] = {
                "elastic_residual_relative": float(relative_residual),
                "solve_seconds": solve_seconds,
                "tensile_max_pa": float(tension.max()),
                "von_mises_max_pa": float(vm.max()),
                "balance_force_n": balance.sum(axis=0).tolist(),
            }
            fields[method] = current
            arrays.update(
                {
                    f"{method}_{key}_stress_pa": value
                    for key, value in current.items()
                    if key not in ["membrane", "bending"]
                }
            )
            arrays.update(
                {
                    f"{method}_membrane_n_per_m": membrane,
                    f"{method}_bending_n": bending,
                    f"{method}_corner_stress_pa": corners,
                    f"{method}_displacement_m": displacement,
                }
            )
        sensitivities = {}
        if name == "coarse":
            for label, quadrature, stress_order in [("boundary_q8_q12", 8, 96), ("boundary_stress16_96", 12, 16)]:
                changed_remote = remote_traction(model, axes, peak, center, quadrature, stress_order)
                _, displacement, _, _ = model.solve(-changed_remote)
                changed = tensor_at_points(model, displacement, points) + particular_fields[16]["volume"]
                sensitivities[label] = float(field_difference(changed, fields["corrected"]["volume"], volume_weights))
        if name == "fine":
            changed_force = pressure_load(model, axes, load, center, 48, 1024)
            _, displacement, _, _ = model.solve(changed_force)
            changed = tensor_at_points(model, displacement, points)
            sensitivities["pressure_q24_q48"] = float(
                field_difference(changed, fields["direct"]["volume"], volume_weights)
            )
        path = args.output / f"{name}-fields.npz"
        np.savez_compressed(
            path,
            volume_points_m=points,
            volume_weights_m3=volume_weights,
            probe_points_m=probes,
            column_points_m=columns,
            column_areas_m2=areas,
            nodal_force_n=force,
            remote_force_n=remote,
            **arrays,
        )
        records.append(
            {
                "name": name,
                "mesh": file_id(mesh_path),
                "fields": file_id(path),
                "tetrahedra": len(tets),
                "scalar_nodes": len(model.vertices),
                "volume_m3": float(volumes.sum()),
                "construction_seconds": construction,
                "particular_force_error_relative": float(force_error),
                "particular_moment_error_relative": float(moment_error),
                "methods": metrics,
                "sensitivities": sensitivities,
            }
        )
        saved_fields.append(fields)
        write_json(args.output / "progress.json", {"rows": records})
        del model
        gc.collect()
    comparisons = {}
    for label, left, right in [
        ("coarse_direct_fine_direct", saved_fields[0]["direct"], saved_fields[2]["direct"]),
        ("coarse_corrected_fine_direct", saved_fields[0]["corrected"], saved_fields[2]["direct"]),
        ("coarse_corrected_fine_corrected", saved_fields[0]["corrected"], saved_fields[2]["corrected"]),
        ("medium_direct_fine_direct", saved_fields[1]["direct"], saved_fields[2]["direct"]),
        ("medium_corrected_fine_corrected", saved_fields[1]["corrected"], saved_fields[2]["corrected"]),
        ("fine_direct_fine_corrected", saved_fields[2]["direct"], saved_fields[2]["corrected"]),
    ]:
        comparisons[label] = {
            key: float(field_difference(left[key], right[key], measure))
            for key, measure in [("volume", volume_weights), ("membrane", areas), ("bending", areas)]
        }
    peak_errors, reference_changes = {}, {}
    for key in ["tensile_max_pa", "von_mises_max_pa"]:
        value = records[2]["methods"]["corrected"][key]
        peak_errors[key] = abs(records[0]["methods"]["corrected"][key] / value - 1)
        reference_changes[key] = abs(records[1]["methods"]["corrected"][key] / value - 1)
    internal_targets_met = (
        max(comparisons["coarse_corrected_fine_corrected"].values()) <= targets["field_relative"]
        and max(comparisons["medium_corrected_fine_corrected"].values()) <= targets["reference_field_change"]
        and max(peak_errors.values()) <= targets["peak_relative"]
        and max(reference_changes.values()) <= targets["reference_peak_change"]
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": records,
            "comparisons": comparisons,
            "axes_m": axes.tolist(),
            "load_n": load,
            "peak_pressure_pa": peak,
            "targets": targets,
            "coarse_corrected_peak_changes": peak_errors,
            "corrected_reference_peak_changes": reference_changes,
            "corrected_internal_targets_met": bool(internal_targets_met),
            "independent_physical_accuracy_qualified": False,
            "scope": "One prescribed normal Hertz pressure on a fixed faceted full hollow polar-land egg, 500um wall. Local edge and wall refinement retain identical outer/inner boundaries and volume. Particular stress is subtracted on every nonloaded boundary, including the cavity; consistent-mass inertial relief is retained. Common volume-weighted tensors, material-column tangential membrane/bending resultants and sampled complete-corner/near-field maxima compare direct pressure elasticity and the residual decomposition. Internal corrected-mesh agreement shares the analytical particular field and is not independent pressure/support or certified continuum stress accuracy.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
