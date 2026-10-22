"""Test particular Hertz stress plus a finite-body traction correction."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from skfem import FacetBasis, LinearForm, MeshTet, asm
from skfem.helpers import dot
from skfem.quadrature import get_quadrature_tet

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .hertz_stress import elliptic_stress as unbatched_elliptic_stress
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures


@LinearForm
def supplied_traction(v, w):
    return dot(w.traction, v)


def elliptic_stress(points, axes, **kwargs):
    return np.concatenate(
        [unbatched_elliptic_stress(p, axes, **kwargs) for p in np.array_split(points, (len(points) + 2047) // 2048)]
    )


def tensor_at_points(model, displacement, points):
    """Bound point-location scratch without changing points or the P2 field."""
    reference = model.reference
    corners, _ = reference.corner_stress(displacement.reshape(-1))
    result = []
    for start in range(0, len(points), 256):
        current = points[start : start + 256]
        elements = reference.mesh.element_finder()(*current.T)
        local = reference.basis.mapping.invF(current.T[:, :, None], tind=elements)[:, :, 0]
        barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
        result.append(np.einsum("nc,ncij->nij", barycentric, corners[elements]))
    return np.concatenate(result)


def hertz_nodal_load(model, axes, load, radial, angular):
    t, weights = np.polynomial.legendre.leggauss(radial)
    t, weights = (t + 1) / 2, weights / 2
    theta = 2 * np.pi * (np.arange(angular) + 0.5) / angular
    points = np.zeros((radial, angular, 3))
    points[:, :, :2] = (
        np.sqrt(1 - t[:, None] ** 2)[..., None] * (np.column_stack((np.cos(theta), np.sin(theta))) * axes)[None]
    )
    force = np.zeros_like(model.vertices)
    locations = points.reshape(-1, 3)
    forces = np.repeat(3 * t * t * weights * load / angular, angular)
    for start in range(0, len(locations), 256):
        force[:, 2] += (
            model.reference.scalar_basis.probes(locations[start : start + 256].T).T @ forces[start : start + 256]
        )
    return force


def remote_hertz_traction(model, axes, peak, quadrature, stress_order):
    mesh = model.reference.mesh
    facets = mesh.boundary_facets()
    facets = facets[np.any(mesh.p[2, mesh.facets[:, facets]] > 0, axis=0)]
    boundary = FacetBasis(mesh, model.reference.element, facets=facets, intorder=quadrature)
    points = boundary.global_coordinates().transpose(1, 2, 0)
    stress = elliptic_stress(points.reshape(-1, 3), axes, peak=peak, order=stress_order).reshape(
        *points.shape[:2], 3, 3
    )
    traction = np.einsum("fqij,jfq->ifq", stress, boundary.normals)
    return asm(supplied_traction, boundary, traction=traction).reshape(-1, 3)


def field_difference(candidate, reference, weights):
    return np.sqrt(
        np.dot(weights, ((candidate - reference) ** 2).sum(axis=(1, 2)))
        / np.dot(weights, (reference**2).sum(axis=(1, 2)))
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "complete_tensor_relative": 0.05,
        "peak_relative": 0.10,
        "reference_field_change": 0.02,
        "reference_peak_change": 0.05,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "scope": "Finite distributed-pressure box control with unchanged consistent free-link inertial relief. Hertz is a prescribed normal load; contact-pressure prediction and hollow-shell applicability are separate.",
        },
    )
    load, young, poisson, density = 0.2, 1e10, 0.3, 2000
    axes, _, _, peak = ellipse(load, young / (1 - poisson * poisson), np.diag([50.0, 80.0]))
    a = axes[0]
    grids = [
        ("coarse", [-20, -10, 0, 10, 20], [0, 10, 20]),
        ("medium", [-20, -10, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 10, 20], [0, 0.25, 0.5, 1, 2, 4, 10, 20]),
        (
            "fine",
            [-20, -10, -4, -2, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 2, 4, 10, 20],
            [0, 0.125, 0.25, 0.5, 1, 2, 4, 10, 20],
        ),
    ]
    meshes = [MeshTet.init_tensor(a * np.array(x), a * np.array(x), a * np.array(z)) for _, x, z in grids]
    common = meshes[1]
    q, weight = get_quadrature_tet(2)
    barycentric = np.column_stack((1 - q.sum(axis=0), q.T))
    corners = common.p.T[common.t.T]
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    points = np.einsum("qc,tci->tqi", barycentric, corners).reshape(-1, 3)
    weights = (volumes[:, None] * (6 * weight)).reshape(-1)
    np.testing.assert_allclose(weights.sum(), 40 * 40 * 20 * a**3, rtol=1e-13)
    r = np.linspace(-1.5, 1.5, 41)
    x, y = np.meshgrid(r * axes[0], r * axes[1], indexing="ij")
    probes = [np.column_stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)))]
    depth = np.r_[0, np.geomspace(0.005, 4, 60)] * a
    for along in (0, 1):
        x, z = np.meshgrid(np.linspace(-2, 2, 61) * axes[along], depth, indexing="ij")
        p = np.zeros((x.size, 3))
        p[:, along], p[:, 2] = x.reshape(-1), z.reshape(-1)
        probes.append(p)
    probes = np.concatenate(probes)
    hertz_field = elliptic_stress(points, axes, peak=peak, order=96)
    hertz_probes = elliptic_stress(probes, axes, peak=peak, order=96)
    candidate_hertz = elliptic_stress(points, axes, peak=peak, order=16)
    candidate_probes = elliptic_stress(probes, axes, peak=peak, order=16)
    rows, direct_fields, corrected_fields, direct_probes, corrected_probes = [], [], [], [], []
    for (name, _, _), mesh in zip(grids, meshes):
        print(f"Preparing {name}: {mesh.t.shape[1]} tetrahedra.", flush=True)
        started = perf_counter()
        model = FreeElasticP2(
            mesh.p.T,
            mesh.t.T,
            young,
            poisson,
            density,
            column_faces=mesh.facets[:, mesh.boundary_facets()].T,
            cvxopt_factor=True,
        )
        preparation = perf_counter() - started
        print(f"Prepared {name} with nested-dissection ordering in {preparation:.3f}s.", flush=True)
        force = hertz_nodal_load(model, axes, load, radial=24, angular=512)
        np.testing.assert_allclose(force.sum(axis=0), [0, 0, load], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.cross(model.vertices, force).sum(axis=0), 0, atol=1e-15)
        remote = remote_hertz_traction(model, axes, peak, quadrature=12, stress_order=96)
        print(f"Integrated {name} pressure and remote boundary traction.", flush=True)
        particular_load = force + remote
        particular_force_error = np.linalg.norm(particular_load.sum(axis=0)) / load
        particular_moment_error = np.linalg.norm(np.cross(model.vertices, particular_load).sum(axis=0)) / (load * a)
        assert particular_force_error < 1e-5 and particular_moment_error < 1e-5, (
            name,
            particular_force_error,
            particular_moment_error,
        )
        fields, observations, displacements, solver_rows, corner_fields = {}, {}, {}, {}, {}
        particular_corners = elliptic_stress(
            model.reference.corner_positions.reshape(-1, 3), axes, peak=peak, order=16 if name == "coarse" else 96
        ).reshape(mesh.t.shape[1], 4, 3, 3)
        for method, applied in (("direct", force), ("corrected", -remote)):
            begin = perf_counter()
            tensor, displacement, balance, residual = model.solve(applied)
            solve_seconds = perf_counter() - begin
            residual_relative = np.linalg.norm(residual) / np.linalg.norm(applied)
            assert residual_relative < 1e-7, (name, method, residual_relative)
            fields[method] = tensor_at_points(model, displacement, points)
            observations[method] = tensor_at_points(model, displacement, probes)
            if method == "corrected":
                fields[method] += candidate_hertz if name == "coarse" else hertz_field
                observations[method] += candidate_probes if name == "coarse" else hertz_probes
            corner_fields[method] = tensor + particular_corners if method == "corrected" else tensor
            displacements[method] = displacement
            tension, vm = stress_measures(observations[method])
            corner_tension, corner_vm = stress_measures(corner_fields[method])
            solver_rows[method] = {
                "elastic_residual_relative": float(residual_relative),
                "solve_seconds": solve_seconds,
                "balance_force_n": balance.sum(axis=0).tolist(),
                "sampled_tensile_peak_pa": float(tension.max()),
                "sampled_tensile_position_m": probes[tension.argmax()].tolist(),
                "sampled_von_mises_peak_pa": float(vm.max()),
                "sampled_von_mises_position_m": probes[vm.argmax()].tolist(),
                "corner_von_mises_peak_before_particular_pa": float(stress_measures(tensor)[1].max()),
                "corner_tensile_max_pa": float(corner_tension.max()),
                "corner_von_mises_max_pa": float(corner_vm.max()),
                "combined_sample_tensile_max_pa": float(max(tension.max(), corner_tension.max())),
                "combined_sample_von_mises_max_pa": float(max(vm.max(), corner_vm.max())),
            }
        sensitivities = {}
        if name == "coarse":
            lower = remote_hertz_traction(model, axes, peak, quadrature=8, stress_order=96)
            _, displacement, _, _ = model.solve(-lower)
            changed = tensor_at_points(model, displacement, points) + candidate_hertz
            sensitivities["remote_q8_q12_complete_tensor_relative"] = float(
                field_difference(changed, fields["corrected"], weights)
            )
            # The actual native candidate uses16-point near-field evaluation.
            lower = remote_hertz_traction(model, axes, peak, quadrature=12, stress_order=16)
            _, displacement, _, _ = model.solve(-lower)
            changed = tensor_at_points(model, displacement, points) + candidate_hertz
            sensitivities["remote_stress_order16_96_complete_tensor_relative"] = float(
                field_difference(changed, fields["corrected"], weights)
            )
        if name == "fine":
            refined = hertz_nodal_load(model, axes, load, radial=48, angular=1024)
            _, displacement, _, _ = model.solve(refined)
            changed = tensor_at_points(model, displacement, points)
            sensitivities["pressure_rule_complete_tensor_relative"] = float(
                field_difference(changed, fields["direct"], weights)
            )
        direct_fields.append(fields["direct"])
        corrected_fields.append(fields["corrected"])
        direct_probes.append(observations["direct"])
        corrected_probes.append(observations["corrected"])
        path = args.output / (name + ".npz")
        np.savez_compressed(
            path,
            vertices_m=mesh.p.T,
            tetrahedra=mesh.t.T,
            coordinates_m=model.vertices,
            applied_nodal_force_n=force,
            remote_hertz_nodal_force_n=remote,
            direct_displacement_m=displacements["direct"],
            residual_displacement_m=displacements["corrected"],
            common_points_m=points,
            common_weights_m3=weights,
            direct_stress_pa=fields["direct"],
            corrected_stress_pa=fields["corrected"],
            probe_points_m=probes,
            direct_probe_stress_pa=observations["direct"],
            corrected_probe_stress_pa=observations["corrected"],
            direct_corner_stress_pa=corner_fields["direct"],
            corrected_corner_stress_pa=corner_fields["corrected"],
        )
        rows.append(
            {
                "mesh": name,
                "tetrahedra": mesh.t.shape[1],
                "scalar_nodes": len(model.vertices),
                "preparation_seconds": preparation,
                "particular_force_error_relative": float(particular_force_error),
                "particular_moment_error_relative": float(particular_moment_error),
                "methods": solver_rows,
                "sensitivities": sensitivities,
                "arrays": file_id(path),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
        del model
    comparisons = {}
    for name, candidate, reference in (
        ("coarse_direct_vs_fine_direct", direct_fields[0], direct_fields[2]),
        ("coarse_corrected_vs_fine_direct", corrected_fields[0], direct_fields[2]),
        ("coarse_corrected_vs_fine_corrected", corrected_fields[0], corrected_fields[2]),
        ("medium_direct_vs_fine_direct", direct_fields[1], direct_fields[2]),
        ("medium_corrected_vs_fine_corrected", corrected_fields[1], corrected_fields[2]),
        ("fine_direct_vs_fine_corrected", direct_fields[2], corrected_fields[2]),
    ):
        comparisons[name] = float(field_difference(candidate, reference, weights))
    peak_errors, reference_changes = {}, {}
    for metric in ("combined_sample_tensile_max_pa", "combined_sample_von_mises_max_pa"):
        value = rows[2]["methods"]["corrected"][metric]
        peak_errors[metric] = abs(rows[0]["methods"]["corrected"][metric] / value - 1)
        reference_changes[metric] = abs(rows[1]["methods"]["corrected"][metric] / value - 1)
    accepted = comparisons["coarse_corrected_vs_fine_corrected"] <= targets["complete_tensor_relative"]
    accepted &= comparisons["medium_corrected_vs_fine_corrected"] <= targets["reference_field_change"]
    accepted &= max(peak_errors.values()) <= targets["peak_relative"]
    accepted &= max(reference_changes.values()) <= targets["reference_peak_change"]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "axes_m": axes.tolist(),
            "peak_pressure_pa": peak,
            "common_volume_queries": len(points),
            "near_field_probes": len(probes),
            "targets": targets,
            "complete_tensor_comparisons": comparisons,
            "coarse_corrected_peak_errors": peak_errors,
            "corrected_reference_peak_changes": reference_changes,
            "corrected_targets_met": bool(accepted),
            "scope": "Finite free-link box under prescribed distributed Hertz pressure. Complete volume-weighted common tensors and common near-field probes compare direct resolved-pressure elasticity and particular-stress residual decompositions. Corrected mesh refinement shares the analytical particular field; it is distinct from independent pressure prediction and arbitrary hollow-shell/friction qualification.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
