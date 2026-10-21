"""Solve a bounded normal hard-contact reference without prescribing Hertz pressure.

A continuous bilinear pressure basis uses the half-space surface Green function
plus a smooth finite-body residual. Only a small normal-source table is prepared
offline; this is not a default runtime pressure solve or a full vector basis.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve
from skfem import FacetBasis, asm

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure, halfspace_stress, rectangular_compliance
from .p2_elastic import FreeElasticP2
from .study_hertz_residual import supplied_traction


@dataclass(frozen=True)
class NormalContactSolution:
    force: np.ndarray
    coordinates: np.ndarray
    gap: np.ndarray
    iterations: int
    maximum_linear_residual: float


def points_on_grid(axes):
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)


def probe_matrix(model, points):
    return sparse.vstack(
        [model.reference.scalar_basis.probes(points[start : start + 128].T) for start in range(0, len(points), 128)],
        format="csr",
    )


def hat_compliance(axes, modulus, quadrature):
    """Convolve two uniform rectangles to obtain continuous linear hat loads."""
    count = len(axes[0])
    widths = np.array([axis[1] - axis[0] for axis in axes])
    offsets = points_on_grid([np.arange(1 - count, count) * width for width in widths])
    gauss, weights = np.polynomial.legendre.leggauss(quadrature)
    sources = points_on_grid([gauss * width / 2 for width in widths])
    weights = np.outer(weights, weights).reshape(-1) / 4
    kernel = (rectangular_compliance(offsets, sources, widths, modulus) @ weights).reshape(2 * count - 1, -1)
    indices = np.indices((count, count)).reshape(2, -1).T
    first = indices[:, 0, None] - indices[None, :, 0] + count - 1
    second = indices[:, 1, None] - indices[None, :, 1] + count - 1
    matrix = kernel[first, second]
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=1e-20)
    return matrix


def hat_source_weights(source_axes, centers, widths):
    """Average the regular source interpolant over each positive pressure hat."""
    gauss, weights = np.polynomial.legendre.leggauss(3)
    nodes = np.r_[(gauss - 1) / 2, (gauss + 1) / 2]
    weights = np.tile(weights, 2) * (1 - np.abs(nodes)) / 2
    offsets = points_on_grid([nodes * width for width in widths])
    weight = np.outer(weights, weights).reshape(-1)
    count = len(source_axes[0])
    interpolation = RegularGridInterpolator(source_axes, np.eye(count * count).reshape(count, count, -1))
    positions = centers[:, None] + offsets[None]
    values = interpolation(positions.reshape(-1, 2)).reshape(len(centers), len(offsets), -1)
    result = np.einsum("q,nqg->gn", weight, values)
    np.testing.assert_allclose(result.sum(axis=0), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.T @ points_on_grid(source_axes), centers, rtol=1e-12, atol=1e-16)
    return result


def contact_minimum(compliance, gap, rigid, initial):
    """Primal active steps for the actual three-coordinate complementarity system."""
    force, active = initial.copy(), initial > 0
    target = np.array([1.0, 0, 0])
    np.testing.assert_allclose(rigid.T @ force, target, rtol=0, atol=1e-12)
    maximum_residual = 0.0
    for iteration in range(5 * len(force)):
        indices = np.flatnonzero(active)
        matrix = np.block(
            [[compliance[np.ix_(indices, indices)], rigid[indices]], [rigid[indices].T, np.zeros((3, 3))]]
        )
        rhs = np.r_[-gap[indices], target]
        solution = solve(matrix, rhs, assume_a="gen")
        residual = np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1)
        assert residual < 1e-10, residual
        maximum_residual = max(maximum_residual, float(residual))
        trial = np.zeros_like(force)
        trial[indices] = solution[:-3]
        negative = indices[trial[indices] < -1e-12]
        if len(negative):
            fractions = force[negative] / (force[negative] - trial[negative])
            leaving = negative[fractions.argmin()]
            force += fractions.min() * (trial - force)
            force[leaving], active[leaving] = 0, False
            continue
        force = trial
        coordinate = solution[-3:]
        dual = compliance @ force + gap + rigid @ coordinate
        inactive = np.flatnonzero(~active)
        if not len(inactive) or dual[inactive].min() >= -1e-10:
            assert force.min() >= -1e-12
            assert np.linalg.norm(rigid.T @ force - target) <= 1e-10
            assert dual.min() >= -1e-10 and np.max(np.abs(force * dual)) <= 1e-10
            return NormalContactSolution(force, coordinate, dual, iteration + 1, maximum_residual)
        active[inactive[dual[inactive].argmin()]] = True
    raise RuntimeError("Normal hard-contact complementarity did not converge")


def prepare_residual_table(model, source_axes, center_z, output):
    """Prepare only normal unit-source responses, including the cavity boundary."""
    mesh = model.reference.mesh
    facets = mesh.boundary_facets()
    on_plane = np.all(mesh.p[2, mesh.facets[:, facets]] == center_z, axis=0)
    boundary = FacetBasis(mesh, model.reference.element, facets=facets[~on_plane], intorder=12)
    points = boundary.global_coordinates().transpose(1, 2, 0)
    local = points.reshape(-1, 3).copy()
    local[:, 2] = center_z - local[:, 2]
    source_points = points_on_grid(source_axes)
    source_positions = np.column_stack((source_points, np.full(len(source_points), center_z)))
    probe_matrix(model, source_positions)
    displacements = np.empty((len(model.vertices), 3, len(source_points)))
    rows = []
    for index, source in enumerate(source_points):
        shifted = local.copy()
        shifted[:, :2] -= source
        tensor = halfspace_stress(shifted, np.zeros((1, 2)), np.ones(1))
        tensor[:, 2, :2] *= -1
        tensor[:, :2, 2] *= -1
        tensor = tensor.reshape(*points.shape[:2], 3, 3)
        traction = -np.einsum("fqij,jfq->ifq", tensor, boundary.normals)
        force = asm(supplied_traction, boundary, traction=traction).reshape(-1, 3)
        force_error = np.linalg.norm(force.sum(axis=0) - [0, 0, -1])
        relative = model.vertices - source_positions[index]
        moment_error = np.linalg.norm(np.cross(relative, force).sum(axis=0)) / 0.0005
        assert max(force_error, moment_error) < 1e-5, (index, force_error, moment_error)
        _, displacement, _, residual = model.solve(force)
        residual_relative = np.linalg.norm(residual) / np.linalg.norm(force)
        assert residual_relative < 1e-7, (index, residual_relative)
        displacements[:, :, index] = displacement
        rows.append(
            {
                "source": index,
                "position_m": source_positions[index].tolist(),
                "force_error_n": float(force_error),
                "moment_error_over_wall_nm": float(moment_error),
                "elastic_residual_relative": float(residual_relative),
            }
        )
        write_json(output / "source-progress.json", {"rows": rows})
    np.savez_compressed(
        output / "normal-source-table.npz",
        source_points_m=source_positions,
        coordinates_m=model.vertices,
        displacement_m_per_n=displacements,
    )
    return displacements, rows


def pressure_observation(axes, solution, load, query_axes, hertz_axes, peak):
    points = points_on_grid(axes)
    widths = np.array([axis[1] - axis[0] for axis in axes])
    nodal_pressure = (solution.force * load / widths.prod()).reshape(len(axes[0]), -1)
    query = points_on_grid(query_axes)
    pressure = RegularGridInterpolator(axes, nodal_pressure)(query)
    expected = ellipse_pressure(query, hertz_axes, np.array([1.0, 0]), peak)
    support = solution.force.reshape(nodal_pressure.shape) > 1e-10
    assert not (support[0].any() or support[-1].any() or support[:, 0].any() or support[:, -1].any()), (
        "Contact reached the declared pressure window"
    )
    first_moment = solution.force @ points
    second = np.einsum("n,ni,nj->ij", solution.force, points, points) + np.diag(widths**2 / 6)
    eigen, vectors = np.linalg.eigh(second)
    moment_axes = np.sqrt(5 * eigen[::-1])
    force = solution.force * load
    return (
        {
            "relative_pressure_l2_to_hertz": float(np.linalg.norm(pressure - expected) / np.linalg.norm(expected)),
            "peak_pressure_pa": float(pressure.max()),
            "peak_pressure_relative_difference": float(pressure.max() / peak - 1),
            "moment_equivalent_axes_m": moment_axes.tolist(),
            "moment_axes_relative_difference": (moment_axes / hertz_axes - 1).tolist(),
            "moment_major_direction": vectors[:, -1].tolist(),
            "force_error_n": float(abs(force.sum() - load)),
            "moment_nm": [-float(first_moment[1] * load), float(first_moment[0] * load), 0.0],
            "active_pressure_nodes": int(support.sum()),
            "minimum_scaled_gap": float(solution.gap.min()),
            "complementarity_scaled": float(np.max(np.abs(solution.force * solution.gap))),
            "maximum_linear_residual": solution.maximum_linear_residual,
            "iterations": solution.iterations,
        },
        pressure,
        nodal_pressure,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load, young, poisson, density = 0.2, 1e10, 0.3, 2000
    modulus = young / (1 - poisson**2)
    curvature = np.diag([50.0, 80.0])
    hertz_axes, _, _, peak = ellipse(load, modulus, curvature)
    pressure_half_width = 1.3 * hertz_axes
    source_axes = [np.linspace(-1.5 * width, 1.5 * width, 5) for width in hertz_axes]
    query_axes = [np.linspace(-width, width, 129) for width in pressure_half_width]
    targets = {
        "pressure_grid_change": 0.02,
        "source_table_change": 0.02,
        "pressure_prediction_relative_l2": 0.05,
        "moment_axes_relative": 0.05,
        "halfspace_kernel_rule_change": 2e-4,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "mesh": file_id(args.mesh),
            "targets": targets,
            "scope": "Offline scalar normal-contact reference: continuous bilinear pressure hats, given force and zero first moments, force-controlled rigid paraboloid above a flat material land. No Hertz pressure/support is prescribed; its scale only chooses a window whose boundary must stay inactive. The finite-body regular correction includes the cavity and free-link inertial relief. This is not a runtime coupled pressure solver or an independent complete-stress reference.",
        },
    )
    with np.load(args.mesh) as mesh:
        vertices, tets, boundary = (mesh[key] for key in ["vertices", "tetrahedra", "surface_triangles"])
    # This captured polar-land domain declares the material plane at 30 mm.
    # A rounded mesh maximum can differ by one ulp and misclassify that plane.
    center_z = 0.03
    np.testing.assert_allclose(vertices[:, 2].max(), center_z, rtol=0, atol=1e-16)
    write_json(
        args.output / "material-plane.json",
        {"declared_z_m": center_z, "mesh_maximum_z_m": float(vertices[:, 2].max())},
    )
    begun = perf_counter()
    model = FreeElasticP2(vertices, tets, young, poisson, density, column_faces=boundary, cvxopt_factor=True)
    construction_seconds = perf_counter() - begun
    begun = perf_counter()
    displacements, source_rows = prepare_residual_table(model, source_axes, center_z, args.output)
    table_seconds = perf_counter() - begun
    query = points_on_grid(query_axes)
    physical_query = np.column_stack((query, np.full(len(query), center_z)))
    query_probe = probe_matrix(model, physical_query)
    center_response = -(query_probe @ displacements[:, 2, 12]) * load
    normalized = query / hertz_axes[0]
    fit = np.column_stack(
        (
            np.ones(len(query)),
            normalized,
            normalized[:, 0] ** 2 / 2,
            normalized[:, 0] * normalized[:, 1],
            normalized[:, 1] ** 2 / 2,
        )
    )
    inside = ((query / hertz_axes) ** 2).sum(axis=1) <= 1
    coefficient, _, _, _ = np.linalg.lstsq(fit[inside], center_response[inside], rcond=None)
    correction = np.array([[coefficient[3], coefficient[4]], [coefficient[4], coefficient[5]]]) / hertz_axes[0] ** 2
    corrected_curvature = curvature + correction
    corrected_axes, corrected_direction, _, corrected_peak = ellipse(load, modulus, corrected_curvature)
    corrected_prediction = ellipse_pressure(query, corrected_axes, corrected_direction, corrected_peak)
    rows, arrays, pressures = [], {}, {}
    source_selections = {3: np.array([i * 5 + j for i in [0, 2, 4] for j in [0, 2, 4]]), 5: np.arange(25)}
    for count in [17, 33, 65]:
        axes = [np.linspace(-width, width, count) for width in pressure_half_width]
        points = points_on_grid(axes)
        widths = np.array([axis[1] - axis[0] for axis in axes])
        print(f"Contact grid {count}x{count}.", flush=True)
        base = hat_compliance(axes, modulus, 8)
        rule_change = None
        if count == 17:
            higher = hat_compliance(axes, modulus, 12)
            rule_change = float(np.linalg.norm(base - higher) / np.linalg.norm(higher))
            assert rule_change < targets["halfspace_kernel_rule_change"], rule_change
        gap = np.einsum("ni,ij,nj->n", points, curvature, points) / 2
        rigid = np.column_stack((np.ones(len(points)), points / hertz_axes[0]))
        initial = np.zeros((count, count))
        for i in [count // 2 - 1, count // 2 + 1]:
            for j in [count // 2 - 1, count // 2 + 1]:
                initial[i, j] = 0.25
        scale = load / (modulus * hertz_axes[0])
        target_points = np.column_stack((points, np.full(len(points), center_z)))
        response = -(probe_matrix(model, target_points) @ displacements[:, 2, :])
        for table in [0, 3, 5]:
            correction_matrix = np.zeros_like(base)
            if table:
                selected = source_selections[table]
                regular_axes = source_axes if table == 5 else [axis[::2] for axis in source_axes]
                interpolation = hat_source_weights(regular_axes, points, widths)
                correction_matrix = response[:, selected] @ interpolation
            matrix = base + correction_matrix
            begun = perf_counter()
            solution = contact_minimum(matrix * load / scale, gap / scale, rigid, initial.reshape(-1))
            seconds = perf_counter() - begun
            initial = solution.force.reshape(count, count)
            observation, pressure, nodal = pressure_observation(axes, solution, load, query_axes, hertz_axes, peak)
            label = f"grid{count}-sources{table}"
            observation.update(
                label=label,
                nodes_per_axis=count,
                source_table_axis=table,
                solve_seconds=seconds,
                kernel_rule_relative_change=rule_change,
            )
            if table == 5:
                observation["curvature_corrected_pressure_relative_l2"] = float(
                    np.linalg.norm(pressure - corrected_prediction) / np.linalg.norm(pressure)
                )
            rows.append(observation)
            pressures[label] = pressure
            arrays.update(
                {
                    label + "-pressure_pa": nodal,
                    label + "-force_n": solution.force * load,
                    label + "-scaled_gap": solution.gap,
                    label + "-coordinates_m": solution.coordinates * scale,
                }
            )
            write_json(args.output / "contact-progress.json", {"rows": rows})
        del base, correction_matrix, matrix
    reference = pressures["grid65-sources5"]
    grid_change = np.linalg.norm(pressures["grid33-sources5"] - reference) / np.linalg.norm(reference)
    table_change = np.linalg.norm(pressures["grid65-sources3"] - reference) / np.linalg.norm(reference)
    np.savez_compressed(args.output / "pressures.npz", common_points_m=query, **arrays)
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "construction_seconds": construction_seconds,
            "normal_source_table_seconds": table_seconds,
            "normal_source_checks": source_rows,
            "grid_relative_pressure_change": float(grid_change),
            "source_table_relative_pressure_change": float(table_change),
            "targets": targets,
            "discrete_pressure_refinement_targets_met": bool(
                grid_change <= targets["pressure_grid_change"] and table_change <= targets["source_table_change"]
            ),
            "hertz_axes_m": hertz_axes.tolist(),
            "hertz_peak_pressure_pa": float(peak),
            "one_pass_curvature_correction_per_m": correction.tolist(),
            "corrected_hertz_axes_m": corrected_axes.tolist(),
            "independent_complete_stress_qualified": False,
            "finite_body_mesh_refinement_qualified": False,
            "scope": "Pressure/support are outputs of force-and-moment-controlled normal complementarity with continuous hat loads. Half-space and two regular source tables share no prescribed Hertz pressure. Source-table and pressure-grid changes bound only the tested discrete components. A fixed finite-body mesh and sampled contact constraints do not certify continuum pressure/stress or arbitrary contact geometry/friction. One-pass curvature correction uses the already computed normal residual displacement and is separately compared.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
