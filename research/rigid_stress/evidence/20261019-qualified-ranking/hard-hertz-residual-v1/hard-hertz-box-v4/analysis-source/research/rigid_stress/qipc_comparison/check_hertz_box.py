"""Compare the native finite-box correction and measure resident complete updates."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from skfem import FacetBasis, asm

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, halfspace_stress
from .native_hertz import NativeHertz
from .native_hertz_box import NativeHertzBox
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import (
    elliptic_stress,
    field_difference,
    remote_hertz_traction,
    supplied_traction,
    tensor_at_points,
)


def array(field):
    return qd_to_numpy(field, transpose=True, copy=True)


def timings(operation):
    operation()
    qd.sync()
    times = []
    for _ in range(5):
        start = perf_counter()
        for _ in range(10):
            operation()
        qd.sync()
        times.append((perf_counter() - start) / 10)
    return {"seconds": times, "median_seconds": float(np.median(times))}


def independent_remote_point_load(model, load):
    mesh = model.reference.mesh
    facets = mesh.boundary_facets()
    facets = facets[np.any(mesh.p[2, mesh.facets[:, facets]] > 0, axis=0)]
    basis = FacetBasis(mesh, model.reference.element, facets=facets, intorder=12)
    points = basis.global_coordinates().transpose(1, 2, 0)
    stress = halfspace_stress(points.reshape(-1, 3), np.zeros((1, 2)), np.array([load])).reshape(
        *points.shape[:2], 3, 3
    )
    traction = np.einsum("fqij,jfq->ifq", stress, basis.normals)
    return asm(supplied_traction, basis, traction=traction).reshape(-1, 3)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--remote", choices=("hertz", "point"), default="hertz")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "reference": file_id(args.reference),
            "remote_traction": args.remote,
            "complete_volume_tensor_relative_target": 1e-3,
            "scope": "Prescribed normal Hertz on the z=0 plane of a fixed finite box; one region per environment. Supplied current centers and curvatures, no contact grouping or friction. Module timing is not moving-scene timing.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(args.reference) as saved:
        vertices, tets = saved["vertices_m"], saved["tetrahedra"]
        volume_points, volume_weights = saved["common_points_m"], saved["common_weights_m3"]
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    mesh = args.output / "mesh.npz"
    np.savez_compressed(
        mesh,
        vertices=vertices,
        tetrahedra=tets,
        surface_triangles=cpu.reference.mesh.facets[:, cpu.reference.mesh.boundary_facets()].T,
    )
    started = perf_counter()
    model = StressModel(RigidStressOptions(mesh=mesh, method="auto"))
    surface = StressSurface(4, model.info)
    qd.sync()
    build_seconds = perf_counter() - started
    coordinates = array(model.info.vertices)
    permutation = cpu.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
    axes, _, _, _ = ellipse(0.2, 1e10 / 0.91, np.diag([50.0, 80.0]))
    angle = np.arange(32) * (2 * np.pi / 32)
    probes = np.column_stack((axes[0] * np.cos(angle), axes[1] * np.sin(angle), np.zeros(32)))
    axial = np.column_stack((np.zeros((61, 2)), np.r_[0, np.geomspace(0.005, 4, 60)] * axes[0]))
    points = np.concatenate((cpu.reference.corner_positions.reshape(-1, 3), probes, axial))
    # The complete P2 field uses each element's own corner side. Additional
    # probes have one material element located during this control's setup.
    probe_points = np.concatenate((probes, axial))
    probe_elements = cpu.reference.mesh.element_finder()(*probe_points.T)
    local = cpu.reference.basis.mapping.invF(probe_points.T[:, :, None], tind=probe_elements)[:, :, 0]
    elements = np.r_[np.repeat(np.arange(len(tets)), 4), probe_elements]
    barycentric = np.concatenate(
        (np.tile(np.eye(4), (len(tets), 1)), np.column_stack((1 - local.sum(axis=0), local.T)))
    )
    hertz = NativeHertz(3)
    hertz.state.force.from_numpy(np.array([0.2, 0.12, 0.25]))
    hertz.state.modulus.fill(1e10 / 0.91)
    hertz.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (3, 2, 2)).copy())
    hertz.update()
    native = NativeHertzBox(3, points, elements, barycentric, point_remote=args.remote == "point")
    state = model.create_state(3, "max")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
    omega.fill(0)
    native.scatter(hertz.state, state, model.info, surface.info)
    assert np.all(array(native.query.status) == 0), array(native.query.status)
    loads = array(state.force)
    model.recover(omega, state, surface_load=True)
    native.evaluate(hertz.state, state, model.info)
    assert np.all(array(state.valid)) and np.all(array(native.query.status) == 0)
    actual, peak = array(native.query.tensor), array(native.query.peak)
    native_displacements = array(state.displacement)
    supplied_axes, supplied_pressure = array(hertz.state.axes), array(hertz.state.peak_pressure)
    reference, rows, volume_fields, refined_volume_fields, exact_fields = [], [], [], [], []
    for env, load in enumerate([0.2, 0.12, 0.25]):
        a, _, _, pressure = ellipse(load, 1e10 / 0.91, np.diag([50.0, 80.0]))
        applied = np.empty_like(loads[env])
        applied[permutation] = loads[env]
        corner, displacement, _, residual = cpu.solve(applied)
        expected = np.concatenate((corner.reshape(-1, 3, 3), tensor_at_points(cpu, displacement, probe_points)))
        expected += elliptic_stress(points, supplied_axes[env], peak=supplied_pressure[env], order=16)
        error = np.linalg.norm(actual[env] - expected) / np.linalg.norm(expected)
        assert error < 1e-7 and np.linalg.norm(residual) / np.linalg.norm(applied) < 1e-7, error
        native_u = np.empty_like(native_displacements[env])
        native_u[permutation] = native_displacements[env]
        volume = tensor_at_points(cpu, native_u, volume_points)
        volume += elliptic_stress(volume_points, supplied_axes[env], peak=supplied_pressure[env], order=16)
        point_load_error = None
        if args.remote == "point":
            point_load = independent_remote_point_load(cpu, load)
            point_load_error = float(np.linalg.norm(applied + point_load) / np.linalg.norm(point_load))
            assert point_load_error < 1e-4, point_load_error
        remote = remote_hertz_traction(cpu, supplied_axes[env], supplied_pressure[env], 12, 96)
        corner, displacement, _, _ = cpu.solve(-remote)
        refined_volume = tensor_at_points(cpu, displacement, volume_points)
        refined_volume += elliptic_stress(volume_points, supplied_axes[env], peak=supplied_pressure[env], order=96)
        volume_error = float(field_difference(volume, refined_volume, volume_weights))
        assert volume_error < 1e-3, volume_error
        volume_fields.append(volume)
        refined_volume_fields.append(refined_volume)
        refined = np.concatenate((corner.reshape(-1, 3, 3), tensor_at_points(cpu, displacement, probe_points)))
        refined += elliptic_stress(points, supplied_axes[env], peak=supplied_pressure[env], order=96)
        refined_error = np.linalg.norm(actual[env] - refined) / np.linalg.norm(refined)
        assert refined_error < 1e-3, refined_error
        exact_remote = remote_hertz_traction(cpu, a, pressure, 12, 96)
        corner, displacement, _, _ = cpu.solve(-exact_remote)
        exact = np.concatenate((corner.reshape(-1, 3, 3), tensor_at_points(cpu, displacement, probe_points)))
        exact += elliptic_stress(points, a, peak=pressure, order=96)
        lookup_error = np.linalg.norm(actual[env] - exact) / np.linalg.norm(exact)
        exact_fields.append(exact)
        np.testing.assert_allclose(peak[env], stress_measures(actual[env])[1].max(), rtol=1e-12)
        rows.append(
            {
                "force_n": load,
                "same_load_tensor_relative": float(error),
                "higher_rule_tensor_relative": float(refined_error),
                "complete_volume_tensor_relative": volume_error,
                "independent_point_traction_relative": point_load_error,
                "independent_ellipse_tensor_relative": float(lookup_error),
                "wrench_error_relative": float(array(native.query.wrench_error)[env]),
                "sampled_von_mises_peak_pa": float(peak[env]),
            }
        )
        reference.append(refined)
    # Reset only env1 and invalidate env2; env0 must remain independent.
    hertz.state.force.from_numpy(np.array([0.2, 0, -0.1]))
    hertz.update()
    native.scatter(hertz.state, state, model.info, surface.info)
    np.testing.assert_array_equal(array(native.query.status), [0, 1, 2])
    np.testing.assert_allclose(array(state.force)[0], loads[0], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(array(state.force)[1:], 0)
    model.recover(omega, state, surface_load=True)
    native.evaluate(hertz.state, state, model.info)
    altered = array(native.query.tensor)
    np.testing.assert_allclose(altered[0], actual[0], rtol=1e-10, atol=1e-5)
    np.testing.assert_array_equal(altered[1], 0)
    assert np.isnan(altered[2]).all() and np.isnan(array(native.query.peak)[2])
    hertz.state.force.from_numpy(np.array([0.2, 0.12, 0.25]))
    hertz.update()
    native.scatter(hertz.state, state, model.info, surface.info)
    model.recover(omega, state, surface_load=True)
    native.evaluate(hertz.state, state, model.info)
    np.testing.assert_allclose(array(native.query.tensor), actual, rtol=1e-10, atol=1e-5)
    centers = np.zeros((3, 3))
    centers[1, :2] = axes * [0.5, 0.25]
    native.query.center.from_numpy(centers)
    theta = np.deg2rad(23)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    curvatures = np.broadcast_to(np.diag([50.0, 80.0]), (3, 2, 2)).copy()
    curvatures[1] = rotation @ curvatures[1] @ rotation.T
    hertz.state.curvature.from_numpy(curvatures)
    hertz.update()
    native.scatter(hertz.state, state, model.info, surface.info)
    model.recover(omega, state, surface_load=True)
    native.evaluate(hertz.state, state, model.info)
    moved = array(native.query.tensor)
    assert np.all(array(native.query.status) == 0) and np.all(array(state.valid))
    np.testing.assert_allclose(moved[[0, 2]], actual[[0, 2]], rtol=1e-10, atol=1e-5)
    applied[permutation] = array(state.force)[1]
    corner, displacement, _, _ = cpu.solve(applied)
    expected = np.concatenate((corner.reshape(-1, 3, 3), tensor_at_points(cpu, displacement, probe_points)))
    d = array(hertz.state.direction)[1]
    frame = np.array([[d[0], -d[1], 0], [d[1], d[0], 0], [0, 0, 1]])
    local = elliptic_stress(
        (points - centers[1]) @ frame, array(hertz.state.axes)[1], peak=array(hertz.state.peak_pressure)[1], order=16
    )
    expected += frame @ local @ frame.T
    moving_error = float(np.linalg.norm(moved[1] - expected) / np.linalg.norm(expected))
    assert moving_error < 1e-7, moving_error
    centers[2, 0] = np.nan
    native.query.center.from_numpy(centers)
    native.scatter(hertz.state, state, model.info, surface.info)
    np.testing.assert_array_equal(array(native.query.status), [0, 0, 2])
    np.testing.assert_array_equal(array(state.force)[2], 0)
    arrays = args.output / "fields.npz"
    np.savez_compressed(
        arrays,
        points_m=points,
        elements=elements,
        barycentric=barycentric,
        native_tensor_pa=actual,
        reference_tensor_pa=np.array(reference),
        independent_ellipse_tensor_pa=np.array(exact_fields),
        volume_points_m=volume_points,
        volume_weights_m3=volume_weights,
        native_volume_tensor_pa=np.array(volume_fields),
        refined_volume_tensor_pa=np.array(refined_volume_fields),
        moved_tensor_pa=moved,
        moved_expected_tensor_pa=expected,
        residual_force_n=loads,
        coordinates_m=coordinates,
    )
    write_json(
        args.output / "validation.json",
        {
            "passed": True,
            "rows": rows,
            "moving_center_curvature_tensor_relative": moving_error,
            "arrays": file_id(arrays),
        },
    )
    benchmark = []
    for batch in (1, 128, 512):
        max_peak = None
        for full in (False, True):
            hertz = NativeHertz(batch)
            hertz.state.force.from_numpy(np.linspace(0.12, 0.25, batch))
            hertz.state.modulus.fill(1e10 / 0.91)
            hertz.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (batch, 2, 2)).copy())
            native = NativeHertzBox(batch, points, elements, barycentric, full, point_remote=args.remote == "point")
            state = model.create_state(batch, "max")
            omega = V_VEC(3, dtype=gs.qd_float, shape=(batch,))
            omega.fill(0)

            def update():
                hertz.update()
                native.scatter(hertz.state, state, model.info, surface.info)
                model.recover(omega, state, surface_load=True)
                native.evaluate(hertz.state, state, model.info)

            total = timings(update)
            assert np.all(array(state.valid)) and np.all(array(native.query.status) == 0)
            measured = {"batch": batch, "output": "full" if full else "max", "inclusive": total}
            if full:
                np.testing.assert_allclose(array(native.query.peak), max_peak, rtol=1e-10)
                measured["transfer"] = timings(lambda: native.scatter(hertz.state, state, model.info, surface.info))
                measured["recovery"] = timings(lambda: model.recover(omega, state, surface_load=True))
                measured["combined_field"] = timings(lambda: native.evaluate(hertz.state, state, model.info))
            else:
                max_peak = array(native.query.peak)
            benchmark.append(measured)
            write_json(args.output / "benchmark-progress.json", {"rows": benchmark})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "remote_traction": args.remote,
            "moving_center_curvature_tensor_relative": moving_error,
            "rows": rows,
            "arrays": file_id(arrays),
            "build_seconds": build_seconds,
            "elastic_method": model.selected_method,
            "tetrahedra": len(tets),
            "scalar_nodes": len(coordinates),
            "queries": len(points),
            "surface_gauss_order": 4,
            "hertz_order": 16,
            "benchmark": benchmark,
            "scope": "Normal flat-box module with fixed supplied queries including every P2 corner and93 local probes. Maximum is over these samples; no certified continuum maximum. Includes estimate, traction correction, elastic recovery/residual, combined field and reduction. Full/max remain resident. Excludes JIT/setup, host outputs, moving dynamics, current region/curvature preparation and tangential history.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
