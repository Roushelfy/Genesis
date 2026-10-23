"""Compare original DPR/FEM with retained independent direct-pressure finite-box fields."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_automatic_dpr_controls import disk_wedge_loads
from .hertz_reference import ellipse
from .p2_elastic import FreeElasticP2
from .study_hertz_residual import field_difference, hertz_nodal_load, tensor_at_points


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.reference / (name + ".npz") for name in ["coarse", "medium", "fine"]]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "reference": [file_id(path) for path in paths],
            "scope": "Finite free-link box under prescribed normal pressure. This control supplies the independently "
            "verified circular formula's radius to the actual DPR transfer and original FEM. Collider estimation is "
            "tested separately. Only independent direct-pressure FEM arrays are used; no particular/corrected field "
            "is included. The reference ellipse has curvatures 50/80 per metre, while the circular approximation uses "
            "their arithmetic mean. Reference mesh uncertainty is reported, not treated as a certified exact solution.",
        },
    )
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    with np.load(paths[0]) as saved:
        vertices, tetrahedra = saved["vertices_m"], saved["tetrahedra"]
        points, weights = saved["common_points_m"], saved["common_weights_m3"]
        coarse_field = saved["direct_stress_pa"]
    with np.load(paths[1]) as saved:
        np.testing.assert_array_equal(saved["common_points_m"], points)
        medium_field = saved["direct_stress_pa"]
    with np.load(paths[2]) as saved:
        np.testing.assert_array_equal(saved["common_points_m"], points)
        fine_field = saved["direct_stress_pa"]
    mesh = MeshTet(vertices.T, tetrahedra.T, sort_t=False)
    asset = args.output / "elastic.npz"
    np.savez_compressed(
        asset, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    model = StressModel(gs.options.RigidStressOptions.dpr(mesh=asset, method="direct", output_mode="full"))
    surface = StressSurface(10, model.info)
    cpu = FreeElasticP2(vertices, tetrahedra, 1e10, 0.3, 2000, factorize=False)
    coordinates = qd_to_numpy(model.info.vertices)
    permutation = cpu.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
    load, modulus = 0.2, 1e10 / 0.91
    axes, _, _, _ = ellipse(load, modulus, np.diag([50.0, 80.0]))
    circular_radius = (0.75 * load / (modulus * 65.0)) ** (1 / 3)
    area_radius = float(np.sqrt(axes.prod()))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    rows, fields = [], {}
    for name, radius, profile in [
        ("manual_area_circle_dpr", area_radius, False),
        ("mean_curvature_circle_dpr", circular_radius, False),
        ("mean_curvature_circle_hertz", circular_radius, True),
    ]:
        contacts = create_contacts(1, 1, radius, hertz_profile=profile)
        contacts.valid.fill(True)
        contacts.position.fill(0)
        contacts.normal.from_numpy(np.array([[[0.0, 0.0, 1.0]]]))
        contacts.force.from_numpy(np.array([[[0.0, 0.0, load]]]))
        contacts.friction.fill(0)
        state = model.create_state(1, "full")
        kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, model.info, surface.info)
        np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid).all()
        native_force = qd_to_numpy(state.force, transpose=True)[0]
        force = np.empty_like(native_force)
        force[permutation] = native_force
        independent_load_error = None
        pressure_quadrature = []
        if profile:
            exact_native = disk_wedge_loads(
                model, surface, np.zeros(3), np.array([0, 0, load]), np.array([radius]), True
            )[0]
            independent_force = np.empty_like(force)
            independent_force[permutation] = exact_native
            independent_load_error = float(
                np.linalg.norm(force - independent_force) / np.linalg.norm(independent_force)
            )
            assert independent_load_error < 1e-6, independent_load_error
            for radial, angular in [(48, 1024), (96, 2048)]:
                integrated = hertz_nodal_load(cpu, np.full(2, radius), load, radial=radial, angular=angular)
                pressure_quadrature.append(
                    {
                        "radial": radial,
                        "angular": angular,
                        "relative_error_vs_analytic_sectors": float(
                            np.linalg.norm(integrated - independent_force) / np.linalg.norm(independent_force)
                        ),
                    }
                )
            assert pressure_quadrature[-1]["relative_error_vs_analytic_sectors"] < 1e-6, pressure_quadrature
        displacement = np.empty_like(coordinates)
        displacement[permutation] = qd_to_numpy(state.displacement, transpose=True)[0]
        field = tensor_at_points(cpu, displacement, points)
        fields[name] = field
        rows.append(
            {
                "mode": name,
                "radius_m": radius,
                "complete_volume_tensor_vs_fine_direct": float(field_difference(field, fine_field, weights)),
                "same_coarse_mesh_vs_elliptic_pressure": float(field_difference(field, coarse_field, weights)),
                "hertz_nodal_load_vs_analytic_sectors": independent_load_error,
                "independent_pressure_quadrature": pressure_quadrature,
                "native_corner_peak_pa": float(qd_to_numpy(state.peak)[0]),
                "force_error_n": float(qd_to_numpy(contacts.force_error)[0, 0]),
                "moment_error_nm": float(qd_to_numpy(contacts.moment_error)[0, 0]),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "fields.npz", common_points_m=points, common_weights_m3=weights, **fields)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "reference_axes_m": axes.tolist(),
            "reference_direct_medium_fine_change": float(field_difference(medium_field, fine_field, weights)),
            "coarse_exact_ellipse_vs_fine_direct": float(field_difference(coarse_field, fine_field, weights)),
            "reference_converged_under_2_percent": bool(field_difference(medium_field, fine_field, weights) <= 0.02),
            "scope": "Numerical conformance passes separately from measured physical/model differences. This finite-box "
            "normal-pressure comparison does not qualify hollow-egg contact, friction history or continuum stress peaks.",
        },
    )


if __name__ == "__main__":
    main()
