import json

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("skfem")

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.native_load_samples import kernel_continuous_boundary_load
from research.rigid_stress.skfem_reference import SkfemElasticReference
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("method", ("direct", "pcg"))
@pytest.mark.parametrize("analytic", ("uniform_strain", "pure_bending"))
def test_independent_fem_assembly_boundary_loads_and_inertia_relief(
    nonconvex_material_mesh,  # noqa: F811
    tmp_path,
    method,
    analytic,
    record_property,
):
    vertices, tets, faces, shape = nonconvex_material_mesh
    mesh = tmp_path / "material.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    options = RigidStressOptions(mesh=mesh, method=method, output_mode="full")
    native = StressModel(options)
    reference = SkfemElasticReference(vertices, tets, options.young, options.poisson, options.density)
    mass_gauge = SkfemElasticReference(vertices, tets, options.young, options.poisson, options.density, gauge="mass")
    coordinates = qd_to_numpy(native.info.vertices)
    permutation = reference.native_dof_permutation(coordinates)
    columns = qd_to_numpy(native.info.columns)
    starts = qd_to_numpy(native.info.row_start)
    stiffness = (
        options.young
        * sparse.bsr_matrix(
            (qd_to_numpy(native.info.stiffness), columns, starts), shape=(3 * len(coordinates), 3 * len(coordinates))
        ).tocsr()
    )
    mass = sparse.kron(
        sparse.csr_matrix((qd_to_numpy(native.info.mass), columns, starts), shape=(len(coordinates), len(coordinates))),
        np.eye(3),
        format="csr",
    )
    stiffness_reference = reference.stiffness[permutation][:, permutation]
    mass_reference = reference.mass[permutation][:, permutation]
    stiffness_error = sparse.linalg.norm(stiffness - stiffness_reference) / sparse.linalg.norm(stiffness_reference)
    mass_error = sparse.linalg.norm(mass - mass_reference) / sparse.linalg.norm(mass_reference)
    assert stiffness_error < 1e-11
    assert mass_error < 1e-11

    if analytic == "uniform_strain":
        strain = np.array([[1.2, 0.2, -0.3], [0.2, -0.7, 0.1], [-0.3, 0.1, 0.4]]) * 1e-6
        lam, mu = reference.lame
        tensor = 2 * mu * strain + lam * np.trace(strain) * np.eye(3)
        force = reference.uniform_stress_load(tensor)
        expected_displacement = np.empty(reference.basis.N)
        expected_displacement[reference.component_dofs] = reference.coordinates @ strain.T
        expected_tensor = np.broadcast_to(tensor, (len(tets), 4, 3, 3))
    else:
        curvature = 2e-4
        x, y, z = reference.coordinates.T
        expected_displacement = np.empty(reference.basis.N)
        expected_displacement[reference.component_dofs] = np.column_stack(
            (
                curvature * x * z,
                -options.poisson * curvature * y * z,
                -curvature * (x * x - options.poisson * y * y + options.poisson * z * z) / 2,
            )
        )
        expected_tensor = np.zeros((len(tets), 4, 3, 3))
        expected_tensor[..., 0, 0] = options.young * curvature * reference.corner_positions[..., 2]
        force = reference.pure_bending_load(curvature)
    surface = StressSurface(10, native.info)
    boundary_state = native.create_state(1)
    native_tensor = tensor if analytic == "uniform_strain" else np.zeros((3, 3))
    bending_scale = options.young * curvature if analytic == "pure_bending" else 0.0
    kernel_continuous_boundary_load(native_tensor, bending_scale, boundary_state, native.info, surface.info)
    integrated = qd_to_numpy(boundary_state.force, transpose=True)[0].reshape(-1)
    np.testing.assert_allclose(integrated, force[permutation], rtol=1e-11, atol=1e-13)
    expected_displacement = reference.align_rigid_gauge(expected_displacement)
    solved = reference.solve(force)
    np.testing.assert_allclose(solved.displacement, expected_displacement, rtol=1e-9, atol=1e-17)
    np.testing.assert_allclose(solved.tensor_pa, expected_tensor, rtol=1e-9, atol=1e-7)
    assert solved.residual_budget_fraction <= 1
    center = vertices[tets].mean(axis=1)
    probe_tensor, probe_vm = reference.stress_at_points(solved.displacement, center)
    np.testing.assert_allclose(probe_tensor, solved.tensor_pa.mean(axis=1), rtol=1e-9, atol=1e-7)
    assert np.isfinite(probe_vm).all()

    barycentric = np.array([[0.1, 0.2, 0.3, 0.4], [0.6, 0.1, 0.2, 0.1], [0.15, 0.35, 0.4, 0.1]])
    points = np.einsum("qc,ecj->eqj", barycentric, vertices[tets]).reshape(-1, 3)
    probe_tensor, _ = reference.stress_at_points(solved.displacement, points)
    if analytic == "uniform_strain":
        expected_probes = np.broadcast_to(tensor, probe_tensor.shape)
    else:
        expected_probes = np.zeros_like(probe_tensor)
        expected_probes[:, 0, 0] = options.young * curvature * points[:, 2]
    np.testing.assert_allclose(probe_tensor, expected_probes, rtol=1e-9, atol=1e-7)

    random = np.random.default_rng(2026101401)
    random_force = random.normal(size=reference.basis.N) * 1e-5
    forces = (force, random_force)
    velocities = np.array([[0, 0, 0], [0.4, -0.7, 0.9]])
    state = native.create_state(2, "full")
    native_forces = np.stack([value[permutation].reshape(-1, 3) for value in forces], axis=1)
    state.force.from_numpy(native_forces)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    omega.from_numpy(velocities)
    native.recover(omega, state)
    assert qd_to_numpy(state.valid).all()
    displacements = qd_to_numpy(state.displacement, transpose=True).reshape(2, -1)
    tensors = qd_to_numpy(state.stress_tensor, transpose=True)
    vm = qd_to_numpy(state.von_mises, transpose=True)[..., 0]
    rhs = qd_to_numpy(state.rhs, transpose=True).reshape(2, -1)
    peaks = qd_to_numpy(state.peak)
    rows = []
    for environment, (load, velocity) in enumerate(zip(forces, velocities)):
        oracle = reference.solve(load, velocity)
        mass_oracle = mass_gauge.solve(load, velocity)
        assert oracle.residual_budget_fraction <= 1
        assert mass_oracle.residual_budget_fraction <= 1
        np.testing.assert_allclose(oracle.displacement, mass_oracle.displacement, rtol=1e-9, atol=1e-17)
        np.testing.assert_allclose(oracle.tensor_pa, mass_oracle.tensor_pa, rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(oracle.energy_j, mass_oracle.energy_j, rtol=1e-10, atol=1e-20)
        native_displacement = np.empty(reference.basis.N)
        native_displacement[permutation] = displacements[environment]
        aligned = reference.align_rigid_gauge(native_displacement)
        np.testing.assert_allclose(aligned, oracle.displacement, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(rhs[environment], oracle.rhs[permutation], rtol=1e-10, atol=1e-13)
        residual = np.linalg.norm(reference.stiffness @ native_displacement - oracle.rhs)
        budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(oracle.rhs))
        assert residual <= budget
        expected = oracle.tensor_pa[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        np.testing.assert_allclose(tensors[environment], expected, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(vm[environment], oracle.von_mises_pa, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(peaks[environment], oracle.peak_pa, rtol=1e-4, atol=1e-3)
        energy = 0.5 * native_displacement @ (reference.stiffness @ native_displacement)
        np.testing.assert_allclose(energy, oracle.energy_j, rtol=1e-8, atol=1e-20)
        rows.append(
            {
                "environment": environment,
                "complete_residual_N": residual,
                "budget_fraction": residual / budget,
                "peak_difference_Pa": abs(peaks[environment] - oracle.peak_pa),
                "energy_J": energy,
                "independent_pinned_residual_budget_fraction": oracle.residual_budget_fraction,
                "independent_mass_residual_budget_fraction": mass_oracle.residual_budget_fraction,
                "independent_gauge_peak_difference_Pa": abs(oracle.peak_pa - mass_oracle.peak_pa),
            }
        )
    record_property(
        "independent_scikit_fem",
        json.dumps(
            {
                "shape": shape,
                "method": method,
                "analytic": analytic,
                "stiffness_relative_error": stiffness_error,
                "mass_relative_error": mass_error,
                "fields": rows,
            }
        ),
    )
