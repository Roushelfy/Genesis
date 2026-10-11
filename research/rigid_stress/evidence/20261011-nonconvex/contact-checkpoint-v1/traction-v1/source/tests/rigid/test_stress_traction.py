import numpy as np
import pytest
import quadrants as qd
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    create_tractions,
    func_coulomb_projection,
    func_sample_traction,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_cone_projection(values, normals, friction):
    """Vectorized FP64 reference for manufactured primal/dual KKT solutions."""
    compression = (values * normals).sum(axis=-1)
    tangent = values - compression[..., None] * normals
    length = np.linalg.norm(tangent, axis=-1)
    inside = (compression >= 0) & (length <= friction * compression)
    boundary = ~inside & (compression + friction * length > 0)
    pressure = (compression + friction * length) / (1 + friction**2)
    result = np.where(inside[..., None], values, 0.0)
    direction = tangent / np.maximum(length[..., None], np.finfo(np.float64).tiny)
    result += np.where(boundary[..., None], pressure[..., None] * (normals + friction[..., None] * direction), 0.0)
    return result


@qd.kernel
def kernel_cone_queries(
    values: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    friction: qd.types.ndarray(),
    projections: qd.Tensor,
    jacobians: qd.Tensor,
):
    for i in range(values.shape[0]):
        value = qd.Vector([values[i, a] for a in qd.static(range(3))])
        normal = qd.Vector([normals[i, a] for a in qd.static(range(3))])
        projections[i], jacobians[i] = func_coulomb_projection(value, normal, friction[i])


@qd.kernel
def kernel_traction_queries(
    forces: qd.Tensor,
    contacts: StressContactState,
    tractions: StressTractionState,
    info: StressTractionInfo,
):
    for i_q, i_c, i_b in qd.ndrange(info.positions.shape[0], contacts.valid.shape[0], contacts.valid.shape[1]):
        forces[i_q, i_c, i_b] = func_sample_traction(i_q, i_c, i_b, contacts, tractions, info)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_coulomb_projection_and_jacobian():
    rng = np.random.default_rng(20261011)
    values = rng.normal(size=(30, 3))
    normals = rng.normal(size=(30, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    friction = np.tile([0.0, 0.25, 0.55, 0.85, 1.2], 6)
    projections = V_VEC(3, dtype=gs.qd_float, shape=(30,))
    jacobians = V_MAT(3, 3, dtype=gs.qd_float, shape=(30,))
    kernel_cone_queries(values, normals, friction, projections, jacobians)
    projected = qd_to_numpy(projections, transpose=True)
    derivative = qd_to_numpy(jacobians, transpose=True)
    reference = cpu_cone_projection(values, normals, friction)
    assert_allclose(projected, reference, rtol=0.0, atol=2e-15)
    # Independently minimize the three-dimensional constrained objective on rotated cones.
    for value, normal, coefficient, actual in zip(values, normals, friction, projected):
        if coefficient > 0 and np.linalg.norm(actual) > 0:
            reference_opt = minimize(
                lambda force, value=value: 0.5 * np.dot(force - value, force - value),
                actual + 0.01 * normal,
                jac=lambda force, value=value: force - value,
                constraints={
                    "type": "ineq",
                    "fun": lambda force, coefficient=coefficient, normal=normal: (
                        coefficient * np.dot(force, normal) - np.linalg.norm(force - np.dot(force, normal) * normal)
                    ),
                },
                method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 1000},
            )
            assert reference_opt.success, reference_opt.message
            assert_allclose(actual, reference_opt.x, rtol=0.0, atol=2e-7)
    step = 1e-6
    for axis in range(3):
        perturbation = np.eye(3)[axis] * step
        difference = (
            cpu_cone_projection(values + perturbation, normals, friction)
            - cpu_cone_projection(values - perturbation, normals, friction)
        ) / (2 * step)
        assert_allclose(derivative[:, :, axis], difference, rtol=1e-8, atol=2e-9)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_vector_tractions_signed_scatter_and_elasticity(nonconvex_material_mesh, asset_tmp_path):  # noqa: F811
    vertices, tetrahedra, triangles, shape = nonconvex_material_mesh
    mesh = asset_tmp_path / f"traction-{shape}.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles)
    options = RigidStressOptions(mesh=mesh, method="direct")
    model = StressModel(options)
    sampled = StressSurface(4, model.info)
    positions = qd_to_numpy(sampled.info.positions, transpose=True)
    face_normals = qd_to_numpy(sampled.info.face_normal, transpose=True)
    n_quadrature = sampled.info.shape.shape[0]
    normals = -np.repeat(face_normals, n_quadrature, axis=0)
    anchor = np.array([0.001, 0.001, 0.0005 if shape == "u" else 0.001])
    radius = 0.00045 if shape == "u" else 0.00075
    relative = (positions - anchor) / radius
    distance = np.linalg.norm(relative, axis=1)
    # At this actual edge/corner, each eligible sample is reachable from the anchor
    # by a segment on its incident planar wall. Euclidean and intrinsic distance agree.
    profile = qd_to_numpy(sampled.info.weights, transpose=True) * np.exp(-0.5 * (distance / 0.45) ** 2)
    profile *= np.maximum(1.0 - distance**2, 0.0) ** 2
    selected = profile > 0
    assert_equal(np.unique(normals[selected], axis=0).shape[0], 2 if shape == "u" else 3)
    info = StressTractionInfo(
        positions=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
        inward_normals=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
    )
    info.positions.from_numpy(positions)
    info.inward_normals.from_numpy(normals)
    friction = np.array([0.0, 0.25, 0.55, 0.85, 1.2])
    n_envs = len(friction)
    rng = np.random.default_rng(20261011 + (shape == "cavity"))
    profiles = profile[:, None] * (1.0 + 0.05 * rng.uniform(-1, 1, size=(len(profile), n_envs)))
    weights = profiles / profiles.sum(axis=0)
    initial_force = np.array([-0.001, -0.0012, -0.0009])
    dual_force = rng.uniform(-1e-4, 1e-4, size=(n_envs, 3))
    dual_moment = rng.uniform(-1.5e-4, 1.5e-4, size=(n_envs, 3))
    values = initial_force - dual_force[None] - np.cross(dual_moment[None], relative[:, None])
    reference = weights[..., None] * cpu_cone_projection(values, normals[:, None], friction[None])
    force = reference.sum(axis=0)
    moment = np.cross(positions[:, None] - anchor, reference).sum(axis=0)
    # Adjusting the force dual by F-initial_force leaves every projection argument
    # unchanged, proving that reference is the unique convex subproblem optimum.
    dual_force += force - initial_force
    points = anchor + rng.uniform(-2e-5, 2e-5, size=(n_envs, 3))
    free_moment = moment - np.cross(points - anchor, force)
    contacts = create_contacts(1, n_envs, radius)
    sources = create_sources(1, n_envs)
    tractions = create_tractions(len(positions), 1, n_envs)
    contacts.center.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    contacts.position.from_numpy(points[None])
    contacts.force.from_numpy(force[None])
    contacts.friction.from_numpy(friction[None])
    contacts.valid.fill(True)
    sources.anchor_moment.from_numpy(moment[None])
    sources.free_moment.from_numpy(free_moment[None])
    tractions.profile.from_numpy(profiles[:, None])
    kernel_fit_tractions(contacts, sources, tractions, info, float(np.finfo(gs.np_float).eps), 1e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), StressTractionStatus.SUCCESS)
    assert_equal(qd_to_numpy(contacts.force, transpose=True)[:, 0], force)
    assert_equal(qd_to_numpy(sources.anchor_moment, transpose=True)[:, 0], moment)
    quadrature_force = V_VEC(3, dtype=gs.qd_float, shape=(len(positions), 1, n_envs))
    kernel_traction_queries(quadrature_force, contacts, tractions, info)
    actual = qd_to_numpy(quadrature_force, transpose=True)[:, :, 0].transpose((1, 0, 2))
    assert_allclose(actual, reference, rtol=1e-4, atol=1e-12)
    compression = (actual * normals[:, None]).sum(axis=2)
    tangent = actual - compression[..., None] * normals[:, None]
    assert (compression >= -1e-15).all()
    assert (np.linalg.norm(tangent, axis=2) <= friction[None] * compression + 1e-15).all()
    states = (model.create_state(n_envs), model.create_state(n_envs, "full"))
    errno = V(dtype=gs.qd_int, shape=(n_envs,))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T[np.arange(len(positions)) // n_quadrature]
    basis = qd_to_numpy(sampled.info.shape, transpose=True)[np.arange(len(positions)) % n_quadrature]
    expected_force = np.zeros((model.info.vertices.shape[0], n_envs, 3))
    np.add.at(expected_force, nodes.reshape(-1), (basis[:, :, None, None] * reference[:, None]).reshape(-1, n_envs, 3))
    oracle = P2Shell(
        vertices, tetrahedra, triangles, options.young, options.poisson, options.density, 1, factor_backend="none"
    )
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    rhs = oracle.balance(np.moveaxis(expected_force, 1, -1).reshape(oracle.ndof, n_envs))
    reference_displacement = np.zeros_like(rhs)
    reference_displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    vertex_positions = qd_to_numpy(model.info.vertices, transpose=True)
    for state in states:
        kernel_scatter_tractions(contacts, tractions, state, model.info, sampled.info, info, errno)
        assert_equal(qd_to_numpy(errno, transpose=True), 0)
        nodal_force = qd_to_numpy(state.force, transpose=True)
        assert_allclose(nodal_force, np.moveaxis(expected_force, 1, 0), rtol=1e-4, atol=1e-12)
        force_error = np.linalg.norm(nodal_force.sum(axis=1) - force, axis=1) / np.linalg.norm(force, axis=1)
        recovered_moment = np.cross(vertex_positions[None] - points[:, None], nodal_force).sum(axis=1)
        moment_error = np.linalg.norm(recovered_moment - free_moment, axis=1) / (radius * np.linalg.norm(force, axis=1))
        assert (force_error <= 1e-8).all() and (moment_error <= 1e-8).all()
        model.recover(omega, state)
        assert qd_to_numpy(state.valid, transpose=True).all()
        displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
        residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
        budget = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))
        assert (residual <= budget).all()
    complete = states[1]
    actual_tensor = qd_to_numpy(complete.stress_tensor, transpose=True)
    actual_vm = qd_to_numpy(complete.von_mises, transpose=True)[..., 0]
    expected_peaks = np.empty(n_envs)
    for i_b in range(n_envs):
        tensor, vm = stress_field(
            oracle.glambda, oracle.elements, reference_displacement[:, i_b], options.young, options.poisson
        )
        assert_allclose(actual_tensor[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(actual_vm[i_b], vm, rtol=1e-4, atol=1e-3)
        expected_peaks[i_b] = vm.max()
    assert_allclose(qd_to_numpy(complete.peak, transpose=True), expected_peaks, rtol=1e-4, atol=1e-3)
    assert_equal(actual_vm.max(axis=(1, 2)), qd_to_numpy(complete.peak, transpose=True))
    assert_allclose(
        qd_to_numpy(states[0].peak, transpose=True), qd_to_numpy(complete.peak, transpose=True), rtol=1e-4, atol=1e-3
    )
