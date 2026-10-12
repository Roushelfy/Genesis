"""Unselected source-prior aggregate fields, including self-balanced full-shell compression."""

import json
import sys
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.optimize import minimize, root as solve_root
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_footprints, create_surface_fan_tractions, kernel_footprint_ports, kernel_surface_fan_profiles
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, create_traction_samples, kernel_traction_samples
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import project_coulomb
from source_prior_candidate_v1 import create_source_priors, kernel_aggregate_pair, kernel_fit_source_priors, kernel_query_prior_forces, kernel_scatter_prior_forces, kernel_snapshot_source_priors


def cpu_fit(xyz, profiles, normals, mu, identifiers, source_forces, anchors, force, moment, radius, scale):
    totals = np.array([profiles[identifiers == i].sum() for i in range(len(source_forces))])
    weights = profiles / profiles.sum()
    densities = source_forces[identifiers] * (profiles.sum() / totals[identifiers])[:, None] / scale
    relative = (xyz - anchors[0]) / radius
    operator = np.zeros((len(xyz), 6, 3))
    operator[:, :3] = np.eye(3)
    x, y, z = relative.T
    operator[:, 3, 1], operator[:, 3, 2] = -z, y
    operator[:, 4, 0], operator[:, 4, 2] = z, -x
    operator[:, 5, 0], operator[:, 5, 1] = -y, x
    target = np.concatenate((force / scale, moment / (radius * scale)))

    def evaluate(multiplier):
        values = densities - np.einsum("qij,i->qj", operator, multiplier)
        projected, jacobian = project_coulomb(values, normals, mu)
        energy = target @ multiplier + 0.5 * np.einsum("q,qi,qi->", weights, projected, projected)
        gradient = target - np.einsum("q,qij,qj->i", weights, operator, projected)
        hessian = np.einsum("q,qik,qkl,qjl->ij", weights, operator, jacobian, operator)
        return energy, gradient, hessian, weights[:, None] * scale * projected

    solved = minimize(lambda value: evaluate(value)[:2], np.zeros(6), jac=True,
                      hess=lambda value: evaluate(value)[2], method="trust-exact",
                      options={"gtol": 1e-13, "maxiter": 1000})
    refined = solve_root(lambda value: evaluate(value)[1], solved.x,
                         jac=lambda value: evaluate(value)[2], method="hybr", options={"xtol": 1e-12})
    multiplier = min((solved.x, refined.x), key=lambda value: np.linalg.norm(evaluate(value)[1]))
    assert np.linalg.norm(evaluate(multiplier)[1]) <= 1e-10
    return evaluate(multiplier)[3]


backend, output = sys.argv[1], Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
mesh_path = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
options = gs.options.RigidStressOptions(mesh=mesh_path, method="direct", contact_model="intrinsic", intrinsic_support="pair_union")
model = StressModel(options)
surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
info = create_traction_info(surface.info)
with np.load(mesh_path) as asset:
    vertices, tetrahedra, faces = (asset[name].copy() for name in ("vertices", "tetrahedra", "surface_triangles"))
oracle = P2Shell(vertices, tetrahedra, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
factor = splu(oracle.k[free][:, free].tocsc())
reports = []
for case in ("step602", "policy876", "balanced_compression"):
    if case == "step602":
        with np.load("tests/rigid/assets/nonconvex_mesh64_tick602.npz") as saved:
            original = {name: saved[name].copy() for name in saved.files}
        anchors, points, forces = (original[name] for name in ("anchor", "position", "force"))
        radii, frictions, anchor_faces = (original[name] for name in ("radius", "friction", "anchor_face"))
        moments = np.cross(points - anchors, forces)
    elif case == "policy876":
        with np.load("tests/rigid/assets/nonconvex_policy_tick876.npz") as saved:
            original = {name: saved[name].copy() for name in saved.files}
        anchors, points, forces = (original["contacts." + name][:2] for name in ("center", "position", "force"))
        radii, frictions, anchor_faces = (original["contacts." + name][:2] for name in ("radius", "friction", "anchor_face"))
        moments = original["raw_sources.anchor_moment"][:2]
    else:
        boundary_vertices = np.unique(faces)
        indices = boundary_vertices[[np.argmin(vertices[boundary_vertices, 0]), np.argmax(vertices[boundary_vertices, 0])]]
        anchors = vertices[indices].copy()
        points = anchors.copy()
        direction = (anchors[1] - anchors[0]) / np.linalg.norm(anchors[1] - anchors[0])
        forces = np.array([0.05 * direction, -0.05 * direction])
        moments = np.zeros((2, 3))
        radii, frictions = np.full(2, 0.006), np.full(2, 0.8)
        anchor_faces = np.array([np.flatnonzero((faces == index).any(axis=1))[0] for index in indices])
    n_envs = 3
    contacts, sources = create_contacts(2, n_envs, 0.006), create_sources(2, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 2, n_envs)
    tractions = create_surface_fan_tractions(surface.info, 2, n_envs, shared_support=True)
    prior = create_source_priors(2, n_envs)
    contacts.valid.fill(True)
    for name, values in [("center", anchors), ("position", points), ("force", forces), ("radius", radii), ("anchor_face", anchor_faces)]:
        contacts.__dict__[name].from_numpy(np.repeat(values[:, None], n_envs, axis=1))
    actual_mu = frictions[None] * np.array([1.0, 1.1, 1.2])[:, None]
    contacts.friction.from_numpy(actual_mu.T.copy())
    sources.geom.fill(24)
    sources.pair.fill(22)
    sources.status.fill(0)
    sources.anchor_moment.from_numpy(np.repeat(moments[:, None], n_envs, axis=1))
    sources.free_moment.from_numpy(np.repeat((moments - np.cross(points - anchors, forces))[:, None], n_envs, axis=1))
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions)
    kernel_snapshot_source_priors(contacts, sources, tractions, prior)
    kernel_aggregate_pair(contacts, sources)
    kernel_fit_source_priors(contacts, sources, tractions, info, epsilon, 9e-9, 256, 0, prior)
    status = qd_to_numpy(tractions.status, transpose=True, copy=True)
    assert (status == 0).all(), {"case": case, "status": status.tolist()}
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    query = V_VEC(3, dtype=gs.qd_float, shape=(*tractions.profile.shape,))
    kernel_query_prior_forces(query, contacts, tractions, info, prior)
    sample_positions = qd_to_numpy(samples.positions, transpose=True, copy=True)
    sample_normals = qd_to_numpy(samples.normals, transpose=True, copy=True)
    sample_basis = qd_to_numpy(samples.shape, transpose=True, copy=True)
    sample_faces = qd_to_numpy(samples.faces, transpose=True, copy=True)
    sample_forces = qd_to_numpy(query, transpose=True, copy=True)
    profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)
    counts = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    nodal_reference = np.zeros((n_envs, len(qd_to_numpy(model.info.vertices, transpose=True)), 3))
    force = forces.sum(axis=0)
    moment = moments.sum(axis=0) + np.cross(anchors - anchors[0], forces).sum(axis=0)
    scale = np.maximum(np.linalg.norm(forces, axis=1), np.linalg.norm(moments, axis=1) / radii).sum()
    force_errors, moment_errors, cone_excesses, source_force_norms, profile_totals = [], [], [], [], []
    for i_b in range(n_envs):
        clouds = [[], [], [], [], [], [], [], []]
        for i_s in range(2):
            count = counts[i_b, i_s]
            for cloud, values in zip(clouds, [sample_positions[i_b, :count, i_s], profiles[i_b, :count, i_s], sample_normals[i_b, :count, i_s], sample_basis[i_b, :count, i_s], sample_faces[i_b, :count, i_s], sample_forces[i_b, :count, i_s], np.full(count, actual_mu[i_b, i_s]), np.full(count, i_s, dtype=int)]):
                cloud.append(values)
        xyz, weights, normals, basis, destination_faces, actual_force, mu, identifiers = (np.concatenate(values) for values in clouds)
        expected_force = cpu_fit(xyz, weights, normals, mu, identifiers, forces, anchors, force, moment, radii[0], scale)
        np.testing.assert_allclose(actual_force, expected_force, rtol=1e-4, atol=1e-10)
        np.add.at(nodal_reference[i_b], surface_nodes[destination_faces].reshape(-1), (basis[:, :, None] * expected_force[:, None]).reshape(-1, 3))
        force_errors.append(float(np.linalg.norm(actual_force.sum(axis=0) - force) / scale))
        moment_errors.append(float(np.linalg.norm(np.cross(xyz - points[0], actual_force).sum(axis=0) - (moment - np.cross(points[0] - anchors[0], force))) / (radii[0] * scale)))
        compression = np.einsum("qi,qi->q", actual_force, normals)
        excess = np.linalg.norm(actual_force - compression[:, None] * normals, axis=1) - mu * compression
        assert compression.min() >= -1e-12 and excess.max() <= 1e-12
        cone_excesses.append(float(excess.max()))
        source_force_norms.append([float(np.linalg.norm(actual_force[identifiers == i].sum(axis=0))) for i in range(2)])
        profile_totals.append([float(weights[identifiers == i].sum()) for i in range(2)])
        np.savez_compressed(output / f"{case}-cloud-{i_b}.npz", positions=xyz, profiles=weights, normals=normals, basis=basis, faces=destination_faces, force=actual_force, source=identifiers, source_forces=forces, anchors=anchors, points=points, radius=radii, friction=mu)
    assert max(force_errors + moment_errors) <= 1e-8
    state = model.create_state(n_envs, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    kernel_scatter_prior_forces(query, state, tractions, info, model.info, surface.info)
    np.testing.assert_allclose(qd_to_numpy(state.force, transpose=True), nodal_reference, rtol=1e-4, atol=1e-10)
    rhs = oracle.balance(np.moveaxis(nodal_reference, 0, -1).reshape(oracle.ndof, n_envs))
    expected = np.zeros_like(rhs)
    expected[free] = factor.solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    fractions = residual / np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))
    assert fractions.max() <= 1.0
    tensor, vm = qd_to_numpy(state.stress_tensor, transpose=True, copy=True), qd_to_numpy(state.von_mises, transpose=True, copy=True)
    peaks = []
    for i_b in range(n_envs):
        expected_tensor, expected_vm = stress_field(oracle.glambda, oracle.elements, expected[:, i_b], options.young, options.poisson)
        np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(vm[i_b, :, :, 0], expected_vm, rtol=1e-4, atol=1e-3)
        peaks.append(float(expected_vm.max()))
    np.testing.assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
    if case == "balanced_compression":
        assert np.linalg.norm(force) == 0 and np.linalg.norm(moment) <= 1e-15
        assert min(min(row) for row in source_force_norms) > 0.01 and min(peaks) > 100
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.center, transpose=True), np.repeat(anchors[None], n_envs, axis=0))
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.radius, transpose=True), np.repeat(radii[None], n_envs, axis=0))
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.friction, transpose=True), actual_mu)
    np.testing.assert_array_equal(qd_to_numpy(prior.force, transpose=True), np.repeat(forces[None], n_envs, axis=0))
    report = {"case": case, "model": "explicit_source_prior_aggregate_unselected", "sample_counts": counts.tolist(), "profile_totals": profile_totals, "actual_friction": actual_mu.tolist(), "aggregate_force_N": force.tolist(), "aggregate_anchor_moment_Nm": moment.tolist(), "original_force_scale_N": scale, "recovered_source_force_norms_N": source_force_norms, "force_error": force_errors, "moment_error": moment_errors, "cone_excess_N": cone_excesses, "complete_residual_budget_fraction": fractions.tolist(), "oracle_peak_Pa": peaks, "native_peak_Pa": qd_to_numpy(state.peak, transpose=True).tolist()}
    reports.append(report)
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(report), flush=True)
