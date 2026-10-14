"""Qualify an explicitly different aggregate-pair traction model without selecting it or changing original inputs.

The conserved object is the complete pair wrench, not each rigid solver point-wrench component. Original source
footprints, material normals, radii and destination friction are retained. This is an offline modeling experiment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints, create_surface_fan_tractions, kernel_footprint_ports, kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info, create_traction_samples, kernel_fit_tractions, kernel_scatter_tractions,
    kernel_traction_samples,
)
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from tests.rigid.test_stress_shared_support import kernel_shared_traction_queries
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit


@qd.kernel
def kernel_aggregate_fixture_pair(contacts: StressContactState, sources: StressSourceState):
    for i_b in range(contacts.force.shape[1]):
        anchor = contacts.center[0, i_b]
        force = contacts.force[0, i_b] + contacts.force[1, i_b]
        moment = sources.anchor_moment[0, i_b] + sources.anchor_moment[1, i_b]
        moment += (contacts.center[1, i_b] - anchor).cross(contacts.force[1, i_b])
        # Only the isolated working target is aggregated. The original input
        # fixture and bound physical source centers/radii/mu remain unchanged.
        contacts.position[0, i_b] = anchor
        contacts.force[0, i_b], sources.anchor_moment[0, i_b] = force, moment
        sources.free_moment[0, i_b] = moment
        contacts.force[1, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sources.anchor_moment[1, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sources.free_moment[1, i_b] = qd.Vector.zero(gs.qd_float, 3)


root = Path(__file__).parent
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
for tick in (602, 876):
    if tick == 602:
        with np.load("tests/rigid/assets/nonconvex_mesh64_tick602.npz") as saved:
            original = {name: saved[name].copy() for name in saved.files}
        anchors, points, forces = (original[name] for name in ("anchor", "position", "force"))
        radii, frictions, anchor_faces = (original[name] for name in ("radius", "friction", "anchor_face"))
        moments = np.cross(points - anchors, forces)
    else:
        with np.load("tests/rigid/assets/nonconvex_policy_tick876.npz") as saved:
            original = {name: saved[name].copy() for name in saved.files}
        anchors, points, forces = (original["contacts." + name][:2] for name in ("center", "position", "force"))
        radii, frictions, anchor_faces = (original["contacts." + name][:2] for name in ("radius", "friction", "anchor_face"))
        moments = original["raw_sources.anchor_moment"][:2]
    n_envs = 3
    contacts, sources = create_contacts(2, n_envs, 0.006), create_sources(2, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 2, n_envs)
    tractions = create_surface_fan_tractions(surface.info, 2, n_envs, shared_support=True)
    contacts.valid.fill(True)
    for name, values in [("center", anchors), ("position", points), ("force", forces), ("radius", radii), ("anchor_face", anchor_faces)]:
        contacts.__dict__[name].from_numpy(np.repeat(values[:, None], n_envs, axis=1))
    actual_mu = frictions[None] * np.array([1.0, 1.1, 1.2])[:, None]
    contacts.friction.from_numpy(actual_mu.T.copy())
    sources.geom.fill(24)
    sources.pair.fill(22)
    sources.status.fill(0)
    sources.anchor_moment.from_numpy(np.repeat(moments[:, None], n_envs, axis=1))
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions)
    kernel_aggregate_fixture_pair(contacts, sources)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    status = qd_to_numpy(tractions.status, transpose=True, copy=True)
    assert (status == 0).all(), {"tick": tick, "status": status.tolist()}
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    query = V_VEC(3, dtype=gs.qd_float, shape=(tractions.profile.shape[0], 2, 2, n_envs))
    kernel_shared_traction_queries(query, contacts, tractions, info)
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
    force_errors, moment_errors, cone_excesses = [], [], []
    for i_b in range(n_envs):
        clouds = [[], [], [], [], [], [], []]
        for i_s in range(2):
            count = counts[i_b, i_s]
            for cloud, values in zip(clouds, [sample_positions[i_b, :count, i_s], profiles[i_b, :count, i_s], sample_normals[i_b, :count, i_s], sample_basis[i_b, :count, i_s], sample_faces[i_b, :count, i_s], sample_forces[i_b, :count, i_s, 0], np.full(count, actual_mu[i_b, i_s])]):
                cloud.append(values)
        xyz, weights, normals, basis, destination_faces, actual_force, mu = (np.concatenate(values) for values in clouds)
        patch = WrenchPatch(anchors[0], force, radii[0], mu, moment)
        expected_force = cpu_resolved_wrench_fit(xyz, weights, normals, patch)
        np.testing.assert_allclose(actual_force, expected_force, rtol=1e-4, atol=1e-10)
        np.add.at(nodal_reference[i_b], surface_nodes[destination_faces].reshape(-1), (basis[:, :, None] * expected_force[:, None]).reshape(-1, 3))
        force_errors.append(float(np.linalg.norm(actual_force.sum(axis=0) - force) / np.linalg.norm(force)))
        moment_errors.append(float(np.linalg.norm(np.cross(xyz - anchors[0], actual_force).sum(axis=0) - moment) / (radii[0] * np.linalg.norm(force))))
        compression = np.einsum("qi,qi->q", actual_force, normals)
        excess = np.linalg.norm(actual_force - compression[:, None] * normals, axis=1) - mu * compression
        assert compression.min() >= -1e-12 and excess.max() <= 1e-12
        cone_excesses.append(float(excess.max()))
    assert max(force_errors + moment_errors) <= 1e-8
    state = model.create_state(n_envs, "full")
    errno, omega = V(dtype=gs.qd_int, shape=(n_envs,)), V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert not qd_to_numpy(errno, transpose=True).any()
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
    tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
    vm = qd_to_numpy(state.von_mises, transpose=True, copy=True)
    peaks = []
    for i_b in range(n_envs):
        expected_tensor, expected_vm = stress_field(oracle.glambda, oracle.elements, expected[:, i_b], options.young, options.poisson)
        np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(vm[i_b, :, :, 0], expected_vm, rtol=1e-4, atol=1e-3)
        peaks.append(float(expected_vm.max()))
    np.testing.assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.center, transpose=True), np.repeat(anchors[None], n_envs, axis=0))
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.radius, transpose=True), np.repeat(radii[None], n_envs, axis=0))
    np.testing.assert_array_equal(qd_to_numpy(tractions.support.friction, transpose=True), actual_mu)
    report = {"tick": tick, "model": "explicit_pair_aggregate_experiment_unselected", "integration": "unmodified_native_surface_fan", "sample_counts": counts.tolist(), "actual_friction": actual_mu.tolist(), "force_error": force_errors, "moment_error": moment_errors, "cone_excess_N": cone_excesses, "complete_residual_budget_fraction": fractions.tolist(), "oracle_peak_Pa": peaks, "native_peak_Pa": qd_to_numpy(state.peak, transpose=True).tolist()}
    reports.append(report)
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(report), flush=True)
