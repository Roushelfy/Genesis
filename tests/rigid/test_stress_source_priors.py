from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize, root
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress import cooperative_fit
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.fan_refinement import kernel_refine_failed_fans, kernel_select_failed_fans
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts, kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step, kernel_reset
from genesis.engine.solvers.rigid.stress.source_priors import (
    kernel_partition_source_priors,
    kernel_refresh_refined_source_priors,
    kernel_snapshot_source_priors,
)
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_traction_samples,
    kernel_fit_tractions,
    kernel_prepare_cached_tractions,
    kernel_scatter_tractions,
    kernel_traction_samples,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, ErrorCode, V, iter_data
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import project_coulomb
from tests.rigid.test_stress_shared_support import kernel_shared_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_source_prior_fit(
    positions, profiles, normals, friction, source, forces, anchor, force, moment, radius, scale, initial=None
):
    """Independent FP64 convex dual, without native multipliers or recovered forces."""
    totals = np.array([profiles[source == i].sum() for i in range(len(forces))])
    weights = profiles / profiles.sum()
    density = forces[source] * (profiles.sum() / totals[source] / scale)[:, None]
    relative = (positions - anchor) / radius
    operator = np.zeros((len(positions), 6, 3))
    operator[:, :3] = np.eye(3)
    x, y, z = relative.T
    operator[:, 3, 1], operator[:, 3, 2] = -z, y
    operator[:, 4, 0], operator[:, 4, 2] = z, -x
    operator[:, 5, 0], operator[:, 5, 1] = -y, x
    target = np.concatenate((force / scale, moment / (radius * scale)))

    def evaluate(multiplier):
        value = density - np.einsum("qij,i->qj", operator, multiplier)
        projected, jacobian = project_coulomb(value, normals, friction)
        energy = target @ multiplier + 0.5 * np.einsum("q,qi,qi->", weights, projected, projected)
        gradient = target - np.einsum("q,qij,qj->i", weights, operator, projected)
        hessian = np.einsum("q,qik,qkl,qjl->ij", weights, operator, jacobian, operator)
        return energy, gradient, hessian, weights[:, None] * scale * projected

    fitted = minimize(
        lambda value: evaluate(value)[:2],
        np.zeros(6),
        jac=True,
        hess=lambda value: evaluate(value)[2],
        method="trust-exact",
        options={"gtol": 1e-13, "maxiter": 1000},
    )
    corrected = root(
        lambda value: evaluate(value)[1], fitted.x, jac=lambda value: evaluate(value)[2], options={"xtol": 1e-12}
    )
    multiplier = min((fitted.x, corrected.x), key=lambda value: np.linalg.norm(evaluate(value)[1]))
    if initial is not None and np.linalg.norm(evaluate(multiplier)[1]) > 1e-10:
        # This independently published SOCP/root initializer carries no native multipliers or forces.
        root_scale = max(1.0, np.linalg.norm(initial))
        polished = root(
            lambda value: evaluate(value * root_scale)[1],
            initial / root_scale,
            jac=lambda value: evaluate(value * root_scale)[2] * root_scale,
            options={"xtol": 1e-12, "maxfev": 2000},
        )
        multiplier = min((multiplier, polished.x * root_scale), key=lambda value: np.linalg.norm(evaluate(value)[1]))
    assert np.linalg.norm(evaluate(multiplier)[1]) <= 1e-10
    return evaluate(multiplier)[3]


def source_prior_fixture(case, vertices, faces):
    assets = Path(__file__).with_name("assets")
    if case == "roundoff_edge":
        with np.load(assets / "nonconvex_shared_edge_roundoff.npz") as saved:
            anchors, points, forces, radii, friction, anchor_faces = (
                saved[key][0].copy() for key in ("witness", "point", "force", "radius", "friction", "old_anchor_face")
            )
        return anchors, points, forces, radii, friction, anchor_faces, np.cross(points - anchors, forces)
    if case in ("varied2048", "varied4096", "varied4096_slow"):
        asset = f"nonconvex_{case}_single_source.npz"
        if case == "varied4096_slow":
            asset = "nonconvex_varied4096_slow_fan.npz"
        with np.load(assets / asset) as saved:
            return tuple(
                saved[key].copy() for key in ("anchor", "point", "force", "radius", "friction", "anchor_face", "moment")
            )
    if case == "balanced_compression":
        boundary = np.unique(faces)
        indices = boundary[[np.argmin(vertices[boundary, 0]), np.argmax(vertices[boundary, 0])]]
        anchors = vertices[indices].copy()
        direction = (anchors[1] - anchors[0]) / np.linalg.norm(anchors[1] - anchors[0])
        forces = np.array([0.05 * direction, -0.05 * direction])
        return (
            anchors,
            anchors.copy(),
            forces,
            np.full(2, 0.006),
            np.full(2, 0.8),
            np.array([np.flatnonzero((faces == index).any(axis=1))[0] for index in indices]),
            np.zeros((2, 3)),
        )
    name = {
        "step602": "nonconvex_mesh64_tick602",
        "policy876": "nonconvex_policy_tick876",
        "policy807": "nonconvex_policy_tick807",
    }[case]
    with np.load(assets / f"{name}.npz") as saved:
        original = {key: saved[key].copy() for key in saved.files}
    if case == "step602":
        anchors, points, forces = (original[key] for key in ("anchor", "position", "force"))
        radius, friction, anchor_face = (original[key] for key in ("radius", "friction", "anchor_face"))
        moment = np.cross(points - anchors, forces)
    else:
        selection = slice(None, 2) if case == "policy876" else slice(None)
        anchors, points, forces = (original[f"contacts.{key}"][selection] for key in ("center", "position", "force"))
        radius, friction, anchor_face = (
            original[f"contacts.{key}"][selection] for key in ("radius", "friction", "anchor_face")
        )
        moment = original["raw_sources.anchor_moment"][selection]
    return anchors, points, forces, radius, friction, anchor_face, moment


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize(
    "case",
    (
        "step602",
        "policy876",
        "policy807",
        "balanced_compression",
        "varied2048",
        "varied4096",
        "roundoff_edge",
        "varied4096_slow",
    ),
)
def test_body_source_prior_wrench_cones_full_fields_and_gauge_rows(case, record_property):
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = RigidStressOptions(mesh=mesh, method="direct", intrinsic_load_partition="body_source_prior")
    model, n_envs = StressModel(options), 3
    with np.load(mesh) as asset:
        vertices, tets, faces = (asset[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    anchors, points, forces, radii, friction, anchor_faces, moments = source_prior_fixture(case, vertices, faces)
    n_sources = len(forces)
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    info = create_traction_info(surface.info)
    contacts, sources = create_contacts(n_sources, n_envs, 0.006), create_sources(n_sources, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], n_sources, n_envs, paths.info.face_ports.shape[1])
    tractions = create_surface_fan_tractions(
        surface.info, n_sources, n_envs, shared_support=True, source_priors=True, cached_acceptance=True
    )
    for name, values in (
        ("center", anchors),
        ("position", points),
        ("force", forces),
        ("radius", radii),
        ("anchor_face", anchor_faces),
    ):
        getattr(contacts, name).from_numpy(np.repeat(values[:, None], n_envs, axis=1))
    actual_mu = np.array([1.0, 1.1, 1.2])[:, None] * friction
    contacts.friction.from_numpy(actual_mu.T.copy())
    contacts.valid.fill(True)
    sources.geom.from_numpy(np.repeat(np.arange(n_sources)[:, None], n_envs, axis=1).astype(gs.np_int))
    sources.pair.from_numpy(np.repeat((100 + np.arange(n_sources))[:, None], n_envs, axis=1).astype(gs.np_int))
    sources.status.fill(0)
    sources.anchor_moment.from_numpy(np.repeat(moments[:, None], n_envs, axis=1))
    sources.free_moment.from_numpy(np.repeat((moments - np.cross(points - anchors, forces))[:, None], n_envs, axis=1))
    epsilon = float(np.finfo(gs.np_float).eps)
    if case == "roundoff_edge":
        with np.load(Path(__file__).with_name("assets") / "nonconvex_shared_edge_roundoff.npz") as saved:
            contacts.frame_position.from_numpy(np.repeat(saved["frame_position"][0, None], n_envs, axis=0))
            sources.uncertainty.from_numpy(np.repeat(saved["uncertainty"][0, :, None], n_envs, axis=1))
        sources.witness.from_numpy(np.repeat(anchors[:, None], n_envs, axis=1))
        kernel_resolve_sources(1e-5, epsilon, contacts, sources, model.info, surface.info)
        assert_equal(qd_to_numpy(sources.status), 0)
        anchors = qd_to_numpy(contacts.center, transpose=True, copy=True)[0]
        moments = qd_to_numpy(sources.anchor_moment, transpose=True, copy=True)[0]
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions, body_union=True)
    kernel_snapshot_source_priors(contacts, sources, tractions)
    kernel_partition_source_priors(contacts, sources, tractions)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    if case in ("varied2048", "varied4096"):
        assert qd_to_numpy(tractions.status, transpose=True)[0, 0] in (3, 4)
    kernel_select_failed_fans(contacts, footprints, tractions)
    kernel_refine_failed_fans(
        contacts, sources, footprints, tractions, model.info, surface.info, paths.info, info, epsilon
    )
    kernel_refresh_refined_source_priors(contacts, tractions)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=3)
    if case == "varied4096_slow":
        assert qd_to_numpy(tractions.status, transpose=True)[0, 0] == 3
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=4)
    if case in ("varied2048", "varied4096"):
        assert qd_to_numpy(tractions.fan_refined, transpose=True)[0, 0]
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    fitted = (
        tractions.dual,
        tractions.weight_sum,
        tractions.scale,
        tractions.status,
        tractions.force_error,
        tractions.moment_error,
        tractions.evaluations,
        tractions.certificate,
        tractions.certificate_margin,
    )
    snapshots = tuple(qd_to_numpy(value, transpose=True, copy=True) for value in fitted)
    cooperative_fit.kernel_fit_tractions(
        contacts, sources, tractions, info, epsilon, 9e-9, 128, warm_start=False, is_retry=True
    )
    for value, expected in zip(fitted, snapshots):
        assert_equal(qd_to_numpy(value, transpose=True), expected)
    assert_equal(qd_to_numpy(contacts.valid, transpose=True), True)
    assert_equal(qd_to_numpy(contacts.position, transpose=True), np.repeat(points[None], n_envs, axis=0))
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    queried = V_VEC(3, dtype=gs.qd_float, shape=(tractions.profile.shape[0], n_sources, 1, n_envs))
    kernel_shared_traction_queries(queried, contacts, tractions, info)
    arrays = [
        qd_to_numpy(value, transpose=True, copy=True)
        for value in (
            samples.positions,
            tractions.profile,
            samples.normals,
            samples.shape,
            samples.faces,
            queried,
        )
    ]
    counts = qd_to_numpy(tractions.sample_count, transpose=True)
    record_property("fan_refined", qd_to_numpy(tractions.fan_refined, transpose=True).tolist())
    record_property("fit_evaluations", qd_to_numpy(tractions.evaluations, transpose=True).tolist())
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    nodal = np.zeros((n_envs, model.info.vertices.shape[0], 3))
    force = forces.sum(axis=0)
    moment = moments.sum(axis=0) + np.cross(anchors - anchors[0], forces).sum(axis=0)
    scale = max(
        np.maximum(np.linalg.norm(forces, axis=1), np.linalg.norm(moments, axis=1) / radii).sum(),
        np.linalg.norm(force),
        np.linalg.norm(moment) / radii[0],
    )
    budget_scale = np.linalg.norm(force) or scale
    force_errors, moment_errors = [], []
    for i_b in range(n_envs):
        clouds = [[], [], [], [], [], [], [], []]
        for i_s in range(n_sources):
            count = counts[i_b, i_s]
            values = [array[i_b, :count, i_s] for array in arrays]
            values[5] = values[5][:, 0]
            values.extend((np.full(count, actual_mu[i_b, i_s]), np.full(count, i_s, dtype=int)))
            for cloud, value in zip(clouds, values):
                cloud.append(value)
        xyz, profiles, normals, basis, sample_faces, actual_force, mu, source = map(np.concatenate, clouds)
        initial = None
        if case == "varied4096_slow":
            with np.load(Path(__file__).with_name("assets") / "nonconvex_varied4096_slow_fan.npz") as saved:
                initial = saved["oracle_dual"]
        expected = cpu_source_prior_fit(
            xyz, profiles, normals, mu, source, forces, anchors[0], force, moment, radii[0], scale, initial
        )
        assert_allclose(actual_force, expected, rtol=1e-4, atol=1e-10)
        np.add.at(
            nodal[i_b], surface_nodes[sample_faces].reshape(-1), (basis[:, :, None] * expected[:, None]).reshape(-1, 3)
        )
        assert np.linalg.norm(actual_force.sum(axis=0) - force) <= 1e-8 * budget_scale
        force_errors.append(float(np.linalg.norm(actual_force.sum(axis=0) - force) / budget_scale))
        assert (
            np.linalg.norm(
                np.cross(xyz - points[0], actual_force).sum(axis=0) - (moment - np.cross(points[0] - anchors[0], force))
            )
            <= 1e-8 * budget_scale * radii[0]
        )
        compression = np.einsum("qi,qi->q", actual_force, normals)
        moment_errors.append(
            float(
                np.linalg.norm(
                    np.cross(xyz - points[0], actual_force).sum(axis=0)
                    - (moment - np.cross(points[0] - anchors[0], force))
                )
                / (budget_scale * radii[0])
            )
        )
        tangent = np.linalg.norm(actual_force - compression[:, None] * normals, axis=1)
        assert compression.min() >= -64 * epsilon * scale
        assert np.max(tangent - mu * compression) <= 64 * epsilon * scale
        if case == "balanced_compression":
            assert_equal(force, 0)
            assert_equal(moment, 0)
            assert_allclose([np.linalg.norm(actual_force[source == i].sum(axis=0)) for i in range(2)], 0.05, atol=1e-9)
    state = model.create_state(n_envs, "full")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)), V(dtype=gs.qd_int, shape=(n_envs,))
    omega.fill(0)
    errno.fill(0)
    kernel_prepare_cached_tractions(
        epsilon, contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True
    )
    assert_equal(qd_to_numpy(errno), 0)
    assert_allclose(qd_to_numpy(state.force, transpose=True), nodal, rtol=1e-4, atol=1e-10)
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    rhs = oracle.balance(np.moveaxis(nodal, 0, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    expected = np.zeros_like(rhs)
    expected[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    assert np.all(residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0)))
    tensor = qd_to_numpy(state.stress_tensor, transpose=True)
    vm = qd_to_numpy(state.von_mises, transpose=True)
    peak = qd_to_numpy(state.peak, transpose=True)
    peak_differences = []
    for i_b in range(n_envs):
        expected_tensor, expected_vm = stress_field(
            oracle.glambda, oracle.elements, expected[:, i_b], options.young, options.poisson
        )
        assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(vm[i_b, :, :, 0], expected_vm, rtol=1e-4, atol=1e-3)
        assert_allclose(peak[i_b], expected_vm.max(), rtol=1e-4, atol=1e-3)
        peak_differences.append(float(abs(peak[i_b] - expected_vm.max())))
    if case == "balanced_compression":
        assert peak.min() > 10000
    record_property("actual_friction", actual_mu.tolist())
    record_property("force_errors", force_errors)
    record_property("moment_errors", moment_errors)
    record_property(
        "complete_residual_budget_fraction",
        (residual / np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).tolist(),
    )
    record_property("native_peak_Pa", peak.tolist())
    record_property("peak_difference_Pa", peak_differences)


@pytest.mark.required
@pytest.mark.precision("64")
def test_body_source_prior_dynamic_workspace_release_reset_and_invalid_member():
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    with np.load(Path(__file__).with_name("assets") / "nonconvex_plane_pair_tick710.npz") as saved:
        points, anchors, forces = (saved[key] for key in ("position", "anchor", "force"))
        radii, mu = saved["radius"], saved["friction"]
    options = RigidStressOptions(
        mesh=assets / "elastic.npz",
        contact_model="intrinsic",
        intrinsic_load_partition="body_source_prior",
        contact_block_size=2,
        method="direct",
    )
    model, n_envs = StressModel(options), 3
    surface = StressSurface(10, model.info)
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=assets / "egg_shell.urdf", align=False, convexify=True, decimate=False, preserve_mesh_precision=True
        )
    )
    scene.build(n_envs=n_envs)
    raw = create_contacts(12, n_envs, 0.006)
    original = np.repeat(np.arange(2), 6)
    source_forces = np.broadcast_to(forces[original, None] / 6, (12, n_envs, 3)).copy()
    raw.position.from_numpy(np.broadcast_to(points[original, None], (12, n_envs, 3)).copy())
    raw.force.from_numpy(source_forces)
    raw.radius.from_numpy(np.broadcast_to(radii[original, None], (12, n_envs)).copy())
    raw.friction.from_numpy(np.broadcast_to(mu[original, None], (12, n_envs)).copy())
    streamed = IntrinsicContacts(entity.base_link, raw, model.info, surface.info, options, None)
    assert streamed.contacts.valid.shape[0] == 2
    streamed.raw_sources.geom.fill(entity.base_link.geoms[0].idx)
    streamed.raw_sources.pair.from_numpy(np.repeat(np.arange(12)[:, None], n_envs, axis=1).astype(gs.np_int))
    witnesses = np.broadcast_to(anchors[original, None], (12, n_envs, 3)).copy()
    streamed.raw_sources.witness.from_numpy(witnesses)
    state = model.create_state(n_envs, "full")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)), V(dtype=gs.qd_int, shape=(n_envs,))
    omega.fill(0)
    epsilon = float(np.finfo(gs.np_float).eps)
    initial = np.ones((12, n_envs), dtype=gs.np_bool)
    reference = None
    for active in (initial, initial & [False, True, True], initial):
        raw.valid.from_numpy(active)
        errno.fill(0)
        kernel_begin_step(state)
        streamed.scatter(raw, state, model.info, surface.info, options, epsilon, errno)
        assert streamed.contacts.valid.shape[0] == 12
        assert any(item.name.endswith("tractions.priors.force") for item in streamed.data("stress"))
        assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
        assert_equal(qd_to_numpy(errno), 0)
        assert_equal(qd_to_numpy(raw.force, transpose=True), np.moveaxis(source_forces, 0, 1))
        model.recover(omega, state)
        kernel_accept(state, raw, errno)
        assert qd_to_numpy(state.step_valid, transpose=True).all()
        snapshot = [
            qd_to_numpy(value, transpose=True, copy=True)
            for value in (state.force, state.stress_tensor, state.von_mises, state.peak)
        ]
        if reference is None:
            reference = snapshot
        for actual, expected in zip(snapshot, reference):
            if active[:, 0].any():
                assert_allclose(actual, expected, rtol=1e-4, atol=1e-10 if actual is snapshot[0] else 1e-3)
            else:
                assert_equal(actual[0], 0)
                assert_allclose(actual[1:], expected[1:], rtol=1e-12, atol=1e-15 if actual is snapshot[0] else 1e-8)
    kernel_reset(np.array([0], dtype=gs.np_int), state)
    assert_equal(qd_to_numpy(state.peak)[0], 0)
    assert np.isnan(qd_to_numpy(state.stress_tensor, transpose=True)[0]).all()
    assert_equal(qd_to_numpy(state.stress_tensor, transpose=True)[1:], snapshot[1][1:])
    # A contact can disappear independently of its collision pair, and the
    # packed group must choose its new leader without reusing a stale prior.
    changed = initial.copy()
    changed[0, 0] = False
    raw.valid.from_numpy(changed)
    errno.fill(0)
    kernel_begin_step(state)
    streamed.scatter(raw, state, model.info, surface.info, options, epsilon, errno)
    model.recover(omega, state)
    kernel_accept(state, raw, errno)
    assert_equal(qd_to_numpy(errno), 0)
    assert_equal(qd_to_numpy(streamed.packing.count), [11, 12, 12])
    assert qd_to_numpy(state.step_valid).all()
    assert_allclose(qd_to_numpy(state.peak)[1:], reference[3][1:], rtol=1e-12, atol=1e-8)
    raw.valid.from_numpy(initial)
    # A local invalid witness invalidates its body group and preserves that
    # member's specific diagnostic instead of the leader's generic fit error.
    witnesses[6, 0] = [0, 0, 1]
    streamed.raw_sources.witness.from_numpy(witnesses)
    errno.fill(0)
    kernel_begin_step(state)
    streamed.scatter(raw, state, model.info, surface.info, options, epsilon, errno)
    assert qd_to_numpy(errno)[0] & ErrorCode.INVALID_STRESS_LOAD
    assert_equal(qd_to_numpy(errno)[1:], 0)
    assert qd_to_numpy(raw.status, transpose=True)[0, :].all()
    assert qd_to_numpy(streamed.failure.contact)[0] == 6
    assert qd_to_numpy(streamed.failure.status)[0] == 13
    assert qd_to_numpy(streamed.failure.group_count)[0] == 12
    assert_allclose(qd_to_numpy(streamed.failure.target_force)[0], source_forces[:, 0].sum(axis=0), atol=1e-15)
    witnesses[6, 0] = anchors[original[6]]
    streamed.raw_sources.witness.from_numpy(witnesses)
    errno.fill(0)
    kernel_begin_step(state)
    streamed.scatter(raw, state, model.info, surface.info, options, epsilon, errno)
    model.recover(omega, state)
    kernel_accept(state, raw, errno)
    assert_equal(qd_to_numpy(streamed.failure.status), 0)
    assert_allclose(qd_to_numpy(state.peak), reference[3], rtol=1e-4, atol=1e-3)


@pytest.mark.required
@pytest.mark.precision("64")
def test_body_source_prior_reports_incompatible_continuous_cones():
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    options = RigidStressOptions(
        mesh=assets / "elastic.npz",
        contact_model="intrinsic",
        intrinsic_load_partition="body_source_prior",
        contact_block_size=2,
        method="direct",
    )
    model, n_envs = StressModel(options), 2
    surface = StressSurface(10, model.info)
    triangles = qd_to_numpy(surface.info.face_origin, transpose=True)
    edges = qd_to_numpy(surface.info.face_edges, transpose=True)
    # A tiny verified disk stays in a single true face; its outward force
    # has no admissible inward, zero-friction traction representation.
    anchor = triangles[0] + (edges[0, 0] + edges[0, 1]) / 3
    outward = qd_to_numpy(surface.info.face_normal, transpose=True)[0]
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=assets / "egg_shell.urdf",
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
        )
    )
    scene.build(n_envs=n_envs)
    raw = create_contacts(2, n_envs, 1e-5)
    raw.valid.fill(True)
    raw.position.from_numpy(np.broadcast_to(anchor, (2, n_envs, 3)).copy())
    forces = np.broadcast_to(0.05 * outward, (2, n_envs, 3)).copy()
    forces[:, 1] *= -1
    raw.force.from_numpy(forces)
    raw.friction.fill(0.0)
    streamed = IntrinsicContacts(entity.base_link, raw, model.info, surface.info, options, None)
    streamed.raw_sources.geom.fill(entity.base_link.geoms[0].idx)
    streamed.raw_sources.pair.from_numpy(np.repeat(np.arange(2)[:, None], n_envs, axis=1).astype(gs.np_int))
    streamed.raw_sources.witness.from_numpy(np.broadcast_to(anchor, (2, n_envs, 3)).copy())
    state = model.create_state(n_envs, "full")
    errno = V(dtype=gs.qd_int, shape=(n_envs,))
    errno.fill(0)
    kernel_begin_step(state)
    streamed.scatter(raw, state, model.info, surface.info, options, float(np.finfo(gs.np_float).eps), errno)
    statuses = qd_to_numpy(raw.status, transpose=True)
    # Both codes certify strict continuous physical incompatibility. Backends
    # may establish the force-only or complete-wrench certificate first.
    assert np.isin(statuses[0], [35, 36]).all()
    # Independently verify the disk stays in the planar triangle. A pure
    # outward force is a strict polar witness on that entire physical disk.
    vertices = np.array([triangles[0], triangles[0] + edges[0, 0], triangles[0] + edges[0, 1]])
    for a in range(3):
        direction = np.cross(outward, vertices[(a + 1) % 3] - vertices[a])
        direction /= np.linalg.norm(direction)
        if direction @ (vertices[(a + 2) % 3] - vertices[a]) < 0:
            direction *= -1
        assert direction @ (anchor - vertices[a]) > 1e-5
    assert_equal(qd_to_numpy(streamed.tractions.circle.face, transpose=True), 0)
    assert_allclose(outward @ -outward, -1.0, atol=1e-15)
    assert_equal(statuses[1], 0)
    assert qd_to_numpy(errno)[0] & ErrorCode.INVALID_STRESS_LOAD
    assert qd_to_numpy(errno)[1] == 0
    assert qd_to_numpy(streamed.failure.group_count)[0] == 2
    assert_allclose(qd_to_numpy(streamed.failure.target_force)[0], 0.1 * outward, atol=1e-15)
    assert_equal(qd_to_numpy(raw.force, transpose=True), np.moveaxis(forces, 0, 1))
