import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.compatibility import (
    create_wrench_search,
    kernel_certify_continuous_force,
    kernel_certify_continuous_wrench,
    kernel_search_wrench_certificate,
)
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_anchor_fan_tractions,
    create_footprints,
    create_surface_fan_tractions,
    kernel_anchor_fan_profiles,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.source_vertices import StressSourceVertices
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionStatus,
    create_traction_info,
    kernel_fit_tractions,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from tests.utils.assertions import assert_allclose, assert_equal

ASSETS = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("tick", (113, 114))
def test_saved_live_edge_sources_are_one_topological_feature(tick):
    with np.load(Path(__file__).with_name("assets") / f"nonconvex_mesh64_tick{tick}.npz") as saved:
        witnesses, positions, forces = (saved[key] for key in ("witness", "position", "force"))
        geometries, pairs = saved["geom"], saved["pair"]
        radii, frictions, methods = (saved[key] for key in ("radius", "friction", "method"))
    model = StressModel(RigidStressOptions(mesh=ASSETS / "elastic.npz", method="pcg"))
    surface = StressSurface(10, model.info)
    contacts = create_contacts(len(witnesses), 1, float(radii[0]))
    sources = create_sources(len(witnesses), 1)
    contacts.valid.fill(True)
    contacts.position.from_numpy(positions[:, None])
    contacts.force.from_numpy(forces[:, None])
    contacts.radius.from_numpy(radii[:, None])
    contacts.friction.from_numpy(frictions[:, None])
    # The saved local witnesses came from a world-to-body transform near the
    # declared 0.65 m initial world pose. Include that operand scale when
    # assessing geometric roundoff, without changing the original load.
    contacts.frame_position.from_numpy(np.array([[0.65, 0.0, 0.031]]))
    sources.witness.from_numpy(witnesses[:, None])
    sources.geom.from_numpy(geometries[:, None])
    sources.pair.from_numpy(pairs[:, None])
    sources.method.from_numpy(methods[:, None])
    kernel_resolve_sources(1e-4, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, surface.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True), 0)
    anchors = qd_to_numpy(contacts.center, transpose=True)[0]
    assert_allclose(anchors, witnesses, rtol=0.0, atol=2e-16)
    moments = qd_to_numpy(sources.anchor_moment, transpose=True)[0]
    original_moments = moments + np.cross(anchors - positions, forces)
    assert_allclose(original_moments, 0.0, rtol=0.0, atol=1e-21)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_continuous_force_certificate_for_import_precision_mismatch():
    """A saved slipping force is infeasible even on the continuous radius footprint."""
    with np.load(Path(__file__).with_name("assets") / "nonconvex_urdf_tick111.npz") as saved:
        anchor, position, force = (saved[key] for key in ("anchor", "position", "force"))
        radius, friction, face = (float(saved["radius"]), float(saved["friction"]), int(saved["anchor_face"]))
    model = StressModel(RigidStressOptions(mesh=ASSETS / "elastic.npz", method="pcg"))
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 2)
    contacts = create_contacts(1, 2, radius)
    sources = create_sources(1, 2)
    tractions = create_anchor_fan_tractions(surface.info, 1, 2, compact=True, packed=True)
    footprints = create_footprints(paths.info.positions.shape[0], 1, 2)
    info = create_traction_info(surface.info)
    contacts.valid.fill(True)
    contacts.center.from_numpy(np.broadcast_to(anchor, (1, 2, 3)).copy())
    contacts.position.from_numpy(np.broadcast_to(position, (1, 2, 3)).copy())
    contacts.force.from_numpy(np.broadcast_to(force, (1, 2, 3)).copy())
    contacts.anchor_face.fill(face)
    # Increasing the actual input friction in the second independent environment
    # makes the same force feasible; a certificate must never leak between them.
    contacts.friction.from_numpy(np.array([[friction, 0.85]]))
    sources.status.fill(0)
    sources.anchor_moment.from_numpy(np.broadcast_to(np.cross(position - anchor, force), (1, 2, 3)).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_anchor_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 8e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[:, 0], [StressTractionStatus.NUMERICAL_FAILURE, 0])
    kernel_certify_continuous_force(contacts, tractions, surface.info, epsilon, 1e-8)
    assert_equal(
        qd_to_numpy(tractions.status, transpose=True)[:, 0],
        [StressTractionStatus.CONTINUOUS_FORCE_INFEASIBLE, StressTractionStatus.SUCCESS],
    )
    ray = qd_to_numpy(tractions.certificate, transpose=True)[0, 0]
    margin = float(qd_to_numpy(tractions.certificate_margin, transpose=True)[0, 0])
    assert 1e-8 < margin < 1.3e-8
    assert_equal(ray[3:], 0.0)
    assert_equal(qd_to_numpy(tractions.certificate, transpose=True)[1], 0.0)
    assert force @ ray[:3] / np.linalg.norm(force) >= margin
    with np.load(ASSETS / "elastic.npz") as mesh:
        triangles = mesh["vertices"][mesh["surface_triangles"]]
    # Independent geometric audit of the whole continuous support: closest
    # points on each triangle/edge bound every path of length < radius.
    eligible = []
    for i_f, triangle in enumerate(triangles):
        edges = (triangle[1:] - triangle[0]).T
        local = np.linalg.lstsq(edges, anchor - triangle[0], rcond=None)[0]
        distance = np.inf
        if min(*local, 1.0 - local.sum()) >= 0.0:
            distance = np.linalg.norm(triangle[0] + edges @ local - anchor)
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = triangle[second] - triangle[first]
            coordinate = np.clip((anchor - triangle[first]) @ edge / (edge @ edge), 0.0, 1.0)
            distance = min(distance, np.linalg.norm(triangle[first] + coordinate * edge - anchor))
        if distance <= radius:
            eligible.append(i_f)
    assert_equal(eligible, [49, 51, 69])
    normals = -qd_to_numpy(surface.info.face_normal, transpose=True)[eligible]
    components = normals @ ray[:3]
    support = components + friction * np.linalg.norm(ray[None, :3] - components[:, None] * normals, axis=1)
    assert (support < 0.0).all()
    # Exact inputs remain unchanged even though a diagnostic certificate exists.
    assert_equal(qd_to_numpy(contacts.force, transpose=True), np.broadcast_to(force, (2, 1, 3)))
    assert_equal(qd_to_numpy(contacts.radius, transpose=True), radius)


@pytest.mark.required
@pytest.mark.precision("64")
def test_saved_continuous_wrench_certificate_respects_support_budget_and_environment_lifecycle():
    """The exact step602 component is incompatible; a feasible input and release must never inherit its certificate."""
    with np.load(Path(__file__).with_name("assets") / "nonconvex_mesh64_tick602.npz") as saved:
        anchors, points, forces = (saved[key] for key in ("anchor", "position", "force"))
        radii, frictions, faces = (saved[key] for key in ("radius", "friction", "anchor_face"))
    model = StressModel(RigidStressOptions(mesh=ASSETS / "elastic.npz", method="pcg"))
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(2, 3, float(radii[0])), create_sources(2, 3)
    footprints = create_footprints(paths.info.positions.shape[0], 2, 3)
    tractions = create_surface_fan_tractions(surface.info, 2, 3, shared_support=True)
    info, search = create_traction_info(surface.info), create_wrench_search(2, 3, 8)
    force = np.broadcast_to(forces[:, None], (2, 3, 3)).copy()
    point = np.broadcast_to(points[:, None], (2, 3, 3)).copy()
    # This manufactured control is a different declared input: a zero-load
    # support and one feasible force/free moment at anchor1. It does not change
    # the production acceptance of the saved original component in env0.
    force[0, 1], force[1, 1] = 0.0, forces.sum(axis=0)
    point[1, 1] = anchors[1]
    free_moment = np.zeros((2, 3, 3))
    free_moment[1, 1] = np.cross(points - anchors[1], forces).sum(axis=0)
    anchor_moment = free_moment + np.cross(point - anchors[:, None], force)
    contacts.valid.from_numpy(np.array([[True, True, False], [True, True, False]]))
    contacts.center.from_numpy(np.broadcast_to(anchors[:, None], (2, 3, 3)).copy())
    contacts.position.from_numpy(point)
    contacts.force.from_numpy(force)
    contacts.radius.from_numpy(np.broadcast_to(radii[:, None], (2, 3)).copy())
    contacts.friction.from_numpy(np.broadcast_to(frictions[:, None], (2, 3)).copy())
    contacts.anchor_face.from_numpy(np.broadcast_to(faces[:, None], (2, 3)).copy())
    sources.status.fill(0)
    sources.geom.fill(24)
    sources.pair.fill(22)
    sources.free_moment.from_numpy(free_moment)
    sources.anchor_moment.from_numpy(anchor_moment)
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    statuses = qd_to_numpy(tractions.status, transpose=True, copy=True)
    assert statuses[0, 1] in (StressTractionStatus.NUMERICAL_FAILURE, StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    assert_equal(statuses[0, 0], StressTractionStatus.SUCCESS)
    assert_equal(statuses[1:], StressTractionStatus.SUCCESS)
    kernel_certify_continuous_force(contacts, tractions, surface.info, epsilon, 1e-8)
    assert qd_to_numpy(tractions.status, transpose=True)[0, 1] != StressTractionStatus.CONTINUOUS_FORCE_INFEASIBLE
    kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    # Bounded geometry work must never turn an unfinished proof into a physical diagnosis.
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 2, 16384, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 8, 1, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 8, 16384, 1e-8)
    assert_equal(
        qd_to_numpy(tractions.status, transpose=True)[0], [0, StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE]
    )
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[1:], 0)
    ray = qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 1]
    assert_allclose(np.linalg.norm(ray), 1.0, rtol=0.0, atol=1e-14)
    assert np.linalg.norm(ray[3:]) > 0.1
    assert qd_to_numpy(search.bound, transpose=True)[0, 1] < -1e-4
    assert qd_to_numpy(search.visited, transpose=True)[0, 1] > 0
    scale = max(np.linalg.norm(forces[1]), np.linalg.norm(anchor_moment[1, 0]) / radii[1])
    target = np.concatenate((forces[1], anchor_moment[1, 0] / radii[1])) / scale
    margin = target @ ray
    assert margin > 1e-3
    assert margin >= qd_to_numpy(tractions.certificate_margin, transpose=True)[0, 1]
    assert_equal(qd_to_numpy(tractions.certificate, transpose=True)[1:], 0.0)
    assert_equal(qd_to_numpy(contacts.force, transpose=True), force.swapaxes(0, 1))
    assert_equal(qd_to_numpy(contacts.position, transpose=True), point.swapaxes(0, 1))
    assert_equal(qd_to_numpy(contacts.radius, transpose=True), np.broadcast_to(radii, (3, 2)))
    # A stale certificate margin cannot override the final original-point
    # error budget when the declared target changes towards a feasible input.
    feasible_force = forces.sum(axis=0)
    feasible_moment = np.cross(points - anchors[1], forces).sum(axis=0)
    feasible_target = np.concatenate((feasible_force, feasible_moment / radii[1]))
    incompatible_target = np.concatenate((forces[1], anchor_moment[1, 0] / radii[1]))
    feasible_dot, incompatible_dot = feasible_target @ ray, incompatible_target @ ray
    assert feasible_dot < 0.0 < incompatible_dot
    fraction = -feasible_dot / (incompatible_dot - feasible_dot) + 1e-10
    force[1, 0] = (1 - fraction) * feasible_force + fraction * forces[1]
    anchor_moment[1, 0] = (1 - fraction) * feasible_moment + fraction * anchor_moment[1, 0]
    free_moment[1, 0] = anchor_moment[1, 0] - np.cross(point[1, 0] - anchors[1], force[1, 0])
    contacts.force.from_numpy(force)
    sources.anchor_moment.from_numpy(anchor_moment)
    sources.free_moment.from_numpy(free_moment)
    tractions.status.from_numpy(np.array([[0, 0, 0], [StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE, 0, 0]]))
    tractions.certificate_margin.fill(1.0)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 8, 16384, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    # Releasing only env0 clears its diagnostic; other environments remain independent.
    contacts.valid.from_numpy(np.array([[False, True, False], [False, True, False]]))
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 8, 16384, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    assert_equal(qd_to_numpy(tractions.certificate, transpose=True), 0.0)
    assert_equal(qd_to_numpy(tractions.certificate_margin, transpose=True), 0.0)


def _continuous_disk_polar_bound(mesh, anchors, radii, friction, ray):
    """Independent clipped-face boundary/arc Lipschitz bound; no native subdivision or search witnesses."""
    triangles, tetrahedra = mesh["vertices"][mesh["surface_triangles"]], mesh["tetrahedra"]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for i_f, face in enumerate(mesh["surface_triangles"]):
        owner = np.flatnonzero(np.isin(tetrahedra, face).sum(axis=1) == 3)[0]
        opposite = tetrahedra[owner][~np.isin(tetrahedra[owner], face)][0]
        if normals[i_f] @ (mesh["vertices"][opposite] - triangles[i_f, 0]) < 0:
            normals[i_f] *= -1
    maximum = -np.inf
    for center, radius, mu in zip(anchors, radii, friction):
        for triangle, normal in zip(triangles, normals):
            distance = (center - triangle[0]) @ normal
            if abs(distance) >= radius:
                continue
            circle_center, circle_radius = center - distance * normal, np.sqrt(radius**2 - distance**2)
            e0 = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
            e1 = np.cross(normal, e0)
            coordinates = np.column_stack(((triangle - circle_center) @ e0, (triangle - circle_center) @ e1))
            dual = np.linalg.pinv(np.column_stack((coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])))
            points = [v for v in coordinates if np.linalg.norm(v) <= circle_radius]
            angles = [0.0, 2 * np.pi]
            for first, second in zip(coordinates, np.roll(coordinates, -1, axis=0)):
                direction = second - first
                a, b, c = direction @ direction, 2 * first @ direction, first @ first - circle_radius**2
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    for fraction in ((-b - np.sqrt(discriminant)) / (2 * a), (-b + np.sqrt(discriminant)) / (2 * a)):
                        if 0 <= fraction <= 1:
                            point = first + fraction * direction
                            points.append(point)
                            angles.append(np.arctan2(point[1], point[0]) % (2 * np.pi))

            def polar(local, center=circle_center, first=e0, second=e1, inward=normal, friction=mu):
                xyz = center + local[:, 0, None] * first + local[:, 1, None] * second
                value = ray[:3] + np.cross(ray[3:], (xyz - anchors[1]) / radii[1])
                compression = value @ inward
                return compression + friction * np.linalg.norm(value - compression[:, None] * inward, axis=1)

            if points:
                maximum = max(maximum, float(polar(np.asarray(points)).max()))
            angles = np.unique(angles)
            for first, second in pairwise(angles):
                middle = (first + second) / 2
                point = circle_radius * np.array([np.cos(middle), np.sin(middle)])
                uv = dual @ (point - coordinates[0])
                if (uv >= 0).all() and uv.sum() <= 1:
                    count = max(1, int(np.ceil((second - first) * 16384 / (2 * np.pi))))
                    theta = np.linspace(first, second, count + 1)
                    values = polar(circle_radius * np.column_stack((np.cos(theta), np.sin(theta))))
                    lipschitz = (1 + mu) * np.linalg.norm(ray[3:]) * circle_radius / radii[1]
                    maximum = max(maximum, float(values.max() + lipschitz * (second - first) / (2 * count)))
    return maximum


@pytest.mark.required
@pytest.mark.precision("64")
def test_saved_policy_boundary_certificate_preserves_feasible_components_and_release():
    """The original policy876 component is incompatible; a bounded failed proof must stay unresolved."""
    with np.load(Path(__file__).with_name("assets") / "nonconvex_policy_tick876.npz") as saved:
        original = {key: saved[key].copy() for key in saved.files}
    model = StressModel(RigidStressOptions(mesh=ASSETS / "elastic.npz", method="pcg"))
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(2, 3, 0.006), create_sources(2, 3)
    info, search = create_traction_info(surface.info), create_wrench_search(2, 3, 12)
    footprints = create_footprints(paths.info.positions.shape[0], 2, 3)
    tractions = create_surface_fan_tractions(surface.info, 2, 3, shared_support=True)
    for name in ("position", "center", "anchor_face", "force", "normal", "friction", "radius"):
        values = original["contacts." + name][:2]
        contacts.__dict__[name].from_numpy(np.repeat(values[:, None], 3, axis=1))
    for name in ("geom", "pair", "status", "free_moment", "anchor_moment", "barycentric"):
        values = original["raw_sources." + name][:2]
        sources.__dict__[name].from_numpy(np.repeat(values[:, None], 3, axis=1))
    valid = np.array([[True, True, False], [True, True, False]])
    force = np.repeat(original["contacts.force"][:2, None], 3, axis=1)
    moment = np.repeat(original["raw_sources.anchor_moment"][:2, None], 3, axis=1)
    # Env1 is a separately declared feasible control. The saved original
    # component and both unchanged support sources remain in env0.
    force[1, 1], moment[1, 1] = 0.0, 0.0
    contacts.valid.from_numpy(valid)
    contacts.force.from_numpy(force)
    sources.anchor_moment.from_numpy(moment)
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    before_force = qd_to_numpy(tractions.force_error, transpose=True, copy=True)
    before_moment = qd_to_numpy(tractions.moment_error, transpose=True, copy=True)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 0], StressTractionStatus.SUCCESS)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[1:], StressTractionStatus.SUCCESS)
    kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 2, 65536, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 12, 1, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[0, 1], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 12, 65536, 1e-8)
    assert_equal(
        qd_to_numpy(tractions.status, transpose=True)[0], [0, StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE]
    )
    assert_equal(qd_to_numpy(tractions.status, transpose=True)[1:], 0)
    assert qd_to_numpy(search.bound, transpose=True)[0, 1] < -1e-4
    assert qd_to_numpy(search.evaluations, transpose=True)[0, 1] <= 640
    assert 0 < qd_to_numpy(search.visited, transpose=True)[0, 1] <= 65536
    assert_equal(qd_to_numpy(tractions.force_error, transpose=True), before_force)
    assert_equal(qd_to_numpy(tractions.moment_error, transpose=True), before_moment)
    ray = qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 1]
    anchors, radii, friction = (original["contacts." + name][:2] for name in ("center", "radius", "friction"))
    target = np.concatenate((force[1, 0], moment[1, 0] / radii[1])) / np.linalg.norm(force[1, 0])
    assert target @ ray > 1e-3
    assert_allclose(np.linalg.norm(ray), 1.0, rtol=0.0, atol=1e-14)
    with np.load(ASSETS / "elastic.npz") as mesh:
        assert _continuous_disk_polar_bound(mesh, anchors, radii, friction, ray) < -1e-4
    assert_equal(qd_to_numpy(contacts.force, transpose=True), force.swapaxes(0, 1))
    for name in ("position", "center", "friction", "radius"):
        assert_equal(
            qd_to_numpy(contacts.__dict__[name], transpose=True),
            np.repeat(original["contacts." + name][None, :2], 3, axis=0),
        )
    assert_equal(qd_to_numpy(tractions.certificate, transpose=True)[1:], 0.0)
    # Release/reset env0 only; feasible env1 must retain its success, and no
    # stale physical classification or certificate may survive in env0.
    valid[:, 0] = False
    contacts.valid.from_numpy(valid)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)
    kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 12, 65536, 1e-8)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    assert_equal(qd_to_numpy(tractions.certificate, transpose=True), 0.0)
    assert_equal(qd_to_numpy(tractions.certificate_margin, transpose=True), 0.0)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("preserve", (False, True))
def test_urdf_source_mesh_precision_preserves_declared_inertia(preserve):
    scene = gs.Scene(show_viewer=False, profiling_options=gs.options.ProfilingOptions(show_FPS=False))
    egg = scene.add_entity(
        gs.morphs.URDF(
            file=ASSETS / "egg_shell.urdf",
            pos=(0.65, 0.013, 0.07),
            quat=(0.9238795325112867, 0.0, 0.3826834323650898, 0.0),
            align=False,
            decimate=False,
            convexify=True,
            preserve_mesh_precision=preserve,
        )
    )
    scene.build()
    model = StressModel(RigidStressOptions(mesh=ASSETS / "elastic.npz", method="pcg"))
    vertices = StressSourceVertices(egg.base_link, model.info, 1e-8)
    offsets = qd_to_numpy(vertices.info.offset, transpose=True)
    if preserve:
        assert offsets.max() < 2e-15
    else:
        assert 5e-10 < offsets.max() < 1e-8
    inertial = ET.parse(ASSETS / "egg_shell.urdf").find("link/inertial")
    mass = float(inertial.find("mass").get("value"))
    com = np.array([float(value) for value in inertial.find("origin").get("xyz").split()])
    values = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
    inertia = np.array(
        [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ]
    )
    assert_allclose(tensor_to_array(egg.get_mass()), mass, rtol=0.0, atol=1e-16)
    assert_allclose(tensor_to_array(egg.get_links_COM())[0], com, rtol=0.0, atol=1e-16)
    rotation = Rotation.from_quat(np.asarray(egg.base_link.desc.inertial_quat)[[1, 2, 3, 0]]).as_matrix()
    authored_inertia = rotation @ tensor_to_array(egg.get_links_inertia())[0] @ rotation.T
    assert_allclose(authored_inertia, inertia, rtol=0.0, atol=1e-16)
