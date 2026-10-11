import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.compatibility import kernel_certify_continuous_force
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_anchor_fan_tractions,
    create_footprints,
    kernel_anchor_fan_profiles,
    kernel_footprint_ports,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
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
    assert_allclose(tensor_to_array(egg.get_links_inertia())[0], inertia, rtol=0.0, atol=1e-16)
