import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from examples.rigid.stress_field_view import StressFieldSurface
from examples.rigid.stress_mesh_asset import prepare_stress_mesh, write_rigid_asset
from tests.utils.assertions import assert_allclose, assert_equal


def test_stress_surface_mesh_preparation(tmp_path):
    source = tmp_path / "cube.stl"
    shifted = trimesh.creation.box(extents=[40, 50, 60])
    shifted.apply_translation([120, -60, 30])
    shifted.export(source)
    prepared = prepare_stress_mesh(source, None, "volume", 0.001, None, tmp_path / "cache")
    with np.load(prepared.elastic) as asset:
        vertices, tetrahedra, faces = asset["vertices"], asset["tetrahedra"], asset["surface_triangles"]
    points = vertices[tetrahedra]
    assert (np.linalg.det(points[:, 1:] - points[:, :1]) > 0).all()
    boundary = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert boundary.is_volume
    assert_allclose(boundary.extents, [0.04, 0.05, 0.06], atol=1e-9)
    assert_allclose(boundary.volume, 0.04 * 0.05 * 0.06, atol=1e-12)
    assert_allclose(prepared.minimum_z, -0.03, atol=1e-9)
    rigid = write_rigid_asset(prepared.collision, boundary, 2000.0)
    mass = ET.parse(rigid).find("link/inertial/mass")
    assert_allclose(float(mass.attrib["value"]), 2000 * 0.04 * 0.05 * 0.06, atol=1e-12)
    repeated = prepare_stress_mesh(source, None, "volume", 0.001, None, tmp_path / "cache")
    assert repeated.elastic == prepared.elastic
    assert_allclose(repeated.minimum_z, prepared.minimum_z, atol=1e-9)
    rescaled = prepare_stress_mesh(source, None, "volume", 0.002, None, tmp_path / "cache")
    assert rescaled.elastic != prepared.elastic
    with np.load(rescaled.elastic) as asset:
        assert_allclose(np.ptp(asset["vertices"], axis=0), [0.08, 0.10, 0.12], atol=1e-9)
    reversed_tetrahedra = tetrahedra[:, [1, 0, 2, 3]]
    np.savez(tmp_path / "authored.npz", vertices=vertices, tetrahedra=reversed_tetrahedra)
    authored = prepare_stress_mesh(tmp_path / "authored.npz", None, "volume", 2.0, None, tmp_path / "cache")
    with np.load(authored.elastic) as asset:
        points = asset["vertices"][asset["tetrahedra"]]
        assert (np.linalg.det(points[:, 1:] - points[:, :1]) > 0).all()
        boundary = trimesh.Trimesh(vertices=asset["vertices"], faces=asset["surface_triangles"], process=False)
        assert boundary.is_volume
        assert_allclose(boundary.volume, 8 * 0.04 * 0.05 * 0.06, atol=1e-12)
    opened = trimesh.creation.box()
    opened.update_faces(np.arange(11))
    opened.export(tmp_path / "open.obj")
    with pytest.raises(ValueError, match="closed"):
        prepare_stress_mesh(tmp_path / "open.obj", None, "volume", 1.0, None, tmp_path / "cache")
    with pytest.raises(ValueError, match="midsurface"):
        prepare_stress_mesh(source, None, "shell", 1.0, None, tmp_path / "cache")


def test_stress_boundary_colors_and_material_sides(tmp_path):
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    tetrahedra = np.array([[1, 0, 2, 3], [0, 2, 1, 4]])
    asset = tmp_path / "volume.npz"
    np.savez(asset, vertices=vertices, tetrahedra=tetrahedra)
    field = StressFieldSurface(asset, "volume")
    mesh = field.colored_mesh(np.array([[0.0] * 4, [10.0] * 4]), 10.0)
    assert_equal(len(mesh.faces), 6)
    assert_equal((mesh.visual.vertex_colors[:, :3] == [20, 10, 70]).all(axis=1).sum(), 9)
    assert_equal((mesh.visual.vertex_colors[:, :3] == [240, 40, 20]).all(axis=1).sum(), 9)
    corner = field.colored_mesh(np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]), 10.0)
    red = (corner.visual.vertex_colors[:, :3] == [240, 40, 20]).all(axis=1)
    assert red.any()
    assert_allclose(corner.vertices[red], [1.0, 0.0, 0.0], atol=1e-12)
    gray = field.colored_mesh(np.full((2, 4), np.nan), 10.0)
    assert_equal(gray.visual.vertex_colors[:, :3], [140, 140, 140])
    with pytest.raises(ValueError, match="positive"):
        field.colored_mesh(np.zeros((2, 4)), 0)
    quads = np.array([[0, 1, 2, 3]])
    shell = tmp_path / "shell.npz"
    np.savez(
        shell, vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), quads=quads
    )
    sides = StressFieldSurface(shell, "shell", thickness=0.2)
    properties = sides.material_mesh.mass_properties
    assert_allclose(properties.mass, 0.2, atol=1e-12)
    assert_allclose(properties.center_mass, [0.5, 0.5, 0], atol=1e-12)
    assert_allclose(np.diag(properties.inertia), 0.2 / 12 * np.array([1.04, 1.04, 2.0]), atol=1e-12)
    colored = sides.colored_mesh(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]]]), 10.0)
    assert_allclose(colored.vertices[:6, 2], 0.1, atol=1e-12)
    assert_allclose(colored.vertices[6:, 2], -0.1, atol=1e-12)
    assert_equal(colored.visual.vertex_colors[:6, :3], [20, 10, 70])
    assert_equal(colored.visual.vertex_colors[6:, :3], [240, 40, 20])
