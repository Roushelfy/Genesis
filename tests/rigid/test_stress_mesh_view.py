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


def test_automatic_shell_units_mass_cache_and_reload(tmp_path):
    surface = trimesh.Trimesh(
        vertices=[[120.0, -60.0, 30.0], [160.0, -60.0, 30.0], [160.0, -10.0, 30.0], [120.0, -10.0, 30.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    source = tmp_path / "plate.stl"
    surface.export(source)
    prepared = prepare_stress_mesh(source, None, "shell", 0.001, None, tmp_path / "cache", thickness=0.002)
    with np.load(prepared.elastic) as asset:
        assert_equal(asset["quads"].shape, (6, 4))
        assert_equal(asset["vertices"].shape, (11, 3))
        assert_allclose(np.ptp(asset["vertices"], axis=0), [0.04, 0.05, 0], atol=1e-12)
        assert_allclose(asset["directors"], [0, 0, 1], atol=1e-12)
    field = StressFieldSurface(prepared.elastic, "shell", thickness=0.002)
    material = field.material_mesh
    assert material.is_volume
    assert_allclose(material.volume, 0.04 * 0.05 * 0.002, atol=1e-14)
    assert_allclose(material.center_mass, [0, 0, 0], atol=1e-12)
    assert_allclose(
        np.diag(material.moment_inertia),
        material.volume / 12 * np.array([0.05**2 + 0.002**2, 0.04**2 + 0.002**2, 0.04**2 + 0.05**2]),
        atol=1e-15,
    )
    collision = trimesh.load_scene(prepared.collision, process=False).to_mesh()
    assert_allclose(collision.vertices, material.vertices, atol=1e-10)
    assert_allclose(prepared.minimum_z, -0.001, atol=1e-12)
    repeated = prepare_stress_mesh(source, None, "shell", 0.001, None, tmp_path / "cache", thickness=0.002)
    assert repeated == prepared
    thicker = prepare_stress_mesh(source, None, "shell", 0.001, None, tmp_path / "cache", thickness=0.004)
    assert thicker.elastic != prepared.elastic
    assert_allclose(thicker.minimum_z, -0.002, atol=1e-12)
    reloaded = prepare_stress_mesh(prepared.elastic, None, "shell", 1.0, None, tmp_path / "cache", thickness=0.002)
    with np.load(prepared.elastic) as automatic, np.load(reloaded.elastic) as explicit:
        for name in ("vertices", "quads", "directors"):
            assert_allclose(automatic[name], explicit[name], atol=1e-12)


@pytest.mark.parametrize("kind", ["closed", "reversed", "open"])
def test_automatic_shell_orientable_surface(tmp_path, kind):
    surface = trimesh.creation.icosphere(subdivisions=0, radius=0.03)
    if kind == "reversed":
        surface.invert()
    elif kind == "open":
        surface.update_faces(surface.triangles_center[:, 2] < 0.01)
        surface.remove_unreferenced_vertices()
    source = tmp_path / "surface.obj"
    surface.export(source)
    prepared = prepare_stress_mesh(source, None, "shell", 1.0, None, tmp_path / "cache")
    field = StressFieldSurface(prepared.elastic, "shell")
    assert field.material_mesh.is_volume
    assert field.material_mesh.volume < surface.convex_hull.volume / 4


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        ("disconnected", "disconnected"),
        ("duplicate", "duplicate"),
        ("degenerate", "degenerate"),
        ("nonmanifold", "nonmanifold"),
        ("thickness", "offset|intersect"),
    ],
)
def test_automatic_shell_rejects_invalid_geometry(tmp_path, kind, diagnostic):
    surface = trimesh.creation.icosphere(subdivisions=0, radius=0.03)
    thickness = 0.0005
    if kind == "disconnected":
        second = surface.copy()
        second.apply_translation([0, 0, 0.060000001])
        surface = trimesh.util.concatenate((surface, second))
    elif kind == "duplicate":
        surface.faces = np.concatenate((surface.faces, surface.faces[:1]))
    elif kind == "degenerate":
        surface.vertices[surface.faces[0, 1]] = surface.vertices[surface.faces[0, 0]]
    elif kind == "nonmanifold":
        surface = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            process=False,
        )
    else:
        thickness = 0.2
    source = tmp_path / "invalid.obj"
    surface.export(source)
    with pytest.raises(ValueError, match=diagnostic):
        prepare_stress_mesh(source, None, "shell", 1.0, None, tmp_path / "cache", thickness=thickness)


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
