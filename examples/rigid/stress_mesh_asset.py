"""Prepare and cache volume or finite-thickness shell assets from rigid surface meshes."""

import errno
import hashlib
import importlib.metadata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tetgen
import trimesh

from examples.rigid.stress_shell_asset import shell_boundary, surface_to_quads


@dataclass(frozen=True)
class PreparedStressMesh:
    collision: Path
    elastic: Path
    minimum_z: float


def load_surface(source: Path) -> trimesh.Trimesh:
    """Load a surface without tolerance welding, and apply every scene transform to its coordinates."""
    if source.suffix.lower() == ".obj":
        # Scene loaders do not propagate constructor options to OBJ's individual geometries.
        with source.open("rb") as stream:
            parsed = trimesh.exchange.obj.load_obj(stream, skip_materials=True, maintain_order=True)
        scene = trimesh.Scene(
            {name: trimesh.Trimesh(**geometry, process=False) for name, geometry in parsed["geometry"].items()}
        )
    else:
        scene = trimesh.load_scene(source, process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("The stress surface must contain triangles, not points or curves.")
        # Do not treat a small scene translation as an identity transform: it can define a real material gap.
        vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        faces = mesh.faces if np.linalg.det(transform[:3, :3]) >= 0 else mesh.faces[:, ::-1]
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    return trimesh.util.concatenate(meshes)


def write_rigid_asset(collision: Path, material_mesh: trimesh.Trimesh, density: float) -> Path:
    """Write a Unified Robot Description Format (URDF) asset with inertia from the actual material volume."""
    properties = material_mesh.mass_properties
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Material density must be finite and positive.")
    if properties.mass <= 0 or not np.isfinite(properties.mass) or not np.isfinite(properties.inertia).all():
        raise ValueError("The material mesh must have finite positive mass properties.")
    root = ET.Element("robot", name="stress_mesh")
    link = ET.SubElement(root, "link", name="mesh")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{value:.17g}" for value in properties.center_mass))
    ET.SubElement(inertial, "mass", value=f"{density * properties.mass:.17g}")
    inertia = density * properties.inertia
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.17g}",
        ixy=f"{inertia[0, 1]:.17g}",
        ixz=f"{inertia[0, 2]:.17g}",
        iyy=f"{inertia[1, 1]:.17g}",
        iyz=f"{inertia[1, 2]:.17g}",
        izz=f"{inertia[2, 2]:.17g}",
    )
    for role in ("visual", "collision"):
        geometry = ET.SubElement(ET.SubElement(link, role), "geometry")
        ET.SubElement(geometry, "mesh", filename=str(collision.resolve()))
    digest = hashlib.sha256(ET.tostring(root)).hexdigest()
    path = collision.parent / f"rigid-{digest}.urdf"
    if not path.is_file():
        with TemporaryDirectory(dir=collision.parent) as temporary:
            staged = Path(temporary) / "rigid.urdf"
            ET.ElementTree(root).write(staged, encoding="utf-8", xml_declaration=True)
            staged.replace(path)
    return path


def volume_boundary(
    tetrahedra: np.ndarray, vertices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boundary faces, owning elements and local corner indices, orienting outward when coordinates are given."""
    local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = tetrahedra[:, local_faces].reshape((-1, 3))
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError("The elastic mesh has a non-manifold tetrahedral face.")
    selected = first[counts == 1]
    boundary, owners, corners = faces[selected], selected // 4, local_faces[selected % 4]
    if vertices is not None:
        points = vertices[boundary]
        normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        toward_material = vertices[tetrahedra[owners]].mean(axis=1) - points[:, 0]
        inward = np.einsum("ij,ij->i", normal, toward_material) > 0
        boundary[inward] = boundary[inward][:, [0, 2, 1]]
        corners[inward] = corners[inward][:, [0, 2, 1]]
    return boundary, owners, corners


def prepare_stress_mesh(
    source: Path,
    elastic: Path | None,
    backend: str,
    scale: float,
    max_volume: float | None,
    cache: Path,
    thickness: float = 0.0005,
) -> PreparedStressMesh:
    """Scale a surface and matching elastic mesh together, or tetrahedralize a closed single solid.

    Input coordinates share one authored frame. Scale converts them to metres; both meshes are centered together.
    A supplied elastic asset preserves cavities and its topology. Automatic meshing fills a closed single solid; it does
    not infer wall thickness from an outer surface. Shell input is a midsurface; its two physical sides and open-edge caps
    form the collision mesh. Thickness is in metres, independent of scale. Cache keys include geometry and dependencies.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mesh scale must be finite and positive.")
    if max_volume is not None and (not np.isfinite(max_volume) or max_volume <= 0):
        raise ValueError("Maximum tetrahedron volume must be finite and positive, in cubic metres.")
    if source.suffix.lower() == ".npz":
        if elastic is not None:
            raise ValueError("Use either --mesh asset.npz or --stress-mesh, not both.")
        elastic = source
    if backend not in ("volume", "shell"):
        raise ValueError("Stress backend must be volume or shell.")
    if backend == "shell" and (not np.isfinite(thickness) or thickness <= 0):
        raise ValueError("Shell thickness must be finite and positive, in metres.")
    if backend == "shell" and max_volume is not None:
        raise ValueError("--max-volume only applies to the volume backend.")
    if max_volume is not None and elastic is not None:
        raise ValueError("--max-volume only applies when tetrahedralizing a new surface, not a prepared elastic asset.")

    if source.suffix.lower() == ".npz":
        surface = None
    else:
        surface = load_surface(source)
        surface.remove_unreferenced_vertices()
        if elastic is None and backend == "volume":
            surface.merge_vertices()
        surface.vertices *= scale
        if not len(surface.faces) or not np.isfinite(surface.vertices).all():
            raise ValueError("The surface mesh must contain finite vertices and triangles.")

    digest = hashlib.sha256(b"rigid-stress-mesh-v4-triangle-three-quads")
    digest.update(source.read_bytes())
    if surface is not None:
        # Scene formats can refer to geometry buffers outside the source file
        digest.update(surface.vertices.tobytes())
        digest.update(surface.faces.tobytes())
    if elastic is not None:
        digest.update(elastic.read_bytes())
    digest.update(f"{backend}:{scale.hex()}:{max_volume}:{tetgen.__version__}:{trimesh.__version__}".encode())
    if backend == "shell":
        digest.update(f"{thickness.hex()}:{importlib.metadata.version('pymeshlab')}".encode())
    directory = cache / digest.hexdigest()
    collision_path, elastic_path = directory / "collision.obj", directory / "elastic.npz"
    if collision_path.is_file() and elastic_path.is_file():
        surface = load_surface(collision_path)
        return PreparedStressMesh(collision_path, elastic_path, surface.bounds[0, 2])

    if elastic is None and backend == "shell":
        vertices, quads = surface_to_quads(surface)
        directors = None
        extra_arrays = ()
    elif elastic is None:
        assert surface is not None
        if not surface.is_volume or len(surface.split(only_watertight=False)) != 1:
            raise ValueError(
                "Automatic meshing needs one closed, consistently oriented solid. "
                "Repair the surface or provide a matching --stress-mesh NPZ for cavities or shells."
            )
        mesher = tetgen.TetGen(
            surface.vertices.astype(np.float64, copy=False), surface.faces.astype(np.int32, copy=False)
        )
        switches = "pq1.5/5Q" if max_volume is None else f"pq1.5/5a{max_volume:.17g}Q"
        vertices, tetrahedra, *_ = mesher.tetrahedralize(switches=switches)
        extra_arrays = ()
    else:
        with np.load(elastic, allow_pickle=False) as asset:
            vertices = asset["vertices"] * scale
            tetrahedra = asset["tetrahedra"] if backend == "volume" else None
            extra_arrays = tuple((name, asset[name]) for name in asset.files if name != "vertices")
            if backend == "shell":
                quads, directors = asset["quads"], asset.get("directors")

    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Elastic vertices must be a finite nonempty (n, 3) array.")
    if backend == "volume":
        if (
            tetrahedra.ndim != 2
            or tetrahedra.shape[1] != 4
            or not len(tetrahedra)
            or not np.issubdtype(tetrahedra.dtype, np.integer)
            or tetrahedra.min() < 0
            or tetrahedra.max() >= len(vertices)
        ):
            raise ValueError("The volume asset needs nonempty in-range integer tetrahedra.")
        tetrahedra = tetrahedra.copy()
        points = vertices[tetrahedra]
        determinant = np.linalg.det(points[:, 1:] - points[:, :1])
        if (determinant == 0).any() or not np.isfinite(determinant).all():
            raise ValueError("The volume mesh contains degenerate tetrahedra.")
        flipped = determinant < 0
        tetrahedra[flipped] = tetrahedra[flipped][:, [1, 0, 2, 3]]
        triangles, _, _ = volume_boundary(tetrahedra)
        if surface is None:
            surface = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        surface, directors = shell_boundary(vertices, quads, directors, thickness)
    assert surface is not None
    center = surface.bounds.mean(axis=0)
    surface.vertices = surface.vertices - center
    vertices = vertices - center
    # Publishing a completed directory prevents another process from reading a partly written mesh pair
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as temporary:
        output = Path(temporary)
        surface.export(output / "collision.obj", digits=17)
        if backend == "volume":
            np.savez_compressed(
                output / "elastic.npz", vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles
            )
        else:
            arrays = dict(extra_arrays)
            arrays.update(quads=quads, directors=directors)
            np.savez_compressed(output / "elastic.npz", vertices=vertices, **arrays)
        try:
            output.rename(directory)
        except OSError as error:
            if (
                error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                or not collision_path.is_file()
                or not elastic_path.is_file()
            ):
                raise
    return PreparedStressMesh(collision_path, elastic_path, surface.bounds[0, 2])
