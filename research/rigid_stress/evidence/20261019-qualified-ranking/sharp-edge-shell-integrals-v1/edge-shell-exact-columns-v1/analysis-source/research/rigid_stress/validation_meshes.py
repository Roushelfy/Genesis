"""Generate independent three-dimensional physical-validation assets with pinned Gmsh."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gmsh
import numpy as np


@dataclass(frozen=True)
class PhysicalGeometry:
    name: str
    description: str
    thickness_m: float | None = None
    gap_m: float | None = None
    minimum_rounding_m: float | None = None


GEOMETRIES = (
    PhysicalGeometry("ellipsoid", "Solid ellipsoid, semiaxes12/15/9mm"),
    PhysicalGeometry("egg", "Full normal-offset hollow egg,22/22/30mm semiaxes, taper0.18", thickness_m=0.0005),
    PhysicalGeometry("capsule", "Radius8mm cylinder, axial length24mm, hemispherical caps"),
    PhysicalGeometry("u", "Rounded U tube, bend radius8mm, tube radius3mm, arms24mm", gap_m=0.010),
    PhysicalGeometry("thin_neck", "Smooth revolved cubic B-spline dumbbell,44mm long, central radius2mm"),
    PhysicalGeometry("cavity", "Full spherical shell, outer radius12mm, inner radius8mm", thickness_m=0.004),
    PhysicalGeometry("narrow_gap", "Rounded U tube, bend radius4.5mm, tube radius4mm, arms24mm", gap_m=0.001),
)


def build_shell(inner_radius, layers):
    radii = np.linspace(1, inner_radius, layers + 1)
    volumes = [(3, gmsh.model.occ.addSphere(0, 0, 0, radius)) for radius in radii]
    _, mapping = gmsh.model.occ.fragment(volumes[:1], volumes[1:])
    gmsh.model.occ.remove(mapping[-1], recursive=True)


def build_u(bend_radius, tube_radius):
    bend = gmsh.model.occ.addTorus(0, 0, 0, bend_radius, tube_radius, angle=np.pi)
    gmsh.model.occ.rotate([(3, bend)], 0, 0, 0, 1, 0, 0, np.pi)
    volumes = [(3, bend)]
    for x in (-bend_radius, bend_radius):
        volumes.append((3, gmsh.model.occ.addCylinder(x, 0, 0, 0, 0.024, 0, tube_radius)))
        volumes.append((3, gmsh.model.occ.addSphere(x, 0.024, 0, tube_radius)))
    gmsh.model.occ.fuse(volumes[:1], volumes[1:])


def build_neck():
    profile = (
        np.array(
            [
                [-22, 0],
                [-22, 4],
                [-20, 7],
                [-15, 8],
                [-10, 7],
                [-6, 2],
                [0, 2],
                [6, 2],
                [10, 7],
                [15, 8],
                [20, 7],
                [22, 4],
                [22, 0],
            ]
        )
        * 0.001
    )
    points = [gmsh.model.occ.addPoint(x, radius, 0) for x, radius in profile]
    curve = gmsh.model.occ.addBSpline(points, degree=3)
    axis = gmsh.model.occ.addLine(points[-1], points[0])
    boundary = gmsh.model.occ.addCurveLoop([curve, axis])
    face = gmsh.model.occ.addPlaneSurface([boundary])
    gmsh.model.occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, 2 * np.pi)


def build_geometry(name, layers):
    if name == "ellipsoid":
        volume = gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.dilate([(3, volume)], 0, 0, 0, 0.012, 0.015, 0.009)
    elif name == "egg":
        gmsh.model.occ.addSphere(0, 0, 0, 1)
    elif name == "cavity":
        build_shell(2 / 3, layers)
    elif name == "capsule":
        cylinder = gmsh.model.occ.addCylinder(0, 0, -0.012, 0, 0, 0.024, 0.008)
        caps = [(3, gmsh.model.occ.addSphere(0, 0, z, 0.008)) for z in (-0.012, 0.012)]
        gmsh.model.occ.fuse([(3, cylinder)], caps)
    elif name == "u":
        build_u(0.008, 0.003)
    elif name == "narrow_gap":
        build_u(0.0045, 0.004)
    elif name == "thin_neck":
        build_neck()
    else:
        raise ValueError(f"Unknown geometry: {name}")


def transform_shell(name, coordinates):
    radius = np.linalg.norm(coordinates, axis=1)
    unit = coordinates / radius[:, None]
    if name == "cavity":
        return coordinates * 0.012
    scale = 1 - 0.18 * unit[:, 2]
    outer = unit * np.column_stack((0.022 * scale, 0.022 * scale, np.full(len(unit), 0.030)))
    gradient = np.column_stack(
        (
            2 * unit[:, 0] / (0.022 * scale),
            2 * unit[:, 1] / (0.022 * scale),
            2 / 0.030 * (unit[:, 2] + 0.18 * (unit[:, 0] ** 2 + unit[:, 1] ** 2) / scale),
        )
    )
    normal = gradient / np.linalg.norm(gradient, axis=1)[:, None]
    depth = (1 - radius) * (0.0005 / 0.08)
    return outer - depth[:, None] * normal


def generate_mesh(name, size, layers, output):
    if size <= 0 or layers < 1:
        raise ValueError("Positive mesh size and shell-layer count are required")
    if output.exists():
        raise FileExistsError(output)
    definition = next(geometry for geometry in GEOMETRIES if geometry.name == name)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.model.add(name)
        build_geometry(name, layers)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        size_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(size_field, "F", str(size))
        gmsh.model.mesh.field.setAsBackgroundMesh(size_field)
        gmsh.model.mesh.generate(2 if name == "egg" else 3)
        if name != "egg":
            gmsh.model.mesh.optimize("Netgen")
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, element_tags = gmsh.model.mesh.getElementsByType(2 if name == "egg" else 4)
        assert len(element_tags) > 0
        order = np.argsort(tags)
        indices = np.searchsorted(tags[order], element_tags).reshape(-1, 3 if name == "egg" else 4)
        used, inverse = np.unique(indices, return_inverse=True)
        cad_coordinates = coordinates.reshape(-1, 3)[order][used]
        if name == "egg":
            surface = np.sort(inverse.reshape(-1, 3), axis=1)
            count = len(cad_coordinates)
            unit = cad_coordinates / np.linalg.norm(cad_coordinates, axis=1)[:, None]
            cad_coordinates = np.concatenate([unit * radius for radius in np.linspace(1, 0.92, layers + 1)])
            layers_start = np.arange(layers)[:, None, None] * count
            bottom = surface[None] + layers_start
            top = bottom + count
            tetrahedra = np.stack(
                (
                    np.stack((bottom[..., 0], bottom[..., 1], bottom[..., 2], top[..., 2]), axis=-1),
                    np.stack((bottom[..., 0], bottom[..., 1], top[..., 1], top[..., 2]), axis=-1),
                    np.stack((bottom[..., 0], top[..., 0], top[..., 1], top[..., 2]), axis=-1),
                ),
                axis=-2,
            )
            tets = tetrahedra.reshape(-1, 4)
        else:
            tets = inverse.reshape(-1, 4)
        vertices = transform_shell(name, cad_coordinates) if name in ("egg", "cavity") else cad_coordinates
        determinant = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])
        flipped = determinant < 0
        tets[flipped, 2:4] = tets[flipped, 3:1:-1]
        local_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces, counts = np.unique(np.sort(tets[:, local_faces].reshape(-1, 3)), axis=0, return_counts=True)
        assert np.max(counts) <= 2
        boundary = faces[counts == 1]
        edges = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
        scaled_jacobian = np.abs(determinant) / np.prod(np.linalg.norm(edges, axis=2), axis=1)
        assert np.min(np.abs(determinant)) > 0
        np.savez_compressed(
            output, vertices=vertices, tetrahedra=tets, surface_triangles=boundary, cad_coordinates=cad_coordinates
        )
        metadata = {
            "geometry": asdict(definition),
            "gmsh_version": gmsh.__version__,
            "volume_construction": "independent Gmsh surface with matched normal-offset layers"
            if name == "egg"
            else "Gmsh OpenCASCADE volume tetrahedralization",
            "mesh_size": size,
            "mesh_size_units": "unit-sphere coordinates" if name in ("egg", "cavity") else "m",
            "shell_layers": layers if name in ("egg", "cavity") else None,
            "nodes": len(vertices),
            "tetrahedra": len(tets),
            "boundary_triangles": len(boundary),
            "dimensions_m": np.ptp(vertices, axis=0).tolist(),
            "minimum_volume_m3": float(np.min(np.abs(determinant)) / 6),
            "total_volume_m3": float(np.sum(np.abs(determinant)) / 6),
            "scaled_jacobian_percentiles": np.percentile(scaled_jacobian, [0, 1, 50, 99, 100]).tolist(),
            "mesh_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        return metadata
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=[geometry.name for geometry in GEOMETRIES], required=True)
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--shell-layers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_mesh(args.shape, args.size, args.shell_layers, args.output)


if __name__ == "__main__":
    main()
