#!/usr/bin/env python3
"""
Apply (x,y,z) → (-z,-x,y) rotation to all STL files in hands/right/,
save as *_modified.stl, and create right_modified.urdf with transformed
joint origins, axes, inertial properties, etc.

Rotation matrix R:
  [[0, 0, -1],
   [-1, 0,  0],
   [0, 1,  0]]

This was identified by comparing right_palm_link.STL with
the Blender-modified right_palm_link_modified.STL.
"""

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# The rotation: (x,y,z) → (-z, -x, y)
R = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def rotate_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply R to a 3D vector: (x,y,z) → (-z, -x, y)."""
    return (-z, -x, y)


def rotate_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    """
    Transform RPY angles under the coordinate rotation R.
    R_new = R @ R_rpy @ R^T, then extract new RPY.
    """
    R_rpy = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    R_new = R @ R_rpy @ R.T
    new_rpy = Rotation.from_matrix(R_new).as_euler("xyz")
    return tuple(new_rpy)


def rotate_inertia(ixx, ixy, ixz, iyy, iyz, izz):
    """
    Rotate inertia tensor by R.
    I_new = R @ I @ R^T

    For R = (x,y,z)→(-z,-x,y):
      ixx_new = izz, ixy_new = ixz, ixz_new = -iyz
      iyy_new = ixx, iyz_new = -ixy, izz_new = iyy
    """
    return {
        "ixx": izz,
        "ixy": ixz,
        "ixz": -iyz,
        "iyy": ixx,
        "iyz": -ixy,
        "izz": iyy,
    }


def read_binary_stl(filepath: Path) -> tuple[np.ndarray, np.ndarray, bytes]:
    """
    Read a binary STL file.
    Returns (normals, vertices, header) where:
      normals: (N, 3) float32
      vertices: (N, 3, 3) float32 - N triangles, 3 vertices each, 3 coords
      header: 80-byte header
    """
    with open(filepath, "rb") as f:
        header = f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]

        normals = np.zeros((num_triangles, 3), dtype=np.float32)
        vertices = np.zeros((num_triangles, 3, 3), dtype=np.float32)

        for i in range(num_triangles):
            data = struct.unpack("<12f", f.read(48))
            normals[i] = data[0:3]
            vertices[i, 0] = data[3:6]
            vertices[i, 1] = data[6:9]
            vertices[i, 2] = data[9:12]
            _ = f.read(2)  # attribute byte count

    return normals, vertices, header


def write_binary_stl(filepath: Path, normals: np.ndarray, vertices: np.ndarray, header: bytes):
    """Write a binary STL file."""
    num_triangles = len(normals)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_triangles))

        for i in range(num_triangles):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *vertices[i, 0]))
            f.write(struct.pack("<3f", *vertices[i, 1]))
            f.write(struct.pack("<3f", *vertices[i, 2]))
            f.write(struct.pack("<H", 0))


def rotate_stl(input_path: Path, output_path: Path):
    """Read STL, apply rotation R to vertices and normals, write output."""
    normals, vertices, header = read_binary_stl(input_path)

    R_f32 = R.astype(np.float32)

    # Rotate normals: (N, 3) @ R^T = (N, 3)
    normals_rot = normals @ R_f32.T
    # Rotate vertices: (N, 3, 3) - rotate last axis
    n_tri = vertices.shape[0]
    verts_flat = vertices.reshape(-1, 3)
    verts_rot = verts_flat @ R_f32.T
    vertices_rot = verts_rot.reshape(n_tri, 3, 3)

    write_binary_stl(output_path, normals_rot, vertices_rot, header)
    print(f"  {input_path.name} → {output_path.name}")


def transform_urdf(input_urdf: Path, output_urdf: Path):
    """
    Create a modified URDF with:
    - Mesh filenames changed to *_modified.stl
    - Joint origin xyz rotated
    - Joint origin rpy transformed
    - Joint axis rotated
    - Inertial origin xyz rotated
    - Inertial origin rpy transformed
    - Inertia tensor rotated
    """
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Update mesh filenames
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            stem = Path(filename).stem
            mesh.set("filename", f"{stem}_modified.stl")

    # Transform joint origins and axes
    for joint in root.findall(".//joint"):
        origin = joint.find("origin")
        if origin is not None:
            # Transform xyz
            xyz_str = origin.get("xyz", "0 0 0")
            x, y, z = map(float, xyz_str.split())
            nx, ny, nz = rotate_xyz(x, y, z)
            origin.set("xyz", f"{nx} {ny} {nz}")

            # Transform rpy
            rpy_str = origin.get("rpy", "0 0 0")
            r, p, ya = map(float, rpy_str.split())
            if abs(r) > 1e-15 or abs(p) > 1e-15 or abs(ya) > 1e-15:
                nr, np_, nya = rotate_rpy(r, p, ya)
                origin.set("rpy", f"{nr} {np_} {nya}")

        # Transform axis
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "0 0 0")
            ax, ay, az = map(float, axis_str.split())
            if abs(ax) > 1e-15 or abs(ay) > 1e-15 or abs(az) > 1e-15:
                nax, nay, naz = rotate_xyz(ax, ay, az)
                axis_elem.set(
                    "xyz",
                    f"{int(nax) if nax == int(nax) else nax} "
                    f"{int(nay) if nay == int(nay) else nay} "
                    f"{int(naz) if naz == int(naz) else naz}",
                )

    # Transform inertial properties
    for link in root.findall(".//link"):
        inertial = link.find("inertial")
        if inertial is not None:
            # Transform inertial origin
            origin = inertial.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                x, y, z = map(float, xyz_str.split())
                nx, ny, nz = rotate_xyz(x, y, z)
                origin.set("xyz", f"{nx} {ny} {nz}")

                rpy_str = origin.get("rpy", "0 0 0")
                r, p, ya = map(float, rpy_str.split())
                if abs(r) > 1e-15 or abs(p) > 1e-15 or abs(ya) > 1e-15:
                    nr, np_, nya = rotate_rpy(r, p, ya)
                    origin.set("rpy", f"{nr} {np_} {nya}")

            # Transform inertia tensor
            inertia = inertial.find("inertia")
            if inertia is not None:
                ixx = float(inertia.get("ixx", "0"))
                ixy = float(inertia.get("ixy", "0"))
                ixz = float(inertia.get("ixz", "0"))
                iyy = float(inertia.get("iyy", "0"))
                iyz = float(inertia.get("iyz", "0"))
                izz = float(inertia.get("izz", "0"))

                new_i = rotate_inertia(ixx, ixy, ixz, iyy, iyz, izz)
                for key, val in new_i.items():
                    inertia.set(key, str(val))

    # Write output
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)


def main():
    base_dir = Path(__file__).parent

    # Find all original STL files (exclude already-modified ones)
    stl_files = sorted(p for p in base_dir.glob("*.stl") if "_modified" not in p.stem)
    print(f"Found {len(stl_files)} STL files to transform.")

    # Rotate each STL
    print("\nRotating STL files ((x,y,z) → (-z,-x,y))...")
    for stl_path in stl_files:
        output_name = stl_path.stem + "_modified.stl"
        output_path = base_dir / output_name
        rotate_stl(stl_path, output_path)

    print(f"\nCreated {len(stl_files)} modified STL files.")

    # Transform URDF
    input_urdf = base_dir / "right.urdf"
    output_urdf = base_dir / "right_modified.urdf"
    print(f"\nTransforming URDF: {input_urdf.name} → {output_urdf.name}")
    transform_urdf(input_urdf, output_urdf)
    print(f"Done! Output: {output_urdf}")


if __name__ == "__main__":
    main()
