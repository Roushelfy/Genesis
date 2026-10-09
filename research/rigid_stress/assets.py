"""Generate the exterior rigid mesh and an explicit hollow-shell URDF inertia."""

import argparse
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU


def write_egg_assets(model, destination):
    destination.mkdir(parents=True, exist_ok=True)
    f = model.fem
    outer = f.outer_faces.copy()
    vertices = f.base_xyz[: model.mesh_metadata["outer_surface_vertices"]]
    triangles = vertices[outer]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.einsum("ij,ij->i", normals, triangles.mean(axis=1) - f.com) < 0
    outer[inward] = outer[inward][:, [0, 2, 1]]
    obj = destination / "egg_outer.obj"
    with obj.open("w") as output:
        for x, y, z in vertices:
            output.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in outer + 1:
            output.write(f"f {a} {b} {c}\n")
    # The collision exterior encloses solid volume, but its rigid mass/inertia
    # must describe only the shell wall. G uses consistent P2 volume mass.
    inertia = f.gram[3:, 3:]
    urdf = destination / "egg_shell.urdf"
    robot = Element("robot", name="egg_shell")
    link = SubElement(robot, "link", name="egg")
    inertial = SubElement(link, "inertial")
    SubElement(inertial, "origin", xyz=" ".join(f"{v:.17g}" for v in f.com), rpy="0 0 0")
    SubElement(inertial, "mass", value=f"{f.mass:.17g}")
    SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.17g}",
        ixy=f"{inertia[0, 1]:.17g}",
        ixz=f"{inertia[0, 2]:.17g}",
        iyy=f"{inertia[1, 1]:.17g}",
        iyz=f"{inertia[1, 2]:.17g}",
        izz=f"{inertia[2, 2]:.17g}",
    )
    for kind in ("visual", "collision"):
        part = SubElement(link, kind)
        SubElement(SubElement(part, "geometry"), "mesh", filename="egg_outer.obj")
        if kind == "visual":
            SubElement(SubElement(part, "material", name="shell"), "color", rgba="0.90 0.76 0.57 1")
    ElementTree(robot).write(urdf, encoding="utf-8", xml_declaration=True)
    return urdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(config=EggConfig(level=args.level))
        print(write_egg_assets(model, args.output))


if __name__ == "__main__":
    main()
