"""Generate the exterior rigid mesh and an explicit hollow-shell URDF inertia."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU


def write_egg_assets(model, destination):
    destination.mkdir(parents=True, exist_ok=True)
    f = model.fem
    outer = f.outer_faces.copy()
    vertices = f.base_xyz[:model.mesh_metadata["outer_surface_vertices"]]
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
    com = " ".join(f"{v:.17g}" for v in f.com)
    urdf = destination / "egg_shell.urdf"
    urdf.write_text(f'''<?xml version="1.0"?>
<robot name="egg_shell"><link name="egg">
  <inertial><origin xyz="{com}" rpy="0 0 0"/>
    <mass value="{f.mass:.17g}"/>
    <inertia ixx="{inertia[0, 0]:.17g}" ixy="{inertia[0, 1]:.17g}" ixz="{inertia[0, 2]:.17g}"
             iyy="{inertia[1, 1]:.17g}" iyz="{inertia[1, 2]:.17g}" izz="{inertia[2, 2]:.17g}"/>
  </inertial>
  <visual><geometry><mesh filename="egg_outer.obj"/></geometry>
    <material name="shell"><color rgba="0.90 0.76 0.57 1"/></material></visual>
  <collision><geometry><mesh filename="egg_outer.obj"/></geometry></collision>
</link></robot>
''')
    return urdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/rigid_stress/generated_assets"))
    parser.add_argument("--level", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(config=EggConfig(level=args.level))
        print(write_egg_assets(model, args.output))


if __name__ == "__main__":
    main()
