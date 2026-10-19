"""Export an existing declared solid mesh as native rigid-stress assets without changing geometry."""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .prepare_inputs import mass_properties


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        vertices, tetrahedra, outer, faces = (
            saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces")
        )
    volumes = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]) / 6
    assert (volumes > 0).all()
    properties = mass_properties(vertices, tetrahedra, volumes)
    shutil.copy2(args.mesh, args.output / "elastic.npz")
    with (args.output / "egg_outer.obj").open("w") as stream:
        np.savetxt(stream, outer, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")
    robot = ET.Element("robot", name="declared_egg")
    link = ET.SubElement(robot, "link", name="egg")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{value:.17g}" for value in properties["com_m"]), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{properties['mass_kg']:.17g}")
    inertia = np.asarray(properties["exact_volume_inertia_kg_m2"])
    ET.SubElement(
        inertial,
        "inertia",
        **{
            name: f"{inertia[i, j]:.17g}"
            for name, i, j in (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2), ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))
        },
    )
    for tag in ("visual", "collision"):
        geometry = ET.SubElement(ET.SubElement(link, tag), "geometry")
        ET.SubElement(geometry, "mesh", filename="egg_outer.obj")
    ET.ElementTree(robot).write(args.output / "egg_shell.urdf", encoding="utf-8", xml_declaration=True)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "source": file_id(args.mesh),
            "elastic_mesh": file_id(args.output / "elastic.npz"),
            "collision_mesh": file_id(args.output / "egg_outer.obj"),
            "urdf": file_id(args.output / "egg_shell.urdf"),
            **properties,
            "scope": "Unmodified tetrahedral stress mesh and matching outer polygonal collider, with exact volume-integrated mass/COM/inertia at density2000. No mass retuning, geometry fit or physical accuracy admission.",
        },
    )


if __name__ == "__main__":
    main()
