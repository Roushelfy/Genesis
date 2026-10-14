"""Construct a fixed-hand two-jaw URDF from qualified actual Panda geometry."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .audit_franka_reference import rotation
from .summarize_fine_references import closed_source


def vector_text(values):
    return " ".join(format(float(value), ".17g") for value in values)


def fit_rigid(source, target):
    centre_source, centre_target = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - centre_source).T @ (target - centre_target))
    orientation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = centre_target - orientation @ centre_source
    error = np.max(np.linalg.norm(source @ orientation.T + translation - target, axis=1))
    return orientation, translation, float(error)


def origin(parent, position, orientation):
    ET.SubElement(
        parent, "origin", xyz=vector_text(position), rpy=vector_text(Rotation.from_matrix(orientation).as_euler("xyz"))
    )


def build_fixture(run: Path, output: Path):
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    report = json.loads((run / "results/summary.json").read_text())
    row = [frame for frame in report["captures"] if frame["mode"] == "stationary_lift_hold"][-1]
    output.mkdir(parents=True, exist_ok=False)
    robot = ET.Element("robot", name="captured_panda_fixed_hand")
    ET.SubElement(robot, "link", name="fixed_hand")
    hand_rotation = rotation(row["hand_quaternion_wxyz"])
    hand_position = np.array(row["hand_position_m"])
    records = []
    expected_open = {}
    xml = ET.parse(run / "source/genesis/assets/xml/franka_emika_panda/panda.xml")
    with np.load(run / "results" / row["geometry_snapshot"]) as world:
        for index, finger in enumerate(row["fingers"]):
            name = ("left_finger", "right_finger")[index]
            link = ET.SubElement(robot, "link", name=name)
            source_inertial = xml.find(f".//body[@name='{name}']/inertial")
            assert source_inertial is not None
            inertial = ET.SubElement(link, "inertial")
            centre = np.fromstring(source_inertial.attrib["pos"], sep=" ")
            origin(inertial, centre, np.eye(3))
            mass = float(source_inertial.attrib["mass"])
            assert mass == row["finger_masses_kg"][index]
            ET.SubElement(inertial, "mass", value=str(mass))
            inertia = np.fromstring(source_inertial.attrib["diaginertia"], sep=" ")
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(inertia[0]),
                iyy=str(inertia[1]),
                izz=str(inertia[2]),
                ixy="0",
                ixz="0",
                iyz="0",
            )
            finger_rotation = rotation(finger["world_quaternion_wxyz"])
            finger_position = np.array(finger["world_position_m"])
            axis = finger_rotation @ [0, 1, 0]
            joint_rotation = hand_rotation.T @ finger_rotation
            zero_position = hand_rotation.T @ (finger_position - hand_position - row["finger_position_m"][index] * axis)
            joint = ET.SubElement(robot, "joint", name=f"finger_joint{index + 1}", type="prismatic")
            ET.SubElement(joint, "parent", link="fixed_hand")
            ET.SubElement(joint, "child", link=name)
            origin(joint, zero_position, joint_rotation)
            ET.SubElement(joint, "axis", xyz="0 1 0")
            ET.SubElement(joint, "limit", lower="0", upper="0.04", effort="4", velocity="0.2")
            for geom in report["geometry"]:
                if geom["link"] != finger["link"]:
                    continue
                target_world = world[f"geom{geom['geom']}_world_vertices_m"]
                target = (target_world - finger_position) @ finger_rotation
                with np.load(run / "results" / geom["mesh"]) as local:
                    source, triangles = local["vertices_m"], local["triangles"]
                orientation, translation, error = fit_rigid(source, target)
                assert error < 1e-10, (geom["geom"], error)
                collision = ET.SubElement(link, "collision", name=f"captured_geom{geom['geom']}")
                origin(collision, translation, orientation)
                geometry = ET.SubElement(collision, "geometry")
                if geom["type"] == "GEOM_TYPE.BOX":
                    sizes = np.array(geom["data"][:3])
                    np.testing.assert_allclose(np.ptp(source, axis=0), sizes, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(source.mean(axis=0), 0, rtol=0, atol=1e-10)
                    ET.SubElement(geometry, "box", size=vector_text(sizes))
                else:
                    assert geom["type"] == "GEOM_TYPE.MESH" and geom["is_convex"]
                    filename = f"captured_geom{geom['geom']}.obj"
                    with (output / filename).open("w") as stream:
                        for vertex in source:
                            stream.write("v " + vector_text(vertex) + "\n")
                        for triangle in triangles:
                            stream.write("f " + " ".join(str(int(v) + 1) for v in triangle) + "\n")
                    ET.SubElement(geometry, "mesh", filename=filename)
                expected_open[f"geom{geom['geom']}_world_vertices_m"] = (
                    target_world + (0.04 - row["finger_position_m"][index]) * axis
                )
                records.append(
                    {
                        "source_geom": geom["geom"],
                        "link_name": name,
                        "type": geom["type"],
                        "rigid_fit_max_error_m": error,
                    }
                )
    assert len(records) == 12
    ET.ElementTree(robot).write(output / "fixed_hand.urdf", encoding="utf-8", xml_declaration=True)
    np.savez_compressed(output / "expected_open_geometry.npz", **expected_open)
    manifest = {
        "source_capture": str(run),
        "source_snapshot": row["geometry_snapshot"],
        "geometry": records,
        "hand_position_m": row["hand_position_m"],
        "hand_quaternion_wxyz": row["hand_quaternion_wxyz"],
        "protocol_difference_from_full_application": "The hand is fixed; the original seven arm PD joints and soft finger-joint equality are omitted in this corresponding controlled fixture. Both jaws retain their own PD actuation and actual compound collision geometry. This is not full application equivalence.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return row, manifest
