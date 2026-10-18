"""Preserve authored Franka assets while selecting the accepted bank's initial keyframe."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .artifacts import file_id, write_json


KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100.0])
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10.0])
CAP = np.array([87, 87, 87, 87, 12, 12, 12, 4, 4.0])
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]


def geom_defaults(element, inherited, classes):
    attributes = inherited.copy()
    geometry = element.find("geom")
    if geometry is not None:
        attributes.update(geometry.attrib)
    classes[element.attrib.get("class", "main")] = attributes
    for child in element.findall("default"):
        geom_defaults(child, attributes, classes)


def expand_geoms(element, inherited_class, classes):
    active_class = element.attrib.get("childclass", inherited_class)
    for geometry in element.findall("geom"):
        attributes = classes[geometry.attrib.get("class", active_class)].copy()
        attributes.update(geometry.attrib)
        if "type" not in attributes and "mesh" in attributes:
            attributes["type"] = "mesh"
        geometry.attrib.update(attributes)
    for child in element.findall("body"):
        expand_geoms(child, active_class, classes)


def prepare_mjcf(original, qpos, output):
    original = Path(original).resolve()
    root = ET.parse(original)
    compiler = root.find("compiler")
    meshdir = (original.parent / compiler.attrib.get("meshdir", "")).resolve()
    compiler.set("meshdir", str(meshdir))
    classes = {"main": {}}
    geom_defaults(root.find("default"), {}, classes)
    expand_geoms(root.find("worldbody"), "main", classes)
    keyframes = root.find("keyframe")
    for element in list(keyframes):
        keyframes.remove(element)
    ET.SubElement(keyframes, "key", {"name": "accepted-bank-initial", "qpos": " ".join(map(str, qpos))})
    destination = output / "panda-bank-initial.xml"
    root.write(destination, encoding="unicode")
    assets = sorted({meshdir / element.attrib["file"] for element in root.findall("asset/mesh")})
    write_json(
        output / "robot-inputs.json",
        {
            "original_xml": file_id(original),
            "generated_xml": file_id(destination),
            "mesh_inputs": [file_id(path) for path in assets],
            "changes": "Resolve compiler meshdir without changing mesh bytes; select bank initial qpos keyframe; explicitly materialize inherited geom attributes. This preserves authored MJCF semantics while avoiding the original QIPC reader's omission of inherited primitive size/pose. Collision shapes, inertials, equality and actuator definitions require canonical MuJoCo equivalence checks.",
        },
    )
    return destination


def robot_target(bank, case, tick):
    phase = max(tick - bank["delays"][case], 0) % 600 / 599
    ramp = np.clip(np.array([phase / 0.15, (phase - 0.3) / 0.2, (phase - 0.6) / 0.15]), 0, 1)
    approach, lift, slide = ramp**2 * (3 - 2 * ramp)
    initial, pick, raised, lateral = (
        bank[key][case] for key in ("initial_joints", "pick_joints", "lift_joints", "slide_joints")
    )
    target = initial * (1 - approach) + pick * approach + (raised - pick) * lift + (lateral - raised) * slide
    target[7:] = 0.04 if phase < 0.15 or phase > 0.9 else 0.018
    cap = CAP.copy()
    cap[7:] = 0.055 if 0.7 <= phase < 0.76 else 4
    return target, cap, phase
