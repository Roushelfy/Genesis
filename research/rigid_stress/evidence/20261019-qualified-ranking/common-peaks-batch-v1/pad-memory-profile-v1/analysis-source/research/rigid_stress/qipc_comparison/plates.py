"""Declare a common force history and finite parallel compression plates."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mesh", type=Path, required=True)
    result.add_argument("--loads", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    result.add_argument("--dt", type=float, default=0.001)
    result.add_argument("--ramp", type=float, default=0.1)
    result.add_argument("--hold", type=float, default=0.2)
    result.add_argument("--damping", type=float, default=50.0)
    result.add_argument("--clearance", type=float, default=2.2e-5)
    result.add_argument("--plate-offset-x", type=float, default=0.0)
    result.add_argument("--shear-distance", type=float, default=0.0)
    result.add_argument("--shear-start", type=float, default=0.3)
    result.add_argument("--shear-duration", type=float, default=0.1)
    result.add_argument("--record-stride", type=int, default=10)
    result.add_argument("--empty", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


def force_history(args):
    ramp, hold = round(args.ramp / args.dt), round(args.hold / args.dt)
    assert ramp > 0 and hold > 0 and args.record_stride > 0
    assert abs(ramp * args.dt - args.ramp) < 1e-12
    assert abs(hold * args.dt - args.hold) < 1e-12
    previous = 0.0
    for level, load in enumerate(args.loads):
        assert load >= 0
        for step in range(ramp + hold):
            force = previous + (load - previous) * min((step + 1) / ramp, 1.0)
            yield level, step, force, step == ramp + hold - 1
        previous = load


def shear_target(args, step):
    assert args.shear_duration > 0
    return args.shear_distance * min(max((step * args.dt - args.shear_start) / args.shear_duration, 0), 1)


def configure_native_plate(plate, args):
    plate.set_dofs_kp([0.0, 1e5] if args.shear_distance else 0.0)
    plate.set_dofs_kv(0.0)
    plate.set_dofs_damping([args.damping, 150.0] if args.shear_distance else args.damping)


def control_native_plate(plate, args, force, step):
    if args.shear_distance:
        plate.control_dofs_position(shear_target(args, step), dofs_idx_local=[1])
        plate.control_dofs_force(force, dofs_idx_local=[0])
    else:
        plate.control_dofs_force(force)


def plate_xml(lower, upper, offset_x=0.0, shear=False):
    """Return fixed lower and axially translating upper plates in one MJCF."""
    model = ET.Element("mujoco", model="parallel_plates")
    ET.SubElement(model, "compiler", angle="radian", inertiafromgeom="false" if shear else "true")
    world = ET.SubElement(model, "worldbody")
    base = ET.SubElement(world, "body", name="lower", pos=f"{offset_x:.17g} 0 {lower - 0.005:.17g}")
    ET.SubElement(
        base, "inertial", pos="0 0 0", mass="0.05", diaginertia="0.00006041666666666667 0.00006041666666666667 0.00012"
    )
    ET.SubElement(base, "geom", name="lower_plate", type="box", size="0.06 0.06 0.005", mass="0.05", group="3")
    parent = base
    if shear:
        parent = ET.SubElement(base, "body", name="normal_carriage", pos=f"0 0 {upper - lower + 0.01:.17g}")
        ET.SubElement(
            parent,
            "inertial",
            pos="0 0 0",
            mass="0.001",
            diaginertia="1e-8 1e-8 1e-8",
        )
        ET.SubElement(
            parent,
            "geom",
            name="carriage",
            type="box",
            pos="0.2 0 0",
            size="0.0005 0.0005 0.0005",
            mass="0.001",
            group="3",
        )
        ET.SubElement(parent, "joint", name="compression", type="slide", axis="0 0 -1", limited="false")
    moving = ET.SubElement(parent, "body", name="upper", pos="0 0 0" if shear else f"0 0 {upper - lower + 0.01:.17g}")
    ET.SubElement(
        moving,
        "inertial",
        pos="0 0 0",
        mass="0.05",
        diaginertia="0.00006041666666666667 0.00006041666666666667 0.00012",
    )
    ET.SubElement(
        moving,
        "joint",
        name="shear" if shear else "compression",
        type="slide",
        axis="1 0 0" if shear else "0 0 -1",
        limited="false",
    )
    ET.SubElement(moving, "geom", name="upper_plate", type="box", size="0.06 0.06 0.005", mass="0.05", group="3")
    return ET.tostring(model, encoding="unicode")


def equilibrium(row, target):
    """Apply declared engineering equilibrium gates to actual mechanical observations."""
    scale = max(target, 1e-3)
    return {
        "both_plate_reactions": max(abs(row["upper_reaction_n"] - target), abs(row["lower_reaction_n"] - target))
        <= 0.01 * scale,
        "plate_velocity": abs(row["plate_speed_m_per_s"]) <= 1e-6,
        "egg_inertial_resultant": row["egg_inertia_resultant_norm_n"] <= 0.01 * scale,
        "egg_kinetic": row["egg_kinetic_j"] <= 1e-10,
        "equation_residual": row["equation_residual_relative"] <= 1e-3,
    }
