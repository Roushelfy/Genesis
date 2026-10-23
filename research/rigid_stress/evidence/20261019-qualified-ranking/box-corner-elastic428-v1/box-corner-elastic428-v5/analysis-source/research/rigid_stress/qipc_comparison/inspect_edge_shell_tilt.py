"""Use retained curved-shell displacement to assess the sharp-edge center discrepancy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions

from .artifacts import file_id, runtime_identity, write_json
from .check_curved_shell_operators import locate_curved
from .check_native_hertz_recovery import array
from .study_quadratic_rest_surface import quadratic_basis


def point_displacement(points, center, basis, force, poisson=0.3, young=1e10):
    position = (points - center) @ basis
    radius = np.sqrt(np.sum(position * position, axis=-1))
    z = position[:, 2]
    height = radius + z
    contrast = 1 - 2 * poisson
    common = z / radius**3 - contrast / (radius * height)
    value = (
        np.column_stack(
            [position[:, 0] * common, position[:, 1] * common, z * z / radius**3 + 2 * (1 - poisson) / radius]
        )
        * force[2]
    )
    dot = position[:, :2] @ force[:2]
    diagonal = 1 / radius + contrast / height
    outer = 1 / radius**3 - contrast / (radius * height**2)
    value[:, :2] += diagonal[:, None] * force[:2] + (outer * dot)[:, None] * position[:, :2]
    value[:, 2] += dot * (z / radius**3 + contrast / (radius * height))
    return value @ basis.T / (4 * np.pi * (young / (2 * (1 + poisson))))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--response", type=Path, required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--physical-inputs", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.response / "source-geometry.json",
        args.response / "fields24.npz",
        args.frame,
        args.boundaries,
        args.mesh,
        args.physical_inputs,
    ]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in paths],
            "scope": "Read retained regular P2 displacements; add each source's nodal point particular displacement to remove its mass-weighted rigid translation/rotation. Estimate the smooth along-edge normal displacement gradient from both gripper sources and the other source's far particular field. This is a first-order shell-tilt diagnostic, not a coupled contact solve. Point particular displacement is used only for the global rigid projection and the distant source; the self singular field is excluded from the smooth correction.",
        },
    )
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    vertices, elements = array(model.info.vertices), array(model.info.elements).T
    nodes = vertices[elements]
    mass_modes, modes, gram_inverse = (
        array(model.info.mass_modes),
        array(model.info.modes),
        array(model.info.gram_inverse),
    )
    cases = json.loads(paths[0].read_text())["cases"]
    with np.load(paths[1]) as saved:
        displacement = saved["displacement_m"]
    with np.load(args.frame) as saved:
        quaternion = saved["frame_quaternion"][0]
    with np.load(args.boundaries) as saved:
        normals, tangents, planes = saved["normal"][0], saved["tangent"][0], saved["planes"][0]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    corrected = []
    for i, case in enumerate(cases):
        center, basis, force = [np.array(case[k]) for k in ["center", "basis", "force"]]
        particular = point_displacement(vertices, center, basis, force)
        coefficients = gram_inverse @ np.einsum("nic,ni->c", mass_modes, displacement[i] + particular)
        rigid = np.einsum("nic,c->ni", modes, coefficients)
        normalized = np.linalg.norm(
            np.einsum("nic,ni->c", mass_modes, displacement[i] + particular - rigid)
        ) / np.linalg.norm(np.einsum("nic,ni->c", mass_modes, displacement[i] + particular))
        assert normalized < 1e-10
        corrected.append(displacement[i] - rigid)
    corrected = np.array(corrected)
    centers = np.array([case["center"] for case in cases])
    owners, bary, valid, _ = locate_curved(nodes, centers)
    assert valid.all()
    _, gradients, _ = quadratic_basis(bary)
    jacobian = np.einsum("qna,qni->qia", gradients, nodes[owners] - nodes[owners, :1])
    gradients = gradients @ np.linalg.inv(jacobian)
    derivative = np.einsum(
        "bqni,qnj->bqij", corrected[:, elements[owners]] - corrected[:, elements[owners, :1]], gradients
    )
    records = json.loads(args.physical_inputs.read_text())["cases"]
    rows = []
    for i, case in enumerate(cases):
        contact = case["contact"]
        edge = (planes[contact, :, 2] / np.linalg.norm(planes[contact, :, :2], axis=1)).argmin()
        inward = -planes[contact, edge, :2] / np.linalg.norm(planes[contact, edge, :2])
        basis2 = np.column_stack([tangents[contact], np.cross(normals[contact], tangents[contact])]) @ np.column_stack(
            [inward, [-inward[1], inward[0]]]
        )
        along = rotation.inv().apply(basis2[:, 1])
        normal = -rotation.inv().apply(normals[contact])
        other = i + 2 if i < 2 else i - 2
        other_case = cases[other]
        far_gradient = np.empty((3, 3))
        for axis in range(3):
            point = centers[i : i + 1].astype(complex)
            point[:, axis] += 1e-30j
            far_gradient[:, axis] = (
                point_displacement(
                    point, np.array(other_case["center"]), np.array(other_case["basis"]), np.array(other_case["force"])
                )[0].imag
                / 1e-30
            )
        gradient = derivative[i, i] + derivative[other, i] + far_gradient
        curvature = next(row["inputs"][3] for row in records if row["contact"] == contact)
        pressure, rigid = centers[(i // 2) * 2], centers[(i // 2) * 2 + 1]
        observed = (rigid - pressure) @ along
        tilt = normal @ gradient @ along
        rows.append(
            {
                "contact": contact,
                "configuration": case["kind"],
                "required_along_shift_m": float(observed),
                "required_gap_tilt_rad": float(curvature * observed),
                "self_regular_tilt_rad": float(normal @ derivative[i, i] @ along),
                "other_total_tilt_rad": float(normal @ (derivative[other, i] + far_gradient) @ along),
                "two_gripper_tilt_rad": float(tilt),
                "predicted_along_shift_m": float(tilt / curvature),
            }
        )
    write_json(
        args.output / "result.json",
        {"passed": True, "rows": rows, "mass_rigid_projection_checked": True, "whole_goal_complete": False},
    )


if __name__ == "__main__":
    main()
