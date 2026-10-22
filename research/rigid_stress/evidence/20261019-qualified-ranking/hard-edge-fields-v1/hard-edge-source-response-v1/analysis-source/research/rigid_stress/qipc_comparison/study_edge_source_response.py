"""Measure a captured source-line shift with native curved-shell and edge particular fields."""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge_fields import StressEdgeFields
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.hertz_tangential import StressHertzShear
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_MAT, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_curved_shell_operators import locate_curved
from .check_native_edge_fields import kernel_evaluate
from .check_native_hertz_recovery import array, matrix
from .edge_asymptotic import edge_scales
from .study_edge_fields import field_metrics
from .study_far_edge_contact import projected_surface
from .study_hertz_residual import field_difference
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--physical-inputs", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame, args.boundaries, args.mesh, args.physical_inputs]],
            "targets": {
                "source_response_volume": 0.05,
                "source_response_sample_tensor": 0.05,
                "source_response_sample_peak": 0.10,
                "boundary_quadrature_field_change": 0.01,
            },
            "scope": "Each captured edge source separately: local pressure-centroid tangent plane versus the rigid-force line intersecting the same curved material. The pressure shape is transported without refitting its exponent. Native all-boundary point correction plus curved P2 elastic recovery and the new native edge particular field form a declared complete-field approximation. The translated profile is a wrench-compatible counterfactual, not a contact solution. Per-element volume quadrature is coarse near the contact; separate common near-field probes retain that limitation. This does not authorize a relaxed input gate or qualify full physical shell/contact errors.",
        },
    )
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    with np.load(args.frame) as data:
        frame = {key: data[key] for key in data.files}
    with np.load(args.boundaries) as data:
        bounds = {key: data[key] for key in data.files}
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    vertices = array(model.info.vertices)
    np.testing.assert_allclose(vertices, frame["material_nodes_m"], atol=1e-14, rtol=0)
    elements = array(model.info.elements).T
    volume_nodes = vertices[elements]
    surface_nodes = frame["material_surface_nodes"].T
    face_nodes = vertices[surface_nodes]
    quaternion = frame["frame_quaternion"][0]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    origin = frame["frame_position_m"][0]
    cases = []
    for record in json.loads(args.physical_inputs.read_text())["cases"]:
        if record["direction"] != 1:
            continue
        contact = record["contact"]
        source = json.loads(Path(record["sources"][0]["path"]).read_text())
        law = edge_scales(*record["inputs"])
        normal, tangent = bounds["normal"][0, contact], bounds["tangent"][0, contact]
        planes = bounds["planes"][0, contact]
        edge = (planes[:, 2] / np.linalg.norm(planes[:, :2], axis=1)).argmin()
        inward = -planes[edge, :2] / np.linalg.norm(planes[edge, :2])
        box_world = np.column_stack((tangent, np.cross(normal, tangent))) @ np.column_stack(
            (inward, [-inward[1], inward[0]])
        )
        box_basis = rotation.inv().apply(box_world.T).T
        normal_material = rotation.inv().apply(normal)
        edge_origin = rotation.inv().apply(np.array(source["edge_minimum_world_m"]) - origin)
        force = frame["contact_force_n"][0, contact]
        witness = frame["source_witness_m"][0, contact]
        target = edge_origin + box_basis[:, 0] * (2 * (1 - law.exponent) * law.width / 5)
        force_axis = force / np.linalg.norm(force)
        first = np.cross(force_axis, box_basis[:, 1])
        first /= np.linalg.norm(first)
        force_plane = np.stack([first, np.cross(force_axis, first)])
        for kind, point, plane in [
            ("pressure", target, box_basis.T),
            ("rigid_line", frame["contact_position_m"][0, contact], force_plane),
        ]:
            centers, normals, anchors = projected_surface(
                face_nodes,
                frame["material_face_normal"],
                point[None],
                plane,
                normal_material,
                witness,
                return_geometry=True,
            )
            center, material_normal = centers[0], normals[0]
            axis = box_basis[:, 0] - material_normal * (material_normal @ box_basis[:, 0])
            axis /= np.linalg.norm(axis)
            basis = np.column_stack((axis, np.cross(material_normal, axis), -material_normal))
            local_force = force @ basis
            assert local_force[2] > 0
            width = law.width / (axis @ box_basis[:, 0])
            amplitude = (
                3 * local_force[2] * np.sin(np.pi * law.exponent) / (4 * np.pi * law.exponent * width * law.along)
            )
            difference = np.cross(center - frame["contact_position_m"][0, contact], force)
            cases.append(
                {
                    "contact": contact,
                    "kind": kind,
                    "center": center,
                    "basis": basis,
                    "anchor": int(anchors[0]),
                    "force": local_force,
                    "width": width,
                    "along": law.along,
                    "exponent": law.exponent,
                    "amplitude": amplitude,
                    "moment_difference_nm": difference.tolist(),
                    "moment_difference_over_Fa": float(
                        np.linalg.norm(difference) / (record["inputs"][0] * source["hertz_axes_m"][0])
                    ),
                }
            )
    write_json(
        args.output / "source-geometry.json",
        {"cases": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()} for c in cases]},
    )
    count = len(cases)
    law = StressHertz(count)
    law.state.force.from_numpy(np.array([c["force"][2] for c in cases]))
    law.state.axes.from_numpy(np.array([[c["along"], c["width"]] for c in cases]))
    law.state.direction.from_numpy(np.tile([1.0, 0.0], (count, 1)))
    law.state.peak_pressure.from_numpy(np.array([c["amplitude"] for c in cases]))
    law.state.status.fill(0)
    shear = StressHertzShear(
        friction=V(dtype=gs.qd_float, shape=(count,)),
        force=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
        coefficients=V_VEC(2, dtype=gs.qd_float, shape=(16, count)),
    )
    shear.force.from_numpy(np.array([c["force"][:2] for c in cases]))
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(count,)), V(dtype=gs.qd_int, shape=(count,))
    omega.fill(0)
    knots, measures = np.polynomial.legendre.leggauss(3)
    knots, measures = (knots + 1) / 2, measures / 2
    u, v, w = np.array(np.meshgrid(knots, knots, knots, indexing="ij")).reshape(3, -1)
    wu, wv, ww = np.array(np.meshgrid(measures, measures, measures, indexing="ij")).reshape(3, -1)
    local = np.array([u, (1 - u) * v, (1 - u) * (1 - v) * w])
    weights = wu * wv * ww * (1 - u) ** 2 * (1 - v)
    bary = np.column_stack([1 - local.sum(axis=0), local.T])
    shape, gradients, _ = quadratic_basis(bary)
    volume_points = np.einsum("qn,tni->tqi", shape, volume_nodes)
    jacobian = np.einsum("qna,tni->tqia", gradients, volume_nodes - volume_nodes[:, :1])
    volumes = (np.abs(np.linalg.det(jacobian)) * weights).reshape(-1)
    volume_gradient = gradients[None] @ np.linalg.inv(jacobian)
    probe_points = []
    for c in cases:
        x = np.array([-1, 0, 0.25, 0.75, 1.5]) * c["width"] - 2 * (1 - c["exponent"]) * c["width"] / 5
        y = np.array([-1.2, -0.5, 0, 0.5, 1.2]) * c["along"]
        depth = np.array([0.1 * c["width"], c["width"], 0.1 * c["along"]])
        offset = np.stack(np.meshgrid(x, y, depth, indexing="ij"), axis=-1).reshape(-1, 3)
        target = c["center"] + offset[:, :2] @ c["basis"][:, :2].T
        surface = projected_surface(
            face_nodes, frame["material_face_normal"], target, c["basis"][:, :2].T, -c["basis"][:, 2], c["center"]
        )
        probe_points.append(surface + offset[:, 2, None] * c["basis"][:, 2])
    probes = np.concatenate(probe_points)
    owners, barys, valid, _ = locate_curved(volume_nodes, probes)
    assert valid.all(), np.flatnonzero(~valid).tolist()
    _, gradient, _ = quadratic_basis(barys)
    jacobian = np.einsum("qna,qni->qia", gradient, volume_nodes[owners] - volume_nodes[owners, :1])
    probe_gradient = gradient @ np.linalg.inv(jacobian)
    common = np.concatenate([volume_points.reshape(-1, 3), probes])
    inputs, positions = [], []
    for c in cases:
        local = (common - c["center"]) @ c["basis"]
        assert local[:, 2].min() >= -1e-12
        local[:, 2] = np.maximum(local[:, 2], 0)
        local[:, 0] += 2 * (1 - c["exponent"]) * c["width"] / 5
        positions.extend(local)
        inputs.extend(
            [
                [
                    c["width"],
                    c["along"],
                    c["exponent"],
                    c["amplitude"],
                    c["force"][2],
                    *(c["force"][:2] / c["force"][2]),
                    0.3,
                ]
            ]
            * len(common)
        )
    point_bank, input_bank = (
        V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
        V_VEC(8, dtype=gs.qd_float, shape=(len(positions),)),
    )
    tensor_bank, status_bank = (
        V_MAT(3, 3, dtype=gs.qd_float, shape=(len(positions),)),
        V(dtype=gs.qd_int, shape=(len(positions),)),
    )
    point_bank.from_numpy(np.array(positions))
    input_bank.from_numpy(np.array(inputs))
    field = StressEdgeFields()
    kernel_evaluate(point_bank, input_bank, field.quadrature, tensor_bank, status_bank)
    np.testing.assert_array_equal(array(status_bank), 0)
    particular = array(tensor_bank).reshape(count, len(common), 3, 3)
    for i, c in enumerate(cases):
        particular[i] = c["basis"] @ particular[i] @ c["basis"].T
    rows, results = [], []
    for order in [16, 24]:
        surface = StressSurface(order, model.info)
        module = StressHertzRecovery(count, 1, output_mode="full", shear=shear)
        module.frames.center.from_numpy(np.array([c["center"] for c in cases]))
        module.frames.basis.from_numpy(np.array([c["basis"] for c in cases]))
        module.frames.anchor.from_numpy(np.array([c["anchor"] for c in cases], dtype=gs.np_int))
        state = model.create_state(count, "full")
        kernel_begin_step(state)
        errno.fill(0)
        module.scatter(law.state, state, model.info, surface.info, 0.3, 1e-12, errno)
        admission = {
            "order": order,
            "status": array(module.state.status).tolist(),
            "errno": array(errno).tolist(),
            "wrench": array(module.state.wrench_error).tolist(),
        }
        write_json(args.output / f"admission{order}.json", admission)
        np.testing.assert_array_equal(array(module.state.status), 0)
        np.testing.assert_array_equal(array(errno), 0)
        model.recover(omega, state, surface_load=True, compute_peak=True)
        qd.sync()
        displacement = array(state.displacement)
        derivative = np.einsum(
            "btni,tqnj->btqij", displacement[:, elements] - displacement[:, elements[:, :1]], volume_gradient
        )
        probe_derivative = np.einsum(
            "bqni,qnj->bqij", displacement[:, elements[owners]] - displacement[:, elements[owners, :1]], probe_gradient
        )
        derivative = np.concatenate([derivative.reshape(count, -1, 3, 3), probe_derivative], axis=1)
        shear_modulus, lame = 1e10 / 2.6, 1e10 * 0.3 / (1.3 * 0.4)
        combined = (
            particular
            + shear_modulus * (derivative + derivative.swapaxes(-1, -2))
            + lame * np.trace(derivative, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
        )
        np.savez_compressed(
            args.output / f"fields{order}.npz",
            points_m=common,
            volume_weights_m3=volumes,
            particular_pa=particular,
            complete_pa=combined,
            displacement_m=displacement,
            point_correction_forces_n=array(state.force),
            corner_correction_pa=matrix(array(state.stress_tensor)),
        )
        results.append(combined)
        comparisons = []
        for i in [0, 2]:
            comparisons.append(
                {
                    "contact": cases[i]["contact"],
                    "volume_tensor_relative": float(
                        field_difference(combined[i + 1, : len(volumes)], combined[i, : len(volumes)], volumes)
                    ),
                    "common_probes": field_metrics(combined[i + 1, len(volumes) :], combined[i, len(volumes) :]),
                    "all_samples": field_metrics(combined[i + 1], combined[i]),
                    "center_distance_m": float(np.linalg.norm(cases[i + 1]["center"] - cases[i]["center"])),
                }
            )
        rows.append({**admission, "source_response": comparisons})
        write_json(args.output / "progress.json", {"rows": rows})
    convergence = [field_metrics(a, b) for a, b in zip(*results)]
    passed = all(c["tensor_relative"] < 0.01 for c in convergence)
    write_json(
        args.output / "result.json",
        {"passed": passed, "rows": rows, "boundary_quadrature_change": convergence, "whole_goal_complete": False},
    )
    assert passed


if __name__ == "__main__":
    main()
