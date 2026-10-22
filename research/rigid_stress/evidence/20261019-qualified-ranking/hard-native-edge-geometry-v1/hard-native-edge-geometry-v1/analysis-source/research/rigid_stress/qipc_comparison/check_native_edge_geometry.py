"""Check connected native box-edge minima against captured physical material geometry."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.edge_geometry import func_box_edge_minimum
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_MAT, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .study_far_edge_contact import projected_surface


@qd.kernel
def kernel_minima(
    anchors: qd.types.ndarray(),
    inputs: qd.types.ndarray(),
    boundaries: qd.types.ndarray(),
    faces: qd.Tensor,
    positions: qd.Tensor,
    bases: qd.Tensor,
    derivatives: qd.Tensor,
    status: qd.Tensor,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_q in range(anchors.shape[0]):
        center = qd.Vector([inputs[i_q, 0, i] for i in qd.static(range(3))])
        root = qd.Vector([inputs[i_q, 1, i] for i in qd.static(range(3))])
        normal = qd.Vector([inputs[i_q, 2, i] for i in qd.static(range(3))])
        tangent = qd.Vector([inputs[i_q, 3, i] for i in qd.static(range(3))])
        planes = qd.Matrix([[boundaries[i_q, j, i] for i in range(3)] for j in range(4)])
        face, point, bary, basis, slope, curvature, residual, code = func_box_edge_minimum(
            anchors[i_q], center, root, normal, tangent, planes, info, surface, epsilon
        )
        faces[i_q], positions[i_q], bases[i_q] = face, point, basis
        derivatives[i_q], status[i_q] = qd.Vector([slope, curvature, residual]), code


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--references", nargs=2, type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(p) for p in [args.frame, args.boundaries, args.mesh, *args.references]],
            "targets": {
                "reference_position_m": 2e-8,
                "reference_geometry_relative": 2e-4,
                "central_difference_relative": 5e-6,
            },
            "scope": "Native connected quadratic-surface edge minimum and analytic height derivatives. Compare independent global-face projection and scalar minimization, direct central differences, duplicated collision samples, rotated/translated geometry and mixed invalid/corner/interior slots. Geometry only; no physical pressure, load allocation, full stress or scene throughput claim.",
        },
    )
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.boundaries) as saved:
        bounds = {key: saved[key] for key in saved.files}
    references = [json.loads(p.read_text()) for p in args.references]
    active = frame["contact_valid"][0] & (np.linalg.norm(frame["contact_force_n"][0], axis=1) > 0)
    contacts = np.flatnonzero(active & (bounds["count"][0] == 4))
    rotation = Rotation.from_quat(frame["frame_quaternion"][0][[1, 2, 3, 0]])
    origin = frame["frame_position_m"][0]
    nodes = frame["material_nodes_m"][frame["material_surface_nodes"].T]
    anchors, inputs, planes, expected, expected_derivatives = [], [], [], [], []
    for contact in contacts:
        normal, tangent = rotation.inv().apply([bounds["normal"][0, contact], bounds["tangent"][0, contact]])
        plane = np.stack([tangent, np.cross(normal, tangent)])
        witness = frame["source_witness_m"][0, contact]
        seed, _, anchor = projected_surface(
            nodes, frame["material_face_normal"], witness[None], plane, normal, witness, return_geometry=True
        )
        anchors.append(anchor[0])
        inputs.append([seed[0], rotation.inv().apply(frame["region_center_m"][0, contact] - origin), normal, tangent])
        planes.append(bounds["planes"][0, contact])
        reference = next(row for row in references if contact in row.get("contacts", [row["contact"]]))
        expected.append(rotation.inv().apply(np.array(reference["edge_minimum_world_m"]) - origin))
        expected_derivatives.append([reference["edge_slope"], reference["edge_along_curvature_per_m"]])
    anchors, inputs, planes = np.array(anchors), np.array(inputs), np.array(planes)
    expected, expected_derivatives = np.array(expected), np.array(expected_derivatives)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    anchors = anchors.astype(gs.np_int)
    count = len(anchors)
    faces = V(dtype=gs.qd_int, shape=(count,))
    points = V_VEC(3, dtype=gs.qd_float, shape=(count,))
    bases = V_MAT(3, 3, dtype=gs.qd_float, shape=(count,))
    derivatives = V_VEC(3, dtype=gs.qd_float, shape=(count,))
    codes = V(dtype=gs.qd_int, shape=(count,))
    rows = []
    for is_rotated in [False, True]:
        transform = Rotation.from_rotvec([0.3, -0.6, 0.8]) if is_rotated else Rotation.identity()
        translation = np.array([0.04, -0.03, 0.02]) if is_rotated else np.zeros(3)
        asset = args.mesh
        if is_rotated:
            with np.load(args.mesh) as saved:
                payload = {key: saved[key] for key in saved.files}
            for key in ["vertices", "edge_positions"]:
                payload[key] = transform.apply(payload[key]) + translation
            asset = args.output / "rotated.npz"
            np.savez_compressed(asset, **payload)
        model = StressModel(RigidStressOptions(mesh=asset, contact_model="hertz", method="direct"))
        surface = StressSurface(4, model.info)
        np.testing.assert_array_equal(array(model.info.surface_nodes), frame["material_surface_nodes"])
        current = transform.apply(inputs.reshape(-1, 3)).reshape(inputs.shape)
        current[:, :2] += translation
        started = perf_counter()
        kernel_minima(
            anchors,
            current,
            planes,
            faces,
            points,
            bases,
            derivatives,
            codes,
            model.info,
            surface.info,
            float(np.finfo(gs.np_float).eps),
        )
        qd.sync()
        compilation = perf_counter() - started
        actual, actual_basis, actual_derivatives, status = array(points), array(bases), array(derivatives), array(codes)
        np.savez_compressed(
            args.output / f"geometry-rotated{int(is_rotated)}.npz",
            inputs=current,
            anchors=anchors,
            planes=planes,
            positions=actual,
            bases=actual_basis,
            derivatives=actual_derivatives,
            status=status,
            faces=array(faces),
        )
        np.testing.assert_array_equal(status, 0)
        position_error = np.linalg.norm(actual - (transform.apply(expected) + translation), axis=1)
        relative = np.abs(actual_derivatives[:, :2] / expected_derivatives - 1)
        assert position_error.max() < 2e-8, position_error
        assert relative.max() < 2e-4, relative
        differences = []
        for index in range(count):
            # An independent whole-mesh projection supplies central differences.
            basis, point = actual_basis[index], actual[index]
            step = 1e-6
            xy = np.array([[step, 0], [-step, 0], [0, step], [0, -step], [0, 0]])
            targets = point + xy @ basis[:, :2].T
            rotated_nodes = transform.apply(nodes.reshape(-1, 3)).reshape(nodes.shape) + translation
            projected = projected_surface(
                rotated_nodes,
                transform.apply(frame["material_face_normal"]),
                targets,
                basis[:, :2].T,
                basis[:, 2],
                point,
            )
            height = -(projected - projected[-1]) @ basis[:, 2]
            central = np.array([(height[0] - height[1]) / (2 * step), (height[2] + height[3]) / step**2])
            differences.append(np.abs(actual_derivatives[index, :2] / central - 1))
        assert np.max(differences) < 5e-6, differences
        invalid_anchors = anchors.copy()
        invalid_anchors[0] = -1
        kernel_minima(
            invalid_anchors,
            current,
            planes,
            faces,
            points,
            bases,
            derivatives,
            codes,
            model.info,
            surface.info,
            float(np.finfo(gs.np_float).eps),
        )
        assert array(codes)[0] == 2 and array(faces)[0] == -1
        np.testing.assert_array_equal(array(points)[1:], actual[1:])
        mixed = planes.copy()
        mixed[0, :, 2] = np.abs(mixed[0, :, 2])
        positive = np.flatnonzero(mixed[1, :, 2] > 0)[0]
        mixed[1, positive, 2] = -mixed[1, positive, 2]
        kernel_minima(
            anchors,
            current,
            mixed,
            faces,
            points,
            bases,
            derivatives,
            codes,
            model.info,
            surface.info,
            float(np.finfo(gs.np_float).eps),
        )
        np.testing.assert_array_equal(array(codes)[:2], [1, 6])
        np.testing.assert_array_equal(array(points)[2:], actual[2:])
        kernel_minima(
            anchors,
            current,
            planes,
            faces,
            points,
            bases,
            derivatives,
            codes,
            model.info,
            surface.info,
            float(np.finfo(gs.np_float).eps),
        )
        np.testing.assert_array_equal(array(codes), 0)
        np.testing.assert_array_equal(array(points), actual)
        rows.append(
            {
                "is_rotated": is_rotated,
                "queries": count,
                "maximum_position_error_m": float(position_error.max()),
                "maximum_reference_geometry_relative": float(relative.max()),
                "maximum_central_difference_relative": float(np.max(differences)),
                "maximum_stationarity_residual": float(actual_derivatives[:, 2].max()),
                "initial_compile_and_call_s": compilation,
                "invalid_anchor_interior_corner_isolation_and_restoration": True,
            }
        )
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
