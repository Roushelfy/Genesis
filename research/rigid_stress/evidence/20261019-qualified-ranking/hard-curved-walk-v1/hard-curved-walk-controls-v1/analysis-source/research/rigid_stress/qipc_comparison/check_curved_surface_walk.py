"""Check native projected walks against independent quadratic material coordinates."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.surface_walk import func_projected_surface_walk
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .study_quadratic_rest_surface import quadratic_basis


@qd.kernel
def kernel_walk(
    anchors: qd.types.ndarray(),
    centers: qd.types.ndarray(),
    targets: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    faces: qd.Tensor,
    points: qd.Tensor,
    coordinates: qd.Tensor,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_q in range(anchors.shape[0]):
        center = qd.Vector([centers[i_q, i] for i in qd.static(range(3))])
        target = qd.Vector([targets[i_q, i] for i in qd.static(range(3))])
        normal = qd.Vector([normals[i_q, i] for i in qd.static(range(3))])
        face, bary, point = func_projected_surface_walk(
            anchors[i_q], center, target - center, normal, info, surface, epsilon
        )
        faces[i_q], points[i_q], coordinates[i_q] = face, point, bary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--curved", type=Path, required=True)
    cli.add_argument("--affine", type=Path, required=True)
    cli.add_argument("--expect-baseline-failure", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.curved), file_id(args.affine)],
            "scope": "Geometry-only projected walks: independent P2 interior/edge/vertex coordinates, adjacent-face crossing, inner/outer faces, reversed normals, disconnected adjacency and rigid rotation/translation. No pressure, traction, friction-history or physical-stress qualification is implied.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    rows = []
    for kind, asset in [("affine", args.affine), ("curved", args.curved)]:
        for is_rotated in [False, True]:
            current_asset = asset
            if is_rotated:
                with np.load(asset) as saved:
                    payload = {key: saved[key] for key in saved.files}
                rotation = Rotation.from_rotvec([0.3, -0.6, 0.8])
                for key in ["vertices", "edge_positions"]:
                    if key in payload:
                        payload[key] = rotation.apply(payload[key]) + [0.04, -0.03, 0.02]
                current_asset = args.output / (kind + "-rotated.npz")
                np.savez_compressed(current_asset, **payload)
            model = StressModel(RigidStressOptions(mesh=current_asset, contact_model="hertz", method="direct"))
            surface = StressSurface(4, model.info)
            vertices, indices = array(model.info.vertices), array(model.info.surface_nodes).T
            neighbors = array(model.info.surface_neighbors).T
            face_normals = array(surface.info.face_normal)
            faces = np.array([2, 17, 31, 46, 81, 97, 121, 142])
            anchors, centers, targets, normals, is_crossing, names = [], [], [], [], [], []
            for face in faces:
                nodes = vertices[indices[face]]
                origin_bary = np.array([[0.37, 0.31, 0.32]])
                values, gradients, _ = quadratic_basis(origin_bary)
                origin = (values @ nodes)[0]
                tangents = np.einsum("qna,ni->qia", gradients, nodes)[0]
                normal = np.cross(tangents[:, 0], tangents[:, 1])
                normal *= np.sign(normal @ face_normals[face])
                normal /= np.linalg.norm(normal)
                for label, destination in [
                    ("stationary", origin_bary[0]),
                    ("interior", np.array([0.19, 0.43, 0.38])),
                    ("vertex", np.array([1.0, 0.0, 0.0])),
                    ("edge", np.array([0.4, 0.6, 0.0])),
                ]:
                    values, _, _ = quadratic_basis(destination[None])
                    anchors.append(face)
                    centers.append(origin)
                    targets.append((values @ nodes)[0])
                    normals.append(normal)
                    is_crossing.append(False)
                    names.append(label)
                for edge in range(3):
                    other = neighbors[face, edge]
                    assert other >= 0
                    shared = indices[face, [edge, (edge + 1) % 3]]
                    destination = np.full(3, 0.16)
                    destination[np.isin(indices[other, :3], shared)] = 0.42
                    values, _, _ = quadratic_basis(destination[None])
                    anchors.append(face)
                    centers.append(origin)
                    targets.append((values @ vertices[indices[other]])[0])
                    normals.append(normal)
                    is_crossing.append(True)
                    names.append("neighbor")
            anchors = np.array(anchors, dtype=gs.np_int)
            centers, targets, normals = np.array(centers), np.array(targets), np.array(normals)
            is_crossing = np.array(is_crossing)
            located = V(dtype=gs.qd_int, shape=(len(anchors),))
            points = V_VEC(3, dtype=gs.qd_float, shape=(len(anchors),))
            bary = V_VEC(3, dtype=gs.qd_float, shape=(len(anchors),))
            kernel_walk(
                anchors,
                centers,
                targets,
                normals,
                located,
                points,
                bary,
                model.info,
                surface.info,
                float(np.finfo(gs.np_float).eps),
            )
            result_faces, result_points, result_bary = array(located), array(points), array(bary)
            errors = np.linalg.norm(result_points - targets, axis=1)
            is_passed = bool(np.all(result_faces >= 0) and errors.max() < 1e-12)
            row = {
                "kind": kind,
                "is_rotated": is_rotated,
                "queries": len(anchors),
                "valid_queries": int(np.count_nonzero(result_faces >= 0)),
                "maximum_position_error_m": float(errors.max()),
                "positive_controls_passed": is_passed,
            }
            output = args.output / f"{kind}-rotated{int(is_rotated)}.npz"
            np.savez_compressed(
                output,
                anchors=anchors,
                centers=centers,
                targets=targets,
                normals=normals,
                faces=result_faces,
                positions=result_points,
                barycentric=result_bary,
                names=names,
            )
            if args.expect_baseline_failure and kind == "curved":
                assert not is_passed
            else:
                assert is_passed, row
                values, _, _ = quadratic_basis(result_bary)
                reconstructed = np.einsum("qn,qni->qi", values, vertices[indices[result_faces]])
                np.testing.assert_allclose(reconstructed, result_points, rtol=0, atol=1e-12)
                kernel_walk(
                    anchors,
                    centers,
                    targets,
                    -normals,
                    located,
                    points,
                    bary,
                    model.info,
                    surface.info,
                    float(np.finfo(gs.np_float).eps),
                )
                assert np.all(array(located) == -1)
                disconnected = V(dtype=gs.qd_int, shape=model.info.surface_neighbors.shape)
                disconnected.fill(-1)
                isolated_info = replace(model.info, surface_neighbors=disconnected)
                kernel_walk(
                    anchors,
                    centers,
                    targets,
                    normals,
                    located,
                    points,
                    bary,
                    isolated_info,
                    surface.info,
                    float(np.finfo(gs.np_float).eps),
                )
                np.testing.assert_equal(array(located)[is_crossing], -1)
                np.testing.assert_allclose(array(points)[~is_crossing], targets[~is_crossing], rtol=0, atol=1e-12)
                row["reversed_normals_refused"] = len(anchors)
                row["disconnected_crossings_refused"] = int(is_crossing.sum())
                row["isolated_same_face_queries_preserved"] = int((~is_crossing).sum())
            rows.append(row)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "expected_baseline_curved_failure_observed": args.expect_baseline_failure,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
