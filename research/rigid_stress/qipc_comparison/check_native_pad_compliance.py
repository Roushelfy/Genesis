"""Compare native directional compliance to independent complete free-link elasticity."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_operator import NativePadBasis, NativePadCompliance
from .p2_elastic import FreeElasticP2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    vertices = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 2], [3, 4, 2]]) * 0.001
    tets = np.array([[0, 1, 2, 3], [4, 2, 1, 3]])
    face_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces, counts = np.unique(np.sort(tets[:, face_indices].reshape(-1, 3), axis=1), axis=0, return_counts=True)
    faces = faces[counts == 1]
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.6, 0.8, 0], [0.2, 0.3, 0.5], [0.3, -0.5, 0.7]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rows, original = [], None
    for index, young in enumerate((1e10, 2e9)):
        points = vertices if index == 0 else vertices @ rotation.T + [0.02, -0.03, 0.01]
        normals = directions if index == 0 else directions @ rotation.T
        path = args.output / f"mesh{index}.npz"
        np.savez_compressed(path, vertices=points, tetrahedra=tets, surface_triangles=faces)
        model = StressModel(RigidStressOptions(mesh=path, young=young, method="direct"))
        coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
        cpu = FreeElasticP2(points, tets, young, 0.3, 2000)
        permutation = cpu.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
        nodes = np.arange(len(normals))
        native = NativePadCompliance(model, nodes, normals, young)
        matrix = qd_to_numpy(native.info.matrix, transpose=True, copy=True).T
        expected = np.empty_like(matrix)
        for column in range(len(nodes)):
            load = np.zeros_like(cpu.vertices)
            load[permutation[nodes[column]]] = normals[column]
            _, displacement, _, residual = cpu.solve(load)
            assert np.linalg.norm(residual) <= 1e-7
            expected[:, column] = np.einsum("ij,ij->i", normals, displacement[permutation[nodes]])
        error = np.linalg.norm(matrix - expected) / np.linalg.norm(expected)
        assert error <= 1e-10
        basis = NativePadBasis(native.info.matrix, np.eye(len(nodes)), np.ones((len(nodes), 1)))
        factor = qd_to_numpy(basis.workspace.lower, transpose=True, copy=True).T
        factor_error = np.linalg.norm(factor @ factor.T - expected) / np.linalg.norm(expected)
        assert factor_error <= 1e-10
        frame_error = 0.0
        if original is None:
            original = matrix * young
        else:
            frame_error = np.linalg.norm(matrix * young - original) / np.linalg.norm(original)
            assert frame_error <= 1e-10
        qd.sync()
        path = args.output / f"compliance{index}.npz"
        np.savez_compressed(path, native=matrix, independent=expected, factor=factor, nodes=nodes, directions=normals)
        rows.append(
            {
                "young_pa": young,
                "relative_compliance_error": error.item(),
                "relative_factor_error": factor_error.item(),
                "frame_material_error": float(frame_error),
                "native_reciprocity_error": native.reciprocity_error,
                "arrays": file_id(path),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "rows": rows,
            "scope": "Native balanced unit-load condensation and mass-gauge alignment agree with independent complete elasticity for six arbitrary nodal directions, including a translated/rotated material-scaled control. This validates operator construction on a compact mesh, not large-mesh preprocessing cost or automatic contact geometry.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
