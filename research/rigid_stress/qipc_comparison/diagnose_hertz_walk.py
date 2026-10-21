"""Retain homogeneous GPU patch statuses for the original and early-exit walks."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_hertz_transfer import project_oracle
from .native_hertz import NativeHertz
from .native_hertz_transfer import HertzSurfaceTransfer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--egg", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(), "egg": file_id(args.egg)})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    epsilon = float(np.finfo(gs.np_float).eps)
    with np.load(args.egg) as source:
        vertices, tets, faces = (source[key] for key in ["vertices", "tetrahedra", "surface_triangles"])
    vertex = vertices[:, 2].argmax()
    center = vertices[vertex].copy()
    owned = np.any(faces == vertex, axis=1)
    raw = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    raw *= np.where(np.einsum("fi,fi->f", raw, vertices[faces].mean(axis=1)) > 0, 1, -1)[:, None]
    normal = raw[owned].sum(axis=0)
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, [1.0, 0, 0])
    tangent /= np.linalg.norm(tangent)
    rotation = Rotation.from_euler("xyz", [21, -37, 19], degrees=True).as_matrix()
    vertices = vertices @ rotation.T + [0.03, -0.02, 0.01]
    center = center @ rotation.T + [0.03, -0.02, 0.01]
    normal, tangent = normal @ rotation.T, tangent @ rotation.T
    mesh = args.output / "mesh.npz"
    np.savez_compressed(mesh, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh, method="direct"))
    surface = StressSurface(3, model.info)
    coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
    nodes = qd_to_numpy(model.info.surface_nodes, transpose=True, copy=True).T
    normals = qd_to_numpy(surface.info.face_normal, transpose=True, copy=True)
    anchor, _, _ = project_oracle(center, np.zeros((1, 3)), normal, nodes, coordinates, normals)
    rows = []
    for count in [1, 3, 128, 512]:
        native = NativeHertz(count)
        native.state.force.fill(0.1)
        native.state.modulus.fill(1e10 / (1 - 0.3**2))
        native.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (count, 2, 2)).copy())
        mapped = HertzSurfaceTransfer(count, count)
        mapped.patches.environment.from_numpy(np.arange(count, dtype=gs.np_int))
        mapped.patches.anchor.fill(int(anchor[0]))
        for field, value in [
            (mapped.patches.center, center),
            (mapped.patches.normal, normal),
            (mapped.patches.tangent, tangent),
            (mapped.patches.force, -0.1 * normal),
        ]:
            field.from_numpy(np.broadcast_to(value, (count, 3)).copy())
        mapped.patches.moment.fill(0)
        target = model.create_state(count)
        native.update()
        mapped.scatter(native.state, target, model.info, surface.info, epsilon)
        qd.sync()
        fields = {
            name: qd_to_numpy(field, transpose=True, copy=True)
            for name, field in [
                ("status", mapped.patches.status),
                ("force_error_n", mapped.patches.force_error),
                ("moment_error_nm", mapped.patches.moment_error),
                ("sample_face", mapped.samples.face),
                ("sample_barycentric", mapped.samples.barycentric),
                ("sample_position_m", mapped.samples.position),
                ("axes_m", native.state.axes),
                ("nodal_force_n", target.force),
            ]
        }
        path = args.output / f"batch{count}.npz"
        np.savez_compressed(path, center=center, normal=normal, tangent=tangent, **fields)
        status, counts = np.unique(fields["status"], return_counts=True)
        rows.append(
            {
                "count": count,
                "statuses": dict(zip(map(str, status.tolist()), counts.tolist())),
                "maximum_force_error_n": float(fields["force_error_n"].max()),
                "maximum_moment_error_nm": float(fields["moment_error_nm"].max()),
                "fields": file_id(path),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "all_patches_admitted": all(set(row["statuses"]) == {"0"} for row in rows),
            "scope": "Same homogeneous rotated-egg normal patches at four batch sizes. Original input, walk, wrench budgets and rejection masking are unchanged; this diagnostic retains all statuses instead of bypassing the original failed assertion.",
        },
    )


if __name__ == "__main__":
    main()
