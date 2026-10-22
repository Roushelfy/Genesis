"""Check that native source binding uses the actual curved material boundary."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--curved", type=Path, required=True)
    cli.add_argument("--affine", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.curved), file_id(args.affine)],
            "scope": "Known corner/midpoint and unbound interior/shared-edge witnesses lie exactly on the independently evaluated six-node material face. Check native source position and unchanged force/free-moment balance, plus one released environment. The1e-4 projection limit and ambiguity semantics remain unchanged. This is source geometry, not pressure or physical stress validation.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    bary = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.4, 0.3, 0.3], [0.5, 0.5, 0], [0.4, 0.3, 0.3]])
    values, gradient, _ = quadratic_basis(bary)
    rows = []
    for kind, asset in [("affine", args.affine), ("curved", args.curved)]:
        model = StressModel(RigidStressOptions(mesh=asset, contact_model="hertz", method="direct"))
        surface = StressSurface(8, model.info)
        vertices = array(model.info.vertices)
        face = 46
        nodes = vertices[array(model.info.surface_nodes)[:, face]]
        points = values @ nodes
        tangents = np.einsum("qna,ni->qia", gradient, nodes)
        normal = np.cross(tangents[:, :, 0], tangents[:, :, 1])
        normal *= np.sign(normal @ array(surface.info.face_normal)[face])[:, None]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        force = -0.1 * normal
        force[4] = 0
        positions = points + [1e-6, 2e-6, -3e-6]
        free = np.broadcast_to([1e-9, -2e-9, 3e-9], points.shape).copy()
        contacts, sources = create_contacts(1, 5, 0.001), create_sources(1, 5)
        contacts.valid.from_numpy(np.array([[True, True, True, True, False]]))
        contacts.position.from_numpy(positions[None])
        contacts.force.from_numpy(force[None])
        contacts.frame_position.fill(0)
        sources.geom.fill(7)
        sources.pair.fill(11)
        sources.witness.from_numpy(points[None])
        sources.free_moment.from_numpy(free[None])
        sources.witness_face.from_numpy(np.array([[face, face, -1, -1, -1]], dtype=gs.np_int))
        sources.witness_barycentric.from_numpy(bary[None])
        kernel_resolve_sources(1e-4, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, surface.info)
        status = array(sources.status)[:, 0]
        centers = array(contacts.center)[:, 0]
        moment = array(sources.anchor_moment)[:, 0] + np.cross(centers - positions, force) - free
        error = np.linalg.norm(centers[:4] - points[:4], axis=1)
        is_passed = bool(
            np.all(status == 0)
            and np.isfinite(error).all()
            and error.max() < 1e-12
            and np.linalg.norm(moment[:4], axis=1).max() < 1e-20
            and np.isnan(centers[4]).all()
        )
        fields = args.output / (kind + ".npz")
        np.savez_compressed(
            fields,
            expected=points,
            centers=centers,
            status=status,
            moment=moment,
            native_barycentric=array(sources.barycentric),
            expected_barycentric=bary,
        )
        rows.append(
            {
                "kind": kind,
                "passed": is_passed,
                "status": status.tolist(),
                "position_error_m": [float(value) if np.isfinite(value) else None for value in error],
                "fields": file_id(fields),
            }
        )
    passed = all(row["passed"] for row in rows)
    write_json(args.output / "result.json", {"passed": passed, "rows": rows, "whole_goal_complete": False})
    assert passed, rows


if __name__ == "__main__":
    main()
