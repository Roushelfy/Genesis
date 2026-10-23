"""Compare ordered source resolution over captured curved-face queries and invalid inputs."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def kernel_seed_sources(
    faces: qd.types.ndarray(),
    points: qd.types.ndarray(),
    known: qd.types.ndarray(),
    bound: qd.types.ndarray(),
    contacts: StressContactState,
    sources: StressSourceState,
    n_faces: int,
):
    for i in range(faces.shape[0]):
        i_c, i_b = i // contacts.valid.shape[1], i % contacts.valid.shape[1]
        kind = i % 101
        point = qd.Vector([points[i, j] for j in qd.static(range(3))])
        contacts.position[i_c, i_b] = point + qd.Vector([3e-4, -1e-4, 2e-4])
        contacts.force[i_c, i_b] = qd.Vector([0.01, 0.0025, -0.005])
        contacts.valid[i_c, i_b] = kind != 0
        sources.geom[i_c, i_b] = qd.select(kind == 2, -1, 0)
        sources.pair[i_c, i_b] = qd.select(kind == 3, -1, 0)
        sources.uncertainty[i_c, i_b] = qd.select(kind == 4, -1.0, qd.select(kind == 1, 5e-4, 0.0))
        sources.has_vertex_ambiguity[i_c, i_b] = kind == 5
        sources.witness[i_c, i_b] = point
        sources.witness_face[i_c, i_b] = qd.select(bound[i] != 0, faces[i], -1)
        sources.witness_barycentric[i_c, i_b] = qd.Vector([known[i, j] for j in qd.static(range(3))])
        sources.free_moment[i_c, i_b] = qd.Vector([1e-6, 2e-6, -1e-6])
        if kind == 6:
            sources.witness_face[i_c, i_b] = n_faces
        elif kind == 7:
            sources.witness_face[i_c, i_b] = faces[i]
            sources.witness_barycentric[i_c, i_b] = qd.Vector([0.5, 0.5, 0.5])
        elif kind == 8:
            sources.witness[i_c, i_b] = qd.Vector([float("nan")] * 3)
        elif kind == 9:
            contacts.force[i_c, i_b] = qd.Vector([float("inf"), 0.0, 0.0])
        if i_c == 0:
            contacts.frame_position[i_b] = i_b * qd.Vector([0.1, -0.05, 0.025])
            contacts.frame_quaternion[i_b] = qd.Vector([1.0, 0.0, 0.0, 0.0])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--precision", choices=["32", "64"], default="64")
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--queries", type=Path, required=True)
    cli.add_argument("--baseline", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.mesh), file_id(args.queries)] + ([file_id(args.baseline)] if args.baseline else []),
            "precision": args.precision,
            "scope": "Captured all-face bound/unbound queries span16 environments, with uncertainty, inactive sources, invalid geom/pair/coordinates, vertex ambiguity and nonfinite inputs. Source status and canonical face must match exactly; center/barycentric/anchor-moment budgets are128 machine epsilons with geometric/force scales. This tests implementation conformance against the frozen serial source, not independent closest-point or physical accuracy.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision=args.precision, logging_level="warning")
    with np.load(args.queries) as saved:
        faces, points, known, bound = [
            saved[name].copy() for name in ["faces", "points_m", "known_barycentric", "bound"]
        ]
    assert points.dtype == gs.np_float and len(points) % 16 == 0
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    surface = StressSurface(4, model.info)
    contacts = create_contacts(len(points) // 16, 16, 0.003)
    sources = create_sources(len(points) // 16, 16)
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_seed_sources(faces, points, known, bound, contacts, sources, model.info.surface_nodes.shape[0])
    kernel_resolve_sources(0.003, epsilon, contacts, sources, model.info, surface.info)
    fields = {
        "status": qd_to_numpy(sources.status, transpose=True, copy=True),
        "anchor_face": qd_to_numpy(contacts.anchor_face, transpose=True, copy=True),
        "center_m": qd_to_numpy(contacts.center, transpose=True, copy=True),
        "barycentric": qd_to_numpy(sources.barycentric, transpose=True, copy=True),
        "anchor_moment_nm": qd_to_numpy(sources.anchor_moment, transpose=True, copy=True),
    }
    status_counts = {str(i): int(np.count_nonzero(fields["status"] == i)) for i in range(4)}
    assert all(status_counts.values())
    assert np.count_nonzero(fields["anchor_face"] >= 0) > 0
    vertices = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
    scale = max(float(np.abs(vertices).max()), float(np.abs(points).max()))
    budgets = {"center_m": 128 * epsilon * scale, "barycentric": 128 * epsilon}
    budgets["anchor_moment_nm"] = 128 * epsilon * (3 * (scale + 3e-4) * 0.01 + 2e-6)
    comparisons = []
    np.savez_compressed(args.output / "fields.npz", **fields)
    if args.baseline:
        with np.load(args.baseline) as original:
            assert set(original.files) == set(fields)
            for key, value in fields.items():
                if key in budgets:
                    np.testing.assert_allclose(
                        value, original[key], rtol=0, atol=budgets[key], equal_nan=True, err_msg=key
                    )
                else:
                    np.testing.assert_array_equal(value, original[key], err_msg=key)
                finite = np.isfinite(value) & np.isfinite(original[key])
                comparisons.append(
                    {
                        "field": key,
                        "exact": bool(np.array_equal(value, original[key], equal_nan=True)),
                        "maximum_absolute_difference": float(
                            np.abs(value[finite] - original[key][finite]).max(initial=0)
                        ),
                        "absolute_budget": budgets.get(key, 0),
                    },
                )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "precision": args.precision,
            "queries": len(points),
            "environments": 16,
            "status_counts": status_counts,
            "comparisons": comparisons,
            "all_fields_exact": all(row["exact"] for row in comparisons) if comparisons else None,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
