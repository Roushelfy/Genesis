"""Check every native port distance against an independent graph solve on unchanged finite-load meshes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_footprints, kernel_footprint_ports
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.reference_run)
    assert int((args.reference_run / "returncode.txt").read_text()) == 0
    mesh_hash = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    inputs = {str(args.mesh): mesh_hash}
    snapshots = []
    for case in CASES:
        path = args.reference_run / f"results/{case}/snapshot.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = json.loads(path.read_text())
        assert snapshot["mesh_sha256"] == mesh_hash and snapshot["patch_radius_m"] == 0.003
        snapshots.append(snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", logging_level="error")
    options = RigidStressOptions(mesh=args.mesh, method="pcg", quadrature=10, surface_edge_subdivisions=8)
    model = StressModel(options)
    surface = StressSurface(options.quadrature, model.info)
    start = time.perf_counter()
    paths = StressSurfacePaths(model.info, options.surface_edge_subdivisions)
    initialization_seconds = time.perf_counter() - start
    assert len(paths.info.distances.shape) == 0
    ports = qd_to_numpy(paths.info.positions, transpose=True)
    starts = qd_to_numpy(paths.info.starts, transpose=True)
    neighbors = qd_to_numpy(paths.info.neighbors, transpose=True)
    face_ports = qd_to_numpy(paths.info.face_ports, transpose=True).T
    lengths = np.linalg.norm(ports[neighbors] - np.repeat(ports, np.diff(starts), axis=0), axis=1)
    np.testing.assert_allclose(qd_to_numpy(paths.info.lengths, transpose=True), lengths, rtol=5e-14, atol=1e-17)
    n_ports, n_envs = len(ports), len(snapshots)
    n_contacts = max(len(snapshot["patches"]) for snapshot in snapshots)
    contacts = create_contacts(n_contacts, n_envs, radius=0.003)
    sources = create_sources(n_contacts, n_envs)
    footprints = create_footprints(n_ports, n_contacts, n_envs, sparse_paths=True)
    shape = (n_contacts, n_envs)
    witnesses = np.zeros((*shape, 3))
    positions = np.zeros_like(witnesses)
    forces = np.zeros_like(witnesses)
    moments = np.zeros_like(witnesses)
    normals = np.zeros_like(witnesses)
    validity = np.zeros(shape, dtype=bool)
    for i_b, snapshot in enumerate(snapshots):
        for i_c, patch in enumerate(snapshot["patches"]):
            witnesses[i_c, i_b] = patch["witness_m"]
            positions[i_c, i_b] = patch["point_m"]
            forces[i_c, i_b] = patch["force_N"]
            moments[i_c, i_b] = patch["free_moment_Nm"]
            normals[i_c, i_b] = patch["normal"]
            validity[i_c, i_b] = True
    contacts.position.from_numpy(positions)
    contacts.force.from_numpy(forces)
    contacts.normal.from_numpy(normals)
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.free_moment.from_numpy(moments)
    base_rows = np.repeat(np.arange(n_ports), np.diff(starts))
    rows = []
    moved = witnesses.copy()
    for phase in range(3):
        active = validity.copy()
        radii = np.full(shape, 0.003)
        if phase == 1:
            active[0, 0] = False
            radii[:, 1] *= 0.8
        contacts.valid.from_numpy(active)
        contacts.radius.from_numpy(radii)
        sources.witness.from_numpy(moved if phase == 1 else witnesses)
        kernel_resolve_sources(1e-8, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, surface.info)
        np.testing.assert_array_equal(qd_to_numpy(sources.status, transpose=True)[active.T], 0)
        centers = qd_to_numpy(contacts.center, transpose=True)
        faces = qd_to_numpy(contacts.anchor_face, transpose=True)
        if phase == 0:
            i_face = faces[2, 0]
            moved[0, 2] = 0.9 * centers[2, 0] + 0.1 * ports[face_ports[i_face, 0]]
        start = time.perf_counter()
        kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
        qd.sync()
        seconds = time.perf_counter() - start
        actual = qd_to_numpy(footprints.port_distance, transpose=True).transpose(0, 2, 1).reshape(-1, n_ports)
        indices = face_ports[faces.reshape(-1)]
        seed_distances = np.linalg.norm(ports[indices] - centers.reshape(-1, 1, 3), axis=2)
        seed_rows = np.repeat(n_ports + np.arange(n_contacts * n_envs), indices.shape[1])
        selected = np.repeat(active.T.reshape(-1), indices.shape[1])
        graph = sparse.csr_matrix(
            (
                np.r_[lengths, seed_distances.reshape(-1)[selected]],
                (np.r_[base_rows, seed_rows[selected]], np.r_[neighbors, indices.reshape(-1)[selected]]),
            ),
            shape=(n_ports + n_contacts * n_envs,) * 2,
        )
        expected = dijkstra(
            graph, directed=True, indices=n_ports + np.arange(n_contacts * n_envs), limit=float(radii.max())
        )[:, :n_ports]
        inside = expected < radii.T.reshape(-1, 1)
        np.testing.assert_array_equal(np.isfinite(actual), inside)
        np.testing.assert_allclose(actual[inside], expected[inside], rtol=2e-12, atol=2e-17)
        np.savez_compressed(
            args.output / f"phase{phase}.npz",
            native_distance_m=actual,
            independent_distance_m=expected,
            radius_m=radii.T,
            active=active.T,
            centers_m=centers,
            faces=faces,
        )
        rows.append(
            {
                "phase": phase,
                "active_contacts": int(active.sum()),
                "all_port_entries_checked": actual.size,
                "within_radius_entries": int(inside.sum()),
                "maximum_distance_difference_m": float(abs(actual[inside] - expected[inside]).max()),
                "native_seconds_including_first_compile": seconds,
                "all_support_membership_matches": True,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    with np.load(args.output / "phase0.npz") as first, np.load(args.output / "phase2.npz") as restored:
        np.testing.assert_array_equal(first["native_distance_m"], restored["native_distance_m"])
    summary = {
        "scope": __doc__,
        "port_count": n_ports,
        "directed_graph_edges": len(neighbors),
        "all_pair_float64_bytes_avoided": 8 * n_ports**2,
        "per_contact_float64_distance_bytes": 8 * n_ports * n_contacts * n_envs,
        "heap_index_bytes": 8 * n_ports * n_contacts * n_envs + 4 * n_contacts * n_envs,
        "initialization_seconds": initialization_seconds,
        "rows": rows,
        "exact_original_restoration": True,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Qualifies the unchanged8-port graph at every port for all five original snapshots, release, radius and anchor changes, and exact restoration. Changed-phase inputs are numerical lifecycle controls. This is not a continuum-geodesic, traction-fit, elastic-field, physical-convergence or warmed-application-cost claim.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
