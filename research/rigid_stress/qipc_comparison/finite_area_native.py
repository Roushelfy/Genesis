"""Compare native elastic core and DPR/SCR source reconstruction under a declared finite-area load."""

import argparse
import faulthandler
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import voigt_tensor
from .native_reference_replay import ContactViews


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1])
    cli.add_argument("--radius", type=float, default=0.002)
    cli.add_argument("--method", choices=("direct", "pcg"), default="direct")
    cli.add_argument("--iterations", type=int, default=10000)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert args.radius > 0 and min(args.loads) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    assets = json.loads((args.assets / "result.json").read_text())
    reference = json.loads((args.reference.parent / "result.json").read_text())
    assert reference["mesh"]["sha256"] == file_id(args.assets / "elastic.npz")["sha256"]
    with np.load(args.assets / "elastic.npz") as saved:
        tetrahedra, outer = saved["tetrahedra"], saved["outer_vertices"]
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), show_viewer=False)
    for method in ("pressure", "intrinsic"):
        egg = scene.add_entity(
            morph=gs.morphs.URDF(
                file=args.assets / "egg_shell.urdf", align=False, decimate=False, preserve_mesh_precision=True
            ),
            material=gs.materials.Rigid(),
        )
        egg.base_link.configure_stress_recovery(
            gs.options.RigidStressOptions(
                mesh=args.assets / "elastic.npz",
                method=args.method,
                max_iterations=args.iterations,
                contact_model=method,
                contact_radius=args.radius,
                intrinsic_load_partition="body_source_prior",
                output_mode="full",
                update_mode="manual",
                fused_pipeline=False,
                surface_inverse=False,
            )
        )
    write_json(
        args.output / "building.json", {"identity": runtime_identity(), "mesh": file_id(args.assets / "elastic.npz")}
    )
    started = time.perf_counter()
    faulthandler.dump_traceback_later(180, repeat=True)
    scene.build(n_envs=1)
    faulthandler.cancel_dump_traceback_later()
    setup_seconds = time.perf_counter() - started
    dpr, scr = scene.rigid_solver.stress_recovery.links
    assert dpr.model is scr.model and dpr.surface is scr.surface
    for entry in (dpr, scr):
        np.testing.assert_allclose(entry.link.get_mass().cpu().numpy(), assets["mass_kg"], rtol=1e-12)
    coordinates = qd_to_numpy(dpr.model.info.vertices, transpose=True)
    elements = qd_to_numpy(dpr.model.info.elements, transpose=True).T
    np.testing.assert_array_equal(elements[:, :4], tetrahedra)
    source_points = np.array([[0, 0, outer[:, 2].min()], [0, 0, outer[:, 2].max()]])
    source_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    errno = qd_to_torch(scene.rigid_solver._errno, transpose=True, copy=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.assets / "elastic.npz"),
            "assets": file_id(args.assets / "result.json"),
            "reference": file_id(args.reference.parent / "result.json"),
            "radius_m": args.radius,
            "setup_seconds": setup_seconds,
            "native_options": [entry.options.model_dump(mode="json") for entry in (dpr, scr)],
            "rigid_masses_kg": [entry.link.get_mass().tolist() for entry in (dpr, scr)],
        },
    )
    rows = []
    for load in args.loads:
        reference_path = args.reference / f"load-{load:g}.npz"
        with np.load(reference_path) as saved:
            reference_coordinates, forces, expected = saved["vertices_m"], saved["forces_n"], saved["stress_pa"]
            assert saved["load_n"].item() == load
        distance, permutation = cKDTree(reference_coordinates).query(coordinates)
        assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(coordinates) == len(reference_coordinates)
        for route, entry in (("core", dpr), ("dpr", dpr), ("scr", scr)):
            output = args.output / route
            output.mkdir(exist_ok=True)
            errno.zero_()
            kernel_begin_step(entry.state)
            force_view = qd_to_torch(entry.state.force, transpose=True, copy=False)
            force_view.zero_()
            entry.omega.fill(0)
            entry.contacts.valid.fill(False)
            started = time.perf_counter()
            if route == "core":
                force_view[0] = torch.as_tensor(forces[permutation], device=gs.device)
            else:
                contacts = ContactViews.allocate(entry.contacts)
                count = contacts.load(
                    {
                        "compressed_positions_material_m": source_points,
                        "compressed_forces_material_n": load * source_normals,
                        "compressed_normals": source_normals,
                        "friction": np.array(0.0),
                        "radius_m": np.array(args.radius),
                    }
                )
                if entry.intrinsic is not None:
                    sources = entry.intrinsic.raw_sources
                    sources.geom.fill(entry.link.geoms[0].idx)
                    sources.method.fill(0)
                    sources.vertex.fill(-1)
                    sources.witness_face.fill(-1)
                    sources.has_vertex_ambiguity.fill(False)
                    sources.free_moment.fill(0)
                    pair = qd_to_torch(sources.pair, transpose=True, copy=False)
                    witness = qd_to_torch(sources.witness, transpose=True, copy=False)
                    pair.fill_(-1)
                    witness.fill_(float("nan"))
                    pair[0, :count] = torch.arange(count, dtype=gs.tc_int, device=gs.device)
                    witness[0, :count] = contacts.position[0, :count]
                    entry.intrinsic.scatter(
                        entry.contacts,
                        entry.state,
                        entry.model.info,
                        entry.surface.info,
                        entry.options,
                        np.finfo(gs.np_float).eps.item(),
                        scene.rigid_solver._errno,
                    )
                else:
                    kernel_anchor(np.finfo(gs.np_float).eps.item(), entry.contacts, entry.surface.info)
                    kernel_pressure(entry.contacts, entry.surface.info, entry.options.cooperative_pressure)
                    kernel_scatter(
                        entry.contacts,
                        entry.state,
                        entry.model.info,
                        entry.surface.info,
                        entry.options.cooperative_scatter,
                        entry.options.cached_face_bounds,
                    )
            entry.model.recover(entry.omega, entry.state, surface_load=False)
            if route != "core":
                kernel_accept(entry.state, entry.contacts, scene.rigid_solver._errno)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            tensor = voigt_tensor(qd_to_numpy(entry.state.stress_tensor, transpose=True)[0])
            actual_force = qd_to_numpy(entry.state.force, transpose=True)[0]
            rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
            residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
            code = int(qd_to_numpy(scene.rigid_solver._errno, transpose=True)[0])
            valid = bool(qd_to_numpy(entry.state.valid, transpose=True)[0] and code == 0 and np.isfinite(tensor).all())
            rhs_norm, force_norm = np.linalg.norm(rhs), np.linalg.norm(actual_force)
            path = output / f"load-{load:g}.npz"
            np.savez_compressed(
                path,
                vertices_m=coordinates,
                tetrahedra=tetrahedra,
                stress_pa=tensor,
                forces_n=actual_force,
                rhs_n=rhs,
                equation_residual_n=residual,
                displacement_m=qd_to_numpy(entry.state.displacement, transpose=True)[0],
                source_positions_m=source_points,
                source_forces_n=load * source_normals,
            )
            rows.append(
                {
                    "route": route,
                    "load_n": load,
                    "native_valid": valid,
                    "errno": code,
                    "contact_status": qd_to_numpy(entry.contacts.status, transpose=True)[0, :2].tolist()
                    if route != "core"
                    else [],
                    "equation_relative_residual": float(np.linalg.norm(residual) / rhs_norm) if rhs_norm else None,
                    "balance_correction_relative": float(np.linalg.norm(rhs - actual_force) / force_norm)
                    if force_norm
                    else None,
                    "core_maximum_stress_relative_difference": float(
                        np.max(np.abs(tensor - expected)) / np.max(np.abs(expected))
                    )
                    if valid and route == "core"
                    else None,
                    "cold_seconds_including_transfers": seconds,
                    "field": file_id(path),
                    "reference_field": file_id(reference_path),
                }
            )
            write_json(args.output / "progress.json", {"cases": rows})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "setup_seconds": setup_seconds,
            "cases": rows,
            "physical_accuracy_qualified": False,
            "scope": "Native Quadrants core receives exact independent P2 nodal loads. DPR/SCR receive two balanced axial point-source wrenches with radius fixed before stress evaluation; they reconstruct their own load profiles. Explicit witnesses are on the matching collider, but no actual scene contact history is claimed. Full fields and reconstructed loads separate elasticity from load-recovery error. Two stationary entities share immutable operators; no scene steps occur. Cold timings are diagnostic, not matched admitted throughput.",
        },
    )


if __name__ == "__main__":
    main()
