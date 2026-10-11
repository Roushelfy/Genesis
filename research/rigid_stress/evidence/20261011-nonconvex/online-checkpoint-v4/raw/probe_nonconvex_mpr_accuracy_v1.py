"""Audit the intrinsic law inside every live rigid step, with an untimed independent FP64 oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--contact-block-size", type=int, default=2)
    parser.add_argument("--oracle-stride", type=int, default=20)
    parser.add_argument("--egg-geometry", choices=("urdf", "mesh64"), default="urdf")
    parser.add_argument("--collision-method", choices=("auto", "mpr", "gjk"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
        Path("genesis/options/rigid_stress.py"),
        Path("examples/rigid/franka_egg_stress.py"),
        Path(__file__),
    ]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    gs.init(backend=gs.cuda, precision="64")
    workload = FrankaEgg(
        args.envs,
        seed=args.seed,
        varied=True,
        output_mode="full",
        contact_model="intrinsic",
        contact_block_size=args.contact_block_size,
        egg_geometry=args.egg_geometry,
        use_gjk_collision={"auto": None, "mpr": False, "gjk": True}[args.collision_method],
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    solver.collider.collider_info.mpr.CCD_TOLERANCE.fill(1e-10)
    print(json.dumps({"diagnostic_mpr_fractional_tolerance": 1e-10}), flush=True)
    with np.load(entry.options.mesh) as asset:
        oracle = P2Shell(
            asset["vertices"],
            asset["tetrahedra"],
            asset["surface_triangles"],
            entry.options.young,
            entry.options.poisson,
            entry.options.density,
            2,
            factor_backend="none",
        )
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(entry.model.info.pins, transpose=True))
    factor = splu(oracle.k[free][:, free].tocsc())
    vertex_info = entry.intrinsic.vertices.info
    np.savez_compressed(
        args.output / "source-vertices.npz",
        positions=qd_to_numpy(vertex_info.positions, transpose=True),
        status=qd_to_numpy(vertex_info.status, transpose=True),
        offsets=qd_to_numpy(vertex_info.offset, transpose=True),
    )
    records, captures = [], []
    previous_valid = None
    completed_steps = 0
    try:
        for tick in range(args.steps):
            workload.step()
            solver.check_errno()
            arrays = {
                "valid": qd_to_numpy(entry.contacts.valid, transpose=True),
                "position": qd_to_numpy(entry.contacts.position, transpose=True),
                "force": qd_to_numpy(entry.contacts.force, transpose=True),
                "radius": qd_to_numpy(entry.contacts.radius, transpose=True),
                "friction": qd_to_numpy(entry.contacts.friction, transpose=True),
                "status": qd_to_numpy(entry.contacts.status, transpose=True),
                "force_error": qd_to_numpy(entry.contacts.force_error, transpose=True),
                "moment_error": qd_to_numpy(entry.contacts.moment_error, transpose=True),
                "evaluations": qd_to_numpy(entry.contacts.evaluations, transpose=True),
                "samples": qd_to_numpy(entry.contacts.candidate_count, transpose=True),
            }
            valid = arrays["valid"]
            loaded = valid & (np.linalg.norm(arrays["force"], axis=2) > 0.0)
            force_scale = np.linalg.norm(arrays["force"][loaded], axis=1)
            normalized_force = arrays["force_error"][loaded] / force_scale
            normalized_moment = arrays["moment_error"][loaded] / (force_scale * arrays["radius"][loaded])
            assert (arrays["status"][valid] == 0).all()
            assert (normalized_force <= 1e-8).all() and (normalized_moment <= 1e-8).all()
            assert np.isfinite(tensor_to_array(entry.link.get_max_stress())).all()
            record = {
                "tick": tick,
                "loaded_contacts": int(loaded.sum()),
                "contact_count_per_env": valid.sum(axis=1).tolist(),
                "contact_slot_changes": int((valid != previous_valid).sum())
                if previous_valid is not None
                else int(valid.sum()),
                "mu_values": np.unique(arrays["friction"][loaded]).tolist(),
                "max_force_error": float(normalized_force.max(initial=0.0)),
                "max_moment_error": float(normalized_moment.max(initial=0.0)),
                "max_evaluations": int(arrays["evaluations"].max()),
                "max_samples": int(arrays["samples"].max()),
                "reset_environments": workload.reset_count,
            }
            records.append(record)
            previous_valid = valid.copy()
            if tick % args.oracle_stride == 0 or tick == args.steps - 1:
                nodal = qd_to_numpy(entry.state.force, transpose=True)
                omega = qd_to_numpy(entry.omega, transpose=True)
                centrifugal = -np.cross(omega[:, None], np.cross(omega[:, None], oracle.xyz[None] - oracle.com))
                raw = np.moveaxis(nodal, 0, -1).reshape(oracle.ndof, args.envs)
                raw += oracle.m @ np.moveaxis(centrifugal, 0, -1).reshape(oracle.ndof, args.envs)
                rhs = oracle.balance(raw)
                reference = np.zeros_like(rhs)
                reference[free] = factor.solve(rhs[free])
                native = qd_to_numpy(entry.state.displacement, transpose=True).reshape(args.envs, oracle.ndof).T
                residual = np.linalg.norm(oracle.k @ native - rhs, axis=0)
                budget = np.maximum(
                    entry.options.absolute_tolerance, entry.options.tolerance * np.linalg.norm(rhs, axis=0)
                )
                assert (residual <= budget).all()
                tensor = qd_to_numpy(entry.state.stress_tensor, transpose=True)
                vm = qd_to_numpy(entry.state.von_mises, transpose=True)[..., 0]
                cpu_peaks = []
                for i_b in range(args.envs):
                    expected_tensor, expected_vm = stress_field(
                        oracle.glambda, oracle.elements, reference[:, i_b], entry.options.young, entry.options.poisson
                    )
                    np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
                    np.testing.assert_allclose(vm[i_b], expected_vm, rtol=1e-4, atol=1e-3)
                    cpu_peaks.append(expected_vm.max())
                np.testing.assert_allclose(
                    qd_to_numpy(entry.state.peak, transpose=True), cpu_peaks, rtol=1e-4, atol=1e-3
                )
                capture = {"tick": tick, "full_residual_max_N": float(residual.max()), "cpu_peaks_Pa": cpu_peaks}
                captures.append(capture)
                np.savez_compressed(
                    args.output / f"snapshot-{tick}.npz",
                    **arrays,
                    nodal_force=nodal,
                    omega=omega,
                    displacement=native,
                    full_residual=residual,
                    full_residual_budget=budget,
                    cpu_peaks=cpu_peaks,
                    rigid_position=tensor_to_array(workload.egg.get_pos()),
                )
                print(json.dumps(record | capture), flush=True)
            completed_steps += 1
    finally:
        qd.sync()
        if completed_steps < args.steps:
            failed = args.output / "failure"
            failed.mkdir(exist_ok=True)
            statuses = qd_to_numpy(entry.contacts.status, transpose=True)
            force = qd_to_numpy(entry.contacts.force, transpose=True)
            force_scale = np.linalg.norm(force, axis=2)
            radius = qd_to_numpy(entry.contacts.radius, transpose=True)
            force_error = qd_to_numpy(entry.contacts.force_error, transpose=True)
            moment_error = qd_to_numpy(entry.contacts.moment_error, transpose=True)
            np.savez_compressed(
                failed / f"snapshot-{completed_steps}.npz",
                valid=qd_to_numpy(entry.contacts.valid, transpose=True),
                position=qd_to_numpy(entry.contacts.position, transpose=True),
                force=force,
                normal=qd_to_numpy(entry.contacts.normal, transpose=True),
                radius=radius,
                friction=qd_to_numpy(entry.contacts.friction, transpose=True),
                frame_position=qd_to_numpy(entry.contacts.frame_position, transpose=True),
                frame_quaternion=qd_to_numpy(entry.contacts.frame_quaternion, transpose=True),
                raw_status=statuses,
                traction_status=np.maximum(statuses - 30, 0),
                source_status=qd_to_numpy(entry.intrinsic.raw_sources.status, transpose=True),
                anchor=qd_to_numpy(entry.contacts.center, transpose=True),
                anchor_face=qd_to_numpy(entry.contacts.anchor_face, transpose=True),
                samples=qd_to_numpy(entry.contacts.candidate_count, transpose=True),
                fit_force_error=np.divide(
                    force_error, force_scale, out=np.zeros_like(force_error), where=force_scale > 0
                ),
                fit_moment_error=np.divide(
                    moment_error, force_scale * radius, out=np.zeros_like(moment_error), where=force_scale > 0
                ),
                witness=qd_to_numpy(entry.intrinsic.raw_sources.witness, transpose=True),
                geom=qd_to_numpy(entry.intrinsic.raw_sources.geom, transpose=True),
                pair=qd_to_numpy(entry.intrinsic.raw_sources.pair, transpose=True),
                method=qd_to_numpy(entry.intrinsic.raw_sources.method, transpose=True),
                certificate=qd_to_numpy(entry.intrinsic.certificates.ray, transpose=True),
                certificate_margin=qd_to_numpy(entry.intrinsic.certificates.margin, transpose=True),
            )
        args.output.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "scope": "Intrusive validation of native intrinsic recovery in every live scene.step, including diagnostic and oracle overhead; no throughput claim.",
                    "arguments": vars(args) | {"output": str(args.output)},
                    "completed_steps": completed_steps,
                    "records": records,
                    "oracle_captures": captures,
                    "source_sha256": source_hashes,
                    "source_sha256_after": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
