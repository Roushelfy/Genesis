"""Diagnose new-law mapping of pre-integration witnesses on the existing egg trajectory.

The current live scene still executes its existing stress pipeline. This probe
uses that association's saved material frame and a separate elastic state at
declared ticks. It neither changes rigid dynamics nor establishes new-law live
throughput. All stress arithmetic in the probe is native Quadrants; CPU FP64
elasticity is an untimed independent oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    kernel_footprint_ports,
    kernel_footprint_profiles,
)
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import (
    StressSourceState,
    create_sources,
    kernel_capture_rigid_sources,
    kernel_resolve_sources,
)
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    create_traction_info,
    create_tractions,
    func_sample_traction,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell


@qd.kernel
def kernel_mapping_errors(
    errors: qd.Tensor,
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        errors[i_c, i_b] = qd.Vector.zero(gs.qd_float, 4)
        if contact_state.valid[i_c, i_b] and traction_state.status[i_c, i_b] == StressTractionStatus.SUCCESS:
            force_sum, moment_sum = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
            maximum_cone, minimum_compression = gs.qd_float(0.0), gs.qd_float(0.0)
            for i_q in range(traction_info.positions.shape[0]):
                force = func_sample_traction(i_q, i_c, i_b, contact_state, traction_state, traction_info)
                force_sum += force
                moment_sum += (traction_info.positions[i_q] - contact_state.position[i_c, i_b]).cross(force)
                normal = traction_info.inward_normals[i_q]
                compression = force.dot(normal)
                tangent = force - compression * normal
                maximum_cone = qd.max(maximum_cone, tangent.norm() - contact_state.friction[i_c, i_b] * compression)
                minimum_compression = qd.min(minimum_compression, compression)
            scale = contact_state.force[i_c, i_b].norm()
            if scale > 0.0:
                errors[i_c, i_b][0] = (force_sum - contact_state.force[i_c, i_b]).norm() / scale
                errors[i_c, i_b][1] = (moment_sum - source_state.free_moment[i_c, i_b]).norm()
                errors[i_c, i_b][1] /= scale * contact_state.radius[i_c, i_b]
                errors[i_c, i_b][2] = maximum_cone / scale
                errors[i_c, i_b][3] = minimum_compression / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--ticks", type=int, nargs="+", default=[0, 100, 180, 400, 460, 560, 620])
    parser.add_argument("--projection-limit", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cuda, precision="64")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, fused_rigid_post=False, fused_pipeline=False)
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    paths = StressSurfacePaths(entry.model.info, 2)
    info = create_traction_info(entry.surface.info)
    n_contacts = entry.contacts.valid.shape[0]
    sources = create_sources(n_contacts, args.envs)
    footprints = create_footprints(paths.info.positions.shape[0], n_contacts, args.envs)
    tractions = create_tractions(info.positions.shape[0], n_contacts, args.envs)
    state = entry.model.create_state(args.envs, "full")
    errno = V(dtype=gs.qd_int, shape=(args.envs,))
    errors = V_VEC(4, dtype=gs.qd_float, shape=(n_contacts, args.envs))
    epsilon = float(np.finfo(gs.np_float).eps)
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
    captures = []
    for tick in range(args.steps):
        workload.step()
        if tick in args.ticks:
            kernel_capture_rigid_sources(entry.link.idx, entry.contacts, sources, solver.collider.collider_state)
            kernel_resolve_sources(
                args.projection_limit, epsilon, entry.contacts, sources, entry.model.info, entry.surface.info
            )
            kernel_footprint_ports(entry.contacts, sources, footprints, entry.surface.info, paths.info)
            kernel_footprint_profiles(
                entry.contacts, footprints, tractions, entry.model.info, entry.surface.info, paths.info
            )
            kernel_fit_tractions(entry.contacts, sources, tractions, info, epsilon, 1e-9, 128)
            kernel_mapping_errors(errors, entry.contacts, sources, tractions, info)
            errno.fill(0)
            kernel_scatter_tractions(
                entry.contacts, tractions, state, entry.model.info, entry.surface.info, info, errno
            )
            entry.model.recover(entry.omega, state)
            arrays = {
                "valid": qd_to_numpy(entry.contacts.valid, transpose=True, copy=True),
                "position": qd_to_numpy(entry.contacts.position, transpose=True, copy=True),
                "force": qd_to_numpy(entry.contacts.force, transpose=True, copy=True),
                "radius": qd_to_numpy(entry.contacts.radius, transpose=True, copy=True),
                "friction": qd_to_numpy(entry.contacts.friction, transpose=True, copy=True),
                "witness": qd_to_numpy(sources.witness, transpose=True, copy=True),
                "geom": qd_to_numpy(sources.geom, transpose=True, copy=True),
                "pair": qd_to_numpy(sources.pair, transpose=True, copy=True),
                "method": qd_to_numpy(sources.method, transpose=True, copy=True),
                "anchor": qd_to_numpy(entry.contacts.center, transpose=True, copy=True),
                "source_status": qd_to_numpy(sources.status, transpose=True, copy=True),
                "footprint_status": qd_to_numpy(footprints.status, transpose=True, copy=True),
                "traction_status": qd_to_numpy(tractions.status, transpose=True, copy=True),
                "evaluations": qd_to_numpy(tractions.evaluations, transpose=True, copy=True),
                "samples": qd_to_numpy(footprints.samples, transpose=True, copy=True),
                "errors": qd_to_numpy(errors, transpose=True, copy=True),
                "errno": qd_to_numpy(errno, transpose=True, copy=True),
                "nodal_force": qd_to_numpy(state.force, transpose=True, copy=True),
                "omega": qd_to_numpy(entry.omega, transpose=True, copy=True),
                "displacement": qd_to_numpy(state.displacement, transpose=True, copy=True),
                "peak": qd_to_numpy(state.peak, transpose=True, copy=True),
            }
            checkpoint = args.output.parent / f"snapshot-{tick}.npz"
            np.savez_compressed(checkpoint, **arrays)
            valid = arrays["valid"]
            loaded = valid & (np.linalg.norm(arrays["force"], axis=2) > 1e-12)
            centrifugal = -np.cross(
                arrays["omega"][:, None],
                np.cross(arrays["omega"][:, None], oracle.xyz[None] - oracle.com),
            )
            raw = np.moveaxis(arrays["nodal_force"], 0, -1).reshape(oracle.ndof, args.envs)
            raw += oracle.m @ np.moveaxis(centrifugal, 0, -1).reshape(oracle.ndof, args.envs)
            rhs = oracle.balance(raw)
            reference = np.zeros_like(rhs)
            reference[free] = factor.solve(rhs[free])
            native = arrays["displacement"].reshape(args.envs, oracle.ndof).T
            residual = np.linalg.norm(oracle.k @ native - rhs, axis=0)
            tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
            vm = qd_to_numpy(state.von_mises, transpose=True, copy=True)[..., 0]
            cpu_peaks = []
            for i_b in range(args.envs):
                expected_tensor, expected_vm = stress_field(
                    oracle.glambda, oracle.elements, reference[:, i_b], entry.options.young, entry.options.poisson
                )
                np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
                np.testing.assert_allclose(vm[i_b], expected_vm, rtol=1e-4, atol=1e-3)
                cpu_peaks.append(expected_vm.max())
            np.testing.assert_allclose(arrays["peak"], cpu_peaks, rtol=1e-4, atol=1e-3)
            assert (
                residual
                <= np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * np.linalg.norm(rhs, axis=0))
            ).all()
            record = {
                "tick": tick,
                "snapshot": checkpoint.name,
                "loaded_contacts": int(loaded.sum()),
                "source_failures": int((loaded & (arrays["source_status"] != 0)).sum()),
                "footprint_failures": int((loaded & (arrays["footprint_status"] != 0)).sum()),
                "fit_failures": int((loaded & (arrays["traction_status"] != 0)).sum()),
                "invalid_envs": int((arrays["errno"] != 0).sum()),
                "max_wrench_errors": arrays["errors"][loaded].max(axis=0).tolist() if loaded.any() else [0.0] * 4,
                "mu_values": np.unique(arrays["friction"][loaded]).tolist(),
                "max_evaluations": int(arrays["evaluations"].max()),
                "full_residual_max_N": float(residual.max()),
                "native_peaks_Pa": arrays["peak"].tolist(),
                "cpu_peaks_Pa": cpu_peaks,
            }
            captures.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Intrusive new-law snapshots on the current live old-law trajectory; no new-law throughput claim.",
                "arguments": vars(args) | {"output": str(args.output)},
                "native_contact_shape": list(entry.contacts.valid.shape),
                "native_sample_count": info.positions.shape[0],
                "reset_environments": workload.reset_count,
                "captures": captures,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [*Path("genesis/engine/solvers/rigid/stress").glob("*.py"), Path(__file__)]
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
