"""Prescribed-load relaxation using original PR22 kernels, without a contact-solver change.

PR22 has no public nodal-force input. This separate harness inserts dt*f into
its original right-hand side, preserving its force/Hessian/coarse/PCG/integration
order. Zeroing velocities between steps is a static-relaxation procedure; these
times are not presented as an original scene-step throughput.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
import torch
from scipy.spatial import cKDTree

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class
from genesis.utils.misc import tensor_to_array

from . import diagnostics
from .common import file_identity, identity, synchronize, vars_json, write_json
from .observation import FullField


@qd.kernel
def add_prescribed_force(dt: float, force: qd.types.ndarray(ndim=3), scratch: array_class.ShellScratch):
    for i_v, i_b in qd.ndrange(force.shape[0], force.shape[1]):
        scratch.verts_rhs[i_v, i_b] += dt * qd.Vector([force[i_v, i_b, 0], force[i_v, i_b, 1], force[i_v, i_b, 2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--geometry", choices=("egg", "sphere"), default="egg")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    dt = 0.01
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, 0.0),
        ),
        shell_options=gs.options.ShellOptions(
            n_pcg_iterations=args.iterations,
            pcg_threshold=args.tolerance,
            coarse_update_interval=1,
        ),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=args.inputs / f"{args.geometry}-mid-level{args.level}.obj",
            decimate=False,
        ),
        material=gs.materials.Shell(
            rho=2000.0,
            E=1e10,
            nu=0.3,
            thickness=0.0005,
        ),
    )
    scene.build(n_envs=1)
    with np.load(args.loads / f"level{args.level}.npz") as saved:
        shell_force = saved["shell_force_n"]
        reference_tensor = saved["reference_tensor_pa"]
    with np.load(args.inputs / f"{args.geometry}-level{args.level}.npz") as saved:
        midpoint, faces = saved["mid_vertices"], saved["mid_faces"]
    distance, vertices = cKDTree(midpoint).query(entity.init_verts)
    assert len(np.unique(vertices)) == len(vertices) == len(midpoint)
    actual_faces = tensor_to_array(entity.get_faces())[0]
    centers = entity.init_verts[actual_faces].mean(axis=1)
    face_distance, face_order = cKDTree(midpoint[faces].mean(axis=1)).query(centers)
    assert len(np.unique(face_order)) == len(face_order) == len(faces)
    force = torch.as_tensor(shell_force[vertices, None], dtype=gs.tc_float, device=gs.device)
    solver = scene.sim.shell_solver
    state, scratch, info, config = solver.shell_state, solver.shell_scratch, solver.shell_info, solver._static_config
    observation = FullField.allocate(solver)
    diagnostics.configure(solver)
    history = []
    previous = entity.init_verts.copy()
    synchronize()
    started = perf_counter()
    for step in range(args.steps):
        entity.set_verts_vel(0.0)
        diagnostics.ACTIVE.clear()
        shell.kernel_shell_compute_forces(dt, solver._gravity, state, scratch, info)
        add_prescribed_force(dt, force, scratch)
        if config.has_coarse_space:
            shell.kernel_shell_coarse_assemble(state, scratch, info)
            shell.kernel_shell_coarse_factorize(state, scratch, info, config)
        shell.kernel_shell_system_product(scratch.verts_dv, scratch.verts_Ap, state, scratch, info)
        shell.kernel_shell_pcg_init(args.tolerance, state, scratch, info, config)
        for _ in range(args.iterations):
            diagnostics.iteration(solver)
            shell.kernel_shell_system_product(scratch.verts_p, scratch.verts_Ap, state, scratch, info)
            shell.kernel_shell_pcg_update(state, scratch, info, config)
        diagnostics.after_solve(solver)
        shell.kernel_shell_apply_dv(state, scratch)
        shell.kernel_shell_integrate(dt, state, scratch)
        observation.update(solver)
        current = tensor_to_array(entity.get_verts_pos())[0]
        history.append(
            {
                "step": step,
                "increment_max_m": float(np.linalg.norm(current - previous, axis=1).max()),
                "peak_pa": float(tensor_to_array(observation.peaks)[0]),
                "iterations": int(tensor_to_array(diagnostics.ACTIVE.counts)[0]),
                "cap_exhausted": int(tensor_to_array(diagnostics.ACTIVE.caps)[0]),
                "residuals": tensor_to_array(diagnostics.ACTIVE.residuals)[0].tolist(),
            }
        )
        previous = current.copy()
    synchronize()
    seconds = perf_counter() - started
    tensor = tensor_to_array(observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
    target = reference_tensor[face_order]
    error = np.linalg.norm(tensor - target) / np.linalg.norm(target)
    np.savez_compressed(
        args.output / "fields.npz",
        shell_tensor_pa=tensor,
        reference_tensor_pa=target,
        shell_vm_pa=tensor_to_array(observation.von_mises)[:, 0].reshape(-1, 3),
        rest_vertices=entity.init_verts,
        final_vertices=current,
        faces=actual_faces,
        applied_force_n=shell_force[vertices],
        face_order=face_order,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "prescribed-finite-compression" if args.geometry == "egg" else "prescribed-spherical-inflation",
            "args": vars_json(args),
            "identity": identity(),
            "load_source": file_identity(args.loads / f"level{args.level}.npz"),
            "maximum_vertex_input_difference_m": float(distance.max()),
            "maximum_face_centroid_input_difference_m": float(face_distance.max()),
            "unweighted_common_fiber_tensor_relative_error": float(error),
            "static_relaxation_seconds_including_first_compile": seconds,
            "history": history,
            "scope": "prescribed-input harness using original kernels; no contact or fracture/plasticity",
        },
    )


if __name__ == "__main__":
    main()
