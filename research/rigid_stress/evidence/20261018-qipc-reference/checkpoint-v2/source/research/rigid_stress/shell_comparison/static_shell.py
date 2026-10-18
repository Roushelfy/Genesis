"""Time accepted shell statics under the same finite compression as the native core."""

from time import perf_counter

import numpy as np
import torch
from scipy.spatial import cKDTree

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils.misc import qd_to_torch, tensor_to_array

from .common import synchronize
from .observation import FullField
from .prescribed import add_prescribed_force
from .static_cost import make_parser, measure


class ShellStatic:
    def __init__(self, args):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, 0.0),
            ),
            shell_options=gs.options.ShellOptions(
                n_pcg_iterations=300,
                pcg_threshold=1e-8,
                coarse_update_interval=1,
            ),
            show_viewer=False,
        )
        self.entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=args.inputs / f"egg-mid-level{args.level}.obj",
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
        self.solver = scene.sim.shell_solver
        self.observation = FullField.allocate(self.solver)
        with np.load(args.loads / f"level{args.level}.npz") as saved:
            force = saved["shell_force_n"]
        with np.load(args.inputs / f"egg-level{args.level}.npz") as saved:
            midpoint = saved["mid_vertices"]
        _, order = cKDTree(midpoint).query(self.entity.init_verts)
        assert len(np.unique(order)) == len(order)
        self.base_force = torch.as_tensor(force[order, None], dtype=gs.tc_float, device=gs.device)
        self.force = self.base_force.clone()
        self.rest = torch.as_tensor(self.entity.init_verts, dtype=gs.tc_float, device=gs.device)
        self.description = {
            "method": "original-kernel static-relaxation harness",
            "outer_steps": 3,
            "dt": 0.01,
            "pcg_cap": 300,
            "pcg_threshold": 1e-8,
            "coarse_update_interval": 1,
            "initial_guess": "zero from rest",
            "static_relative_force_budget": 1e-8,
            "nodes": len(order),
            "note": "No contact solver, plasticity or fracture. This is separate from original scene-step rates.",
        }

    def solve(self, amplitude):
        solver = self.solver
        state, scratch, info, config = (
            solver.shell_state,
            solver.shell_scratch,
            solver.shell_info,
            solver._static_config,
        )
        synchronize()
        started = perf_counter()
        self.entity.set_verts_pos(self.rest)
        self.entity.set_verts_vel(0.0)
        scratch.verts_dv.fill(0.0)
        self.force.copy_(self.base_force * amplitude)
        cap_flags = []
        for _ in range(3):
            self.entity.set_verts_vel(0.0)
            shell.kernel_shell_compute_forces(0.01, solver._gravity, state, scratch, info)
            add_prescribed_force(0.01, self.force, scratch)
            if config.has_coarse_space:
                shell.kernel_shell_coarse_assemble(state, scratch, info)
                shell.kernel_shell_coarse_factorize(state, scratch, info, config)
            shell.kernel_shell_system_product(scratch.verts_dv, scratch.verts_Ap, state, scratch, info)
            shell.kernel_shell_pcg_init(1e-8, state, scratch, info, config)
            for _ in range(300):
                shell.kernel_shell_system_product(scratch.verts_p, scratch.verts_Ap, state, scratch, info)
                shell.kernel_shell_pcg_update(state, scratch, info, config)
            cap_flags.append(qd_to_torch(scratch.envs_is_solving, copy=True))
            shell.kernel_shell_apply_dv(state, scratch)
            shell.kernel_shell_integrate(0.01, state, scratch)
        self.observation.update(solver)
        self.entity.set_verts_vel(0.0)
        shell.kernel_shell_compute_forces(0.01, solver._gravity, state, scratch, info)
        add_prescribed_force(0.01, self.force, scratch)
        residual = torch.linalg.vector_norm(qd_to_torch(scratch.verts_rhs)) / (
            0.01 * torch.linalg.vector_norm(self.force)
        )
        synchronize()
        seconds = perf_counter() - started
        relative = float(tensor_to_array(residual))
        cap_exhausted = any(tensor_to_array(flag).any() for flag in cap_flags)
        assert not cap_exhausted and relative <= 1e-8
        return {
            "amplitude": amplitude,
            "seconds": seconds,
            "numerically_valid": True,
            "complete_relative_force_residual": relative,
            "cap_exhausted": bool(cap_exhausted),
        }

    def export(self, output):
        np.savez_compressed(
            output / "fields.npz",
            tensor_pa=tensor_to_array(self.observation.tensors)[:, 0],
            vm_pa=tensor_to_array(self.observation.von_mises)[:, 0],
            rest_vertices=self.entity.init_verts,
            final_vertices=tensor_to_array(self.entity.get_verts_pos())[0],
            faces=tensor_to_array(self.entity.get_faces())[0],
        )


def main():
    args = make_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    started = perf_counter()
    solver = ShellStatic(args)
    synchronize()
    solver.description["setup_seconds_in_this_process"] = perf_counter() - started
    measure(args, solver)


if __name__ == "__main__":
    main()
