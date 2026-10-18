"""Matched-bank full-field Franka/egg rollouts for pinned native and external implementations.

The same controller targets and force caps drive both models. The shell remains
deformable; its vertex/SDF coupling is not replaced by prescribed contact forces.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from . import diagnostics
from .common import file_identity, identity, synchronize, write_json
from .common import vars_json
from .memory import ProcessMemory
from .rollout import make_parser, profile_run, run
from .observation import FullField


class ShellEgg:
    def __init__(self, args):
        self.n_envs = args.envs
        self.tick = 0
        self.reset_count = 0
        self.stress = False
        self.arm = np.arange(7)
        self.fingers = np.array([7, 8])
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=args.sim_dt,
                substeps=args.substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                batch_dofs_info=True,
                friction_cone=gs.friction_cone.elliptic,
                contact_resolution=gs.contact_resolution.convex,
                iterations=100,
                tolerance=1e-12,
                contact_pruning_tolerance=None,
                use_gjk_collision=False,
                enable_contact_patch=False,
            ),
            shell_options=gs.options.ShellOptions(
                n_pcg_iterations=args.iterations,
                pcg_threshold=args.tolerance,
                n_coarse_patches=args.patches,
                coarse_update_interval=args.coarse_interval,
            ),
            show_viewer=False,
        )
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(
                friction=0.1,
                coup_friction=args.coupling_friction,
            ),
        )
        self.robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
            material=gs.materials.Rigid(
                friction=0.1,
                coup_friction=args.coupling_friction,
            ),
        )
        if args.model == "shell":
            self.egg = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=args.inputs / f"egg-mid-level{args.level}.obj",
                    pos=(0.65, 0.0, 0.031),
                    decimate=False,
                ),
                material=gs.materials.Shell(
                    rho=2000.0,
                    E=1e10,
                    nu=0.3,
                    thickness=0.0005,
                    damping=args.damping,
                ),
            )
        else:
            assert args.model == "rigid" and args.native_assets is not None
            self.egg = self.scene.add_entity(
                morph=gs.morphs.URDF(
                    file=args.native_assets / f"level{args.level}/egg_shell.urdf",
                    pos=(0.65, 0.0, 0.031),
                    align=False,
                    convexify=True,
                    decimate=False,
                ),
                material=gs.materials.Rigid(
                    friction=0.6,
                ),
            )
        self.scene.build(n_envs=args.envs)
        self.robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], self.arm)
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], self.arm)
        self.robot.set_dofs_kp([100, 100], self.fingers)
        self.robot.set_dofs_kv([10, 10], self.fingers)
        with np.load(args.conditions, allow_pickle=False) as bank:
            ids = np.arange(args.envs) % len(bank["position"])
            self.initial_joints = torch.as_tensor(bank["initial_joints"][ids], dtype=gs.tc_float, device=gs.device)
            self.pick_joints = torch.as_tensor(bank["pick_joints"][ids], dtype=gs.tc_float, device=gs.device)
            self.lift_joints = torch.as_tensor(bank["lift_joints"][ids], dtype=gs.tc_float, device=gs.device)
            self.slide_joints = torch.as_tensor(bank["slide_joints"][ids], dtype=gs.tc_float, device=gs.device)
            self.delays = bank["delays"][ids]
            self.initial_position = torch.as_tensor(bank["position"][ids], dtype=gs.tc_float, device=gs.device)
            quaternion = torch.as_tensor(bank["quaternion"][ids], dtype=gs.tc_float, device=gs.device)
            friction = torch.as_tensor(bank["friction"][ids], dtype=gs.tc_float, device=gs.device)
        self.device_delays = torch.as_tensor(self.delays, dtype=gs.tc_float, device=gs.device)
        self.robot.set_qpos(self.initial_joints)
        if args.model == "shell":
            reference = self.egg.init_verts - np.array([0.65, 0.0, 0.031])
            # Public mesh setter takes world coordinates. Rotate about the same link origin.
            rotations = gu.quat_to_R(quaternion)
            self.initial_vertices = torch.as_tensor(
                reference, dtype=gs.tc_float, device=gs.device
            ) @ rotations.transpose(1, 2)
            self.initial_vertices += self.initial_position[:, None]
            self.egg.set_verts_pos(self.initial_vertices)
        else:
            self.egg.set_pos(self.initial_position)
            self.egg.set_quat(quaternion)
            self.egg.set_friction_ratio(friction)
        self.reset_state = self.scene.get_state()
        self.observation = FullField.allocate(self.scene.sim.shell_solver) if args.model == "shell" else None
        if args.instrumented:
            diagnostics.configure(self.scene.sim.shell_solver)

    def reset(self, ids=None):
        self.scene.reset(state=self.reset_state, envs_idx=ids)
        self.reset_count += self.n_envs if ids is None else len(ids)


def main():
    args = make_parser().parse_args()
    assert args.model in ("shell", "rigid")
    assert args.output_mode == "full"
    args.output.mkdir(parents=True, exist_ok=False)
    memory = ProcessMemory(args.memory)
    memory.start()
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    started = perf_counter()
    workload = ShellEgg(args)
    synchronize()
    build_seconds = perf_counter() - started
    result = (
        profile_run(args, workload, args.model == "shell")
        if args.trace
        else run(args, workload, args.model == "shell", diagnostics.ACTIVE)
    )
    result.update(
        kind="matched-franka-egg",
        args=vars_json(args),
        identity=identity(),
        condition_bank=file_identity(args.conditions),
        build_seconds=build_seconds,
        observation_bytes=workload.observation.nbytes if workload.observation is not None else 0,
        source_scope="instrumented-read-only" if args.instrumented else "original-plus-output-sidecar",
        process_memory=memory.finish(),
        rigid_dofs=workload.scene.rigid_solver.n_dofs,
        shell_numerics={
            "vertices": workload.egg.n_verts,
            "faces": workload.egg.n_faces,
            "patches": workload.egg.patches.n_patches if workload.egg.patches is not None else 0,
            "fracture_enabled": workload.scene.sim.shell_solver._static_config.has_fracture,
            "plasticity_enabled": workload.scene.sim.shell_solver._static_config.has_plasticity,
            "allocated_vertices": workload.egg.n_verts_max,
            "lumped_shell_mass_kg": float(workload.egg.topology.faces_mass.sum()),
            "coupling_friction": args.coupling_friction,
            "bank_friction_ratio_applied_to_shell_coupling": False,
        }
        if args.model == "shell"
        else None,
    )
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
