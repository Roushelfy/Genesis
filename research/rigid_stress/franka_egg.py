"""Franka/egg scene seed using current public Genesis APIs.

The CPU recovery adapter exposes finite-patch force/moment discrepancies.
It is not an accepted end-to-end benchmark until the agent validates the
trajectory, contact footprint, wrench mapping and substep sampling. CUDA
physics + CPU recovery is an integration/debug mode, NOT a GPU stress path.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils.misc import tensor_to_array

from .assets import write_egg_assets
from .cpu import ContactBatch, EggConfig, EggRecoveryCPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--stress", choices=("cpu", "off"), default="cpu")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-radius-m", type=float, default=0.006)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("research/rigid_stress/runs/franka_seed.json"))
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 100:
        raise ValueError("Positive environment count and at least 100 steps required")
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(args.envs, EggConfig(level=args.level), rtol=args.rtol)
        assets = args.output.parent / "assets"
        urdf = write_egg_assets(model, assets)
        gs.init(backend=gs.gpu if args.backend == "cuda" else gs.cpu, precision="32")
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=args.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(
                contact_pruning_tolerance=None, enable_torsional_friction=False,
                enable_rolling_friction=False, use_hibernation=False,
            ),
            viewer_options=gs.options.ViewerOptions(camera_pos=(1.4, -1.1, 0.8),
                                                   camera_lookat=(0.55, 0., 0.1)),
            show_viewer=args.vis,
        )
        scene.add_entity(gs.morphs.Plane())
        franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        egg = scene.add_entity(
            gs.morphs.URDF(file=str(urdf.resolve()), pos=(0.65, 0., 0.031),
                           fixed=False, align=False, convexify=True, decimate=False),
            material=gs.materials.Rigid(friction=0.6),
        )
        scene.build(n_envs=args.envs, env_spacing=(1., 1.))
        arm = [franka.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        fingers = [franka.get_joint(f"finger_joint{i}").dofs_idx_local[0] for i in (1, 2)]
        franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], arm)
        franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], arm)
        franka.set_dofs_kp([100., 100.], fingers)
        franka.set_dofs_kv([10., 10.], fingers)
        franka.set_dofs_force_range([-4., -4.], [4., 4.], fingers)
        initial = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, .04, .04],
                               dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1)
        franka.set_qpos(initial)
        # Different centres/amplitudes are allowed inputs, never an algorithm
        # dependency. The complete production suite must also vary orientations,
        # patch sizes, friction, contact transitions and environment phases.
        centers = torch.tensor([.65, 0., .031], dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1)
        centers[:, 1] += torch.linspace(-.006, .006, args.envs, device=gs.device) if args.envs > 1 else 0
        egg.set_pos(centers)
        hand = franka.get_link("hand")
        orientation = torch.tensor([0., 1., 0., 0.], dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1)
        pickup = centers.clone()
        pickup[:, 2] = .135
        raised = pickup.clone()
        raised[:, 2] += .12
        q_pick = franka.inverse_kinematics(link=hand, pos=pickup, quat=orientation)
        q_lift = franka.inverse_kinematics(link=hand, pos=raised, quat=orientation)
        records = []
        start = perf_counter()
        for step in range(args.steps):
            phase = step / (args.steps - 1)
            blend = np.clip((phase - .3) / .2, 0., 1.)
            blend = blend**3 * (10 - 15 * blend + 6 * blend**2)
            target = q_pick * (1 - blend) + q_lift * blend
            franka.control_dofs_position(target[:, arm], arm)
            grip = .04 if phase < .15 or phase > .9 else .018
            franka.control_dofs_position(torch.full((args.envs, 2), grip, dtype=gs.tc_float, device=gs.device), fingers)
            scene.step()
            if args.stress == "cpu":
                # Keep filtering/transforms batched on device; transfer once per
                # selected field into this explicitly CPU-only reference path.
                contacts = egg.get_contacts(is_padded=True)
                on_a = (contacts["geom_a"] >= egg.geom_start) & (contacts["geom_a"] < egg.geom_end)
                force = torch.where(on_a[..., None], contacts["force_a"], contacts["force_b"])
                quat = egg.get_quat()
                local_position = gu.inv_transform_by_quat(contacts["position"] - egg.get_pos()[:, None], quat[:, None])
                local_force = gu.inv_transform_by_quat(force, quat[:, None])
                cpu_contacts = ContactBatch(
                    tensor_to_array(local_position), tensor_to_array(local_force),
                    np.full(contacts["valid_mask"].shape, args.patch_radius_m),
                    tensor_to_array(contacts["valid_mask"]),
                )
                loads = model.map_contacts(cpu_contacts)
                gravity = gu.inv_transform_by_quat(torch.tensor([0., 0., -9.81], dtype=gs.tc_float,
                                                                 device=gs.device).repeat(args.envs, 1), quat)
                gravity_cpu = tensor_to_array(gravity)
                raw = loads.nodal_force_n.copy(order="F")
                for env in range(args.envs):
                    raw[:, env] += model.fem.m @ np.tile(gravity_cpu[env], len(model.fem.xyz))
                omega = tensor_to_array(gu.inv_transform_by_quat(egg.get_ang(), quat))
                result = model.recover(model.compatible_rhs(raw, omega))
                records.append({"step": step, "peak_pa": result.peak_pa.tolist(),
                                "relative_residual_nonzero": result.relative_residual.tolist(),
                                "finite_patch_point_moment_difference_nm":
                                    np.linalg.norm(loads.resultant_moment_nm - loads.input_moment_nm, axis=1).tolist()})
        if args.backend == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "integration_seed_not_accepted_benchmark", "backend": args.backend,
            "stress_backend": args.stress, "envs": args.envs, "steps": args.steps,
            "dt": args.dt, "substeps": 1, "seconds_with_first_compile": elapsed,
            "env_steps_per_second_with_first_compile": args.steps * args.envs / elapsed,
            "stress_mesh_converged": False, "material_calibrated_to_real_egg": False,
            "sampling": "last rigid contact solve of each step; substeps=1 only",
            "limitations": "validate grasp success, wrench/footprint consistency and impacts before acceptance",
            "records": records,
        }, indent=2))
        print(args.output)


if __name__ == "__main__":
    main()
