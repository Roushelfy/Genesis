import argparse
import os
import time

import numpy as np
import pynvml
import torch
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=60, help="Number of timed steps, the egg held and lifted")
    parser.add_argument("--rigid-egg", action="store_true", help="Grasp a rigid egg instead, as a baseline")
    parser.add_argument("--substeps", type=int, default=10, help="Number of substeps per control step")
    parser.add_argument("--pcg-tolerance", type=float, default=1e-4, help="Relative tolerance of the linear solves")
    parser.add_argument("--contact-stiffness", type=float, default=25.0, help="Stiffness ratio of the egg contacts")
    args = parser.parse_args()

    # The benchmark measures the throughput of a thousand parallel environments, which needs a GPU.
    gs.init(backend=gs.gpu, precision="32", performance_mode=True, seed=0, logging_level="warning")

    EGG_LENGTH, EGG_WIDTH = 0.06, 0.045
    sphere = trimesh.creation.icosphere(subdivisions=2)
    egg_verts = sphere.vertices * 0.5 * np.array([EGG_LENGTH, EGG_WIDTH, EGG_WIDTH])
    egg_verts[:, 1:] *= 1.0 - 0.12 * sphere.vertices[:, :1]
    os.makedirs("out", exist_ok=True)
    egg_path = os.path.join("out", "franka_egg_benchmark_egg.obj")
    trimesh.Trimesh(egg_verts, sphere.faces, process=False).export(egg_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=args.substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
        ),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=args.pcg_tolerance,
            contact_stiffness=args.contact_stiffness,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    if args.rigid_egg:
        # A solid egg of the same shape and mass, about 70 g
        egg = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=egg_path,
                pos=(0.55, 0.0, 0.5 * EGG_WIDTH + 1e-3),
            ),
            material=gs.materials.Rigid(
                rho=1100.0,
            ),
        )
    else:
        egg = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=egg_path,
                pos=(0.55, 0.0, 0.5 * EGG_WIDTH + 1e-3),
            ),
            material=gs.materials.Shell(
                rho=2e4,
                E=1e10,
                nu=0.3,
                thickness=4e-4,
                damping=1e-4,
                tensile_strength=1e7,
                fracture=False,
            ),
        )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            coup_friction=0.6,
            gravity_compensation=1.0,
        ),
    )
    n_envs = args.num_envs
    scene.build(n_envs=n_envs)

    arm_dofs = np.arange(7)
    finger_dofs = np.arange(7, 9)
    franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000])
    franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 100, 100])
    franka.set_dofs_force_range([-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12], arm_dofs)

    # Every environment grasps the egg off-center, at its own yaw and with its own grip force
    rng = np.random.default_rng(0)
    grasp_offset = np.stack((rng.uniform(-0.015, 0.015, n_envs), rng.uniform(-0.004, 0.004, n_envs)), axis=-1)
    grasp_yaw = rng.uniform(-0.3, 0.3, n_envs)
    grip_force = rng.uniform(5.0, 40.0, n_envs)
    franka.set_dofs_force_range(-np.stack((grip_force,) * 2, -1), np.stack((grip_force,) * 2, -1), finger_dofs)
    hand = franka.get_link("hand")
    hand_quat = np.stack(
        (np.zeros(n_envs), np.cos(0.5 * grasp_yaw), np.sin(0.5 * grasp_yaw), np.zeros(n_envs)), axis=-1
    )
    grasp_pos = np.concatenate(((0.55, 0.0) + grasp_offset, np.full((n_envs, 1), 0.5 * EGG_WIDTH + 0.103)), axis=-1)
    qpos_seed = np.array([0.0, -0.3, 0.0, -2.4, 0.0, 2.1, 0.8, 0.04, 0.04])
    qpos_above = tensor_to_array(
        franka.inverse_kinematics(link=hand, pos=grasp_pos + (0.0, 0.0, 0.08), quat=hand_quat, init_qpos=qpos_seed)
    )
    qpos_grasp = tensor_to_array(
        franka.inverse_kinematics(link=hand, pos=grasp_pos, quat=hand_quat, init_qpos=qpos_above)
    )
    qpos = qpos_above.copy()
    qpos[:, 7:] = 0.04
    franka.set_qpos(qpos)

    # The untimed steps descend over the egg, then close the fingers on it, then the timed ones lift it
    n_steps_grasp = 120
    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    iterations = []
    for i_step in range(n_steps_grasp + args.steps):
        time_step = i_step * scene.dt
        ratio = min(time_step / 0.5, 1.0)
        qpos = (1.0 - ratio) * qpos_above + ratio * qpos_grasp
        if time_step >= 1.2:
            ratio = min((time_step - 1.2) / 0.6, 1.0)
            qpos = (1.0 - ratio) * qpos_grasp + ratio * qpos_above
        franka.control_dofs_position(qpos[:, arm_dofs], arm_dofs)
        franka.control_dofs_position(0.04 * min(max((1.2 - time_step) / 0.4, 0.0), 1.0), finger_dofs)
        if i_step == n_steps_grasp:
            torch.cuda.synchronize()
            time_start = time.perf_counter()
        scene.step()
        # The iteration counts stay on the device until the timing ends
        if not args.rigid_egg and i_step >= n_steps_grasp:
            iterations.append(scene.sim.shell_solver.get_envs_pcg_iterations())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - time_start

    memory_used = sum(
        process.usedGpuMemory
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(device)
        if process.pid == os.getpid()
    )
    print(f"device {torch.cuda.get_device_name()}, {n_envs} environments, {'rigid' if args.rigid_egg else 'shell'} egg")
    print(f"timed steps {args.steps}, wall time {elapsed:.3f} s, compilation and grasp approach excluded")
    print(
        f"scene steps per second {args.steps / elapsed:.2f}, "
        f"environment steps per second {n_envs * args.steps / elapsed:.1f}"
    )
    print(f"GPU memory of the process {memory_used / 2**20:.0f} MiB")
    if not args.rigid_egg:
        iterations = tensor_to_array(torch.stack(iterations))
        print(
            f"PCG iterations of the last substep of a step: mean {iterations.mean():.1f}, slowest environment "
            f"{iterations.max(axis=-1).mean():.1f}"
        )
        envs_solver_failure = tensor_to_array(scene.sim.shell_solver.get_envs_solver_failure())
        print(f"environments whose solver failed {(envs_solver_failure > 0).sum()}")
        print(f"environments whose egg failed {(tensor_to_array(egg.get_peak_damage()) >= 1.0).sum()}")


if __name__ == "__main__":
    main()
