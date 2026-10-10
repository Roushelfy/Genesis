import argparse
import os

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("-t", "--seconds", type=float, default=3.0, help="Simulated duration of the grasp, in s")
    parser.add_argument("--dt", type=float, default=1e-2, help="Control timestep, in s")
    parser.add_argument("--substeps", type=int, default=10, help="Number of substeps per control step")
    parser.add_argument("--fracture", action="store_true", help="Tear the shell where it fails instead of reporting it")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="warning")

    # A low-resolution hollow egg: an icosphere whose cross-section tapers from the blunt to the pointed end
    EGG_LENGTH, EGG_WIDTH = 0.06, 0.045
    sphere = trimesh.creation.icosphere(subdivisions=2)
    egg_verts = sphere.vertices * 0.5 * np.array([EGG_LENGTH, EGG_WIDTH, EGG_WIDTH])
    egg_verts[:, 1:] *= 1.0 - 0.12 * sphere.vertices[:, :1]
    os.makedirs("out", exist_ok=True)
    egg_path = os.path.join("out", "franka_egg_grasp_egg.obj")
    trimesh.Trimesh(egg_verts, sphere.faces, process=False).export(egg_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.5, 0.4),
            camera_lookat=(0.55, 0.0, 0.08),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=egg_path,
            pos=(0.55, 0.0, 0.5 * EGG_WIDTH + 1e-3),
        ),
        # The density of the shell carries the mass of a whole egg, about 60 g, contents included
        material=gs.materials.Shell(
            rho=2e4,
            E=1e10,
            nu=0.3,
            thickness=4e-4,
            damping=1e-4,
            tensile_strength=1e7,
            fracture=args.fracture,
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
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    arm_dofs = np.arange(7)
    finger_dofs = np.arange(7, 9)
    franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000])
    franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 100, 100])
    franka.set_dofs_force_range([-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12], arm_dofs)

    # Every environment grasps the egg off-center, at its own yaw, with its own grip force. The fingers close by
    # position control, the force range of their joints capping the grip. The grip of the second half of the
    # environments loosens while carrying the egg, which lets it slide between the fingers.
    rng = np.random.default_rng(0)
    grasp_offset = np.stack((rng.uniform(-0.015, 0.015, n_envs), rng.uniform(-0.004, 0.004, n_envs)), axis=-1)
    grasp_yaw = rng.uniform(-0.3, 0.3, n_envs)
    grip_force = np.linspace(5.0, 60.0, n_envs)
    loose_force = np.where(np.arange(n_envs) >= n_envs // 2, 0.02, 1.0) * grip_force

    hand = franka.get_link("hand")
    hand_quat = np.stack(
        (np.zeros(n_envs), np.cos(0.5 * grasp_yaw), np.sin(0.5 * grasp_yaw), np.zeros(n_envs)), axis=-1
    )
    # The fingertip pads are centered 0.103 m below the hand, at the height of the center of the egg
    grasp_pos = np.concatenate(((0.55, 0.0) + grasp_offset, np.full((n_envs, 1), 0.5 * EGG_WIDTH + 0.103)), axis=-1)
    # Hand positions of the motion keyframes: above the egg, at the grasp, lifted, then carried sideways
    keyframes_offset = np.array([[0.0, 0.0, 0.08], [0.0, 0.0, 0.0], [0.0, 0.0, 0.08], [0.0, 0.12, 0.08]])
    # Seeding the inverse kinematics with an elbow-up configuration keeps every environment on the same arm branch
    keyframes_qpos = [np.array([0.0, -0.3, 0.0, -2.4, 0.0, 2.1, 0.8, 0.04, 0.04])]
    for offset in keyframes_offset:
        qpos = franka.inverse_kinematics(
            link=hand,
            pos=grasp_pos + offset,
            quat=hand_quat,
            init_qpos=keyframes_qpos[-1][0] if keyframes_qpos[-1].ndim > 1 else keyframes_qpos[-1],
        )
        keyframes_qpos.append(tensor_to_array(qpos))
    keyframes_qpos = keyframes_qpos[1:]
    # Phases as (end time, start keyframe, end keyframe, finger opening at the start and the end, grip force): the
    # descent, a pause for the arm to settle, the closing, the lift, the carry, the loosened carry, and the release
    phases = (
        (0.5, 0, 1, 0.04, 0.04, grip_force),
        (0.8, 1, 1, 0.04, 0.04, grip_force),
        (1.2, 1, 1, 0.04, 0.0, grip_force),
        (1.8, 1, 2, 0.0, 0.0, grip_force),
        (2.4, 2, 3, 0.0, 0.0, grip_force),
        (2.8, 3, 3, 0.0, 0.0, loose_force),
        (args.seconds, 3, 3, 0.04, 0.04, grip_force),
    )

    qpos = keyframes_qpos[0].copy()
    qpos[:, 7:] = 0.04
    franka.set_qpos(qpos)

    contact_force_max = np.zeros(n_envs)
    i_phase_prev = -1
    for i_step in range(int(args.seconds / args.dt)):
        time = i_step * args.dt
        phase_start = 0.0
        for i_phase, (phase_end, i_k0, i_k1, opening_0, opening_1, force) in enumerate(phases):
            if time < phase_end or i_phase == len(phases) - 1:
                ratio = min((time - phase_start) / (phase_end - phase_start), 1.0)
                qpos = (1.0 - ratio) * keyframes_qpos[i_k0] + ratio * keyframes_qpos[i_k1]
                opening = (1.0 - ratio) * opening_0 + ratio * opening_1
                franka.control_dofs_position(qpos[:, arm_dofs], arm_dofs)
                franka.control_dofs_position(np.full((n_envs, 2), opening), finger_dofs)
                if i_phase != i_phase_prev:
                    forces = np.stack((force, force), axis=-1)
                    franka.set_dofs_force_range(-forces, forces, finger_dofs)
                    i_phase_prev = i_phase
                break
            phase_start = phase_end
        scene.step()

        verts_force = tensor_to_array(egg.get_verts_contact_force())
        contact_force_max = np.maximum(contact_force_max, np.linalg.norm(verts_force, axis=-1).sum(axis=-1))
        if i_step % 25 == 0 or i_step == int(args.seconds / args.dt) - 1:
            egg_pos = tensor_to_array(egg.get_verts_pos())[:, : egg.n_verts].mean(axis=-2)
            print(
                f"t={time + args.dt:.2f}s egg height {np.array2string(egg_pos[:, 2], precision=3)} "
                f"peak damage {np.array2string(tensor_to_array(egg.get_peak_damage()), precision=3)} "
                f"pcg iterations {tensor_to_array(scene.sim.shell_solver.get_envs_pcg_iterations())}"
            )

    print("grip force (N)", np.array2string(grip_force, precision=1))
    print("largest total contact force on the egg (N)", np.array2string(contact_force_max, precision=1))
    print("peak damage index", np.array2string(tensor_to_array(egg.get_peak_damage()), precision=3))
    print("first failed face", tensor_to_array(egg.get_failure_face()))
    print("linear solver failures", tensor_to_array(scene.sim.shell_solver.get_envs_solver_failure()))


if __name__ == "__main__":
    main()
