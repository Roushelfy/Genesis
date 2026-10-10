import argparse
import os

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


def run_grasp(scene, egg, franka, keyframes_qpos, clock, n_steps):
    """Step the scripted grasp of every environment on its own clock, and return the observables of every step."""
    arm_dofs = np.arange(7)
    finger_dofs = np.arange(7, 9)
    trajectory = []
    for _ in range(n_steps):
        time = clock * 1e-2
        ratio_descend = np.clip(time / 0.5, 0.0, 1.0)[:, None]
        ratio_lift = np.clip((time - 1.2) / 0.6, 0.0, 1.0)[:, None]
        qpos = np.where(
            time[:, None] < 0.5,
            (1.0 - ratio_descend) * keyframes_qpos[0] + ratio_descend * keyframes_qpos[1],
            (1.0 - ratio_lift) * keyframes_qpos[1] + ratio_lift * keyframes_qpos[0],
        )
        franka.control_dofs_position(qpos[:, arm_dofs], arm_dofs)
        opening = 0.04 * (1.0 - np.clip((time - 0.8) / 0.4, 0.0, 1.0))
        franka.control_dofs_position(np.repeat(opening[:, None], 2, axis=1), finger_dofs)
        scene.step()
        clock += 1
        trajectory.append(
            (
                tensor_to_array(egg.get_verts_pos())[:, : egg.n_verts],
                np.linalg.norm(tensor_to_array(egg.get_verts_contact_force()), axis=-1).sum(axis=-1),
                tensor_to_array(egg.get_peak_damage()),
            )
        )
    return trajectory


def max_deviation(trajectory_a, trajectory_b, envs_idx):
    """Largest deviation of the egg vertex positions (um), the total contact force (N) and the peak damage index."""
    return [
        max(np.abs(obs_a[k][envs_idx] - obs_b[k][envs_idx]).max() for obs_a, obs_b in zip(trajectory_a, trajectory_b))
        * scale
        for k, scale in enumerate((1e6, 1.0, 1.0))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("-s", "--steps", type=int, default=60, help="Number of steps of every compared run")
    parser.add_argument("--snapshot-step", type=int, default=100, help="Step of the snapshot, mid-grasp")
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.steps, args.snapshot_step = 2, 2

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="warning", seed=0)

    EGG_LENGTH, EGG_WIDTH = 0.06, 0.045
    sphere = trimesh.creation.icosphere(subdivisions=2)
    egg_verts = sphere.vertices * 0.5 * np.array([EGG_LENGTH, EGG_WIDTH, EGG_WIDTH])
    egg_verts[:, 1:] *= 1.0 - 0.12 * sphere.vertices[:, :1]
    os.makedirs("out", exist_ok=True)
    egg_path = os.path.join("out", "shell_egg_consistency_egg.obj")
    trimesh.Trimesh(egg_verts, sphere.faces, process=False).export(egg_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
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
    n_envs = 4
    scene.build(n_envs=n_envs)

    franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000])
    franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 100, 100])
    franka.set_dofs_force_range([-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12], np.arange(7))
    # Every environment grasps the egg at its own offset and yaw, with its own grip force
    grip_force = np.array([10.0, 25.0, 40.0, 15.0])
    franka.set_dofs_force_range(-np.stack((grip_force,) * 2, -1), np.stack((grip_force,) * 2, -1), np.arange(7, 9))
    grasp_yaw = np.array([0.0, 0.2, -0.25, 0.1])
    hand_quat = np.stack((np.zeros(n_envs), np.cos(0.5 * grasp_yaw), np.sin(0.5 * grasp_yaw), np.zeros(n_envs)), -1)
    grasp_offset = np.array([[0.0, 0.0], [0.01, 0.002], [-0.008, -0.003], [0.004, 0.0]])
    grasp_pos = np.concatenate(((0.55, 0.0) + grasp_offset, np.full((n_envs, 1), 0.5 * EGG_WIDTH + 0.103)), axis=-1)
    hand = franka.get_link("hand")
    qpos_seed = np.array([0.0, -0.3, 0.0, -2.4, 0.0, 2.1, 0.8, 0.04, 0.04])
    qpos_above = tensor_to_array(
        franka.inverse_kinematics(link=hand, pos=grasp_pos + (0.0, 0.0, 0.08), quat=hand_quat, init_qpos=qpos_seed)
    )
    qpos_grasp = tensor_to_array(
        franka.inverse_kinematics(link=hand, pos=grasp_pos, quat=hand_quat, init_qpos=qpos_above)
    )
    qpos_init = qpos_above.copy()
    qpos_init[:, 7:] = 0.04
    franka.set_qpos(qpos_init)
    keyframes_qpos = (qpos_above, qpos_grasp)
    state_init = scene.get_state()
    clock = np.zeros(n_envs, dtype=int)
    all_envs = np.arange(n_envs)

    # A fresh run, then the same run after a full reset
    trajectory_fresh = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.snapshot_step + args.steps)
    scene.reset(state=state_init)
    clock[:] = 0
    trajectory_reset = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.snapshot_step + args.steps)
    print("full reset against the fresh run:", max_deviation(trajectory_fresh, trajectory_reset, all_envs))

    # A snapshot and a checkpoint mid-grasp, the run that continues them, then the runs from either one restored. The
    # snapshot holds the reduced state, from which the rigid solver restarts its constraint solve cold, and the
    # checkpoint the whole state of every solver.
    scene.reset(state=state_init)
    clock[:] = 0
    run_grasp(scene, egg, franka, keyframes_qpos, clock, args.snapshot_step)
    snapshot = scene.get_state()
    checkpoint = scene.__getstate__()
    trajectory_continued = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.steps)
    scene.reset(state=snapshot)
    clock[:] = args.snapshot_step
    trajectory_restored = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.steps)
    print(
        "restored snapshot against the continued run:",
        max_deviation(trajectory_continued, trajectory_restored, all_envs),
    )
    scene.__setstate__(checkpoint)
    clock[:] = args.snapshot_step
    trajectory_checkpoint = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.steps)
    print(
        "restored checkpoint against the continued run:",
        max_deviation(trajectory_continued, trajectory_checkpoint, all_envs),
    )

    # One step from the same state, twice: the variation of the arithmetic alone
    scene.reset(state=snapshot)
    clock[:] = args.snapshot_step
    trajectory_step_a = run_grasp(scene, egg, franka, keyframes_qpos, clock, 1)
    scene.reset(state=snapshot)
    clock[:] = args.snapshot_step
    trajectory_step_b = run_grasp(scene, egg, franka, keyframes_qpos, clock, 1)
    print("one step repeated from the same state:", max_deviation(trajectory_step_a, trajectory_step_b, all_envs))

    # Environment 1 reset alone from the snapshot replays the fresh run, the others carry on
    scene.reset(state=snapshot)
    scene.reset(state=state_init, envs_idx=[1])
    clock[:] = args.snapshot_step
    clock[1] = 0
    trajectory_partial = run_grasp(scene, egg, franka, keyframes_qpos, clock, args.steps)
    print(
        "reset environment against the fresh run:",
        max_deviation(trajectory_fresh[: args.steps], trajectory_partial, [1]),
    )
    print(
        "other environments against the continued run:",
        max_deviation(trajectory_continued, trajectory_partial, [0, 2, 3]),
    )


if __name__ == "__main__":
    main()
