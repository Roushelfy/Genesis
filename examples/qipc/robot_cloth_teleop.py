"""QIPC coupler: keyboard teleoperation of a Franka Panda with a hanging cloth.

QIPC port of examples/IPC_Solver/ipc_robot_cloth_teleop.py. The cloth hangs from
two pinned top corners (P2 soft vertex constraints); drive the gripper to pinch
it, press P to release the pins, and carry the cloth around.

Keyboard Controls:
    Arrow keys  - Move in XY plane
    J/K         - Move down/up
    N/M         - Yaw left/right
    U/O         - Pitch up/down
    L/;         - Roll left/right
    Space       - Toggle gripper
    P           - Release cloth pins
    Backslash   - Reset EE target
    Esc         - Quit

Notes:
  - contact_resistance ~1e7 (cgq default 1e4 lets heavy squashes penetrate).
  - Cloth needs bending_stiffness >= ~5 or a hard pinch squirts it out.
"""

import argparse

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003  # per HOLD callback, i.e. per sim step (0.3 m/s at dt=0.01)
DELTA_ROT = 0.02  # per sim step (2 rad/s at dt=0.01)
# How far the IK target may run ahead of the measured end effector (see leash_target).
LEASH_POS = 0.03
LEASH_ROT = 0.25

CLOTH_SIZE = 0.35
CLOTH_CENTER = np.array([0.55, 0.0, 0.35])
CLOTH_TOP = CLOTH_CENTER[2] + CLOTH_SIZE / 2


def make_grid_cloth_obj(path: str, n: int = 21, size: float = 1.0) -> None:
    xs = np.linspace(-size / 2, size / 2, n)
    ys = np.linspace(-size / 2, size / 2, n)
    with open(path, "w") as fh:
        for y in ys:
            for x in xs:
                fh.write(f"v {x:.6f} {y:.6f} 0.0\n")
        for j in range(n - 1):
            for i in range(n - 1):
                v0 = j * n + i + 1
                v1 = j * n + i + 2
                v2 = (j + 1) * n + i + 1
                v3 = (j + 1) * n + i + 2
                fh.write(f"f {v0} {v1} {v3}\n")
                fh.write(f"f {v0} {v3} {v2}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    obj_path = "/tmp/qipc_grid21.obj"
    make_grid_cloth_obj(obj_path, n=21, size=1.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.002,
            contact_resistance=1e7,
            init_collision_pair_capacity=60000,
            fem_constraint_strength=3000.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.7, -1.3, 0.9),
            camera_lookat=(0.45, 0.0, 0.4),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
            pos=(0.0, 0.0, 0.005),
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
            qipc_home_qpos=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
        ),
    )

    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=obj_path,
            scale=CLOTH_SIZE,
            pos=tuple(CLOTH_CENTER),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=3e4,
            nu=0.3,
            rho=200.0,
            thickness=0.001,
            bending_stiffness=5.0,
            friction_mu=1.0,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)

    scene.build()

    ee_link = franka.get_link("hand")

    coupler_jc = scene.sim._coupler._jc
    coupler_jc[7:9].set_dofs_kp(500.0)
    coupler_jc[7:9].set_dofs_kv(50.0)

    # Sync Genesis kinematic state to the QIPC home pose (Scene.build's final
    # reset restores Genesis's own init-qpos FK, which is stale for IK).
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=gs.np_float)
    franka.set_qpos(home)
    franka.control_dofs_position(home[:7], dofs_idx_local=motor_dofs_idx)
    franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

    # Pin the two top corners of the cloth
    verts0 = cloth.get_state().pos[0].cpu().numpy()
    top_left = int(np.argmin(np.linalg.norm(verts0 - [CLOTH_CENTER[0] - CLOTH_SIZE / 2, 0.0, CLOTH_TOP], axis=1)))
    top_right = int(np.argmin(np.linalg.norm(verts0 - [CLOTH_CENTER[0] + CLOTH_SIZE / 2, 0.0, CLOTH_TOP], axis=1)))
    cloth.set_vertex_constraints([top_left, top_right], is_soft_constraint=True)

    # Start the target at the actual post-build EE pose, position AND orientation, so the
    # first IK step is a no-op instead of a jump to an unrelated wrist configuration.
    target_init_pos = ee_link.get_pos().reshape(-1)[:3].cpu().numpy().astype(gs.np_float).copy()
    target_init_quat = ee_link.get_quat().reshape(-1)[:4].cpu().numpy().astype(gs.np_float).copy()
    target_pos = target_init_pos.copy()
    target_quat = target_init_quat.copy()
    qpos_cmd = home.copy()

    if scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return
    scene.viewer.update(force=True)

    is_gripper_closed = np.array(False, dtype=gs.np_bool)
    is_running = True

    def move(dpos_xyz):
        target_pos[:] += dpos_xyz

    def rotate(axis_idx, delta):
        delta_xyz = np.zeros(3, dtype=gs.np_float)
        delta_xyz[axis_idx] = delta
        target_quat[:] = gu.transform_quat_by_quat(target_quat, gu.xyz_to_quat(delta_xyz))

    def reset_pose():
        target_pos[:] = target_init_pos
        target_quat[:] = target_init_quat

    def set_gripper(is_closed):
        is_gripper_closed[()] = is_closed

    def release_pins():
        cloth.remove_vertex_constraints()
        gs.logger.info("Cloth pins released.")

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_pose", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        Keybind("release_pins", Key.P, KeyAction.RELEASE, callback=release_pins),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    def leash_target():
        """Keep the target within reach of where the arm actually is.

        HOLD keybinds fire once per sim step (scene.step -> visualizer.update ->
        viewer.update -> update_on_sim_step), so a held key advances the target
        0.3 m/s in sim time -- far faster than the arm tracks it. Unleashed the
        target runs away (measured: 0.6 m of error after ~1.6 s of holding) and
        the IK, asked for a pose the arm cannot reach, hops between arm
        configurations: single-step command jumps up to 0.48 rad. Clamping the
        target keeps every solve local (jumps drop ~50x, to <0.01 rad) and
        leaves the release behaviour unchanged.
        """
        ee_pos = ee_link.get_pos().reshape(-1)[:3].cpu().numpy()
        lag = target_pos - ee_pos
        dist = float(np.linalg.norm(lag))
        if dist > LEASH_POS:
            target_pos[:] = ee_pos + lag * (LEASH_POS / dist)

        ee_quat = ee_link.get_quat().reshape(-1)[:4].cpu().numpy()
        # Geodesic angle between the two orientations, convention-free.
        angle = 2.0 * np.arccos(min(1.0, abs(float(np.dot(target_quat, ee_quat)))))
        if angle > LEASH_ROT:
            # gu.slerp is numba-jitted and reshapes t to the quaternions' batch
            # shape, so t must be a 1-element array, not a Python float.
            frac = np.array([LEASH_ROT / angle], dtype=ee_quat.dtype)
            target_quat[:] = gu.slerp(ee_quat, target_quat.copy(), frac)

    try:
        while is_running and scene.viewer.is_alive():
            leash_target()
            # Command-space servoing: seed with the previous command so the commands form a
            # smooth chain, and max_samples=1 so a convergence miss cannot random-resample
            # the PD target onto another arm branch.
            qpos = franka.inverse_kinematics(
                link=ee_link,
                pos=target_pos,
                quat=target_quat,
                init_qpos=qpos_cmd,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=motor_dofs_idx,
            )
            qpos_cmd[:7] = tensor_to_array(qpos)[:7]
            franka.control_dofs_position(qpos_cmd[:7], dofs_idx_local=motor_dofs_idx)
            franka.control_dofs_position(-0.02 if is_gripper_closed[()] else 0.04, dofs_idx_local=finger_dofs_idx)

            scene.step()
            scene.rigid_solver._func_update_geoms(scene._envs_idx)
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
