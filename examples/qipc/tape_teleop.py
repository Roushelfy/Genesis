"""QIPC coupler: keyboard teleoperation of a Franka Panda with a tape roll.

A wound tape roll (imported from a cgq adhesive_tape_wind npz asset) stands on
the ground next to the robot. Two adhesion modes:

    --mode bond   Phase-2 distance bonds (default): the coil holds itself
                  firmly; peeling a layer takes ~RCC_RELEASE_FORCE (0.5 N with
                  the default preset). Wound with --lock.
    --mode soft   pure Phase-1 soft adhesion: the coil is held only by the
                  beta-state stickiness (Cn=1 by default) and will gradually
                  unroll when disturbed -- matching cgq's --no-lock behavior.

Assets ship in genesis/assets/qipc/ (tape_roll_lock.npz, tape_roll_soft.npz); override with --asset
or QIPC_TAPE_ASSET_BOND / QIPC_TAPE_ASSET_SOFT. Regenerate them with cgq's
examples/adhesive_tape_wind.py:
    python -m examples.adhesive_tape_wind --preset speed-r150-bend5k --lock    --save tape_roll_lock.npz
    python -m examples.adhesive_tape_wind --preset speed-r150-bend5k --no-lock --save tape_roll_soft.npz

The gripper is adhesive against the tape (beta0=1: touch -> stick), so you can
pick the roll up by touching it, and in bond mode peel the tail off by pulling.

Keyboard Controls:
    Arrow keys  - Move in XY plane
    J/K         - Move down/up
    N/M         - Yaw left/right
    U/O         - Pitch up/down
    L/;         - Roll left/right
    Space       - Toggle gripper
    Backslash   - Reset EE target
    Esc         - Quit
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003
DELTA_ROT = 0.02

ROLL_XY = (0.55, 0.0)
ROLL_EULER = (90.0, 0.0, 0.0)  # asset hub axis +z -> world +y (roll stands on its tread)


def _default_asset(mode: str) -> str:
    env = os.environ.get(f"QIPC_TAPE_ASSET_{mode.upper()}", "")
    if env:
        return env
    suffix = {"bond": "lock", "soft": "soft"}[mode]
    return os.path.join(get_assets_dir(), "qipc", f"tape_roll_{suffix}.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bond", "soft"], default="bond")
    parser.add_argument("--asset", type=str, default=None, help="tape roll npz (default per --mode)")
    parser.add_argument("--sticky-gripper", action=argparse.BooleanOptionalAction, default=True,
                        help="adhesive contact between the robot and the tape (touch -> stick)")
    parser.add_argument("--headless-steps", type=int, default=0,
                        help="debug: run N steps without the viewer and exit")
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    from genesis.engine.couplers.qipc_coupler.tape import TapeAsset, add_tape_roll, recommended_coupler_options

    asset_path = args.asset or _default_asset(args.mode)
    asset = TapeAsset.from_npz(asset_path)
    opts = recommended_coupler_options(asset)
    if args.mode == "bond":
        opts.update(adhesion_bond_distance_lock=True, adhesion_bond_max_bonds=16384)
    else:
        opts.update(adhesion_bond_distance_lock=False, adhesion_bond_max_bonds=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.QIPCCouplerOptions(**opts),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, -1.0, 0.7),
            camera_lookat=(0.5, 0.0, 0.15),
            camera_fov=40,
        ),
        show_viewer=args.headless_steps == 0,
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
            # widen the robot's contact band: coarse robot meshes vs 0.18mm tape d_hat
            qipc_d_hat=1e-3,
        ),
    )

    # Seat the roll on the ground (cgq drop convention: lowest coil vertex one
    # contact band above the plane). add_tape_roll bakes the transform into the
    # meshes, so this analytic height is exact. Do NOT spawn the roll high and
    # let it fall: in bond mode the distance locks resist rigid translation
    # (qipc engine behavior), so an airborne locked coil hovers instead of falling.
    quat = gu.xyz_to_quat(np.asarray(ROLL_EULER, dtype=np.float64), degrees=True)
    rot = gu.quat_to_R(quat)
    coil_rot = asset.tape_positions @ rot.T
    roll_z = -float(coil_rot[:, 2].min()) + asset.thick + 0.5 * asset.d_hat
    tape, hub = add_tape_roll(
        scene,
        asset,
        pos=(ROLL_XY[0], ROLL_XY[1], roll_z),
        euler=ROLL_EULER,
        with_hub=True,
        hub_fixed=False,
        tape_surface=gs.surfaces.Plastic(color=(0.85, 0.75, 0.3, 1.0)),
        hub_surface=gs.surfaces.Plastic(color=(0.4, 0.25, 0.15, 1.0)),
    )
    if args.sticky_gripper:
        scene.sim.coupler.add_adhesion(
            tape, franka,
            Cn=float(asset.params.get("CN", 1.0)),
            Ct=float(asset.params.get("CT", 1.0)),
            W=float(asset.params.get("ADH_W", 1.0)),
            eta=float(asset.params.get("ETA", 100.0)),
            bonding_rate=1.0,
            beta0=1.0,  # touch -> stick
            friction=1.0,
        )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)

    scene.build()
    gs.logger.info(f"tape mode={args.mode} asset={asset_path} roll_z={roll_z:.4f}")

    ee_link = franka.get_link("hand")
    coupler_jc = scene.sim._coupler._jc
    coupler_jc[7:9].set_dofs_kp(500.0)
    coupler_jc[7:9].set_dofs_kv(50.0)

    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=gs.np_float)
    franka.set_qpos(home)
    franka.control_dofs_position(home[:7], dofs_idx_local=motor_dofs_idx)
    franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

    target_init_pos = ee_link.get_pos().reshape(-1)[:3].cpu().numpy().astype(gs.np_float).copy()
    target_init_quat = ee_link.get_quat().reshape(-1)[:4].cpu().numpy().astype(gs.np_float).copy()
    target_pos = target_init_pos.copy()
    target_quat = target_init_quat.copy()
    qpos_cmd = home.copy()

    def servo_step(gripper_closed: bool):
        qpos = franka.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            init_qpos=qpos_cmd,
            max_samples=1,
            max_solver_iters=30,
            dofs_idx_local=motor_dofs_idx,
        )
        qpos_np = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
        qpos_cmd[:7] = qpos_np[:7]
        franka.control_dofs_position(qpos_cmd[:7], dofs_idx_local=motor_dofs_idx)
        franka.control_dofs_position(-0.02 if gripper_closed else 0.04, dofs_idx_local=finger_dofs_idx)
        scene.step()
        scene.rigid_solver._func_update_geoms(scene._envs_idx)

    if args.headless_steps > 0:
        for i in range(args.headless_steps):
            servo_step(False)
        pos = tape.get_state().pos[0].cpu().numpy()
        print(f"[headless] tape z=[{pos[:, 2].min():.4f},{pos[:, 2].max():.4f}] finite={np.isfinite(pos).all()}",
              flush=True)
        return

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
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    try:
        while is_running and scene.viewer.is_alive():
            servo_step(bool(is_gripper_closed[()]))
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
