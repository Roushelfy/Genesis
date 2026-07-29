"""Minimal QIPC teleoperation scene for debugging the bimanual Wuji hands.

This intentionally contains no table, tape, FEM, contact, or adhesion.  It
isolates palm IK and dexterous-hand position control so controller transients
cannot be confused with object interaction.

Keyboard Controls:
    1 / 2 / 3   - control the right hand / left hand / both
    Arrow keys  - move the selected palm target in XY
    J/K         - move down/up
    N/M         - yaw left/right
    U/O         - pitch up/down
    L/;         - roll left/right
    Space       - hold to close the selected thumb/index, release to open
    Backslash   - reset targets to the measured palm poses
    Esc         - quit

For an automated close-hand stability check:
    python examples/qipc/dexhand_teleop_debug.py --headless-steps 100
"""

import argparse

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

from dexhand_tape_teleop import (
    ARM_JOINTS,
    ARM_KP,
    ARM_KV,
    HAND_JOINTS,
    INIT_ARM_DEG,
    PALM_LINK,
    ROBOT_POS,
    resolve_urdf,
)

DT = 0.01
DELTA_POS = 0.003
DELTA_ROT = 0.02

# Do not step a tiny finger from 0 directly to ~1 rad.  At 2 rad/s a full
# close takes about half a second while remaining responsive to Space.
GRIP_SPEED = 2.0
MAX_GRIP_STEP = GRIP_SPEED * DT
HAND_KP = 50.0
# gs-gym's kv=5 is tuned for Genesis's articulated rigid solver.  With QIPC's
# tiny independently represented finger links it over-damps the URDF-effort-
# limited joints (a distal joint can then move only ~0.03 rad/s).
HAND_KV = 0.5
# The URDF effort is a conservative continuous rating.  A bounded 8x peak is
# enough to follow the rate-limited close trajectory, while remaining roughly
# an order of magnitude below the old unbounded 50 Nm impulse.
HAND_EFFORT_SCALE = 8.0

# A missed IK solve must not turn into a discontinuous arm command.
MAX_ARM_COMMAND_STEP = 0.03
ARM_FORCE_LIMIT = 2000.0


def _hand_posture(robot, side: str, closed: bool) -> np.ndarray:
    """Return a thumb/index pinch posture, clipped to URDF limits."""
    values = []
    for name in HAND_JOINTS[side]:
        finger = int(name.split("finger", 1)[1].split("_", 1)[0])
        joint = int(name.rsplit("joint", 1)[1])
        if closed and finger in (1, 2):
            if finger == 1:
                want = (0.9, 0.35, 0.9, 0.9)[joint - 1]
            else:
                want = (0.95, 0.0, 0.95, 0.95)[joint - 1]
        else:
            want = 0.08 if (finger == 1 and joint == 1) else 0.0
        lo, hi = (float(v) for v in robot.get_joint(name).dofs_limit[0])
        values.append(np.clip(want, lo, hi))
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=None, help="marvin_wuji_capsule_scaled.urdf")
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="run an automated right-hand close probe for N steps and exit",
    )
    parser.add_argument(
        "--probe-close-step",
        type=int,
        default=5,
        help="headless step at which the right hand starts closing",
    )
    parser.add_argument(
        "--probe-release-step",
        type=int,
        default=None,
        help="optional headless step at which the right hand is released",
    )
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            debug_viewer=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.25, -1.15, 1.45),
            camera_lookat=(0.35, 0.0, 1.0),
            camera_fov=45,
        ),
        show_viewer=args.headless_steps == 0,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=resolve_urdf(args.urdf),
            pos=ROBOT_POS,
            fixed=True,
            merge_fixed_links=False,
            requires_jac_and_IK=True,
            convexify=True,
            collision=True,
            links_to_keep=list(PALM_LINK.values()),
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
        ),
    )

    dofs = {
        (kind, side): [robot.get_joint(name).dofs_idx_local[0] for name in names[side]]
        for kind, names in (("arm", ARM_JOINTS), ("hand", HAND_JOINTS))
        for side in ("right", "left")
    }
    home = np.zeros(robot.n_qs, dtype=np.float64)
    for side in ("right", "left"):
        home[dofs[("arm", side)]] = np.deg2rad(INIT_ARM_DEG[side])
    robot.material.qipc_home_qpos = home.tolist()

    # QIPC does not currently inherit Genesis force ranges.  Capture each
    # hand joint's URDF effort before build and install it on JointCollection.
    hand_effort = {
        (side, d): max(abs(float(v)) for v in robot.get_joint(name).dofs_force_range[0])
        for side in ("right", "left")
        for name, d in zip(HAND_JOINTS[side], dofs[("hand", side)], strict=True)
    }

    scene.build()

    coupler = scene.sim.coupler
    jc = coupler._jc
    jc_row = {int(d): i for i, d in enumerate(coupler._genesis_dof_order.cpu().numpy())}
    for side in ("right", "left"):
        for k, d in enumerate(dofs[("arm", side)]):
            row = jc[jc_row[d]]
            row.set_dofs_kp(float(ARM_KP[k]))
            row.set_dofs_kv(float(ARM_KV[k]))
            row.set_dofs_force_range(-ARM_FORCE_LIMIT, ARM_FORCE_LIMIT)
        for d in dofs[("hand", side)]:
            row = jc[jc_row[d]]
            effort = HAND_EFFORT_SCALE * hand_effort[(side, d)]
            row.set_dofs_kp(HAND_KP)
            row.set_dofs_kv(HAND_KV)
            row.set_dofs_force_range(-effort, effort)

    robot.set_qpos(home)
    for key, idx in dofs.items():
        robot.control_dofs_position(home[idx], dofs_idx_local=idx)

    palms = {side: robot.get_link(PALM_LINK[side]) for side in ("right", "left")}
    target_pos = {
        side: link.get_pos().reshape(-1)[:3].cpu().numpy().astype(gs.np_float).copy()
        for side, link in palms.items()
    }
    target_quat = {
        side: link.get_quat().reshape(-1)[:4].cpu().numpy().astype(gs.np_float).copy()
        for side, link in palms.items()
    }
    postures = {
        (side, state): _hand_posture(robot, side, state)
        for side in ("right", "left")
        for state in (False, True)
    }
    hand_cmd = {side: postures[(side, False)].copy() for side in ("right", "left")}
    qpos_cmd = home.copy()
    closed = {"right": False, "left": False}

    def servo_step():
        for side in ("right", "left"):
            arm = dofs[("arm", side)]
            solution = robot.inverse_kinematics(
                link=palms[side],
                pos=target_pos[side],
                quat=target_quat[side],
                init_qpos=qpos_cmd,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=arm,
            )
            qn = solution.cpu().numpy() if hasattr(solution, "cpu") else np.asarray(solution)
            delta = np.clip(qn[arm] - qpos_cmd[arm], -MAX_ARM_COMMAND_STEP, MAX_ARM_COMMAND_STEP)
            qpos_cmd[arm] += delta
            robot.control_dofs_position(qpos_cmd[arm], dofs_idx_local=arm)

            goal = postures[(side, closed[side])]
            hand_cmd[side] += np.clip(goal - hand_cmd[side], -MAX_GRIP_STEP, MAX_GRIP_STEP)
            robot.control_dofs_position(hand_cmd[side], dofs_idx_local=dofs[("hand", side)])

        scene.step()
        scene.rigid_solver._func_update_geoms(scene._envs_idx)

    if args.headless_steps > 0:
        initial_palm = None
        for step in range(args.headless_steps):
            if step == args.probe_close_step:
                closed["right"] = True
                initial_palm = palms["right"].get_pos().reshape(-1)[:3].cpu().numpy().copy()
                print("[probe] closing right hand", flush=True)
            if step == args.probe_release_step:
                closed["right"] = False
                print("[probe] opening right hand", flush=True)
            servo_step()
            report_steps = {
                0,
                args.probe_close_step,
                args.probe_close_step + 1,
                args.probe_release_step,
                None if args.probe_release_step is None else args.probe_release_step + 1,
                args.headless_steps - 1,
            }
            if step in report_steps:
                theta = jc.get_dofs_position().cpu().numpy()
                torque = jc.get_dofs_applied_force().cpu().numpy()
                rows = [jc_row[d] for d in dofs[("hand", "right")]]
                palm = palms["right"].get_pos().reshape(-1)[:3].cpu().numpy()
                print(
                    f"[probe] step={step:03d} palm={np.round(palm, 4)} "
                    f"hand=[{theta[rows].min():.3f},{theta[rows].max():.3f}] "
                    f"max_tau={np.abs(torque[rows]).max():.3f}",
                    flush=True,
                )

        if initial_palm is None:
            raise ValueError("--probe-close-step must be smaller than --headless-steps")
        final_palm = palms["right"].get_pos().reshape(-1)[:3].cpu().numpy()
        actual = robot.get_dofs_position(dofs_idx_local=dofs[("hand", "right")]).cpu().numpy()
        final_goal = postures[("right", closed["right"])]
        palm_drift = float(np.linalg.norm(final_palm - initial_palm))
        posture_error = float(np.max(np.abs(actual - final_goal)))
        finite = bool(np.isfinite(actual).all() and np.isfinite(final_palm).all())
        passed = finite and palm_drift < 0.03 and posture_error < 0.15
        print(
            f"[probe] {'PASS' if passed else 'FAIL'} finite={finite} "
            f"palm_drift={palm_drift:.4f}m max_posture_error={posture_error:.4f}rad",
            flush=True,
        )
        print(
            "[probe] final joints: "
            + ", ".join(
                f"{name.rsplit('_', 1)[-1]}={value:.3f}/{goal:.3f}"
                for name, value, goal in zip(
                    HAND_JOINTS["right"],
                    actual,
                    final_goal,
                    strict=True,
                )
            ),
            flush=True,
        )
        if not passed:
            raise RuntimeError("DexHand close probe did not meet stability thresholds")
        return

    if scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return
    scene.viewer.update(force=True)

    active = {"right"}
    gizmo = {
        side: scene.draw_debug_frame(
            T=gu.trans_quat_to_T(target_pos[side], target_quat[side]),
            axis_length=0.12,
            origin_size=0.008,
            axis_radius=0.005,
        )
        for side in ("right", "left")
    }
    is_running = True
    space_active = set()

    def select(sides):
        active.clear()
        active.update(sides)
        gs.logger.info(f"controlling: {'+'.join(sorted(active))}")

    def move(delta):
        for side in active:
            target_pos[side][:] += delta

    def rotate(axis_idx, delta):
        delta_xyz = np.zeros(3, dtype=gs.np_float)
        delta_xyz[axis_idx] = delta
        for side in active:
            target_quat[side][:] = gu.transform_quat_by_quat(
                target_quat[side], gu.xyz_to_quat(delta_xyz)
            )

    def reset_targets():
        for side in ("right", "left"):
            target_pos[side][:] = palms[side].get_pos().reshape(-1)[:3].cpu().numpy()
            target_quat[side][:] = palms[side].get_quat().reshape(-1)[:4].cpu().numpy()

    def set_grip(is_closed):
        if is_closed:
            space_active.clear()
            space_active.update(active)
            for side in space_active:
                closed[side] = True
        else:
            for side in space_active:
                closed[side] = False
            space_active.clear()
        gs.logger.info(
            f"grip: right={'closed' if closed['right'] else 'open'} "
            f"left={'closed' if closed['left'] else 'open'}"
        )

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("select_right", Key._1, KeyAction.RELEASE, callback=select, args=({"right"},)),
        Keybind("select_left", Key._2, KeyAction.RELEASE, callback=select, args=({"left"},)),
        Keybind("select_both", Key._3, KeyAction.RELEASE, callback=select, args=({"right", "left"},)),
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_targets", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_targets),
        Keybind("close_grip", Key.SPACE, KeyAction.PRESS, callback=set_grip, args=(True,)),
        Keybind("open_grip", Key.SPACE, KeyAction.RELEASE, callback=set_grip, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )
    gs.logger.info(
        "keys: 1/2/3 pick hand(s), arrows+J/K move, N/M U/O L/; rotate, "
        "hold Space to pinch, release to open, \\ reset, Esc quit"
    )

    try:
        while is_running and scene.viewer.is_alive():
            scene.update_debug_objects(
                tuple(gizmo[side] for side in ("right", "left")),
                tuple(gu.trans_quat_to_T(target_pos[side], target_quat[side]) for side in ("right", "left")),
            )
            servo_step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
