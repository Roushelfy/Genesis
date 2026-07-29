"""QIPC coupler: bimanual dexhand teleoperation of a tape roll on a table.

A bimanual Marvin torso with two 20-DoF Wuji hands (54 dofs) stands over a work
table with a wound tape roll lying flat on it, hub axis up -- the gs-gym-internal
RoboWits arena geometry, driven through QIPC so the tape is a real FEM shell with
adhesive self-contact.

Pick a hand with 1/2 (or 3 for both), fly its palm target with the keys, and
hold Space to pinch with its thumb and index finger. The hands are adhesive against the tape
(``beta0=1``: touch -> stick), so a closing hand that reaches the roll holds it.
RGB gizmos mark the two palm targets; the selected one is the larger.

The robot URDF is the gs-gym internal asset, resolved from --urdf,
$QIPC_MARVIN_URDF, or the HuggingFace cache
(``Genesis-Intelligence/internal_assets``: marvin_wuji_capsule_scaled.urdf).

Keyboard Controls:
    1 / 2 / 3   - control the right hand / left hand / both
    Arrow keys  - Move the selected palm target in XY
    J/K         - Move down/up
    N/M         - Yaw left/right
    U/O         - Pitch up/down
    L/;         - Roll left/right
    Space       - Hold to close the selected thumb/index, release to open
    Backslash   - Reset targets to the current palm poses
    Esc         - Quit
"""

import argparse
import glob
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003  # per HOLD callback, i.e. per sim step
DELTA_ROT = 0.02
DT = 0.01
GRIP_SPEED = 2.0
MAX_GRIP_STEP = GRIP_SPEED * DT
MAX_ARM_COMMAND_STEP = 0.03

# gs-gym-internal RoboWits arena: a 0.76m work table 0.6m in front of a torso
# mounted at 1.08m (gs_gym/envs/robowits/robowits.py).
TABLE_POS = (0.597, 0.0, 0.38)
TABLE_SIZE = (0.85, 1.5, 0.76)
TABLE_TOP = TABLE_POS[2] + 0.5 * TABLE_SIZE[2]
ROBOT_POS = (0.0, 0.0, 1.08)
# Right palm's resting xy at INIT_ARM_DEG (measured; see the build check below).
ROLL_XY = (0.379, -0.25)

ARM_JOINTS = {"right": [f"Joint{i}_R" for i in range(1, 8)], "left": [f"Joint{i}_L" for i in range(1, 8)]}
HAND_JOINTS = {
    side: [f"{side}_hand_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
    for side in ("right", "left")
}
PALM_LINK = {"right": "right_hand_palm_link", "left": "left_hand_palm_link"}
# gs-gym's reset pose (BimanualMarvinWujiRobot._INIT_{RIGHT,LEFT}_DEG)
INIT_ARM_DEG = {"right": (-110, -75, 90, -110, -75, 0, 0), "left": (110, -75, -90, -110, 75, 0, 0)}
# gs-gym's gains; the URDF carries no actuator gains, so without these the
# coupler leaves every joint at its kp=100 fallback and the arms sag.
ARM_KP = (7200, 7200, 7200, 3600, 3600, 3600, 3600)
ARM_KV = (600, 600, 600, 400, 200, 200, 200)
HAND_KP, HAND_KV = 50.0, 0.5
HAND_EFFORT_SCALE = 8.0
ARM_FORCE_LIMIT = 2000.0


def resolve_urdf(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("QIPC_MARVIN_URDF", "")
    if env:
        return env
    roots = [
        os.environ.get("HF_HOME", ""),
        os.path.join(os.environ.get("XDG_CACHE_HOME", ""), "huggingface") if os.environ.get("XDG_CACHE_HOME") else "",
        os.path.expanduser("~/.cache/huggingface"),
    ]
    for root in filter(None, roots):
        hits = sorted(glob.glob(os.path.join(
            root, "hub/datasets--Genesis-Intelligence--internal_assets/snapshots/*/"
            "*/marvin_robots/assemble/marvin_wuji_capsule_scaled.urdf")))
        if hits:
            return hits[-1]
    gs.raise_exception(
        "Marvin/Wuji URDF not found. Pass --urdf, set QIPC_MARVIN_URDF, or download "
        "Genesis-Intelligence/internal_assets from HuggingFace "
        f"(searched: {[r for r in roots if r]})."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=None, help="marvin_wuji_capsule_scaled.urdf")
    parser.add_argument("--mode", choices=["bond", "soft"], default="bond", help="tape adhesion mode")
    parser.add_argument("--sticky-hands", action=argparse.BooleanOptionalAction, default=True,
                        help="adhesive contact between the hands and the tape (touch -> stick)")
    parser.add_argument("--headless-steps", type=int, default=0,
                        help="debug: run N steps without the viewer and exit")
    parser.add_argument("--probe-close-step", type=int, default=None,
                        help="headless step at which the right thumb/index start closing")
    parser.add_argument("--probe-release-step", type=int, default=None,
                        help="headless step at which the right thumb/index are released")
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")
    from genesis.engine.couplers.qipc_coupler.tape import (
        TapeAsset,
        add_tape_roll,
        recommended_coupler_options,
    )

    suffix = {"bond": "lock", "soft": "soft"}[args.mode]
    asset = TapeAsset.from_npz(os.path.join(get_assets_dir(), "qipc", f"tape_roll_{suffix}.npz"))
    opts = recommended_coupler_options(asset)
    if args.mode == "soft":
        opts.update(adhesion_bond_distance_lock=False, adhesion_bond_max_bonds=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**opts),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.1, -0.95, 1.35),
            camera_lookat=(0.45, 0.0, 0.85),
            camera_fov=45,
        ),
        show_viewer=args.headless_steps == 0,
    )
    # No ground Plane: the table box reaches down to z=0, and QIPC's halfplane
    # kernel asserts d > 0 for every vertex, so a table flush on the ground trips
    # the coupler's build preflight. Everything here rests on the table.

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=resolve_urdf(args.urdf),
            pos=ROBOT_POS,
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,  # the palm links hang off fixed joints
            requires_jac_and_IK=True,
            convexify=True,
            collision=True,
            links_to_keep=list(PALM_LINK.values()),
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
            # Widen the robot's contact band: capsule collision meshes vs a
            # 0.18mm tape band.
            qipc_d_hat=1e-3,
            # This robot's capsules overlap by construction (Genesis filters the
            # same pairs for its own solver at qpos0). Left on, the IPC barrier
            # fights those overlaps every step and the arms track their PD
            # targets 3.5x worse (joint error 0.30 rad vs 0.085).
            qipc_self_contact=False,
        ),
    )

    # Joint indices resolve before build, so the home pose can be mapped BY NAME
    # into the entity's own dof order (Genesis orders dofs by tree traversal:
    # right arm, right hand, left arm, left hand -- not the URDF's document
    # order) and handed to the coupler, which builds the ABD bodies at that pose.
    dofs = {
        (kind, side): [robot.get_joint(n).dofs_idx_local[0] for n in names[side]]
        for kind, names in (("arm", ARM_JOINTS), ("hand", HAND_JOINTS))
        for side in ("right", "left")
    }
    home = np.zeros(robot.n_qs, dtype=np.float64)
    for side in ("right", "left"):
        home[dofs[("arm", side)]] = np.deg2rad(INIT_ARM_DEG[side])
    robot.material.qipc_home_qpos = home.tolist()
    hand_effort = {
        (side, d): max(abs(float(v)) for v in robot.get_joint(name).dofs_force_range[0])
        for side in ("right", "left")
        for name, d in zip(HAND_JOINTS[side], dofs[("hand", side)], strict=True)
    }

    scene.add_entity(
        morph=gs.morphs.Box(pos=TABLE_POS, size=TABLE_SIZE, fixed=True, collision=True),
        material=gs.materials.Rigid(coup_friction=0.8),
        surface=gs.surfaces.Default(color=(0.62, 0.6, 0.58)),
    )

    # Put the roll under the right palm's home position: the arm's workspace is
    # smaller than the table, and a roll at the table centre (0.56m from the
    # base) is out of reach -- IK stalls 15cm short there. FK needs a built
    # scene, so this is the measured home palm xy for INIT_ARM_DEG; the build
    # check below flags it if the home pose changes.
    palm_home = np.array(ROLL_XY, dtype=np.float64)
    lowest = min(float(asset.tape_positions[:, 2].min()), -0.5 * asset.hub_height)
    roll_z = TABLE_TOP - lowest + asset.thick + 0.5 * asset.d_hat
    tape, _hub = add_tape_roll(
        scene, asset,
        pos=(float(palm_home[0]), float(palm_home[1]), roll_z),
        euler=(0.0, 0.0, 0.0),  # hub axis up: the roll lies flat
        with_hub=True, hub_fixed=False,
        tape_surface=gs.surfaces.Plastic(color=(0.85, 0.75, 0.3, 1.0)),
        hub_surface=gs.surfaces.Plastic(color=(0.4, 0.25, 0.15, 1.0)),
    )
    if args.sticky_hands:
        scene.sim.coupler.add_adhesion(
            tape, robot,
            Cn=float(asset.params.get("CN", 1.0)),
            Ct=float(asset.params.get("CT", 1.0)),
            W=float(asset.params.get("ADH_W", 1.0)),
            eta=float(asset.params.get("ETA", 100.0)),
            bonding_rate=1.0,
            beta0=1.0,  # touch -> stick
            friction=1.0,
        )

    scene.build()

    # PD gains, mapped through the coupler's JointCollection row order.
    coupler = scene.sim.coupler
    jc = coupler._jc
    jc_row = {int(d): i for i, d in enumerate(coupler._genesis_dof_order.cpu().numpy())}
    for side in ("right", "left"):
        for k, d in enumerate(dofs[("arm", side)]):
            jc[jc_row[d]].set_dofs_kp(float(ARM_KP[k]))
            jc[jc_row[d]].set_dofs_kv(float(ARM_KV[k]))
            jc[jc_row[d]].set_dofs_force_range(-ARM_FORCE_LIMIT, ARM_FORCE_LIMIT)
        for d in dofs[("hand", side)]:
            jc[jc_row[d]].set_dofs_kp(HAND_KP)
            jc[jc_row[d]].set_dofs_kv(HAND_KV)
            effort = HAND_EFFORT_SCALE * hand_effort[(side, d)]
            jc[jc_row[d]].set_dofs_force_range(-effort, effort)

    robot.set_qpos(home)
    for key, idx in dofs.items():
        robot.control_dofs_position(home[idx], dofs_idx_local=idx)

    # Check the roll landed under the hand only AFTER set_qpos: Scene.build's own
    # reset leaves Genesis's link states at ITS init qpos (arms straight out),
    # not at the coupler's home pose, so reading the palm any earlier reports a
    # pose the simulation never has.
    palm_now = robot.get_link(PALM_LINK["right"]).get_pos().reshape(-1)[:3].cpu().numpy()
    if np.linalg.norm(palm_now[:2] - palm_home) > 0.05:
        gs.logger.warning(
            f"right palm rests at {np.round(palm_now[:2], 3)} but the roll was placed at "
            f"{np.round(palm_home, 3)}: update ROLL_XY for this home pose or the roll is out of reach."
        )
    gs.logger.info(f"tape mode={args.mode} roll at {np.round(palm_home, 3)} z={roll_z:.4f} "
                   f"right palm {np.round(palm_now, 3)}")

    # Thumb/index pinch postures. The other three fingers remain open.
    postures = {}
    for side in ("right", "left"):
        for closed in (False, True):
            vals = []
            for name in HAND_JOINTS[side]:
                finger = int(name.split("finger", 1)[1].split("_", 1)[0])
                joint = int(name.rsplit("joint", 1)[1])
                lo, hi = (float(v) for v in robot.get_joint(name).dofs_limit[0])
                if closed and finger == 1:
                    want = (0.9, 0.35, 0.9, 0.9)[joint - 1]
                elif closed and finger == 2:
                    want = (0.95, 0.0, 0.95, 0.95)[joint - 1]
                else:
                    want = 0.08 if (finger == 1 and joint == 1) else 0.0
                vals.append(min(max(want, lo), hi))
            postures[(side, closed)] = np.array(vals)

    palms = {side: robot.get_link(PALM_LINK[side]) for side in ("right", "left")}
    target_pos, target_quat, init_pos, init_quat = {}, {}, {}, {}
    for side, link in palms.items():
        init_pos[side] = link.get_pos().reshape(-1)[:3].cpu().numpy().astype(gs.np_float).copy()
        init_quat[side] = link.get_quat().reshape(-1)[:4].cpu().numpy().astype(gs.np_float).copy()
        target_pos[side] = init_pos[side].copy()
        target_quat[side] = init_quat[side].copy()
    qpos_cmd = home.copy()
    closed = {"right": False, "left": False}
    hand_cmd = {side: postures[(side, False)].copy() for side in ("right", "left")}

    def servo_step():
        for side in ("right", "left"):
            arm = dofs[("arm", side)]
            qpos = robot.inverse_kinematics(
                link=palms[side],
                pos=target_pos[side],
                quat=target_quat[side],
                init_qpos=qpos_cmd,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=arm,
            )
            qn = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
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
                print("[probe] closing right thumb/index", flush=True)
            if step == args.probe_release_step:
                closed["right"] = False
                print("[probe] opening right thumb/index", flush=True)
            servo_step()
            if step in {
                args.probe_close_step,
                args.probe_release_step,
                args.headless_steps - 1,
            }:
                palm = palms["right"].get_pos().reshape(-1)[:3].cpu().numpy()
                rows = [jc_row[d] for d in dofs[("hand", "right")]]
                theta = jc.get_dofs_position().cpu().numpy()
                torque = jc.get_dofs_applied_force().cpu().numpy()
                print(
                    f"[probe] step={step:03d} palm={np.round(palm, 4)} "
                    f"hand=[{theta[rows].min():.3f},{theta[rows].max():.3f}] "
                    f"max_tau={np.abs(torque[rows]).max():.3f}",
                    flush=True,
                )
        pos = tape.get_state().pos[0].cpu().numpy()
        drift = (
            float(np.linalg.norm(palms["right"].get_pos().reshape(-1)[:3].cpu().numpy() - initial_palm))
            if initial_palm is not None
            else 0.0
        )
        print(f"[headless] tape z=[{pos[:, 2].min():.4f},{pos[:, 2].max():.4f}] "
              f"finite={np.isfinite(pos).all()} palm_drift={drift:.4f}m "
              f"right_palm={np.round(palms['right'].get_pos().reshape(-1)[:3].cpu().numpy(), 3)}",
              flush=True)
        return

    if scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return
    scene.viewer.update(force=True)

    active = {"right"}
    gizmo = {
        side: scene.draw_debug_frame(
            T=gu.trans_quat_to_T(target_pos[side], target_quat[side]),
            axis_length=0.12, origin_size=0.008, axis_radius=0.005,
        )
        for side in ("right", "left")
    }
    is_running = True
    space_active = set()

    def select(sides):
        active.clear()
        active.update(sides)
        gs.logger.info(f"controlling: {'+'.join(sorted(active))}")

    def move(dpos_xyz):
        for side in active:
            target_pos[side][:] += dpos_xyz

    def rotate(axis_idx, delta):
        delta_xyz = np.zeros(3, dtype=gs.np_float)
        delta_xyz[axis_idx] = delta
        for side in active:
            target_quat[side][:] = gu.transform_quat_by_quat(target_quat[side], gu.xyz_to_quat(delta_xyz))

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
        gs.logger.info(f"grip: right={'closed' if closed['right'] else 'open'} "
                       f"left={'closed' if closed['left'] else 'open'}")

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
                tuple(gizmo[s] for s in ("right", "left")),
                tuple(gu.trans_quat_to_T(target_pos[s], target_quat[s]) for s in ("right", "left")),
            )
            servo_step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
