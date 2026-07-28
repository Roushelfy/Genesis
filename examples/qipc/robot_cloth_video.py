"""QIPC coupler: Franka pinch-grasps a hanging cloth (scripted teleop, rendered video).

FEM/P2 features exercised: cloth (Baraff-Witkin shell) + IPC contact against the
robot fingers, soft vertex constraints (clothesline pins), remove_vertex_constraints
(release), zero-copy pos+vel writeback for rendering.

Sequence:
  1. Cloth hangs from two pinned top corners (soft constraints); robot at home.
  2. EE descends so the cloth top edge sits between the open fingers.
  3. Gripper closes -> friction pinch.
  4. Pins released -> cloth held only by the gripper.
  5. Robot lifts and sways -> cloth swings.
  6. Gripper opens -> cloth drops to the ground.

Run (GPU container):
    python examples/qipc/robot_cloth_video.py
"""

import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu

OUT_DIR = os.environ.get("QIPC_FEM_OUT", "/mnt/home/zhaofeng/workspace/qipc-test/output")

CLOTH_SIZE = 0.35
CLOTH_CENTER = np.array([0.55, 0.0, 0.35])
CLOTH_TOP = CLOTH_CENTER[2] + CLOTH_SIZE / 2  # 0.525
HAND_TO_PINCH = 0.103  # hand frame to fingertip pinch point
RENDER_EVERY = 2
FPS = 50


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
    os.makedirs(OUT_DIR, exist_ok=True)
    gs.init(precision="64", logging_level="info")

    obj_path = os.path.join(OUT_DIR, "grid21.obj")
    make_grid_cloth_obj(obj_path, n=21, size=1.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.002,
            contact_resistance=1e7,
            init_collision_pair_capacity=60000,
            fem_constraint_strength=3000.0,
        ),
        show_viewer=False,
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

    # Cloth hanging vertically in the xz plane (normal = y)
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

    cam = scene.add_camera(
        res=(1280, 720),
        pos=(1.7, -1.3, 0.9),
        lookat=(0.45, 0.0, 0.4),
        fov=35,
        GUI=False,
    )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)

    scene.build()

    ee_link = franka.get_link("hand")

    # Finger PD gains via the QIPC joint collection (runtime-settable)
    coupler_jc = scene.sim._coupler._jc
    coupler_jc[7:9].set_dofs_kp(500.0)
    coupler_jc[7:9].set_dofs_kv(50.0)

    # Sync the Genesis-side kinematic state to the QIPC home pose. Scene.build's
    # final reset restores Genesis's own init-qpos FK (all zeros for this MJCF),
    # which would make ee_link.get_pos()/IK operate on a stale state and command
    # a violent arm swing on the first step.
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=gs.np_float)
    franka.set_qpos(home)
    franka.control_dofs_position(home[:7], dofs_idx_local=motor_dofs_idx)
    franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

    # --- Pin the two top corners of the cloth (clothesline) ---
    verts0 = cloth.get_state().pos[0].cpu().numpy()
    top_left = int(np.argmin(np.linalg.norm(verts0 - [CLOTH_CENTER[0] - CLOTH_SIZE / 2, 0.0, CLOTH_TOP], axis=1)))
    top_right = int(np.argmin(np.linalg.norm(verts0 - [CLOTH_CENTER[0] + CLOTH_SIZE / 2, 0.0, CLOTH_TOP], axis=1)))
    cloth.set_vertex_constraints([top_left, top_right], is_soft_constraint=True)
    print(f"pinned corners: {top_left} @ {verts0[top_left]}, {top_right} @ {verts0[top_right]}")

    # --- Scripted EE trajectory ---
    # The trajectory starts at the ACTUAL post-build EE pose (zero jump: a
    # teleporting IK target makes the PD-driven arm whip through the scene and
    # blows up the contact solve), then moves smoothly. The target orientation
    # likewise blends from the current EE quat to straight-down.
    down_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=gs.np_float), degrees=True)
    start_quat = ee_link.get_quat().reshape(-1)[:4].cpu().numpy().astype(gs.np_float)
    if float(np.dot(start_quat, down_quat)) < 0.0:
        down_quat = -down_quat

    def nlerp(q0, q1, t):
        q = (1.0 - t) * q0 + t * q1
        return (q / np.linalg.norm(q)).astype(gs.np_float)

    pinch_z = CLOTH_TOP - 0.035  # pinch 3.5cm below the top edge (palm stays above the edge)
    retract = ee_link.get_pos().reshape(-1)[:3].cpu().numpy().astype(np.float64)
    hover = np.array([CLOTH_CENTER[0], 0.0, pinch_z + HAND_TO_PINCH + 0.15])
    grasp = np.array([CLOTH_CENTER[0], 0.0, pinch_z + HAND_TO_PINCH])
    print(f"start EE (retract): {retract}", flush=True)
    target_pos = retract.copy()
    target_quat = start_quat.copy()
    gripper_target = 0.04  # open
    # Overshoot target (like the IPC teleop example's -0.03): the barrier stops
    # the pads at the fabric surface and the PD residual provides the grip.
    # (0.001 was too weak -> slip; 0.0 with floppy cloth caused pinch squirt,
    # cured by bending_stiffness=5.)
    GRIP_CLOSED = -0.01

    # Release the pins right after the close: holding a pinch on the taut,
    # strain-limited sheet squeezes the pads off the fabric ("wet watermelon
    # seed"); once the pins are gone the fabric slackens and the pinch holds.
    PHASE = dict(settle=80, approach=180, descend=300, close=360, release=380, lift=540, drop=620, end=680)

    def lerp(a, b, t):
        return a + (b - a) * np.clip(t, 0.0, 1.0)

    grasped_probe = []
    qpos_cmd = home.copy()
    last_tgt_key = None

    cam.start_recording()
    for i in range(PHASE["end"]):
        if i < PHASE["settle"]:
            # Blend orientation to straight-down while holding position
            t = i / PHASE["settle"]
            target_pos = retract.copy()
            target_quat = nlerp(start_quat, down_quat, t)
        elif i < PHASE["approach"]:
            t = (i - PHASE["settle"]) / (PHASE["approach"] - PHASE["settle"])
            target_pos = lerp(retract, hover, t)
            target_quat = down_quat
        elif i < PHASE["descend"]:
            t = (i - PHASE["approach"]) / (PHASE["descend"] - PHASE["approach"])
            target_pos = lerp(hover, grasp, t)
        elif i < PHASE["close"]:
            if i == PHASE["descend"]:
                # Close the loop on PD steady-state error: measure where the EE
                # actually settled and overdrive the target by the residual so
                # the fingertips reach the intended pinch height.
                ee_actual = ee_link.get_pos().reshape(-1)[:3].cpu().numpy().astype(np.float64)
                residual = grasp - ee_actual
                grasp = grasp + residual
                print(f"[step {i}] EE residual before close: {residual} -> overdriven grasp {grasp}", flush=True)
            target_pos = grasp.copy()
            gripper_target = GRIP_CLOSED
        elif i == PHASE["release"]:
            cloth.remove_vertex_constraints()
            print(f"[step {i}] pins released", flush=True)
        elif PHASE["release"] < i < PHASE["lift"]:
            t = (i - PHASE["release"]) / (PHASE["lift"] - PHASE["release"])
            sway = 0.12 * np.sin(2.0 * np.pi * t)
            target_pos = grasp + np.array([sway * 0.3, sway, 0.10 * t])
        elif i < PHASE["drop"]:
            t = (i - PHASE["lift"]) / (PHASE["drop"] - PHASE["lift"])
            target_pos = grasp + np.array([0.0, 0.0, 0.10 - 0.05 * t])
            if i == PHASE["lift"] + 40:
                gripper_target = 0.04  # open -> drop the cloth
                print(f"[step {i}] gripper opened", flush=True)

        # Command-space IK servoing:
        # - seed with the PREVIOUS COMMAND (init_qpos=qpos_cmd), not the
        #   contact-perturbed measured state, so commands form a smooth chain;
        # - max_samples=1 disables Genesis IK's random resampling (which
        #   otherwise teleports the PD target to another arm branch on any
        #   convergence miss -- the IPC teleop example dodges the same issue by
        #   hardcoding its initial qpos);
        # - skip IK entirely while the target is unchanged.
        tgt_key = (tuple(np.round(target_pos, 6)), tuple(np.round(target_quat, 6)))
        if tgt_key != last_tgt_key:
            last_tgt_key = tgt_key
            qpos = franka.inverse_kinematics(
                link=ee_link,
                pos=np.asarray(target_pos, dtype=gs.np_float),
                quat=target_quat,
                init_qpos=qpos_cmd,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=motor_dofs_idx,
            )
            qpos_np = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
            qpos_cmd[:7] = qpos_np[:7]
        franka.control_dofs_position(qpos_cmd[:7], dofs_idx_local=motor_dofs_idx)
        franka.control_dofs_position(gripper_target, dofs_idx_local=finger_dofs_idx)

        scene.step()
        scene.rigid_solver._func_update_geoms(scene._envs_idx)

        if i % RENDER_EVERY == 0:
            cam.render()

        if i % 20 == 0:
            qs = scene.sim._coupler._scene.solver
            print(f"[step {i}] newton={qs.newton_iters} t={qs.step_ms:.1f}ms", flush=True)

        if i in (PHASE["close"], PHASE["release"], PHASE["lift"], PHASE["drop"] - 1) or (
            PHASE["close"] <= i < PHASE["drop"] and i % 40 == 0
        ):
            pos = cloth.get_state().pos[0].cpu().numpy()
            print(
                f"[step {i}] cloth z: min={pos[:, 2].min():.3f} max={pos[:, 2].max():.3f} "
                f"top_edge_max={pos[:, 2].max():.3f}"
            )
            grasped_probe.append((i, pos[:, 2].max()))

    video_path = os.path.join(OUT_DIR, "qipc_robot_cloth.mp4")
    cam.stop_recording(save_to_filename=video_path, fps=FPS)
    print(f"[video] {video_path}")

    # --- Post-run checks ---
    pos = cloth.get_state().pos[0].cpu().numpy()
    ok = bool(np.isfinite(pos).all())
    # After the drop the cloth should be low (on/near the ground), not stuck at the pins
    if pos[:, 2].min() > 0.15:
        print(f"WARN: cloth did not drop (min_z={pos[:, 2].min():.3f})")
    # During the lift phase the cloth top must have been held up by the gripper
    lift_probe = [z for (s, z) in grasped_probe if s == PHASE["lift"]]
    if lift_probe and lift_probe[0] < CLOTH_TOP - 0.15:
        print(f"WARN: cloth top fell during lift ({lift_probe[0]:.3f}) -> grasp may have failed")
        ok = False
    print(f"final: min_z={pos[:, 2].min():.3f} max_z={pos[:, 2].max():.3f} -> {'OK' if ok else 'CHECK VIDEO'}")


if __name__ == "__main__":
    main()
