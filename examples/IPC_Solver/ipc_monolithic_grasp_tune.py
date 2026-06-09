"""Interactive tuning playground for the IPC grasp under different coupling modes.

Runs the Franka grasp-a-cube sequence with a viewer and exposes the parameters that
matter for whether `ipc_monolithic` can actually grasp/lift the cube. Per-phase it prints
the end-effector pos vs the IK target, the gripper opening, and the cube height — so you
can see *where* it fails (arm not reaching, gripper not closing, or grip slipping on lift).

Run directly from the Genesis repo (its uv env now carries the libuipc fork + cu129
torch via the default 'ipc' dependency group):
  cd ~/work/Genesis && uv run examples/IPC_Solver/ipc_monolithic_grasp_tune.py --vis --realtime

Compare against the working reduced-coordinate path:
  uv run examples/IPC_Solver/ipc_monolithic_grasp_tune.py --vis --coup external_articulation

(Fallback, to reuse the prebuilt gs-gym-internal venv as-is:
  examples/IPC_Solver/run.sh ipc_monolithic_grasp_tune.py --vis --realtime)

Knobs to try for ipc_monolithic (default reproduces the non-lifting case):
  --kp-scale       scale all arm PD gains (lower = softer/faster IPC solve, less tracking)
  --joint-strength AffineBody joint constraint stiffness (higher = stiffer joints, firmer)
  --finger-kp/-kv  gripper PD gains (grip force ~ finger-kp * penetration)
  --grip           finger close target (0.0 = fully closed; negative = squeeze harder)
  --grasp-z        height the gripper descends to before closing (align with cube top)
  --lift-z         height to lift to after grasping
  --friction       coupling friction for franka + cube (higher = less slip)
  --cube           abd (rigid) | fem (deformable) | rigid (no-ipc baseline)
  --rho            cube density
  --newton-tol     IPC newton translation tolerance (higher = looser/faster)
  --dt             timestep (smaller = more stable, slower)
  --steps-scale    multiply all phase step counts (e.g. 2.0 to give more settling time)
  --realtime       throttle playback to ~real time so it's watchable
  --backend        cpu | gpu
"""

import argparse
import time

import numpy as np

import genesis as gs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis", action="store_true", default=False)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--coup", default="ipc_monolithic",
                   choices=["ipc_monolithic", "external_articulation", "two_way_soft_constraint", "none"])
    p.add_argument("--cube", default="abd", choices=["abd", "fem", "rigid"])
    p.add_argument("--rho", type=float, default=1000.0)
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--joint-strength", type=float, default=100.0)
    p.add_argument("--finger-kp", type=float, default=1000.0)
    p.add_argument("--finger-kv", type=float, default=100.0)
    p.add_argument("--arm-kv", type=float, default=10.0,
                   help="arm joint damping kv. The Franka MJCF default (~200) is far over the "
                        "monolithic explicit-damping bound (kv*dt/I<2) and causes jitter; this overrides it.")
    p.add_argument("--grip", type=float, default=0.0, help="finger close target (m); negative squeezes")
    p.add_argument("--grasp-z", type=float, default=0.135)
    p.add_argument("--lift-z", type=float, default=0.3)
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--cube-friction", type=float, default=0.5)
    p.add_argument("--newton-tol", type=float, default=10.0)
    p.add_argument("--contact-d-hat", type=float, default=None)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps-scale", type=float, default=1.0)
    p.add_argument("--realtime", action="store_true", default=False)
    p.add_argument("--hold-steps", type=int, default=300, help="extra steps holding the lift pose (watch grip)")
    args = p.parse_args()

    gs.init(backend=getattr(gs, args.backend), logging_level="info")

    no_ipc = args.coup == "none"
    ipc_kwargs = dict(
        enable_rigid_rigid_contact=(args.cube != "fem"),
        enable_rigid_ground_contact=(args.cube != "fem"),
        newton_translation_tolerance=args.newton_tol,
        joint_strength_ratio=args.joint_strength,
    )
    if args.contact_d_hat is not None:
        ipc_kwargs["contact_d_hat"] = args.contact_d_hat
    coupler_options = None if no_ipc else gs.options.IPCCouplerOptions(**ipc_kwargs)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, 1.0, 1.0), camera_lookat=(0.3, 0.0, 0.5)),
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only") if not no_ipc else None)

    franka_mat = None
    if not no_ipc:
        kw = dict(coup_friction=args.friction, coup_type=args.coup, coup_stiffness=(10.0, 10.0))
        if args.coup == "two_way_soft_constraint":
            kw["coup_links"] = ("left_finger", "right_finger")
        franka_mat = gs.materials.Rigid(**kw)
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"), material=franka_mat
    )

    if no_ipc:
        cube_mat = gs.materials.Rigid()
    elif args.cube == "abd":
        cube_mat = gs.materials.Rigid(rho=args.rho, coup_type="ipc_only", coup_friction=args.cube_friction,
                                      enable_coup_collision=True)
    elif args.cube == "fem":
        cube_mat = gs.materials.FEM.Elastic(E=5.0e4, nu=0.45, rho=args.rho, friction_mu=args.cube_friction,
                                            model="stable_neohookean")
    else:
        cube_mat = gs.materials.Rigid(rho=args.rho)
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.03), size=(0.05, 0.05, 0.05)),
        material=cube_mat, surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build()

    # --- Mass report: ground-truth URDF link masses to compare against IPC ---
    # For ipc_monolithic the coupler injects each ABD body's true 12x12 mass matrix
    # and logs "[IPC ABD] link=...: true-inertia ABD mass (m=... kg)". The sum of those
    # injected masses should equal the TOTAL below (mass is conserved under fixed-joint
    # merging); each per-body m should match the URDF link/merged-subtree mass, NOT
    # rho*mesh_volume. With the old density model the fingers came out near-massless.
    if not no_ipc:
        link_masses = [(lk.name, float(lk.inertial_mass or 0.0)) for lk in franka.links]
        total = sum(m for _, m in link_masses)
        print("\n=== Franka URDF link inertial masses (ground truth) ===", flush=True)
        for name, m in link_masses:
            print(f"  {name:24s} {m:8.4f} kg", flush=True)
        print(f"  {'TOTAL':24s} {total:8.4f} kg", flush=True)
        print(
            "Compare with the '[IPC ABD] ... true-inertia ABD mass (m=...)' lines above: revolute\n"
            "links match the URDF mass (armature is folded into their INERTIA); prismatic joints\n"
            "fold armature into the MASS too (fingers: 0.015 + 0.1 armature = 0.115 kg).\n",
            flush=True,
        )

    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    ee = franka.get_link("hand")
    ee_quat = [0.0, 1.0, 0.0, 0.0]

    # franka.set_dofs_kp(base_kp * np.array([args.kp_scale] * 7 + [100.0, 100.0], dtype=np.float32))
    # IMPORTANT: also override the ARM kv. The MJCF actuator bias (biasprm[2]) gives the arm
    # a default damping kv ~200, which set_dofs_kp does NOT touch. For ipc_monolithic that
    # explicit -kv*qd term blows the stability bound (kv*dt/I<2) and jitters at ANY kp.

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0])
    franka.set_dofs_kv([100.0, 100.0], fingers_dof)
    def diag(tag):
        ee_pos = ee.get_pos().cpu().numpy().reshape(-1)
        q = franka.get_dofs_position().cpu().numpy().reshape(-1)
        cz = float(cube.get_pos().cpu().numpy().reshape(-1)[2]) if args.cube != "fem" else float("nan")
        print(f"[{tag:9s}] ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
              f"fingers=({q[7]:.4f},{q[8]:.4f}) cube_z={cz:.4f}", flush=True)

    def run(n, target_ee_z=None):
        n = int(n * args.steps_scale)
        for _ in range(n):
            t0 = time.perf_counter()
            scene.step()
            if args.realtime:
                dtr = args.dt - (time.perf_counter() - t0)
                if dtr > 0:
                    time.sleep(dtr)

    # Phase 1: move above the cube, gripper open
    q = franka.inverse_kinematics(link=ee, pos=[0.65, 0.0, 0.4], quat=ee_quat)
    q[fingers_dof] = 0.04
    franka.control_dofs_position(q)
    run(200); diag("approach")

    # Phase 2: descend halfway
    q = franka.inverse_kinematics(link=ee, pos=[0.65, 0.0, 0.25], quat=ee_quat)
    franka.control_dofs_position(q[motors_dof], dofs_idx_local=motors_dof)
    run(100); diag("descend")

    # Phase 3: reach grasp height
    q = franka.inverse_kinematics(link=ee, pos=[0.65, 0.0, args.grasp_z], quat=ee_quat)
    franka.control_dofs_position(q[motors_dof], dofs_idx_local=motors_dof)
    run(50); diag("at-grasp")

    # Phase 4: close gripper
    franka.control_dofs_position(q[motors_dof], dofs_idx_local=motors_dof)
    franka.control_dofs_position(args.grip, dofs_idx_local=fingers_dof)
    run(30); diag("grip")

    # Phase 5: lift
    q = franka.inverse_kinematics(link=ee, pos=[0.65, 0.0, args.lift_z], quat=ee_quat)
    franka.control_dofs_position(q[motors_dof], dofs_idx_local=motors_dof)
    run(80); diag("lifted")

    # Hold the lift pose so you can watch whether the grip survives
    run(args.hold_steps); diag("hold-end")

    if args.vis:
        print("Holding viewer open (Ctrl-C to exit)...", flush=True)
        try:
            while scene.viewer.is_alive() if hasattr(scene, "viewer") and scene.viewer else False:
                scene.step()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
