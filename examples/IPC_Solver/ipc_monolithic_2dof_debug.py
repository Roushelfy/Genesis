"""2-DOF debug bench: hinge + slide, ipc_monolithic vs external_articulation.

A fixed-base 2-DOF arm — a REVOLUTE (hinge) joint carrying a PRISMATIC (slide) joint —
each driven by the Genesis position controller to follow its own simple harmonic motion:
  hinge(t) = A_h * sin(2*pi*f_h*t)      [rad]
  slide(t) = A_s * sin(2*pi*f_s*t)      [m]
This exercises BOTH joint types at once (revolute torque + prismatic force actuation),
a kinematic CHAIN (link1 is the slide's parent and the hinge's child, so it accumulates
both wrenches), and armature injection on each. In driver mode it runs BOTH coupling
modes in isolated subprocesses and prints a side-by-side comparison, using
external_articulation (Genesis owns the reduced-coordinate dynamics) as the reference.

Run (from the Genesis repo; uv env has the libuipc fork + cu129 torch):
  # side-by-side comparison (one subprocess per mode):
  uv run examples/IPC_Solver/ipc_monolithic_2dof_debug.py --armature 0.1
  # single mode, full per-step trace:
  uv run examples/IPC_Solver/ipc_monolithic_2dof_debug.py --coup ipc_monolithic --armature 0.1 --verbose
  # watch it (single mode, real time):
  uv run examples/IPC_Solver/ipc_monolithic_2dof_debug.py --coup ipc_monolithic --armature 0.1 --vis --realtime

Knobs: --armature --damping --hinge-amp --hinge-freq --hinge-kp/-kv
       --slide-amp --slide-freq --slide-kp/-kv --dt --steps --effort --no-limit --vis --realtime
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

DOF_NAMES = ("hinge[rad]", "slide[m]")


def build_model(armature: float, damping: float, l1: float, l2: float) -> str:
    """Write the 2-DOF (hinge->slide) MJCF. Welded base body = fixed base.

    link1: box along +Z (length l1, mass 1.0), hinged about Y at the origin.
    link2: box along +Z (length l2, mass 0.5), mounted at the end of link1 and sliding
           along link1's local Z. armature/damping set on both joints.
    """
    w = 0.08
    mjcf = f"""<mujoco model="arm2dof">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom name="base_g" type="box" size="0.03 0.03 0.03" mass="0.2"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" pos="0 0 0"
               armature="{armature}" damping="{damping}" range="-3.1416 3.1416"/>
        <geom name="g1" type="box" size="{w / 2.0} {w / 2.0} {l1 / 2.0}" pos="0 0 {l1 / 2.0}"
              mass="1.0" rgba="0.85 0.35 0.35 1"/>
        <body name="link2" pos="0 0 {l1}">
          <joint name="j2" type="slide" axis="0 0 1" pos="0 0 0"
                 armature="{armature}" damping="{damping}" range="-0.12 0.12"/>
          <geom name="g2" type="box" size="{w * 0.35} {w * 0.35} {l2 / 2.0}" pos="0 0 {l2 / 2.0}"
                mass="0.5" rgba="0.35 0.55 0.9 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="arm2dof_")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf)
    return path


def run_one(mode: str, args) -> dict:
    """Run a single coupling mode; return per-step (steps, 2) dynamics arrays."""
    import genesis as gs

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    model_path = build_model(args.armature, args.damping, args.l1, args.l2)

    coupler_options = gs.options.IPCCouplerOptions(
        enable_rigid_rigid_contact=False,
        enable_rigid_ground_contact=False,
        monolithic_joint_limit_enable=not args.no_limit,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, 1.4, 0.8), camera_lookat=(0.0, 0.0, 0.4)
        ),
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file=model_path),
        material=gs.materials.Rigid(coup_type=mode),
    )
    scene.build()

    n_dofs = robot.n_dofs
    assert n_dofs == 2, f"expected 2 DOFs (hinge+slide), got {n_dofs}"
    robot.set_dofs_kp(np.array([args.hinge_kp, args.slide_kp], dtype=np.float32))
    robot.set_dofs_kv(np.array([args.hinge_kv, args.slide_kv], dtype=np.float32))
    robot.set_dofs_force_range(-args.effort, args.effort)

    coupler = getattr(scene.sim, "coupler", None)

    def vec():
        return np.asarray(robot.get_dofs_position().cpu().numpy()).reshape(-1)[:2].copy()

    def vel():
        return np.asarray(robot.get_dofs_velocity().cpu().numpy()).reshape(-1)[:2].copy()

    def injected_torques():
        taus = np.array([math.nan, math.nan], dtype=np.float64)
        data = getattr(coupler, "_ipc_monolithic_data_by_entity", {}) or {}
        ad = data.get(robot)
        if ad is None or getattr(ad, "torque", None) is None:
            return taus
        for j, dof_local in enumerate(ad.joints_dof_idx_local):
            if 0 <= dof_local < 2:
                taus[dof_local] = float(ad.torque[0, j])
        return taus

    print(f"\n========== {mode} ==========", flush=True)
    print(f"dt={args.dt}  armature={args.armature}  damping={args.damping}  effort={args.effort}", flush=True)
    for lk in robot.links:
        print(f"  link {lk.name:8s} inertial_mass={float(lk.inertial_mass or 0.0):.4f} kg", flush=True)
    print(f"  hinge: amp={args.hinge_amp} rad  freq={args.hinge_freq} Hz  kp={args.hinge_kp} kv={args.hinge_kv}",
          flush=True)
    print(f"  slide: amp={args.slide_amp} m    freq={args.slide_freq} Hz  kp={args.slide_kp} kv={args.slide_kv}",
          flush=True)

    w_h = 2.0 * math.pi * args.hinge_freq
    w_s = 2.0 * math.pi * args.slide_freq
    rec = {k: [] for k in ("t", "q_cmd", "q", "qd", "tau_inj")}
    if args.verbose:
        print(f"\n{'step':>4} {'t':>6} | {'h_cmd':>8} {'h':>8} {'h_err':>8} {'h_qd':>8} {'h_tau':>8} "
              f"| {'s_cmd':>8} {'s':>8} {'s_err':>8} {'s_qd':>8} {'s_tau':>8}", flush=True)

    def cmd_at(t):
        return np.array([args.hinge_amp * math.sin(w_h * t), args.slide_amp * math.sin(w_s * t)], dtype=np.float64)

    for i in range(args.steps):
        t = i * args.dt
        q_cmd = cmd_at(t)
        robot.control_dofs_position(q_cmd.astype(np.float32))
        t0 = time.perf_counter()
        scene.step()
        if args.realtime:
            dtr = args.dt - (time.perf_counter() - t0)
            if dtr > 0:
                time.sleep(dtr)

        q, qd, tau = vec(), vel(), injected_torques()
        rec["t"].append(t)
        rec["q_cmd"].append(q_cmd)
        rec["q"].append(q)
        rec["qd"].append(qd)
        rec["tau_inj"].append(tau)

        if args.verbose and (i % args.log_every == 0 or i == args.steps - 1):
            e = q - q_cmd
            print(f"{i:>4} {t:>6.3f} | {q_cmd[0]:>8.4f} {q[0]:>8.4f} {e[0]:>8.4f} {qd[0]:>8.3f} {tau[0]:>8.3f} "
                  f"| {q_cmd[1]:>8.4f} {q[1]:>8.4f} {e[1]:>8.4f} {qd[1]:>8.3f} {tau[1]:>8.3f}", flush=True)

    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    err = out["q"] - out["q_cmd"]
    print(f"\n--- {mode} summary ---", flush=True)
    for d in range(2):
        print(f"  {DOF_NAMES[d]:10s} tracking RMS={np.sqrt(np.mean(err[:, d] ** 2)):.5f}  "
              f"max|q|={np.max(np.abs(out['q'][:, d])):.5f}  max|qd|={np.max(np.abs(out['qd'][:, d])):.4f}  "
              f"final={out['q'][-1, d]:.5f}", flush=True)

    if args.vis and getattr(scene, "viewer", None) is not None:
        print(f"\n[{mode}] holding viewer open — continuing SHM (Ctrl-C / close window to exit)...", flush=True)
        try:
            i = args.steps
            while scene.viewer.is_alive():
                robot.control_dofs_position(cmd_at(i * args.dt).astype(np.float32))
                t0 = time.perf_counter()
                scene.step()
                i += 1
                if args.realtime:
                    dtr = args.dt - (time.perf_counter() - t0)
                    if dtr > 0:
                        time.sleep(dtr)
        except KeyboardInterrupt:
            pass

    gs.destroy() if hasattr(gs, "destroy") else None
    try:
        os.remove(model_path)
    except OSError:
        pass
    return out


def compare(a: dict, b: dict, args):
    """Driver-side comparison: a=ipc_monolithic, b=external_articulation."""
    t = a["t"]
    n = len(t)
    print("\n\n================= COMPARISON (monolithic vs external_articulation) =================", flush=True)
    print(f"{'t':>6} | {'h_cmd':>7} {'h_mono':>8} {'h_ext':>8} {'dh':>7} | "
          f"{'s_cmd':>7} {'s_mono':>8} {'s_ext':>8} {'ds':>7}", flush=True)
    step = max(1, n // 22)
    for i in range(0, n, step):
        print(f"{t[i]:>6.3f} | {a['q_cmd'][i,0]:>7.3f} {a['q'][i,0]:>8.4f} {b['q'][i,0]:>8.4f} "
              f"{a['q'][i,0]-b['q'][i,0]:>7.4f} | {a['q_cmd'][i,1]:>7.3f} {a['q'][i,1]:>8.4f} "
              f"{b['q'][i,1]:>8.4f} {a['q'][i,1]-b['q'][i,1]:>7.4f}", flush=True)

    print("\n----- summary -----", flush=True)
    diag = []
    for d in range(2):
        em = a["q"][:, d] - a["q_cmd"][:, d]
        ee = b["q"][:, d] - b["q_cmd"][:, d]
        cross = a["q"][:, d] - b["q"][:, d]
        amp = args.hinge_amp if d == 0 else args.slide_amp
        print(f"  {DOF_NAMES[d]:10s} tracking RMS  mono={np.sqrt(np.mean(em**2)):.5f}  "
              f"ext={np.sqrt(np.mean(ee**2)):.5f}  | cross-mode RMS |mono-ext|={np.sqrt(np.mean(cross**2)):.5f}",
              flush=True)
        if np.max(np.abs(a["q"][:, d])) < 0.1 * amp:
            diag.append(f"{DOF_NAMES[d]} monolithic ~DEAD (barely moves) -> not actuated")
        elif np.sqrt(np.mean(cross**2)) > 0.3 * amp:
            diag.append(f"{DOF_NAMES[d]} monolithic differs strongly from ext_art")
    print("  diagnosis: " + ("; ".join(diag) if diag else "both DOFs: monolithic tracks like ext_art (no gross anomaly)"),
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coup", default=None, choices=["ipc_monolithic", "external_articulation"],
                   help="worker mode: run a single coupling type. Omit for side-by-side driver.")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--l1", type=float, default=0.4, help="hinge link length (m)")
    p.add_argument("--l2", type=float, default=0.3, help="slide link length (m)")
    p.add_argument("--armature", type=float, default=0.1, help="joint armature (reflected rotor inertia)")
    p.add_argument("--damping", type=float, default=0.0, help="passive joint damping")
    p.add_argument("--effort", type=float, default=200.0, help="force_range clamp on the control force/torque")
    p.add_argument("--hinge-amp", type=float, default=0.5, help="hinge SHM amplitude (rad)")
    p.add_argument("--hinge-freq", type=float, default=0.5, help="hinge SHM frequency (Hz)")
    p.add_argument("--hinge-kp", type=float, default=300.0)
    p.add_argument("--hinge-kv", type=float, default=8.0)
    p.add_argument("--slide-amp", type=float, default=0.04, help="slide SHM amplitude (m)")
    p.add_argument("--slide-freq", type=float, default=0.7, help="slide SHM frequency (Hz)")
    p.add_argument("--slide-kp", type=float, default=1000.0)
    p.add_argument("--slide-kv", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-limit", action="store_true", help="disable monolithic joint limit penalty")
    p.add_argument("-v", "--vis", action="store_true", help="show the viewer (driver mode shows each mode in turn)")
    p.add_argument("--realtime", action="store_true", help="throttle stepping to ~real time")
    p.add_argument("--verbose", action="store_true", help="print every --log-every steps")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default=None, help="(worker) npz path to dump per-step arrays")
    args = p.parse_args()

    if args.coup is not None:
        data = run_one(args.coup, args)
        if args.out:
            np.savez(args.out, **data)
        return

    common = [
        sys.executable, os.path.abspath(__file__),
        "--backend", args.backend, "--l1", str(args.l1), "--l2", str(args.l2),
        "--armature", str(args.armature), "--damping", str(args.damping), "--effort", str(args.effort),
        "--hinge-amp", str(args.hinge_amp), "--hinge-freq", str(args.hinge_freq),
        "--hinge-kp", str(args.hinge_kp), "--hinge-kv", str(args.hinge_kv),
        "--slide-amp", str(args.slide_amp), "--slide-freq", str(args.slide_freq),
        "--slide-kp", str(args.slide_kp), "--slide-kv", str(args.slide_kv),
        "--dt", str(args.dt), "--steps", str(args.steps), "--log-every", str(args.log_every),
    ]
    if args.no_limit:
        common.append("--no-limit")
    if args.vis:
        common.append("--vis")
    if args.realtime:
        common.append("--realtime")
    if args.verbose:
        common.append("--verbose")

    with tempfile.TemporaryDirectory() as td:
        out_m = os.path.join(td, "mono.npz")
        out_e = os.path.join(td, "ext.npz")
        subprocess.run(common + ["--coup", "ipc_monolithic", "--out", out_m], check=True)
        subprocess.run(common + ["--coup", "external_articulation", "--out", out_e], check=True)
        a = dict(np.load(out_m))
        b = dict(np.load(out_e))
    compare(a, b, args)


if __name__ == "__main__":
    main()
