"""Minimal 1-DOF debug bench: ipc_monolithic vs external_articulation.

A fixed-base arm with a SINGLE revolute joint is driven by the Genesis position
controller to follow simple harmonic motion  q_target(t) = A*sin(2*pi*f*t).  Each
step we record the full joint dynamics (commanded angle/vel, IPC-resolved angle,
read-back velocity, finite-difference vel/acc, and — for monolithic — the joint
torque IPC actually received).  In driver mode the script runs BOTH coupling modes
in *isolated subprocesses* and prints a side-by-side comparison, using
external_articulation (Genesis owns the reduced-coordinate dynamics) as the trusted
reference so you can see exactly where ipc_monolithic diverges.

Why a 1-DOF box pendulum: it is fully analyzable. With axis=y the link swings under
gravity (tests gravity + inertia + control); with axis=z gravity drops out so the
dynamics is pure I*theta_ddot = tau (clean inertia check).

Run (from the Genesis repo; uv env has the libuipc fork + cu129 torch):
  # side-by-side comparison (spawns one subprocess per mode):
  uv run examples/IPC_Solver/ipc_monolithic_1dof_debug.py
  # single mode, full per-step trace:
  uv run examples/IPC_Solver/ipc_monolithic_1dof_debug.py --coup ipc_monolithic --verbose
  # gravity-free pure-inertia check:
  uv run examples/IPC_Solver/ipc_monolithic_1dof_debug.py --axis z
  # watch it in the viewer (single mode, real-time):
  uv run examples/IPC_Solver/ipc_monolithic_1dof_debug.py --coup ipc_monolithic --vis --realtime --kp 10 --kv 0.1
  # (driver --vis shows monolithic first, then external_articulation; close each window to advance)

  # armature check: monolithic should match ext_art once armature is folded into the ABD inertia:
  uv run examples/IPC_Solver/ipc_monolithic_1dof_debug.py --axis z --armature 0.1 --kp 200 --kv 5

Knobs: --axis {y,z} --mass --length --armature --damping --kp --kv --amp --freq --dt --steps
       --no-limit --vis --realtime --effort
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def build_model(axis: str, mass: float, length: float, armature: float, damping: float) -> tuple[str, float, float]:
    """Write a 1-hinge box-pendulum MJCF; return (path, I_com_axis, I_joint_axis).

    MJCF (not URDF) so the joint's ``armature`` (reflected rotor inertia) and ``damping``
    are settable — armature is a MuJoCo concept Genesis adds to the effective joint
    inertia. The link is a box (0.08 x 0.08 x length), COM at (0,0,length/2), hinged at
    the origin (fixed base = worldbody). I_com_axis / I_joint_axis are the scalar inertia
    about the rotation axis, about the COM and about the joint (parallel axis), for
    reference — they EXCLUDE armature; the effective joint inertia is I_joint + armature.
    """
    w = 0.08
    ixx = mass / 12.0 * (w * w + length * length)
    izz = mass / 12.0 * (w * w + w * w)
    axis_vec = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[axis]
    # scalar inertia about the chosen axis, about COM and about the hinge (origin)
    i_com = {"x": ixx, "y": ixx, "z": izz}[axis]
    lever = length / 2.0
    # parallel-axis adds m*d^2 only for axes perpendicular to the COM offset (offset is +z)
    i_joint = i_com + (mass * lever * lever if axis in ("x", "y") else 0.0)

    # The root body "base" has NO joint -> welded to world -> fixed base (Genesis infers
    # fixity structurally; MJCF has no 'fixed' morph flag). link1 hinges off the base.
    mjcf = f"""<mujoco model="arm1dof">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom name="base_g" type="box" size="0.025 0.025 0.025" mass="0.1"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="{axis_vec}" pos="0 0 0"
               armature="{armature}" damping="{damping}" range="-3.1416 3.1416"/>
        <geom name="g1" type="box" size="{w / 2.0} {w / 2.0} {length / 2.0}" pos="0 0 {length / 2.0}"
              mass="{mass}" rgba="0.8 0.35 0.35 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="arm1dof_")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf)
    return path, i_com, i_joint


def run_one(mode: str, args) -> dict:
    """Run a single coupling mode; return per-step dynamics arrays."""
    import genesis as gs

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    model_path, i_com, i_joint = build_model(args.axis, args.mass, args.length, args.armature, args.damping)

    coupler_options = gs.options.IPCCouplerOptions(
        enable_rigid_rigid_contact=False,
        enable_rigid_ground_contact=False,
        monolithic_joint_limit_enable=not args.no_limit,
        monolithic_implicit_damping=not args.no_implicit_damping,
        ipc_monolithic_actuation=args.actuation,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, 1.2, 0.9), camera_lookat=(0.0, 0.0, 0.2)
        ),
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file=model_path),
        material=gs.materials.Rigid(coup_type=mode),
    )
    scene.build()

    # 1 revolute DOF
    n_dofs = robot.n_dofs
    assert n_dofs == 1, f"expected 1 DOF, got {n_dofs}"
    robot.set_dofs_kp(np.array([args.kp], dtype=np.float32))
    robot.set_dofs_kv(np.array([args.kv], dtype=np.float32))
    robot.set_dofs_force_range(-args.effort, args.effort)

    link = robot.get_link("link1")
    coupler = getattr(scene.sim, "coupler", None)

    def scalar(t):
        return float(np.asarray(t.cpu().numpy()).reshape(-1)[0])

    def injected_torque():
        # monolithic only: the per-joint torque IPC actually received this step
        data = getattr(coupler, "_ipc_monolithic_data_by_entity", {}) or {}
        ad = data.get(robot)
        if ad is None or getattr(ad, "torque", None) is None:
            return math.nan
        return float(np.asarray(ad.torque).reshape(-1)[0])

    print(f"\n========== {mode} ==========", flush=True)
    print(f"axis={args.axis}  mass={args.mass}  length={args.length}  dt={args.dt}", flush=True)
    print(f"link1 inertial_mass = {float(link.inertial_mass):.5f} kg", flush=True)
    print(f"I about axis: COM={i_com:.6f}  joint(parallel-axis)={i_joint:.6f} kg*m^2", flush=True)
    print(f"armature={args.armature:.4f}  damping={args.damping:.4f}  -> "
          f"effective joint inertia I_joint+armature = {i_joint + args.armature:.6f}", flush=True)
    print(f"kp={args.kp} kv={args.kv}  SHM amp={args.amp} rad freq={args.freq} Hz", flush=True)

    omega = 2.0 * math.pi * args.freq
    rec = {k: [] for k in ("t", "q_cmd", "qd_cmd", "q", "qd", "fd_vel", "fd_acc", "tau_inj")}
    prev_q = prev_qd = None
    if args.verbose:
        print(f"\n{'step':>4} {'t':>6} {'q_cmd':>8} {'q':>8} {'err':>8} "
              f"{'qd_cmd':>8} {'qd':>8} {'fd_vel':>8} {'tau_inj':>8}", flush=True)

    for i in range(args.steps):
        t = i * args.dt
        q_cmd = args.amp * math.sin(omega * t)
        qd_cmd = args.amp * omega * math.cos(omega * t)
        robot.control_dofs_position(np.array([q_cmd], dtype=np.float32))
        t0 = time.perf_counter()
        scene.step()
        if args.realtime:
            dtr = args.dt - (time.perf_counter() - t0)
            if dtr > 0:
                time.sleep(dtr)

        q = scalar(robot.get_dofs_position())
        qd = scalar(robot.get_dofs_velocity())
        fd_vel = (q - prev_q) / args.dt if prev_q is not None else math.nan
        fd_acc = (qd - prev_qd) / args.dt if prev_qd is not None else math.nan
        tau = injected_torque() if mode == "ipc_monolithic" else math.nan

        rec["t"].append(t); rec["q_cmd"].append(q_cmd); rec["qd_cmd"].append(qd_cmd)
        rec["q"].append(q); rec["qd"].append(qd); rec["fd_vel"].append(fd_vel)
        rec["fd_acc"].append(fd_acc); rec["tau_inj"].append(tau)
        prev_q, prev_qd = q, qd

        if args.verbose and (i % args.log_every == 0 or i == args.steps - 1):
            print(f"{i:>4} {t:>6.3f} {q_cmd:>8.4f} {q:>8.4f} {q - q_cmd:>8.4f} "
                  f"{qd_cmd:>8.3f} {qd:>8.3f} {fd_vel:>8.3f} {tau:>8.4f}", flush=True)

    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    # per-mode summary
    err = out["q"] - out["q_cmd"]
    vel_mismatch = out["qd"] - out["fd_vel"]
    print(f"\n--- {mode} summary ---", flush=True)
    print(f"  tracking RMS error      = {np.sqrt(np.mean(err**2)):.5f} rad", flush=True)
    print(f"  max |q|                 = {np.max(np.abs(out['q'])):.5f} rad "
          f"(cmd amp {args.amp})", flush=True)
    print(f"  max |qd|                = {np.max(np.abs(out['qd'])):.5f} rad/s", flush=True)
    print(f"  vel readback vs fin-diff RMS = {np.sqrt(np.nanmean(vel_mismatch**2)):.5f} rad/s", flush=True)
    print(f"  final q                 = {out['q'][-1]:.5f} rad", flush=True)
    if mode == "ipc_monolithic" and args.axis == "z" and np.isfinite(out["tau_inj"]).any():
        # gravity-free: inferred inertia = tau / angular_accel (use mid-run samples)
        m = slice(len(out["t"]) // 4, 3 * len(out["t"]) // 4)
        acc = out["fd_acc"][m]
        tau = out["tau_inj"][m]
        good = np.isfinite(acc) & np.isfinite(tau) & (np.abs(acc) > 1.0)
        if good.any():
            inferred = np.median(tau[good] / acc[good])
            i_eff = i_joint + args.armature
            print(f"  inferred I (tau/accel)  = {inferred:.6f}  "
                  f"(I_joint+armature={i_eff:.6f}, I_joint={i_joint:.6f})", flush=True)

    # Keep the viewer open, continuing the SHM so the arm keeps moving (Ctrl-C to exit).
    if args.vis and getattr(scene, "viewer", None) is not None:
        print(f"\n[{mode}] holding viewer open — continuing SHM (Ctrl-C / close window to exit)...", flush=True)
        try:
            i = args.steps
            while scene.viewer.is_alive():
                q_cmd = args.amp * math.sin(omega * i * args.dt)
                robot.control_dofs_position(np.array([q_cmd], dtype=np.float32))
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
    """Driver-side side-by-side comparison: a=ipc_monolithic, b=external_articulation."""
    t = a["t"]
    n = len(t)
    print("\n\n================= COMPARISON (monolithic vs external_articulation) =================", flush=True)
    print(f"{'t':>6} {'q_cmd':>8} | {'q_mono':>8} {'q_ext':>8} {'dq':>8} | "
          f"{'qd_mono':>8} {'qd_ext':>8} | {'tau_mono':>9}", flush=True)
    step = max(1, n // 25)
    for i in range(0, n, step):
        print(f"{t[i]:>6.3f} {a['q_cmd'][i]:>8.4f} | {a['q'][i]:>8.4f} {b['q'][i]:>8.4f} "
              f"{a['q'][i] - b['q'][i]:>8.4f} | {a['qd'][i]:>8.3f} {b['qd'][i]:>8.3f} | "
              f"{a['tau_inj'][i]:>9.4f}", flush=True)

    err_m = a["q"] - a["q_cmd"]
    err_e = b["q"] - b["q_cmd"]
    cross = a["q"] - b["q"]
    print("\n----- summary -----", flush=True)
    print(f"  tracking RMS error   monolithic = {np.sqrt(np.mean(err_m**2)):.5f} rad", flush=True)
    print(f"  tracking RMS error   ext_artic  = {np.sqrt(np.mean(err_e**2)):.5f} rad", flush=True)
    print(f"  cross-mode RMS |q_mono - q_ext|  = {np.sqrt(np.mean(cross**2)):.5f} rad", flush=True)
    print(f"  max |q|  mono={np.max(np.abs(a['q'])):.4f}  ext={np.max(np.abs(b['q'])):.4f}  "
          f"(cmd amp {args.amp})", flush=True)
    print(f"  max |qd| mono={np.max(np.abs(a['qd'])):.4f}  ext={np.max(np.abs(b['qd'])):.4f}", flush=True)

    # crude failure-mode classifier on monolithic
    diag = []
    if np.max(np.abs(a["q"])) < 0.1 * args.amp:
        diag.append("monolithic joint is ~DEAD (barely moves) -> torque not reaching the body / too weak")
    if np.max(np.abs(a["q"])) > 3.0 * args.amp:
        diag.append("monolithic is DIVERGING (q >> cmd) -> unstable gains / wrong sign / wrong inertia")
    if np.sqrt(np.mean(cross**2)) > 0.3 * args.amp:
        diag.append("monolithic differs strongly from ext_art reference -> dynamics mismatch")
    # phase: correlation between mono and cmd
    if np.std(a["q"]) > 1e-6:
        corr = np.corrcoef(a["q"], a["q_cmd"])[0, 1]
        if corr < 0.5:
            diag.append(f"monolithic poorly correlated with command (corr={corr:.2f}) -> lag/oscillation")
    print("  diagnosis: " + ("; ".join(diag) if diag else "monolithic tracks like ext_art (no gross anomaly)"),
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coup", default=None,
                   choices=["ipc_monolithic", "external_articulation"],
                   help="worker mode: run a single coupling type. Omit for side-by-side driver.")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--axis", choices=["x", "y", "z"], default="y",
                   help="revolute axis. y=gravity pendulum, z=gravity-free pure-inertia.")
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--length", type=float, default=0.4)
    p.add_argument("--armature", type=float, default=0.0,
                   help="joint armature (reflected rotor inertia); Franka uses ~0.1. Tests the "
                        "monolithic armature injection against ext_art.")
    p.add_argument("--damping", type=float, default=0.0, help="passive joint damping")
    p.add_argument("--effort", type=float, default=200.0, help="force_range clamp on the control torque")
    p.add_argument("--kp", type=float, default=20.0)
    p.add_argument("--kv", type=float, default=1.0)
    p.add_argument("--amp", type=float, default=1.0, help="SHM amplitude (rad)")
    p.add_argument("--freq", type=float, default=0.5, help="SHM frequency (Hz)")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--no-limit", action="store_true", help="disable monolithic joint limit penalty")
    p.add_argument("--no-implicit-damping", action="store_true",
                   help="disable the monolithic implicit-damping torque scaling (apply raw explicit PD)")
    p.add_argument("--actuation", choices=["torque", "pd_prototype"], default="torque",
                   help="ipc_monolithic actuation: explicit torque (default) or implicit-PD driving-joint prototype (M6 P0)")
    p.add_argument("-v", "--vis", action="store_true", help="show the viewer (driver mode shows each mode in turn)")
    p.add_argument("--realtime", action="store_true", help="throttle stepping to ~real time (watchable)")
    p.add_argument("--verbose", action="store_true", help="print every --log-every steps")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default=None, help="(worker) npz path to dump per-step arrays")
    args = p.parse_args()

    if args.coup is not None:
        data = run_one(args.coup, args)
        if args.out:
            np.savez(args.out, **data)
        return

    # driver: run both modes in isolated subprocesses, then compare
    common = [
        sys.executable, os.path.abspath(__file__),
        "--backend", args.backend, "--axis", args.axis,
        "--mass", str(args.mass), "--length", str(args.length),
        "--armature", str(args.armature), "--damping", str(args.damping), "--effort", str(args.effort),
        "--kp", str(args.kp), "--kv", str(args.kv),
        "--amp", str(args.amp), "--freq", str(args.freq),
        "--dt", str(args.dt), "--steps", str(args.steps),
        "--log-every", str(args.log_every),
        "--actuation", str(args.actuation),
    ]
    if args.no_limit:
        common.append("--no-limit")
    if args.no_implicit_damping:
        common.append("--no-implicit-damping")
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
