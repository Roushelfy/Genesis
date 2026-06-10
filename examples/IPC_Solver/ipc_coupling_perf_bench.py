"""Runtime-performance benchmark: ipc_monolithic vs external_articulation.

A fixed-base Franka is driven through LARGE-amplitude, FAST, repeated multi-joint
sinusoidal motion and the per-step wall time is measured (with CUDA sync) to compare
the two coupling modes' speed. Free-space by default (no contact) so the comparison
isolates the coupling/dynamics cost; pass --ground to add a contact plane.

In driver mode it runs BOTH modes in isolated subprocesses (avoids libuipc global
state) and prints a side-by-side speed comparison.

Run (from the Genesis repo; uv env has the libuipc fork + cu129 torch):
  # side-by-side benchmark:
  uv run examples/IPC_Solver/ipc_coupling_perf_bench.py
  # single mode:
  uv run examples/IPC_Solver/ipc_coupling_perf_bench.py --coup ipc_monolithic
  # harder: faster / bigger / with contact:
  uv run examples/IPC_Solver/ipc_coupling_perf_bench.py --freq 1.5 --amp-scale 1.2 --ground
  # also write an mp4 per mode (coupling_perf_<mode>.mp4 in cwd):
  uv run examples/IPC_Solver/ipc_coupling_perf_bench.py --video

NOTE on timing: the GPU must be otherwise IDLE for the numbers to mean anything --
the workload is deterministic, so run-to-run variance comes from other GPU processes
contending for the device (check `nvidia-smi` shows ~0% util first). Rendering for
--video happens OUTSIDE the timed region, but a video run is slower overall and may
shift GPU clocks, so measure speed WITHOUT --video and use --video only to eyeball
the motion / behaviour.

Knobs: --freq --amp-scale --steps --warmup --dt --kp --kv --ground --backend --video
"""

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

# Franka "ready" home + per-joint oscillation amplitude (rad). Centred + amplitudes chosen
# to stay inside the joint limits at amp-scale 1.0; phases spread so the whole arm sweeps.
HOME = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.0], dtype=np.float64)
AMP = np.array([1.2, 0.8, 1.2, 0.8, 1.2, 1.0, 1.2], dtype=np.float64)
PHASE = np.array([i * 2.0 * math.pi / 7.0 for i in range(7)], dtype=np.float64)


def run_one(coup: str, args) -> dict:
    """Benchmark a single coupling mode; return timing stats."""
    import genesis as gs
    import torch

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    ipc_kwargs = dict(
        enable_rigid_rigid_contact=args.ground,
        enable_rigid_ground_contact=args.ground,
    )
    if coup == "ipc_monolithic":
        ipc_kwargs["ipc_monolithic_actuation"] = args.actuation
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt),
        coupler_options=gs.options.IPCCouplerOptions(**ipc_kwargs),
        show_viewer=False,
    )
    if args.ground:
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
        material=gs.materials.Rigid(coup_type=coup),
    )
    cam = None
    if args.video_out:
        cam = scene.add_camera(
            res=(1280, 720), pos=(2.2, -1.6, 1.3), lookat=(0.2, 0.0, 0.45), fov=45, GUI=False
        )
    scene.build()

    motors = slice(0, 7)
    franka.set_dofs_kp(np.array([args.kp] * 7 + [args.kp, args.kp], dtype=np.float32))
    franka.set_dofs_kv(np.array([args.kv] * 7, dtype=np.float32), motors)
    franka.set_dofs_kv(np.array([args.kv, args.kv], dtype=np.float32), slice(7, 9))

    n_dofs = franka.n_dofs
    omega = 2.0 * math.pi * args.freq
    amp = AMP * args.amp_scale

    def target(t):
        q = np.zeros(n_dofs, dtype=np.float32)
        q[:7] = HOME + amp * np.sin(omega * t + PHASE)
        return q

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # warmup (JIT compile + transient settle)
    for i in range(args.warmup):
        franka.control_dofs_position(target(i * args.dt))
        scene.step()
    sync()

    if cam is not None:
        cam.start_recording()

    per_step = np.empty(args.steps, dtype=np.float64)
    max_qd = 0.0
    t_all = time.perf_counter()
    for i in range(args.steps):
        franka.control_dofs_position(target((args.warmup + i) * args.dt))
        sync()
        t0 = time.perf_counter()
        scene.step()
        sync()
        per_step[i] = (time.perf_counter() - t0) * 1000.0
        if cam is not None:  # render is OUTSIDE the timed region (after the sync above)
            cam.render()
        if i % 10 == 0:
            qd = np.abs(franka.get_dofs_velocity().cpu().numpy().reshape(-1)[:7])
            max_qd = max(max_qd, float(qd.max()))
    total_wall = time.perf_counter() - t_all

    if cam is not None:
        fps = max(1, round(1.0 / args.dt))  # real-time playback
        cam.stop_recording(save_to_filename=args.video_out, fps=fps)
        print(f"[{coup}] saved video -> {args.video_out}  ({fps} fps, {args.steps} frames)", flush=True)

    med = float(np.median(per_step))
    out = dict(
        coup=coup,
        per_step=per_step,
        mean=float(np.mean(per_step)),
        median=med,
        p95=float(np.percentile(per_step, 95)),
        steps_per_s=1000.0 / med,
        realtime_factor=args.dt * 1000.0 / med,
        total_wall=total_wall,
        max_qd=max_qd,
    )
    print(
        f"\n[{coup}]  median={out['median']:.2f}ms/step  mean={out['mean']:.2f}  p95={out['p95']:.2f}  "
        f"-> {out['steps_per_s']:.0f} steps/s  ({out['realtime_factor']:.2f}x realtime)  "
        f"max|qd|={max_qd:.1f} rad/s  [{args.steps} steps, dt={args.dt}, ground={args.ground}]",
        flush=True,
    )
    gs.destroy() if hasattr(gs, "destroy") else None
    return out


def compare(a: dict, b: dict, args):
    """a = ipc_monolithic, b = external_articulation."""
    print("\n================= COUPLING PERFORMANCE (Franka, large fast motion) =================", flush=True)
    print(f"motion: freq={args.freq}Hz amp-scale={args.amp_scale} dt={args.dt} ground={args.ground}  "
          f"(max|qd| mono={a['max_qd']:.1f}, ext={b['max_qd']:.1f} rad/s)", flush=True)
    print(f"{'mode':>22} {'median':>9} {'mean':>8} {'p95':>8} {'steps/s':>9} {'xrealtime':>10}", flush=True)
    for d in (a, b):
        print(f"{d['coup']:>22} {d['median']:>8.2f}m {d['mean']:>7.2f}m {d['p95']:>7.2f}m "
              f"{d['steps_per_s']:>9.0f} {d['realtime_factor']:>10.2f}", flush=True)
    faster, slower = (a, b) if a["median"] < b["median"] else (b, a)
    speedup = slower["median"] / faster["median"]
    print(f"\n  --> {faster['coup']} is {speedup:.2f}x faster "
          f"({faster['median']:.2f} vs {slower['median']:.2f} ms/step)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coup", default=None, choices=["ipc_monolithic", "external_articulation"],
                   help="worker mode: benchmark one coupling type. Omit for the side-by-side driver.")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--freq", type=float, default=1.0, help="oscillation frequency (Hz)")
    p.add_argument("--amp-scale", type=float, default=1.0, help="scale on the per-joint amplitudes")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--kp", type=float, default=2000.0)
    p.add_argument("--kv", type=float, default=100.0)
    p.add_argument("--actuation", choices=["torque", "pd", "pd_eac", "pd_native"], default="pd_native",
                   help="(ipc_monolithic only) actuation channel. 'pd_native' (default) is the "
                        "branch-cut-robust incremental-driving constitution -- robust under fast motion "
                        "AND light-link stiff gains (force_range now clamps the converged torque, no "
                        "velocity cap). 'pd_eac' is its EAC-delivered twin. 'torque' is fast/heavy but "
                        "explicit-kp/kv diverges on light links; 'pd' DIVERGES under fast motion.")
    p.add_argument("--steps", type=int, default=300, help="timed steps")
    p.add_argument("--warmup", type=int, default=50, help="warmup steps (JIT + transient, untimed)")
    p.add_argument("--ground", action="store_true", help="add a contact plane (else free-space)")
    p.add_argument("--video", action="store_true",
                   help="(driver) also write coupling_perf_<mode>.mp4 per mode (slower; for eyeballing motion)")
    p.add_argument("--video-out", default=None, help="(worker) mp4 path to record this run to")
    p.add_argument("--out", default=None, help="(worker) npz path to dump per-step times")
    args = p.parse_args()

    if args.coup is not None:
        data = run_one(args.coup, args)
        if args.out:
            np.savez(args.out, per_step=data["per_step"],
                     **{k: v for k, v in data.items() if k not in ("per_step", "coup")})
        return

    common = [
        sys.executable, os.path.abspath(__file__),
        "--backend", args.backend, "--freq", str(args.freq), "--amp-scale", str(args.amp_scale),
        "--dt", str(args.dt), "--kp", str(args.kp), "--kv", str(args.kv),
        "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--actuation", args.actuation,
    ]
    if args.ground:
        common.append("--ground")

    def video_args(mode):
        return ["--video-out", os.path.abspath(f"coupling_perf_{mode}.mp4")] if args.video else []

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_m = os.path.join(td, "mono.npz")
        out_e = os.path.join(td, "ext.npz")
        subprocess.run(common + ["--coup", "ipc_monolithic", "--out", out_m] + video_args("ipc_monolithic"),
                       check=True)
        subprocess.run(common + ["--coup", "external_articulation", "--out", out_e]
                       + video_args("external_articulation"), check=True)
        a = {k: (np.load(out_m)[k] if k == "per_step" else float(np.load(out_m)[k])) for k in
             ("per_step", "mean", "median", "p95", "steps_per_s", "realtime_factor", "total_wall", "max_qd")}
        a["coup"] = "ipc_monolithic"
        b = {k: (np.load(out_e)[k] if k == "per_step" else float(np.load(out_e)[k])) for k in
             ("per_step", "mean", "median", "p95", "steps_per_s", "realtime_factor", "total_wall", "max_qd")}
        b["coup"] = "external_articulation"
    compare(a, b, args)


if __name__ == "__main__":
    main()
