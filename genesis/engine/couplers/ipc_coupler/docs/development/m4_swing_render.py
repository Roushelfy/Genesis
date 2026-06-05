"""M4 readback + render check for the ipc_monolithic coupling mode.

A fixed-base 2-DOF revolute arm is built in IPC (ipc_monolithic), started in a
bent pose, and released under GRAVITY with zero joint torque. IPC integrates the
articulation; Genesis reconstructs joint angles from the IPC body transforms
(signed-angle readback), writes them to qpos, runs FK, and renders via a camera.

This validates the core value proposition (A2): IPC drives the robot, Genesis
renders it. Motion comes from gravity alone (torque actuation is M3).

Outputs an MP4 and prints joint-angle history (must change over time = real
IPC-driven motion that the readback captured).

Run with the gs-gym-internal py3.10 venv:
  /home/zhaofeng/work/gs-gym-internal/.venv/bin/python \
      genesis/engine/couplers/ipc_coupler/docs/development/m4_swing_render.py
"""

import os

import imageio.v2 as imageio
import numpy as np

import genesis as gs

URDF = os.path.join(os.path.dirname(__file__), "m2_arm.urdf")
OUT = "/tmp/ipc_monolithic_swing.mp4"
N_STEPS = 150
FPS = 30
INIT_QPOS = np.array([1.4, -0.6], dtype=np.float32)  # bent start so gravity clearly swings it


def render_frame(camera):
    rendered = camera.render(rgb=True, force_render=True)
    frame = rendered[0] if isinstance(rendered, tuple) else rendered
    if frame is None:
        raise RuntimeError("camera returned no RGB frame")
    return np.asarray(frame)[..., :3]


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.IPCCouplerOptions(),
        show_viewer=False,
    )
    arm = scene.add_entity(
        gs.morphs.URDF(file=URDF, fixed=True, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_type="ipc_monolithic"),
    )
    camera = scene.add_camera(res=(640, 480), pos=(1.4, -1.4, 0.9), lookat=(0.0, 0.0, 0.45), fov=45, GUI=False)
    scene.build(n_envs=1)

    # Start in a bent pose (teleports into IPC via the mark_abd_updated path).
    arm.set_dofs_position(INIT_QPOS)

    frames = []
    angle_hist = []
    for step in range(N_STEPS):
        scene.step()
        frames.append(render_frame(camera))
        q = arm.get_dofs_position().cpu().numpy().reshape(-1)
        angle_hist.append(q.copy())
        if step % 25 == 0 or step == N_STEPS - 1:
            print(f"[step {step:3d}] joint angles (rad) = {np.array2string(q, precision=4)}")

    imageio.mimsave(OUT, frames, fps=FPS)
    angle_hist = np.stack(angle_hist, axis=0)  # (N, n_dofs)

    finite = bool(np.all(np.isfinite(angle_hist)))
    total_motion = float(np.sum(np.abs(np.diff(angle_hist, axis=0))))
    swing_range = float(np.max(angle_hist) - np.min(angle_hist))
    bounded = swing_range < 20.0  # sane bound (no blowup)
    print(f"[result] video={OUT}  frames={len(frames)}  finite={finite}  "
          f"total_motion={total_motion:.3f} rad  swing_range={swing_range:.3f} rad")

    ok = finite and bounded and total_motion > 0.2  # the arm actually moved
    print("M4 SWING-RENDER PASS" if ok else "M4 SWING-RENDER FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
