"""Verify the ipc_monolithic revolute joint limit (AffineBodyRevoluteJointLimit).

A 1-DOF fixed-base pendulum (URDF limit ±0.4 rad) is released under gravity from a
small bias and driven into the limit. We measure how far the joint angle gets past
the limit for: limit OFF, and limit ON at increasing strengths.

Expectation:
  - OFF        -> joint blows past the limit (falls toward ~pi/2 or beyond)
  - ON, soft   -> overshoots the limit, but bounded
  - ON, stiff  -> stays close to the limit (small overshoot)

Run one config:
  .../python m5_joint_limit_experiment.py --limit off
  .../python m5_joint_limit_experiment.py --limit on --strength 10
  .../python m5_joint_limit_experiment.py --limit on --strength 1000
"""

import argparse
import os

import numpy as np

import genesis as gs

URDF = os.path.join(os.path.dirname(__file__), "m5_pendulum.urdf")
LIMIT = 0.4          # rad, matches the URDF <limit>
START = 0.15         # rad, small bias so gravity drives it toward the limit
N_STEPS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", choices=["on", "off"], default="on")
    ap.add_argument("--strength", type=float, default=100.0)
    args = ap.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.IPCCouplerOptions(
            monolithic_joint_limit_enable=(args.limit == "on"),
            monolithic_joint_limit_strength=args.strength,
        ),
        show_viewer=False,
    )
    arm = scene.add_entity(
        gs.morphs.URDF(file=URDF, fixed=True, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_type="ipc_monolithic"),
    )
    scene.build(n_envs=1)
    arm.set_dofs_position(np.array([START], dtype=np.float32))

    angles = []
    for _ in range(N_STEPS):
        scene.step()
        angles.append(float(arm.get_dofs_position().cpu().numpy().reshape(-1)[0]))
    angles = np.asarray(angles)

    max_abs = float(np.max(np.abs(angles)))
    final = float(angles[-1])
    overshoot = max(0.0, max_abs - LIMIT)
    tag = "OFF" if args.limit == "off" else f"ON(strength={args.strength:g})"
    print(
        f"[limit {tag:>18s}]  limit=±{LIMIT}  max|angle|={max_abs:.4f}  "
        f"final={final:+.4f}  overshoot_past_limit={overshoot:.4f} rad"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
