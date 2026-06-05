"""M2 scaffold smoke for the ipc_authoritative coupling mode.

Builds a fixed-base 2-DOF revolute arm in a Genesis IPC scene with
``coup_type='ipc_authoritative'`` and gravity OFF. With zero joint torque the
arm must hold its loader pose. We then check that IPC's read-back link
transforms match the loader pose — validating: config/validation accepted the
mode, the robot was built inside IPC (ABD links + AffineBodyRevoluteJoint +
AffineBodyRevoluteJointExternalForce), advance() runs, and state reads back.

This does NOT exercise nonzero torque (M3) or qpos joint reconstruction (M4).

Run with the gs-gym-internal py3.10 venv (cuda backend + the fork):
  /home/zhaofeng/work/gs-gym-internal/.venv/bin/python \
      genesis/engine/couplers/ipc_coupler/docs/development/m2_holdpose_smoke.py
"""

import os

import numpy as np

import genesis as gs

URDF = os.path.join(os.path.dirname(__file__), "m2_arm.urdf")
N_STEPS = 30
POS_TOL = 2e-3  # m
QUAT_TOL = 1e-2


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(),
        show_viewer=False,
    )
    arm = scene.add_entity(
        gs.morphs.URDF(file=URDF, fixed=True, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_type="ipc_authoritative"),
    )
    scene.build(n_envs=1)

    coupler = scene._sim._coupler
    print(f"[build] coupler={type(coupler).__name__}  "
          f"ipc_authoritative entities={len(coupler._ipc_authoritative_data_by_entity)}  "
          f"abd links={len(coupler._abd_data_by_link)}")
    assert arm in coupler._ipc_authoritative_data_by_entity, "arm not registered as ipc_authoritative"

    # Structural: the two revolute joints were actually built into IPC (so a zero
    # drift below reflects a real held articulation, not frozen disconnected bodies).
    ad = coupler._ipc_authoritative_data_by_entity[arm]
    n_joints = len(ad.joint_slots[0])
    print(f"[build] ipc_authoritative joints={n_joints}  child_links={[l.name for l in ad.joints_child_link]}  "
          f"ext_force={coupler._ipc_rev_ext_force is not None}")
    assert n_joints == 2, f"expected 2 revolute joints built in IPC, got {n_joints}"
    assert coupler._ipc_rev_ext_force is not None, "AffineBodyRevoluteJointExternalForce not created"

    # Loader world transforms (Genesis side) per arm link, by global link idx.
    links_pos0 = arm.get_links_pos().cpu().numpy()   # (B, n_links, 3)
    links_quat0 = arm.get_links_quat().cpu().numpy()  # (B, n_links, 4)
    loader_pos = {}
    loader_quat = {}
    for li, link in enumerate(arm.links):
        loader_pos[link] = links_pos0[0, li].copy()
        loader_quat[link] = links_quat0[0, li].copy()

    # Confirm Genesis disabled its own contact/constraint solve (rule 2).
    rs = scene._sim.rigid_solver
    print(f"[build] rigid_solver enable_collision={rs._enable_collision}  disable_constraint={rs._disable_constraint}")
    assert rs._enable_collision is False and rs._disable_constraint is True

    for _ in range(N_STEPS):
        scene.step()

    # Compare IPC read-back transforms to loader pose for every ABD link of the arm.
    max_pos_drift = 0.0
    max_quat_drift = 0.0
    n_checked = 0
    for link, abd in coupler._abd_data_by_link.items():
        if link.entity is not arm:
            continue
        if link not in loader_pos:
            continue
        T = np.asarray(abd.ipc_transforms[0])
        pos = T[:3, 3]
        pos_drift = float(np.linalg.norm(pos - loader_pos[link]))
        # quaternion drift via rotation matrix Frobenius proxy
        R = T[:3, :3]
        # build loader R from quat (wxyz)
        w, x, y, z = loader_quat[link]
        R0 = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        quat_drift = float(np.linalg.norm(R - R0))
        max_pos_drift = max(max_pos_drift, pos_drift)
        max_quat_drift = max(max_quat_drift, quat_drift)
        n_checked += 1
        print(f"  link {link.name:6s}: pos_drift={pos_drift:.2e} m  rot_drift={quat_drift:.2e}")

    print(f"[after {N_STEPS} steps] checked {n_checked} ABD links  "
          f"max_pos_drift={max_pos_drift:.2e} m  max_rot_drift={max_quat_drift:.2e}")

    ok = n_checked >= 3 and max_pos_drift < POS_TOL and max_quat_drift < QUAT_TOL
    print("M2 HOLD-POSE PASS" if ok else "M2 HOLD-POSE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
