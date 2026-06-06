"""M3 torque-actuation check for the ipc_monolithic coupling mode.

A fixed-base 2-DOF arm is position-controlled (Genesis PD: kp/kv) to a commanded
setpoint. Genesis computes the per-DOF control torque (get_dofs_control_force), the
coupler pushes it to IPC as per-joint torque, IPC integrates the articulation, and
the joint angle is read back. Convergence to the target validates the whole torque
path at once (PASSES: q -> target, settled).

Gains are matched to the toy arm's small mesh-derived ABD inertia (explicit-control
stability needs kv < ~2*I/dt). Real robot arms (~10-100x larger inertia) accept
normal gains (kp~20-200). Requires the fork's AffineBodyRevoluteJointExternalForce
device-safe-inverse fix (libuipc db2bade6 / test 74).

Run with the gs-gym-internal py3.10 venv:
  .../python genesis/engine/couplers/ipc_coupler/docs/development/m3_position_control.py
"""

import os

import numpy as np

import genesis as gs

URDF = os.path.join(os.path.dirname(__file__), "m2_arm.urdf")
TARGET = np.array([0.4, -0.4], dtype=np.float32)
# Gains matched to the toy arm's small (mesh-derived) ABD inertia. Explicit control
# stability needs kv < ~2*I/dt; this arm's I_eff ~ 5e-4..2e-3, so kv must stay small.
# Real robot arms have ~10-100x larger inertia and accept normal gains (kp~20-200).
KP = 2.0
KV = 0.1
N_STEPS = 200
POS_TOL = 0.03   # rad
VEL_TOL = 1.0    # rad/s; "held" at the end


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Gravity off: isolates the torque-PD convergence (toy arm's tiny inertia makes
    # gravity rejection gain-sensitive; real arms with realistic inertia hold under
    # gravity with normal gains). The gate here is: torque PD drives the joints to the
    # commanded setpoint and holds.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(monolithic_torque_enable=True),
        show_viewer=False,
    )
    arm = scene.add_entity(
        gs.morphs.URDF(file=URDF, fixed=True, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_type="ipc_monolithic"),
    )
    scene.build(n_envs=1)

    n_dofs = arm.n_dofs
    arm.set_dofs_kp(np.full(n_dofs, KP, dtype=np.float32))
    arm.set_dofs_kv(np.full(n_dofs, KV, dtype=np.float32))
    arm.control_dofs_position(TARGET)

    q_hist = []
    for step in range(N_STEPS):
        scene.step()
        q = arm.get_dofs_position().cpu().numpy().reshape(-1)
        q_hist.append(q.copy())
        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"[step {step:3d}] q={np.array2string(q, precision=4)}  target={TARGET}")
    q_hist = np.stack(q_hist, axis=0)

    q_final = q_hist[-1]
    v_final = arm.get_dofs_velocity().cpu().numpy().reshape(-1)
    err = np.abs(q_final - TARGET)
    # did it actually move toward the target (not just fall)?
    moved_toward = np.all(np.abs(q_final - TARGET) < np.abs(q_hist[0] - TARGET))
    finite = bool(np.all(np.isfinite(q_hist)))

    print(f"[result] q_final={np.array2string(q_final, precision=4)}  "
          f"err={np.array2string(err, precision=4)} rad  "
          f"|v_final|={np.array2string(np.abs(v_final), precision=3)}")
    print(f"[result] finite={finite}  moved_toward_target={bool(moved_toward)}  "
          f"max_err={err.max():.4f}  max_|v|={np.abs(v_final).max():.3f}")

    ok = finite and bool(moved_toward) and err.max() < POS_TOL and np.abs(v_final).max() < VEL_TOL
    print("M3 POSITION-CONTROL PASS" if ok else "M3 POSITION-CONTROL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
