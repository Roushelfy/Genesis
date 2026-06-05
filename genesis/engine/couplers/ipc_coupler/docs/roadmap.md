# IPC-Monolithic Coupling Mode — Roadmap

Status surface for the new `ipc_monolithic` coupling option in the Genesis
IPC coupler. Keep this page short and current; historical detail lives in
[development/ipc_monolithic_journal.md](development/ipc_monolithic_journal.md).
Stable design lives in [architecture.md](architecture.md); enforceable rules in
[conventions.md](conventions.md).

## North-star principle

> In `ipc_monolithic` mode, **libuipc is the single source of truth for robot
> state**. Genesis computes only the per-DOF control torque (reusing its existing
> PD/`ctrl_mode` evaluation), pushes scalar joint torques into IPC, advances IPC,
> and reads body transforms back to populate `qpos`/`links_state` for getters and
> rendering. **Genesis runs no rigid dynamics, no integration, and no
> contact/constraint solve.**

Primary use case (decided): *render an IPC-driven robot that coexists with FEM /
cloth / RCC in one libuipc scene* — not RL-observation fidelity. Correct
rendering + correct robot↔deformable contact win over joint-velocity fidelity.

## Decisions locked for v1

| Axis | Decision |
|---|---|
| Base | **Fixed base only** (free base deferred) |
| Joints | **revolute + fixed** only (fixed folded via existing merge) |
| Actuation | **Torque path** — `AffineBodyRevoluteJointExternalForce`, per-joint scalar torque |
| PD law | **Genesis-side** via `get_dofs_control_force` (folds all 4 ctrl modes + `force_range`) |
| Control modes | position / velocity / force / position_velocity → **all folded to torque** |
| Joint readback | **signed-angle-from-transforms** (sim2sim-style), velocity = finite difference |
| Zero offset | **per-joint reference offset calibrated at build** (maps Genesis qpos zero ↔ IPC `init_angle`) |
| Genesis dynamics | **off**: skip `kernel_predict_integrate`; force `enable_collision=False`, `disable_constraint=True` |
| Integrator | **lock BDF1** (clean teleport; see journal) |
| Batched envs | **single env** (B=1) for v1 |
| Coexistence | robot + FEM/cloth/RCC in one scene (no extra work expected) |
| Engine | local fork `Roushelfy/libuipc` branch **`RCC-Adhesion`** (follow head), cuda backend only |
| Target env | match **gs-gym-internal**: Python 3.10, CUDA 12.x (cu129 torch / nvcc 12.8), `CUDA_ARCHITECTURES=native` |

## Current phase

**M0 — Feasibility & source verification: COMPLETE.** See journal §M0.
**M1 — Environment: COMPLETE (2026-06-05).** No build needed — gs-gym-internal's
py3.10 venv already has the cuda-backend fork (uipc 0.9.0, SHA `07f3be94`, RTX
4090 / CUDA 12.8). Smoke ([development/m1_uipc_smoke.py](development/m1_uipc_smoke.py))
PASSED: torque-driven revolute joint + state-accessor readback on `cuda`/bdf1.
**M2 — Scaffold: COMPLETE (2026-06-05).** `coup_type='ipc_monolithic'` wired end
to end (enum + option + material + config/validation + ABD/joint/ext-force build).
Hold-pose smoke ([development/m2_holdpose_smoke.py](development/m2_holdpose_smoke.py))
PASSED: fixed-base 2-DOF arm built in IPC with 2 revolute joints, Genesis
collision/constraint disabled, IPC read-back transforms match loader pose. See
journal §M2.
**M3 — Torque actuation: NEXT.** Per-step push `get_dofs_control_force` → per-joint
scalar torque via `AffineBodyRevoluteJointExternalForce`; add predict-skip in
`substep_pre_coupling`; gate on tracking vs a forward-dynamics oracle under gravity.

## Phase plan & acceptance gates

Gates are runnable commands. Until M1 lands, every gate below is a **planned
gate** (see "Planned gates"); no gate is runnable today.

| Phase | Goal | Acceptance gate (planned) |
|---|---|---|
| **M1 Env** | Build `cuda` backend of `RCC-Adhesion` into a py3.10 + cu12 venv; `import uipc` works; one-body `World.advance()` smoke | `python -c "import uipc; ..."` advances a 1-body scene without error |
| **M2 Scaffold** | New `COUPLING_TYPE.IPC_MONOLITHIC`, config + validation, build robot in IPC (ABD links + `AffineBodyRevoluteJoint` + external-force constitution), **zero torque** | Fixed-base arm builds, holds pose under one `couple()`, transforms read back match the loader pose within 1e-4 |
| **M3 Actuation** | Genesis-side torque (`get_dofs_control_force`) → per-joint scalar torque → IPC; all 4 ctrl modes | A known constant-torque / step-position trajectory tracks an independent forward-dynamics oracle within tolerance (see test ladder) |
| **M4 Readback+render** | signed-angle joint readback + finite-diff vel → `qpos`/`links_state`; getters + viewer/camera render IPC-driven robot | `get_dofs_position` round-trips IPC state; headless render smoke produces a frame of the moving arm |
| **M5 Coexistence** | Robot + cloth/RCC in one scene; no double-solve; RCC adhesion intact | A gs-gym-internal deformable scene (arm + cloth) runs N steps and renders with the robot in `ipc_monolithic` |

## Next tasks (M1)

1. Reproduce the `cuda_mixed_fused_pcg` cmake config but with
   `-DUIPC_WITH_CUDA_MIXED_BACKEND=OFF -DUIPC_WITH_CUDA_BACKEND=ON`, pointed at a
   **py3.10** interpreter (gs-gym-internal's venv), nvcc 12.8, `native` arch.
2. Editable-install the resulting `pyuipc` into that venv (matches
   gs-gym-internal's `third_party/libuipc` editable wiring).
3. Smoke: `import uipc`; build a 2-body + 1 revolute-joint scene; `world.advance()`.
4. Record exact commands + machine (GPU, driver, CUDA) in the journal.

## Blockers

- **B1 (RESOLVED):** working `uipc` cuda backend confirmed in gs-gym-internal's
  py3.10 venv; M1 smoke passed.
- **B3 (RESOLVED):** consumer's genesis editable finder redirected to this repo
  (`...Genesis` instead of `third_party/Genesis`); verified import from neutral
  cwd. Caveat: `uv sync` reverts it — see journal §M1 B3 for the durable fix.
- **B2 (watch):** branch-head policy (`follow head` of `RCC-Adhesion`) means the
  fork API can drift under us; known-good SHA today is `07f3be94`. Pin per
  milestone in the journal.

## Out of scope (v1, tracked for later)

Free/floating base · prismatic & spherical joints · batched envs (B>1) ·
per-env gains · `get_dofs_force` / `get_links_acc` readback · partial (per-env)
reset · BDF2 · differentiable coupling.
