# Goal: reliable shell contact and damage observations

Improve the existing Genesis shell implementation for high-throughput damage-avoidance reinforcement learning (RL).
Start from Kashu7100/Genesis PR #22 at `fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`, on the
`shell_contact_consistency` branch of `Roushelfy/Genesis`.

Read `CODING_GUIDELINES.md`, `CLAUDE.md`, and [the audit and acceptance guide](SHELL_CONTACT_REVIEW.md).
Choose the implementation and numerical method based on measurements. The goal is working, validated engine code.

## Intended result

- Physical contact covers the deformed triangle surface, including face interiors and edges. Physical collision and
  tactile surface queries use consistent geometry and thickness conventions. Contact supports normal reactions,
  friction, and force/torque transfer to movable articulated rigid links.
- Elasticity and contact produce a consistent response under pressing, holding, sliding, and release. Investigate the
  existing velocity projection, missing penetration correction, and delayed rigid reaction together with the geometry.
- Repeated runs, full and partial resets, state restore, and independent batched environments give consistent physical
  observables. Identify the cause of variability through controlled reproductions and fix it.
- Preconditioned conjugate gradient (PCG) accuracy is tolerance-driven, with a maximum iteration safety limit. Verify the
  true residual and distinguish convergence, iteration-limit failure, breakdown, and non-finite values. Converged work
  stops efficiently on the GPU without a CPU synchronization every iteration.
- Damage evaluation has documented units and a named mechanical criterion. A lightweight damage-only mode evaluates
  failure before topology changes and supports episode termination, with fixed connectivity and no split-vertex reserve.

## Working priorities

Keep the implementation inside Genesis's existing shell solver, rigid coupling, and state APIs. Use Quadrants for the
simulation hot path and keep batched state on the device. Use low-resolution scenes first, with CPU diagnostics and GPU
profiling where available. Fix observable contact and consistency defects, then optimize throughput at matched accuracy.
Large spatial mesh-convergence studies are deferred. Cheap same-mesh time-step and tolerance checks are part of this goal.

Use a Franka gripper grasping a low-resolution egg-shaped hollow shell as the end-to-end scene. Only rest shape and
material are fixed. Contact positions, orientations, number of contacts, symmetry, and stick/slip state can vary.
Smooth motion within a grasp can support history reuse, with validation and invalidation when contact changes or resets.

## Completion

Deliver the engine changes, focused regression coverage, a runnable headless Franka/egg example and benchmark, and a
report containing exact reproduction commands, tolerances, correctness results, detailed timings, and memory usage.
Show before/after contact and reset behavior, matched-accuracy throughput, and any remaining supported-domain limits.
Check relevant existing shell and rigid-coupling tests. Commit and push the completed work to this branch.
