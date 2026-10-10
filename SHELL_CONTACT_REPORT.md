# Shell contact and consistency: report

This report answers [the goal](SHELL_CONTACT_GOAL.md) and [the audit and acceptance guide](SHELL_CONTACT_REVIEW.md)
for the `shell_contact_consistency` branch, starting from Kashu7100/Genesis PR #22 at `fd051fe3` (the "source engine"
below). Every number below was measured, on the machine and with the commands given in [Reproduction](#reproduction).

## Setup

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120), driver 595.71, Quadrants 1.3.3, Python 3.10.
  CPU runs: Intel Xeon Platinum 8562Y+.
- Precision: single (`precision="32"`) unless stated. Control step 10 ms, 10 substeps (1 ms) unless stated.
- End-to-end scene: Franka Panda (actual MJCF collision geoms, fingertip pads included) grasping a hollow egg shell of
  162 vertices and 320 triangles (icosphere subdivision 2, 60 x 45 mm, tapered), E = 10 GPa, nu = 0.3, 0.4 mm thick,
  70 g, tensile strength 10 MPa, damage-only mode. Every environment grasps off-center at its own yaw and grip force,
  lifts, carries, loosens its grip (half of them, which lets the egg slip out) and releases.

## Summary

| Area | Source engine (fd051fe3) | This branch |
| --- | --- | --- |
| Ball resting inside a triangle, away from its vertices | falls through, 0 N reaction | rests at the exact height (10 um), reaction = weight within 0.2% |
| Torque through a hinge (lever on a sheet) | not supported (one-way velocity projection) | reaction = m g x_com / L within 0.2% |
| Coulomb friction on an incline | velocity projection only | stick below 1e-4 m/s; slip acceleration within 0.5%, across mesh edges |
| Momentum through a 3 m/s impact | not conserved (lagged reaction, no penetration correction) | conserved to 1e-9 kg m/s, no tunneling |
| Full reset vs fresh run (grasp, CPU) | 63 mm apart | bitwise identical |
| One step repeated from the same state (CPU) | 0.45 mm apart | bitwise identical |
| Partially reset environment vs fresh run (CPU) | 1.9 mm apart | bitwise identical |
| Checkpoint restore vs continuation (grasp, CPU) | a scene holding a sheet cannot be checkpointed | bitwise identical, the rigid warm start included |
| Reduced-state snapshot (`get_state`) vs continuation (CPU) | 38 mm apart | shell state complete, the rigid constraint solve restarting cold by design: 2.7 mm |
| PCG stopping | fixed iteration count, failures unreported | tolerance-driven device loop, true residual checked, statuses reported |
| Damage | fracture criterion only, topology always mutated | Rankine damage index, damage-only mode, peak and first failure face |

## Root causes and fixes

### Contact geometry

The source engine sampled every free vertex against every coupled rigid geom (see the review). A face interior or
edge pressed by a geom smaller than a triangle produced no candidate, hence the 0 N reaction of a ball inside a
triangle. A more accurate solve could not repair it.

Every face now looks for its point deepest into each coupled geom on either side of its mid-surface: exact closest
points for spheres, a soft minimum of the vertices for planes, and a projected gradient descent from that soft minimum
on the signed distance of boxes, capsules (both analytic) and meshes (signed distance field). The contact surface sits
half the shell thickness off the mid-surface, the same convention as the elastic model. A bounding sphere per geom
prunes the pairs. Contacts are speculative within twice the distance the face and the geom can close over a substep,
so that a fast approach is stopped by the implicit solve before the surfaces cross (no continuous collision detection
needed). One contact per face and side is a one-point quadrature of the contact patch over that face. A point shared by
adjacent faces is counted once per face: this stiffens the contact there without changing the force it transmits, which
equilibrium sets (verified by the reaction tests below).

A speculative contact whose closest point lies on an edge is kept only if the push of the geom stays within the normal
cone of the surface there: it may lean into the face by the dihedral angle of a convex edge at most. Otherwise the
geom lies over the neighbor face, which keeps the contact. Without this rule, a box sliding across a row of vertices of
a flat sheet met a phantom corner: the face ahead linearized its gap about the shared edge and predicted a collision
the flat surface does not have, which cost the box 13% of its speed as its edge crossed the row (and stopped it
altogether combined with the friction defect below). Penetrating contacts keep every face, as a plate pressed into the
convex egg must push every vertex inside it: restricting them too raised the error of the press below at 10 substeps
from 0.25% to 0.4%.

The physical contact force on a sheet is read with `ShellEntity.get_verts_contact_force` (the contact forces spread on
the vertices by the barycentric coordinates of their points). The tactile sensors of Genesis measure rigid meshes, and
share the closest-point routine of the shell contacts (`raycast_qd.closest_point_on_triangle` delegates to
`geom.qd_closest_point_barycentric`).

### Contact response and two-way coupling

The source engine projected velocities with the lumped vertex mass, had no penetration correction, and applied the
rigid reaction in the next substep. Contacts now join the implicit step of the shell:

- Normal: a penalty on the penetration at the end of the substep, of stiffness `contact_stiffness` (25) times the
  effective mass of the contact point over dt^2 (shell diagonal block plus rigid inverse weight).
- Friction: smoothed Coulomb (stick below a slip velocity of max(1e-4 m/s, 2 mu lambda / h)), whose bound mu lambda
  is the normal impulse of the same contact slot over the previous substep. The potential is then convex within a
  substep, and a contact starts to rub one substep after it starts to push. Taking the bound from an iterate of the
  solve instead overestimates friction: freezing it at an early iterate gave a box with mu = 0.2 friction up to 6x too
  strong, and giving a contact that hopped off for one substep a bound from its first pushing iterate gave friction
  with no normal force at all, which stopped a box sliding down an incline.
- Rigid side: the degrees of freedom of every kinematic tree the contacts touch join the same solve through their
  Schur complement (mass matrix plus contact stiffness, inverted per tree), so that an articulated link responds with
  its effective mass, and the rigid integration of the substep is deferred until the shell contact solve has added the
  rigid velocity change. Both sides receive the opposite impulse at the same point in the same substep.
- Solve: Newton iterations on the incremental potential, each a PCG solve of the linearized system and a line search
  to the minimum along the step (slope bracketing, then safeguarded Newton on the slope). Backtracking stalled
  ("Zeno") on contacts the step starts to push and on stick/slip transitions narrower than its resolution.
- Overlap policy: the penalty corrects an already-overlapping state, removing most of the penetration over the next
  substep, which separates the surfaces at about the penetration over the substep. A scene should start without
  overlap, as the examples place the egg 1 mm above the ground.

### Linear solver (PCG)

- The iterations loop on the device (`qd.graph.do_while`, CUDA graph conditional nodes), so that converged
  environments stop costing work and the host never synchronizes per iteration. An iteration is 5 fused kernels (7 with
  rigid contacts), since a graph node costs about 8.5 us here.
- Stopping: the residual in the block-Jacobi norm below `pcg_tolerance` times the right-hand side, or the rigorous
  bound r^T M^-1 r on the mass-weighted velocity error below `pcg_velocity_tolerance`. The coarse correction is left
  out of the stopping norm: it weights rigid motions about 1e4 above deformation, which made the damage error 2.5x
  larger with the coarse space than without. A z^T M z estimate (z = P r) stopped a falling egg at rest, block-Jacobi
  underestimating smooth errors by 1e4.
- The true residual b - A dv is recomputed before stopping (covering the warm start too). Statuses: CONVERGED,
  MAX_ITERATIONS, BREAKDOWN, NON_FINITE (errno then halts the simulation) and STAGNATION (the true residual stalled at
  the floating-point floor, about 1e-5 relative in single precision for this egg, which is not a failure).
  `ShellSolver.get_envs_solver_failure` returns the sticky union of failure flags per environment since its reset.
  Solver failure never feeds the damage observables.
- The iteration limit applies per linear solve, and the Newton loop has its own (32).

### History, reset and restore

- The warm start `verts_dv` was neither saved nor cleared by `get_state` / `set_state`. It is now part of the state.
- The coarse preconditioner age was global. It is per environment, invalidated by `set_state`, and rebuilt every
  substep by default (`coarse_update_interval=1`, a negligible cost), so that a restored state replays exactly.
- The lagged friction bound and its geom per contact slot are part of the state.
- A scene holding a sheet could not be checkpointed: the shell entity had no description and the shell solver did
  not list its arrays. A shell entity now keeps its description (its resolved rest mesh with its morph, material and
  surface, `ShellEntityDescription`), and the shell solver lists its arrays by kind (`ShellSolver.data`), so that
  `Scene.__getstate__` / `Scene.__setstate__`, pickling, `save_checkpoint`, `export` and `load` cover sheets. A
  checkpoint holds every array a step reads from one step to the next, the rigid constraint warm start included, and
  the grasp restored from one replays its continuation bit for bit on CPU.
- `get_state` / `set_state` restore the reduced state. The shell part of it is complete (a bisection over every array
  of both solvers found no shell array missing), and the rigid solver restarts its constraint solve cold from it by
  design (a restored reduced state is a discontinuity, its warm start and inertia rounding left behind). With an
  active joint limit, a restored Franka alone differs from its continuation by 7e-9 m/s after one step, and the grasp
  by 0.26 um on the egg after one step. Restoring the same snapshot twice gives the same run.

### Damage

The damage index of a face is its largest principal stress over both outer surfaces, membrane plus or minus bending
(sigma_m +- 6 M / h^2, M from the hinge curvature), over `tensile_strength` (Rankine criterion, dimensionless, 1 at
failure, times the tensile strength for the stress in Pa). `get_peak_damage` and `get_failure_face` return its
running maximum and the first face reaching 1 since the last reset. `Shell(fracture=False)` evaluates it with fixed
connectivity and no split-vertex reserve: an episode ends on `get_peak_damage() >= 1`.

## Validation

### Regression tests (`tests/deformable/test_shell.py`, all 12 pass on CPU and GPU)

| Test | What it asserts |
| --- | --- |
| `test_rigid_contact_forces[0,2]` | ball inside a pinned triangle: height within 10 um, reaction = weight within 0.2%; hinged lever: reaction = m g x_com / L within 0.2% at rest; mu = 0.6 box sticks on a 20 degree incline (< 1e-4 m/s), mu = 0.2 box slides at g (sin - mu cos) within 0.5% while crossing a row of vertices |
| `test_rigid_contact_conserves_momentum[0,2]` | 3 m/s ball hits a free sheet off center: momentum conserved to 1e-9, ball stays above the sheet, no vertex inside it |
| `test_state_restore_reproduces_contact[0,2]` | full reset, snapshot restore, checkpoint restore, a pickled copy of the scene and partial reset replay the trajectory within 1e-9 (fails if the warm start is not restored, or if a scene holding a sheet cannot be checkpointed) |
| `test_tearing_at_tensile_strength` | damage-only strip: damage index = E G / tensile strength within 2%, equal to the fracturing strip before failure (1e-9), mesh unchanged |

Before the fixes of the last two defects above, the incline box accelerated 21% slower than Coulomb friction allows over
the window crossing the vertex row (phantom corner), or stopped there (friction with no normal force), both failing the
0.5% assertion. It now accelerates within 0.2% of it. The tactile and raycaster sensor tests
(`tests/sensors/test_tactile.py`, `tests/sensors/test_raycaster.py`, 37 tests) pass on CPU, as do all deformable and
coupling tests (`tests/deformable`, `tests/coupling`: 49 passed, 1 expected failure), the serialization tests the
shell description shares its machinery with (`tests/rigid/test_serialization.py`, 14 tests) and the example runs of the
four shell examples (`tests/test_examples.py -m examples`).

### Consistency (`examples/coupling/shell_egg_consistency.py`, 4 environments, 160 steps)

Maximum deviation of the egg vertex positions:

| Comparison | Source, CPU | This branch, CPU | This branch, GPU (separate runs) |
| --- | --- | --- | --- |
| Full reset vs fresh run | 63 mm | 0 | 0.59 mm, 4.3 mm, 1.9 mm |
| Checkpoint restore vs continuation | not supported | 0 | 1.6 mm |
| Reduced-state snapshot restore vs continuation | 38 mm | 2.7 mm (rigid restarts cold, see above) | 58 um, 0.77 mm, 2.1 mm |
| One step repeated from the same state | 0.45 mm | 0 | 0.06 um in every run |
| Partially reset environment vs fresh run | 1.9 mm | 0 | 50 um, 54 um, 155 um |
| Other environments vs continuation | 59 mm | 5.8 um (rigid restarts cold) | 0.87 um, 3.5 um, 4.8 um |

The CPU backend is bitwise deterministic. On the GPU, float atomics reorder reductions: one step from the same state
differs by 0.06 um (the solver tolerance level). Over the whole grasp the difference grows to millimeters, and it
varies from run to run. A deterministic CPU run whose egg starts 1 nm higher diverges the same way (above 1 um by step
6, above 100 um by step 99, 1.3 mm at the end): the grasp scene itself amplifies perturbations (the egg rocks on the
ground before the grasp, then contact timing decides the impact peak). This is a property of the scenario rather than
of the solver. The remaining lever is a bitwise deterministic GPU reduction order, which the review rules out of scope.

### Tolerance and time step sweeps (same mesh)

Quasi-static press between two plates (`examples/coupling/shell_egg_press.py`), against a double precision,
40-substep, 1e-9 tolerance reference:

| Setting | Force at 100 um | Damage at 100 um | Compression at first failure | Error vs reference |
| --- | --- | --- | --- | --- |
| reference (fp64, 40 substeps) | 10.910 N | 0.589 | 169.40 um | - |
| fp32, 5 substeps | 10.824 N | 0.583 | 170.96 um | 0.7-0.9% |
| fp32, 10 substeps | 10.885 N | 0.587 | 169.88 um | 0.2-0.3% |
| fp32, 20 substeps | 10.902 N | 0.588 | 169.55 um | 0.1% |
| fp32, 10 substeps, tolerance 1e-3 / 1e-4 / 1e-5 | 10.885 N | 0.587 | 169.86-169.88 um | same as above |

The time step error is first order, as backward Euler is, and the solve tolerance does not move the quasi-static
observables. The failing face differs between runs among faces of equal damage index (the egg is symmetric).

Impacts (`examples/coupling/shell_egg_drop.py`: 16 eggs dropped at 0.63 to 1.25 m/s onto the ground at random
orientations, peak damage index against a double precision 1e-9 reference): tolerance 1e-3: median error 15%, max 83%.
Tolerance 1e-4 (the default): median 0.07%, max 0.7%. Tolerance 1e-5: median 0.03%, max 0.13%. No environment failed
at any tolerance. With a fixed iteration count and the old stopping test, some impacts produced 1e5 N forces.

### End-to-end grasp (`examples/coupling/franka_egg_grasp.py -g -b 8`)

All 8 eggs are lifted 80 mm and carried. Of the 4 environments whose grip loosens to 2% while carrying, 2 let their egg
slip out before the release, and every egg falls on release. Before the release, the peak damage index exceeds 1 from
a grip of 29 N on (0.78 at 5 N, 2.1 at 52 N), the drop on the ground breaking every egg. In the benchmark scene at 256
environments over 180 steps, one substep of one environment reached the Newton iteration limit (reported as a solver
failure), and 13.6% of the substeps of an environment had a solve stop at the floating-point floor (STAGNATION, not a
failure).

## Performance

All timings are synchronized wall times on the RTX PRO 6000 with no other job on the GPU, compilation, rendering and
the grasp approach excluded: 60 timed control steps (600 substeps) while the fingers close and lift the egg. One
control step is one policy transition, so environment steps per second are policy transitions per second.

### Throughput and memory (`examples/speed_benchmark/franka_egg.py`)

| Environments | This branch, shell egg | Rigid egg baseline | Source engine, 50 PCG iterations | Source engine, 200 PCG iterations |
| --- | --- | --- | --- | --- |
| 64 | 114 env-steps/s (1.79 scene steps/s), 1650 MiB | 3482, 1646 MiB | 1658, 1648 MiB | 444 |
| 256 | 338 (1.32), 1820 MiB | 12118, 1774 MiB | 6609, 1782 MiB | 1768 |
| 1024 | 759 (0.74), 2490 MiB | 43611, 2318 MiB | 22045, 2308 MiB | 7183 |
| 4096 | 1108 (0.27), 5174 MiB | 141469, 4502 MiB | 31229, 4482 MiB | 17768 |

Memory is that of the process (the CUDA context and allocator pools included): the shell egg adds 672 MiB to the rigid
scene at 4096 environments, 168 KiB per environment. The source engine is faster because it runs a fixed number of
iterations with no contact solve, at an accuracy this branch cannot be matched against: its ball falls through a
triangle, its friction is a velocity projection, its reaction lags a substep, and its fixed iteration count produced
1e5 N impact forces. The comparison at matched accuracy is between the settings of this branch below. Throughput keeps
growing up to 4096 environments, by 1.5x from 1024, as the solve becomes bound by the memory traffic of its products.

At 4096 environments, 20 environments (0.5%) report a solver failure over the 60 steps. In the profiled window below,
every failure is the Newton iteration limit, no PCG solve reaching its own.

### Speed and accuracy settings (1024 environments)

| Setting | env-steps/s | Speedup | Press force error | Impact damage error (median / max) |
| --- | --- | --- | --- | --- |
| default: 10 substeps, `pcg_tolerance=1e-4`, `contact_stiffness=25` | 759 | 1.0x | 0.2% | 0.07% / 0.7% |
| `contact_stiffness=5` | 1590 | 2.1x | 2.3% | 0.3% / 0.6% |
| `pcg_tolerance=1e-3` | 918 | 1.2x | 0.2% | 15% / 83% |
| 5 substeps | 1034 | 1.4x | 0.8% | 47% / 54% |
| all three | 2644 | 3.5x | 5.1% | 52% / 91% |

The press force error is that of the force at 100 um of compression against the double precision reference, and the
impact damage error that of the peak damage index of the 16 drops of `shell_egg_drop.py` against its double precision
reference. A softer contact is the cheap lever: it lets 5x more penetration in, which takes a few percent of an imposed
displacement from the deformation of the egg, and leaves impacts accurate. A 2 ms substep under-resolves the impacts
(about half the peak damage), and a looser tolerance makes impact damage unreliable.

### Profile

Per substep, timing 30 steps from the closing of the fingers on (every kernel synchronized):

| Environments | Step | Contact solve | PCG iterations per substep (mean / slowest environment) | Newton iterations (mean / slowest) | Contact solve per iteration of the slowest environment |
| --- | --- | --- | --- | --- | --- |
| 64 | 898 ms | 97.4% | 351 / 911 | 3.9 / 9.2 | 96 us |
| 256 | 1204 ms | 97.8% | 331 / 1043 | 3.7 / 11.0 | 113 us |
| 1024 | 1742 ms | 97.7% | 323 / 1163 | 3.6 / 13.7 | 146 us |
| 4096 | 4004 ms | 97.6% | 327 / 1242 | 3.7 / 15.8 | 315 us |

At 4096 environments, the rest of the substep is the elastic forces (2.6 ms), the coarse space assembly and
factorization (2.3 and 1.0 ms), contact detection (1.7 ms), the rigid substep (1.1 ms), and the contact finalization,
damage, integration and update (0.5 ms together). The cost of an iteration of the slowest environment grows as
roughly 90 us + 0.055 us per environment:

- The fixed part is the launch of the graph nodes, about 8.5 us each, 7 per PCG iteration with rigid contacts, plus the
  nodes of the Newton iterations and line searches spread over them. Fusing the reductions of an iteration into the
  passes that produce their terms cut the cost of an iteration 3x.
- The variable part is the work of the environments still solving: run alone, the system product (stiffness of the
  faces and hinges) takes 271 us at 4096 environments over the 16.5 us of an empty launch, against 5 us with a single
  environment solving. Converged environments cost nothing measurable, so compacting the active environments would not
  pay. The product reads and atomically accumulates an estimated 0.5 GB per pass at 4096 environments, which bounds it
  by the memory bandwidth of the GPU.
- The slowest environment runs 3.8x the PCG iterations of the mean one, mostly through more Newton iterations (15.8
  against 3.7 per substep), its contacts leaving the linear model of the previous iteration.

Gained in the last round of changes, chiefly the friction bound no longer taken from an early iterate (see Contact
response), which drove extra Newton iterations: against the same benchmark before them, environment steps per second
went from 100 to 114 at 64 environments, 266 to 338 at 256, 737 to 759 at 1024 and 829 to 1108 at 4096. In the
profile at 256 environments, the step went from 1699 to 1204 ms, the Newton iterations of the slowest environment from
16.2 to 11.0 per substep, and the substeps reaching the Newton limit from 19 to 3.

Measured and rejected: a larger coarse space (12 or 20 patches gave the same iterations and time as the default 6,
and none or 3 patches ran 18% slower), inexact Newton (a looser PCG tolerance in the early Newton iterations ran slower
overall), and the compaction of active environments (see above).

## Remaining limits

- Contact resolution is one point per face, side and geom: a rigid body touching a single large triangle rests on one
  point. Faces must be smaller than the contact patches to resolve them (the egg faces are about 7 mm, the finger pads
  20 mm).
- The friction is regularized: a stuck contact creeps at about its tangential impulse over its contact stiffness
  (millimeters per second under the egg weight on the Franka fingertips at the default stiffness). Its bound lags the
  normal impulse by one substep, so that a new contact rubs from its second substep on.
- The penalty lets in a penetration of load x dt^2 / (`contact_stiffness` x contact mass). A lower stiffness runs
  faster but softens displacement-driven loads.
- Backward Euler does not conserve the angular momentum of a freely spinning sheet, an integrator property. The contact
  impulses themselves are equal and opposite at one point.
- Single precision resolves the stiff egg to a relative residual of about 1e-5 (STAGNATION, see above).
- The Newton iteration limit is reached by about 0.5% of the environments of the grasp benchmark over 60 steps,
  reported as a solver failure.
- Impacts need the default 1 ms substep: at 2 ms, the peak damage index of a drop comes out about half.
- A reduced-state snapshot (`get_state`) restarts the rigid constraint solve cold, as the rigid solver does by design.
  A checkpoint restores the run exactly (see History).
- Shell-rigid contact refuses hibernation and differentiable mode (an explicit error).
- Run-to-run GPU differences grow in sensitive scenarios as described above.
- Large spatial mesh-convergence studies were deferred, as the goal asks.

## Reproduction

Every command runs from the repository root inside the Genesis container image, with `PYTHONPATH=$PWD` so that the
checkout is the engine imported, and its own compilation cache (`QD_OFFLINE_CACHE_FILE_PATH`, `GS_CACHE_FILE_PATH`).
Examples write their meshes under `out/`.

```bash
# Regression tests, CPU and GPU, and the sensors sharing the closest-point routine
pytest -n 4 tests/deformable/test_shell.py --backend cpu
pytest -n 4 tests/deformable/test_shell.py --backend gpu
pytest -n 8 tests/sensors/test_tactile.py tests/sensors/test_raycaster.py --backend cpu
pytest -n 8 tests/rigid/test_serialization.py --backend cpu

# Consistency of resets, restores and repeated steps (drop -g for the CPU)
python examples/coupling/shell_egg_consistency.py -g

# Quasi-static press: reference, time steps, tolerances
python examples/coupling/shell_egg_press.py -g --precision 64 --substeps 40 --pcg-tolerance 1e-9 --pcg-velocity-tolerance 1e-9
python examples/coupling/shell_egg_press.py -g --substeps 5   # also 10, 20, 40
python examples/coupling/shell_egg_press.py -g --pcg-tolerance 1e-3   # also 1e-5

# Impacts: reference, then tolerances, comparing the printed peak damage index of every environment
python examples/coupling/shell_egg_drop.py -g --precision 64 --pcg-tolerance 1e-9 --pcg-velocity-tolerance 1e-9 --pcg-max-iterations 20000
python examples/coupling/shell_egg_drop.py -g --pcg-tolerance 1e-4   # also 1e-3, 1e-5

# End-to-end grasp
python examples/coupling/franka_egg_grasp.py -g -b 8

# Throughput and memory, then the rigid egg baseline and the speed/accuracy settings
python examples/speed_benchmark/franka_egg.py -b 4096   # also 64, 256, 1024
python examples/speed_benchmark/franka_egg.py -b 4096 --rigid-egg
python examples/speed_benchmark/franka_egg.py -b 1024 --substeps 5 --pcg-tolerance 1e-3 --contact-stiffness 5
```

The source engine runs the same benchmark from a checkout of `fd051fe3`, adapted to its API: `ShellOptions` takes
`n_pcg_iterations=50` (or 200) instead of the tolerances and the contact stiffness, and the egg material has no
`tensile_strength` or `fracture` (no damage-only mode there). The per-phase profile wraps every kernel of the shell
solve and both halves of the rigid substep with a synchronized timer (diagnostic script `profile_egg.py`, outside the
repository), timing 30 steps from the closing of the gripper (t = 1.0 s) on.
