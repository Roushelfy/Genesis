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
| Coulomb friction on an incline | velocity projection only | stick below 1e-4 m/s, slip acceleration within 0.5% across mesh edges |
| Momentum through a 3 m/s impact | not conserved (lagged reaction, no penetration correction) | conserved to 1e-9 kg m/s, no tunneling |
| Full reset vs fresh run (grasp, CPU and GPU) | 63 mm apart | identical (0) |
| One step repeated from the same state (CPU and GPU) | 0.45 mm apart | identical (0) |
| Partially reset environment vs fresh run (CPU and GPU) | 1.9 mm apart | identical (0) |
| Snapshot restore (`get_state` / `reset(state)`) vs continuation (grasp, CPU and GPU) | 38 mm apart | identical (0) |
| Checkpoint restore (`__getstate__`, pickle, `save_checkpoint`) vs continuation (CPU and GPU) | a scene holding a sheet cannot be checkpointed | identical (0) |
| PCG stopping | fixed iteration count, failures unreported | tolerance-driven device loop, true residual checked, statuses reported, converged environments leave every pass |
| Solver failures over the 60-step grasp benchmark (64 / 256 / 1024 / 4096 environments) | not reported | 0 / 0 / 0 / 0 |
| Damage | fracture criterion only, topology always mutated | Rankine damage index, damage-only mode, peak and first failure face |
| Franka egg grasp throughput (env-steps/s at 64 / 1024 / 4096 environments) | 1658 / 22045 / 31229 with 50 fixed PCG iterations and no contact solve | 126 / 944 / 1244 bit for bit, 182 / 1252 / 1608 in the performance mode (previous round 155 / 977 / 1193), 1.9x with a softer contact (see Performance) |

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

- The iterations loop on the device (`qd.graph.do_while`, CUDA graph conditional nodes), so that the host never
  synchronizes per iteration. Every pass of an iteration runs over the list of the environments still solving, which
  the decision of the previous iteration builds in one block by a scan of their flags: an environment that converged
  launches no work at all, and the lanes of the environments still solving stay side by side.
- Every pass is launched on a fixed grid of lanes (eight per GPU core) that strides over the work of the listed
  environments. A loop whose bound is read on the device (the count of listed environments, or the shape of an
  ndarray, as in the default mode) costs about 9 us per graph node, against about 2.5 us for the fixed grid.
- An iteration is the product, its Schur complement with rigid contacts, the residual update, the coarse solve, the
  decision and the next direction (plus one pass summing the residual norms in the deterministic mode, see below).
- Preconditioner: block-Jacobi plus a coarse correction in which every patch of a sheet moves affinely (9 degrees of
  freedom per patch). A sheet takes up to 12 patches by default, at most one per 8 vertices. Its dense inverse is
  rebuilt every substep (Cholesky factorization dropping the pivots that vanish, fixed or degenerate patches, then
  explicit inversion), so that the coarse solve of an iteration is a matrix-vector product whose lanes read columns of
  the symmetric inverse side by side. The patches read the vertices relative to the sheet, so that two copies of a
  sheet placed apart get the same ones.
- Stopping: the residual in the block-Jacobi norm below `pcg_tolerance` times the right-hand side, or the rigorous
  bound r^T M^-1 r on the mass-weighted velocity error below `pcg_velocity_tolerance`, the true residual b - A dv
  recomputed before stopping (covering the warm start too). Statuses: CONVERGED, MAX_ITERATIONS, BREAKDOWN, NON_FINITE
  (errno then halts the simulation) and STAGNATION (the true residual stalled at the floating-point floor, about 1e-5
  relative in single precision for this egg, which is not a failure). `ShellSolver.get_envs_solver_failure` returns
  the sticky union of failure flags per environment since its reset. Solver failure never feeds the damage
  observables. The iteration limit applies per linear solve, and the Newton loop has its own (32).
- Newton: the contact solve is an inexact Newton method. Its first linear solve of a substep, and every one from the
  moment an iterate matches its contact model or past its eighth iteration, run to `pcg_tolerance`. The solves in
  between, whose iterate only steers the next linearization, stop at 30 times it. An environment stops on an iterate
  solved to the full tolerance, once every contact impulse matches its linear model and the last step was full or the
  Newton direction moved no contact by more than `pcg_velocity_tolerance`. The last condition is new: every
  environment that used to reach the Newton iteration limit had converged long before (its contacts matched their
  model to 1e-3 of the tolerance, and each step moved them by 1e-10 to 5e-9 m/s), but a direction that small leaves
  the slope of the line search to rounding, whose step then never came out full. Loose solves without the bound of
  eight iterations kept switching the contacts of the slowest environments between regimes, which tripled the
  environments reaching the limit.

### Bitwise repeatability on GPU

A sum that threads accumulate with float atomic additions rounds in the order the threads arrive, which varies from run
to run on a GPU: before this round, one step from the same state differed by 0.06 um, and the grasp by millimeters.

Every sum that several threads accumulate (the system product, the residual norms, the coarse restriction and
assembly, the elastic forces, the contact terms and the line search slopes) now adds 64-bit integers: every term is
rounded to an integer multiple of a power of two that follows from a bound on the terms of the sum, and integer
addition does not depend on the order. The bounds come from the largest entry magnitudes the passes before record
(integer atomic maxima of the float bits, themselves order-independent), or from the shares of the residual norms the
vertices record before a second pass sums them. A term keeps about 62 - log2(number of terms) binary digits below its
bound, finer than single precision. A non-finite term reaches its bound and turns the sum into a NaN, so that errno
still halts the simulation.

The power of two of a sum derives from the binary exponents of its bounds by integer arithmetic alone: a product adds
exponents, a sum takes the larger one plus one. A first version evaluated the bounds in floating point, in the pass
that accumulates a sum and again in the passes that read it, which compile apart and round apart under fast math: a
bound straddling a power of two then read a sum back twice too large or too small. That broke the conjugate gradient
of a few environments (a Schur energy above the curvature it belongs to) and drove others to the Newton iteration
limit, 20 failing environments of 1024 over the 60 benchmark steps against 1 to 2 with float sums. A bisection that
switched one group of sums back to floats at a time, then float shadows of the failing sums, located it.

This is the default (`gs.use_deterministic_algorithms`, which Genesis enables unless `performance_mode=True`). The
performance mode keeps float atomics and drops the bound bookkeeping, which is faster but repeats only to the
solver tolerance on GPU. Apple Metal has no 64-bit integers and always sums floats.

### History, reset and restore

- The warm start `verts_dv` was neither saved nor cleared by `get_state` / `set_state`. It is now part of the state.
- The coarse preconditioner age was global. It is per environment, invalidated by `set_state`, and rebuilt every
  substep by default (`coarse_update_interval=1`, 6% of the grasp step at 4096 environments), so that a restored state
  replays exactly.
- The lagged friction bound and its geom per contact slot are part of the state.
- A restored snapshot (`get_state` / `reset(state)`) differed from its continuation by 0.26 um on the egg after one
  step, then 2.7 mm after 60, through the rigid solver. A bisection over every array of both solvers found the shell
  state complete and two rigid causes. The rigid state left out the warm start of the constraint solve, which the solve
  read on, and the forward kinematics `set_state` runs round the inertia of the links (1e-9) apart from those of the
  step, the same arithmetic compiled into different kernels rounding apart. The rigid state now carries the warm start
  (`qacc_ws`, `is_warmstart`), an initial guess the solve checks against its cold start, so that a restored state
  resumes the solve its original continued whatever was edited in it. A substep coupled to sheets refreshes the
  Cartesian space of the links with the kernel `set_state` runs. A restored snapshot then continues bit for bit on CPU
  in the grasp, full, partial or mid-grasp. A rigid scene without sheets keeps its own refresh, so that its restore
  still rounds the inertia apart (a Franka with an active joint limit: 1e-10 m/s after one step with the warm start,
  7e-9 without).
- A scene holding a sheet could not be checkpointed: the shell entity had no description and the shell solver did
  not list its arrays. A shell entity now keeps its description (its resolved rest mesh with its morph, material and
  surface, `ShellEntityDescription`), and the shell solver lists its arrays by kind (`ShellSolver.data`), so that
  `Scene.__getstate__` / `Scene.__setstate__`, pickling, `save_checkpoint`, `export` and `load` cover sheets, and a
  scene restored from a checkpoint continues bit for bit on CPU.

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
| `test_state_restore_reproduces_contact[0,2]` | full reset, snapshot restore, checkpoint restore, a pickled copy of the scene and partial reset replay the trajectory bit for bit, on CPU and GPU (fails if the warm start is not restored, if a scene holding a sheet cannot be checkpointed, or on GPU with float sums: 3 runs of 3 with `use_deterministic_algorithms(False)` differed) |
| `test_tearing_at_tensile_strength` | damage-only strip: damage index = E G / tensile strength within 2%, equal to the fracturing strip before failure (1e-9), mesh unchanged |

Before the fixes of the last two defects above, the incline box accelerated 21% slower than Coulomb friction allows over
the window crossing the vertex row (phantom corner), or stopped there (friction with no normal force), both failing the
0.5% assertion. It now accelerates within 0.2% of it. All deformable and coupling tests and the serialization tests the
shell description shares its machinery with (`tests/deformable`, `tests/coupling`, `tests/rigid/test_serialization.py`:
63 passed, 1 expected failure) pass on CPU and on GPU, as do the tactile and raycaster sensor tests
(`tests/sensors/test_tactile.py`, `tests/sensors/test_raycaster.py`, 37 tests, GPU) and the example runs of the four
shell examples (`tests/test_examples.py -m examples`).

### Consistency (`examples/coupling/shell_egg_consistency.py`, 4 environments, 160 steps)

Maximum deviation of the egg vertex positions:

| Comparison | Source, CPU | This branch, CPU | This branch, GPU (default mode) | Previous round, GPU (float sums) |
| --- | --- | --- | --- | --- |
| Full reset vs fresh run | 63 mm | 0 | 0 | 0.59 to 4.3 mm |
| Snapshot restore vs continuation | 38 mm | 0 | 0 | 58 um to 2.1 mm |
| Checkpoint restore vs continuation | not supported | 0 | 0 | 1.6 to 2.4 mm |
| One step repeated from the same state | 0.45 mm | 0 | 0 | 0.06 um |
| Partially reset environment vs fresh run | 1.9 mm | 0 | 0 | 6.1 to 155 um |
| Other environments vs continuation | 59 mm | 0 | 0 | 0.87 to 7.8 um |

The total contact force and the peak damage index compare identical too, in every comparison, on both backends and in
every run. The GPU comparisons came out identical in every run since the sums became fixed-point (see Bitwise
repeatability), as the GPU determinism probe confirms at the level of the solver arrays: running 20 substeps twice from
one checkpoint of the grasp at 16 environments, every array of the shell solver and every state array of the rigid
solver came out bit for bit equal (one rigid scratch array, the order of a factorization worklist, differs). The
previous round's GPU differences grew from 0.06 um after one step to millimeters over the grasp: the grasp amplifies
perturbations (a CPU run whose egg starts 1 nm higher diverged the same way, to 1.3 mm), so that bitwise sums are what
makes a GPU rollout repeat. In the performance mode the GPU repeats to the solver tolerance only, as before.

### Tolerance and time step sweeps (same mesh)

Quasi-static press between two plates (`examples/coupling/shell_egg_press.py`), against a double precision,
40-substep, 1e-9 tolerance reference:

| Setting | Force at 100 um | Damage at 100 um | Compression at first failure | Error vs reference |
| --- | --- | --- | --- | --- |
| reference (fp64, 40 substeps) | 10.910 N | 0.589 | 169.40 um | - |
| fp32, 5 substeps | 10.824 N | 0.583 | 170.96 um | 0.7-0.9% |
| fp32, 10 substeps | 10.885 N | 0.587 | 169.88 um | 0.2-0.3% |
| fp32, 20 substeps | 10.902 N | 0.588 | 169.55 um | 0.1% |
| fp32, 10 substeps, tolerance 1e-3 / 1e-4 / 1e-5 | 10.885 N | 0.587 | 169.87 um | same as above |

The time step error is first order, as backward Euler is, and the solve tolerance does not move the quasi-static
observables. The failing face differs between runs among faces of equal damage index (the egg is symmetric).

Impacts (`examples/coupling/shell_egg_drop.py`: 16 eggs dropped at 0.63 to 1.25 m/s onto the ground at random
orientations, peak damage index against a double precision 1e-9 reference): tolerance 1e-3: median error 11%, max 67%.
Tolerance 1e-4 (the default): median 0.09%, max 1.6%. Tolerance 1e-5: median 0.01%, max 0.27%. No environment failed
at any tolerance. With a fixed iteration count and the old stopping test, some impacts produced 1e5 N forces.

The maximum at the default tolerance is one drop (the eighth) whose peak damage is sensitive to the path of the
solve: it moves by up to 1.6% between equally converged runs (the deterministic and the performance mode, with and
without the inexact Newton iterations of this round), against 0.01 to 0.3% for the other drops, and keeps 0.13% at
tolerance 1e-5. Over 64 drops against their own double precision reference, the default tolerance gives a median error
of 0.16%, a 90th percentile of 0.60%, a mean of 0.36% and a maximum of 5.3% (one drop, the same in every
configuration). The inexact Newton iterations of this round cost a little of
this accuracy: with every Newton iteration solved to the full tolerance, the 64 drops give a median of 0.15%, a 90th
percentile of 0.45%, a mean of 0.33% and the same maximum, and the 16 drops a median of 0.07% and a maximum of 0.62%.
The difference stays within the spread of equally converged runs of these drops (up to 0.5 percentage points on one
drop between the deterministic and the performance modes), against 11% and 67% at tolerance 1e-3, for 12 to 14% of
throughput.

### End-to-end grasp (`examples/coupling/franka_egg_grasp.py -g -b 8`)

All 8 eggs are lifted 80 mm and carried. Of the 4 environments whose grip loosens to 2% while carrying, one lets its egg
slip out before the release and the others let theirs slide 3 to 12 mm down between the fingers, and every egg falls
on release. Before the release, the peak damage index exceeds 1 from a grip of 21 N on (0.78 at 5 N, 0.77 at 13 N,
2.1 at 52 N), the drop on the ground breaking every egg. No environment reports a solver failure (one did in the
previous round). The grasp is sensitive to the path of the contact solve, as the impacts above: with the solver of the
previous round, the 21 N grip peaked at 0.84.

## Performance

All timings are synchronized wall times on the RTX PRO 6000 with no other job on the GPU, compilation, rendering and
the grasp approach excluded: 60 timed control steps (600 substeps) while the fingers close and lift the egg. One
control step is one policy transition, so environment steps per second are policy transitions per second. The default
mode repeats bit for bit (see Bitwise repeatability); `gs.init(performance_mode=True)` keeps float sums, as Genesis does
for the rest of its solvers in that mode, and repeats to the solver tolerance on GPU.

### Throughput and memory (`examples/speed_benchmark/franka_egg.py`, `--deterministic` for the default mode)

| Environments | Default mode (bit for bit) | Performance mode | Previous round (performance mode) | Rigid egg baseline | Source engine, 50 / 200 PCG iterations |
| --- | --- | --- | --- | --- | --- |
| 64 | 126 env-steps/s, 1684 MiB | 182 (2.84 scene steps/s), 1682 MiB | 155, 1682 MiB | 3472, 1646 MiB | 1658 / 444 |
| 256 | 375, 1886 MiB | 503 (1.97), 1852 MiB | 423, 1852 MiB | 12118 (previous round), 1774 MiB | 6609 / 1768 |
| 1024 | 944, 2684 MiB | 1252 (1.22), 2618 MiB | 977, 2586 MiB | 43619, 2318 MiB | 22045 / 7183 |
| 4096 | 1244, 5912 MiB | 1608 (0.39), 5654 MiB | 1193, 5622 MiB | 141465, 4502 MiB | 31229 / 17768 |

No environment reports a solver failure over the 60 steps in either mode at any size (previous round: 1, 3, 9 and 23
environments), nor over three more seeds of the grasp parameters at 1024 environments in either mode. Memory is that
of the process (the CUDA context and allocator pools included): the shell egg adds 1152 MiB to the rigid scene at 4096
environments in the performance mode, 288 KiB per environment, and the default mode 1410 MiB, its fixed-point sums
taking 64-bit integers. The coarse space of 12 patches takes 137 KiB per environment (its assembled matrix, its factor
and its inverse, 108 x 108 each). The source engine is faster because it runs a fixed number of iterations with no
contact solve, at an accuracy this branch cannot be matched against: its ball falls through a triangle, its friction is
a velocity projection, its reaction lags a substep, and its fixed iteration count produced 1e5 N impact forces. The
comparison at matched accuracy is between the settings of this branch below.

### Gains of this round

| Change | Measured effect |
| --- | --- |
| Passes over the environments still solving only | a converged environment costs nothing measurable (see Converged environments) |
| Fixed grids of lanes instead of loop bounds read on the device | 9 us to 2.5-4 us per graph node, a PCG iteration at 64 environments from about 47 to 33 us of passes |
| Component-major vertex rows of the system product | the product of 4096 environments all solving from 330 to 195 us (float sums), 549 to 288 us (fixed-point) |
| Inexact Newton iterations (30 times looser, the first eight iterations) | +12% at 1024 and +14% at 4096 environments (1046 to 1172, 1358 to 1553 env-steps/s), impact errors within the spread of converged runs (see Validation) |
| Newton step-norm convergence test, exact fixed-point scales | no solver failure left, +5% at 1024 environments in the performance mode |
| total, performance mode | +17%, +19%, +28% and +35% at 64, 256, 1024 and 4096 environments |

- A loop whose bound is read on the device runs on a saturating grid with a serial launch of its bound, about 9 us per
  graph node whatever its work, against about 2 us for a bound known when the kernel compiles. The passes of an
  iteration now stride a fixed grid of eight lanes per GPU core over the environments still solving, every runtime
  count read inside the lane loop. In the default mode, whose arrays are ndarrays of runtime shape, every pass of the
  previous round paid the 9 us.
- The system product scatters the rows of the faces and hinges to their vertices by atomic additions, and is bound by
  the bandwidth of the L2 cache (ncu: 0.92 GB of L2 traffic per product at 4096 environments). With the rows of a
  vertex stored as an interleaved vector, the lanes of a warp, which hold consecutive environments, added one
  component across 12 sectors, against 4 with the rows stored component-major.
- Inexact Newton: see Linear solver. Solving the early Newton iterations loosely saves PCG iterations, but loose solves
  past the first few iterations kept the slowest environments switching contact regimes: unbounded, they tripled the
  environments reaching the Newton iteration limit.
- The determinism costs 25 to 30% of the throughput (944 against 1252 env-steps/s at 1024 environments, 1244 against
  1608 at 4096): the 64-bit sums double the atomic traffic of the product, the residual norms take a second pass, and
  every pass maintains the bounds of its sums.

### Converged environments

One environment grasping while every other one rests (`idle_envs2.py`, its egg falling freely, converged in about 12
PCG iterations per substep):

| Setting | Contact solve per substep | PCG iterations of the grasping environment | Its extra iterations cost |
| --- | --- | --- | --- |
| 1 environment alone | 3.61 ms | 80.1 | 45 us per iteration |
| 1 grasping among 1023 resting | 5.68 ms (2.23 ms with all 1024 resting) | 80.3 | 3.45 ms for 68 iterations, 51 us each |
| 1 grasping among 4095 resting | 11.08 ms (7.80 ms with all 4096 resting) | 76.7 | 3.28 ms for 65 iterations, 51 us each |

The iterations the grasping environment runs past the resting ones cost the same beside 1023 or 4095 converged
environments, a few microseconds above alone (the decision of every iteration still reads the flags of every
environment). In the previous round, the same iterations took 80 us alone, 132 us among 1024 and 298 us among 4096
resting environments. In the default mode the same runs take 5.54, 7.52 and 14.56 ms.

### Speed and accuracy settings (1024 environments, performance mode)

| Setting | env-steps/s | Speedup | Press force error | Impact damage error (median / max) |
| --- | --- | --- | --- | --- |
| default: 10 substeps, `pcg_tolerance=1e-4`, `contact_stiffness=25` | 1252 | 1.0x | 0.2% | 0.09% / 1.6% |
| `contact_stiffness=5` | 2430 | 1.9x | 2.3% | 0.26% / 0.65% |
| `pcg_tolerance=1e-3` | 1469 | 1.2x | 0.2% | 11% / 67% |
| 5 substeps | 1881 | 1.5x | 0.8% | 48% / 57% |
| all three | 4091 | 3.3x | 5.1% | 52% / 84% |

The press force error is that of the force at 100 um of compression against the double precision reference, and the
impact damage error that of the peak damage index of the 16 drops of `shell_egg_drop.py` against its double precision
reference. A softer contact is the cheap lever: it lets 5x more penetration in, which takes a few percent of an imposed
displacement from the deformation of the egg, and leaves impacts accurate. A 2 ms substep under-resolves the impacts
(about half the peak damage), and a looser tolerance makes impact damage unreliable.

### Profile

Per substep, timing 30 steps from the closing of the fingers on (every kernel synchronized):

| Environments | Mode | Step | Contact solve | PCG iterations per substep (mean / slowest environment) | Newton iterations (mean / slowest) | Contact solve per iteration of the slowest environment |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | performance | 358 ms | 93.6% | 217 / 548 | 4.5 / 9.5 | 61 us |
| 256 | performance | 519 ms | 94.8% | 203 / 623 | 4.3 / 10.3 | 79 us |
| 1024 | performance | 826 ms | 93.9% | 202 / 684 | 4.3 / 11.2 | 113 us |
| 4096 | performance | 2426 ms | 91.1% | 204 / 743 | 4.3 / 12.2 | 298 us |
| 64 | default | 516 ms | 92.8% | 219 / 547 | 4.6 / 9.5 | 87 us |
| 256 | default | 709 ms | 93.3% | 204 / 613 | 4.4 / 10.3 | 108 us |
| 1024 | default | 1102 ms | 93.3% | 202 / 679 | 4.3 / 11.3 | 152 us |
| 4096 | default | 3139 ms | 92.0% | 203 / 748 | 4.3 / 12.3 | 387 us |

No substep of any environment reaches the Newton or the PCG iteration limit, and no search direction breaks down, in
either mode (the previous round: 9 to 47 substeps at the Newton limit and up to 7 breakdowns per profile). About 5.6% of
the substeps of an environment end a linear solve at the floating-point floor (STAGNATION, not a failure), against
16% in the previous round, which ran every Newton iteration to the full tolerance. At 4096 environments, the rest of the
substep in the performance mode is the coarse space factorization and assembly (14.3 and 2.1 ms), the elastic forces
(2.5 ms), contact detection (1.4 ms) and the rigid substep (0.3 ms).

At 64 environments, a PCG iteration of the performance mode costs about 33 us of passes (product 11, residual 5.4,
coarse 9.6, decision 4.1, direction 3.2 us), the rest of the 61 us being the Newton iterations, the line searches and
the device loop, about 3.5 us per iteration. At 4096 environments with every environment solving, the product takes
195 us, the coarse solve 178 us (reading 46 KiB of inverse per environment, which no longer fits the L2 cache past
about 2000 environments) and the residual 42 us.

### Measured and rejected this round

- Packing the symmetric coarse inverse (half the bytes): 13% faster coarse solves at 4096 environments, twice slower at
  64, every lane reading its own row of the packed triangle.
- Splitting the dot product of a coarse row over four lanes: no faster at 64 environments, slower at 1024.
- Inverting the coarse matrix by Gauss-Jordan sweeps in shared memory: the shared tile of the device that holds 108
  unknowns is 152 wide (93 KB), one block per streaming multiprocessor, which left the sweep latency bound, faster than
  the Cholesky factorization at 64 and 4096 environments, slower at 1024, and much slower for coarse spaces above the
  tile.
- Static topology (corner vertices read from the rest mesh without fracture): the product reads were not its bottleneck.
- Running two PCG iterations per device loop iteration: the loop costs about 3.5 us per iteration.
- 16 or 20 coarse patches: the factorization of the coarse matrix grows with the cube of its size (36 and 86 ms per
  substep at 4096 environments), outweighing the PCG iterations saved.

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
- No environment of the grasp benchmark reaches a solver limit over 60 steps. The Newton iteration limit (32) and the
  PCG iteration limit (1000) remain safety limits, a solve stopped by either being reported as a solver failure.
- Bitwise repeatability costs throughput: the default mode runs 25 to 30% slower than the performance mode, which
  repeats on GPU to the solver tolerance only. Apple Metal has no 64-bit integers and always sums floats.
- The fixed-point sums round their terms at about 62 - log2(terms) binary digits below a bound on them. Where that
  bound is an a priori estimate (the system product, the coarse correction) rather than a measured maximum (the
  residual norms), a bound far above the actual terms rounds them more coarsely. On the egg, the default and the
  performance modes give the same press force and impact damage within the solver tolerance (see Validation).
- The coarse space takes 137 KiB per environment with 12 patches on the egg, half of the memory the sheet adds.
- Impacts need the default 1 ms substep: at 2 ms, the peak damage index of a drop comes out about half.
- A rigid scene without sheets restores its snapshots to the rounding of the inertia of its links (see History).
- Shell-rigid contact refuses hibernation and differentiable mode (an explicit error).
- Large spatial mesh-convergence studies were deferred, as the goal asks.

## Reproduction

Every command runs from the repository root inside the Genesis container image, with `PYTHONPATH=$PWD` so that the
checkout is the engine imported, and its own compilation cache (`QD_OFFLINE_CACHE_FILE_PATH`, `GS_CACHE_FILE_PATH`).
Examples write their meshes under `out/`.

```bash
# Regression tests, CPU and GPU, and the sensors sharing the closest-point routine
pytest -n 4 tests/deformable tests/coupling tests/rigid/test_serialization.py --backend cpu
pytest -n 4 tests/deformable tests/coupling tests/rigid/test_serialization.py --backend gpu
pytest -n 4 tests/sensors/test_tactile.py tests/sensors/test_raycaster.py --backend gpu
pytest -n 2 -m examples tests/test_examples.py -k "shell or egg"

# Consistency of resets, restores and repeated steps (drop -g for the CPU)
python examples/coupling/shell_egg_consistency.py -g

# Quasi-static press: reference, time steps, tolerances
python examples/coupling/shell_egg_press.py -g --precision 64 --substeps 40 --pcg-tolerance 1e-9 --pcg-velocity-tolerance 1e-9
python examples/coupling/shell_egg_press.py -g --substeps 5   # also 10, 20, 40
python examples/coupling/shell_egg_press.py -g --pcg-tolerance 1e-3   # also 1e-5

# Impacts: reference, then tolerances, comparing the printed peak damage index of every environment
python examples/coupling/shell_egg_drop.py -g --precision 64 --pcg-tolerance 1e-9 --pcg-velocity-tolerance 1e-9 --pcg-max-iterations 20000
python examples/coupling/shell_egg_drop.py -g --pcg-tolerance 1e-4   # also 1e-3, 1e-5
python examples/coupling/shell_egg_drop.py -g -b 64   # 64 drops, against the same reference run with -b 64

# End-to-end grasp
python examples/coupling/franka_egg_grasp.py -g -b 8

# Throughput and memory (performance mode, then the default bit-for-bit mode), then the rigid egg baseline and the
# speed/accuracy settings
python examples/speed_benchmark/franka_egg.py -b 4096   # also 64, 256, 1024
python examples/speed_benchmark/franka_egg.py -b 4096 --deterministic
python examples/speed_benchmark/franka_egg.py -b 4096 --rigid-egg
python examples/speed_benchmark/franka_egg.py -b 1024 --substeps 5 --pcg-tolerance 1e-3 --contact-stiffness 5
```

The source engine runs the same benchmark from a checkout of `fd051fe3`, adapted to its API: `ShellOptions` takes
`n_pcg_iterations=50` (or 200) instead of the tolerances and the contact stiffness, and the egg material has no
`tensile_strength` or `fracture` (no damage-only mode there). The per-phase profile wraps every kernel of the shell
solve and both halves of the rigid substep with a synchronized timer, timing 30 steps from the closing of the gripper
(t = 1.0 s) on and recording the PCG iterations, Newton iterations and statuses of every environment and substep. The
converged-environment measurement runs one grasping environment among resting ones, and the GPU determinism probe runs
20 substeps twice from one checkpoint of the grasp and compares every array of both solvers (diagnostic scripts
`profile_egg_record.py`, `idle_envs2.py` and `determinism_probe.py`, outside the repository). The bisection of the
solver failures of the first fixed-point version ran the benchmark with one group of sums at a time switched back to
floats, then with float shadows of the failing sums recorded at every breakdown.
