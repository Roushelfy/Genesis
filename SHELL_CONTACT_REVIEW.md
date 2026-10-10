# Shell contact and consistency: audit and acceptance guide

This guide records observations from a static review of [Kashu7100/Genesis PR #22](https://github.com/Kashu7100/Genesis/pull/22)
at `fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`. It supports [the goal](SHELL_CONTACT_GOAL.md).
The observations identify investigation starting points. They leave the agent free to choose the numerical method.

## Scope and evidence

The source PR implements triangular shells with Saint Venant-Kirchhoff membrane elasticity, dihedral-angle bending,
stiffness-proportional damping, and one linearized backward-Euler solve per substep. The linear solve uses matrix-free
preconditioned conjugate gradient (PCG), block-Jacobi, and an optional patch coarse space.

The collaborators reported the following results for a private grasp benchmark:

| Observation | Reported result |
| --- | --- |
| Small sphere pressing inside a triangle | 0.300 mm geometric depth and 0 N reaction |
| 150 vertices, 4 substeps, 10 PCG iterations | Stopping tolerance missed on every substep |
| Same case with 200 PCG iterations | Reported stopping tolerance passed |
| Peak damage, 4 / 8 / 16 / 32 substeps | 259.23 / 78.54 / 65.33 / 12.44 MPa-equivalent |
| Repeat of 4-substep, 200-PCG case | 342.16 MPa-equivalent, about 32% higher |

The public PR does not include the corresponding spherical tactile probe, damage observation definition, or benchmark
script. These measurements have not been independently reproduced. Build controlled reproductions and record their
complete settings. If the exact private case is unavailable, distinguish the new reproduction from the reported case.

The immediate target is low-resolution contact and numerical consistency with useful batched throughput. Preserve shell
deformation, including its effect on contact geometry. A rigid-only stress recovery pipeline would be a separate model.
Incremental Potential Contact (IPC) can provide a small reference or an implementation direction if measured costs fit.

## Confirmed code observations

### Surface sampling and response

`LegacyCoupler.kernel_shell_rigid_collide` in `genesis/engine/couplers/legacy_coupler.py` samples each free shell vertex
against every coupling-enabled rigid geom's signed distance field (SDF). It offsets the contact by half shell thickness.
Triangle interiors and edges have no independent contact detection. This explains how a geometric face query can detect
overlap while the physical vertex contacts remain empty. A more accurate elastic solve cannot repair missing candidates.

The shared `_func_collide_in_rigid_geom` responds only when relative normal velocity points inward. It modifies normal
and tangential velocity, derives impulse from the individual vertex's lumped mass, and accumulates a rigid reaction.
The function contains no depth-based position correction. Its local mass response also omits implicit shell compliance.

`Simulator.substep` runs solver pre-coupling work, the coupler, and solver post-coupling work in that order. Under Legacy
coupling, `RigidSolver.substep_pre_coupling` completes the rigid substep before shell contact writes the coupling wrench.
Rigid forward dynamics consumes this accumulated wrench on the following substep and then clears it. Inspect this
ordering when addressing time-step sensitivity and force/torque consistency.

### Elastic solve and convergence reporting

`ShellSolver.substep_pre_coupling` performs one linearized solve, applies its velocity increment, and then contact modifies
velocity. `substep_post_coupling` advances positions. PCG convergence therefore controls a linear system, rather than a
fully converged nonlinear elastic-contact problem.

The membrane tangent keeps the positive part of the geometric stress term. Bending uses a positive semidefinite tangent
approximation. This is a deliberate design for a positive definite mass-plus-stiffness operator on free vertices.
Account for this approximation when diagnosing buckling or strong geometric nonlinearity.

`kernel_shell_pcg_init` warm-starts from the previous substep's `verts_dv`. Its threshold is relative to a block-Jacobi
weighted right-hand side, while the residual numerator also includes coarse correction when enabled. This is not the
ordinary Euclidean relative residual. State the chosen norm and use a consistent reference norm for the stopping test.

`kernel_shell_pcg_update` turns off `envs_is_solving` when `p^T A p <= 0` as well as when its residual test passes.
The outer loop still dispatches the configured maximum number of iterations, even after every environment stops.
After the loop, the solver applies the result even when the iteration cap was reached without convergence.

### Reset and history

`ShellSolver.get_state`, `set_state`, and `kernel_shell_set_state` restore physical state and connectivity. They do not
save or clear the warm-start `verts_dv` buffer. Force assembly clears that buffer for unused/fixed vertices, while free
vertices keep their previous values. This is a confirmed numerical-history omission during reset or state restore.

The omission is a candidate contributor to repeat variability. It has not been shown to explain the reported 32%.
An accurately converged linear solve should substantially reduce sensitivity to its initial guess. Trace the earliest
diverging quantity before attributing a whole grasp's peak difference to this issue or to floating-point precision.

`set_state` already invalidates the coarse cache through `_coarse_age = None`. Its age is global, so partial reset can
trigger rebuilding coarse data for the whole batch. Consider environment-local cache ownership when profiling resets.

### Damage and topology

`func_membrane_stress` returns a membrane stress resultant in N/m, using Green strain. `kernel_shell_fracture` adds an
outer-layer bending strain estimate and computes a tensile separation criterion around each vertex. This criterion is
distinct from maximum von Mises stress. A reported MPa-equivalent value needs an explicit definition and thickness units.

The fracture kernel also selects splits, calls `func_alloc_vert`, and rewrites `corners_vert`. The face count stays fixed,
while connectivity changes independently per environment. There is no adaptive remeshing in this PR.

`tensile_strength=None` disables the fracture kernel and extra vertex reserve. Setting a tensile strength enables both
the criterion and topology changes. With default `fracture_capacity=1.0`, 150 original vertices reserve 300 vertex slots.
Setting capacity to zero leaves fracture evaluation enabled. Damage evaluation and topology mutation need separate
execution paths for a fixed-topology damage-only workload.

## Contact outcomes to demonstrate

Use a physical spherical probe as a minimal reproduction before the articulated grasp. Cover contacts inside faces,
along edges, and near vertices, including points whose surrounding vertices are outside the probe. Use a restrained or
otherwise loaded shell so sustained compression has a physically meaningful nonzero reaction.

The physical contact geometry and tactile distance query should share closest-point and thickness conventions, or have
a documented transformation between them. A virtual distance query by itself does not imply a physical force.
For physical surface contacts, construct a consistent contact Jacobian and force distribution. Check the resultant
force and moment, normal direction, contact activation/release, and frictional work. Avoid counting the same shared-edge
contact several times through adjacent faces.

Demonstrate pressing, holding, sliding with stick/slip transitions, and release. Already-overlapping states require a
defined correction or refusal policy. Fast approach requires a supported collision-safety policy. Choose continuous
collision detection, conservative advancement, or justified adaptive stepping according to the intended workload.

Use actual Franka collision geoms in the end-to-end scene. Geometry coverage must support the advertised input domain.
Document remaining limits precisely. Two-way coupling must include articulated rigid effective response and matching
force/torque transfer, with temporal ordering that meets the consistency tests.

Joint implicit contact, compliant contact, or iterative elastic-contact coupling are reasonable options. A new geometric
detector followed by the unchanged vertex-mass projection only addresses candidate detection. Verify the full response
with the same low-resolution benchmark before choosing the production method.

## Consistency outcomes to demonstrate

1. Compare the next substep from a fresh state, full reset, restored snapshot, and partial reset. Numerical history,
   contact caches, and preconditioners must be valid or invalidated. Other environments retain independent physical state.
2. Run identical prescribed inputs repeatedly and compare force, penetration, stress/damage, and first failure.
   Use frozen-state single-step comparisons to isolate variation before examining longer trajectories.
3. Compare independent environments with their batched equivalents at matched precision and tolerances. Mix easy and
   difficult contacts, completed episodes, and reset environments so batch scheduling cannot hide contamination.
4. Use a named residual norm with relative and absolute tolerances. Recompute the true residual for validation and when
   needed for reliable stopping. Report convergence, maximum-iteration failure, breakdown, and non-finite values.
   Use an explicit recovery or failure path when accuracy cannot be achieved. Solver failure is distinct from material
   failure, and must not quietly become a physical damage reward.
5. On the same mesh, sweep a small set of solve tolerances and time steps. Keep the continuous input motion, controller
   schedule, physical duration, and observation definitions identical. Compare reaction, penetration, pre-failure stress,
   and first failure. Use analytic or high-accuracy small references where available. Expect finite time-step error and
   demonstrate a bounded, explainable trend instead of demanding identical dynamics at every time step.
6. Use small higher-precision diagnostics when needed to locate arithmetic sensitivity. Choose production precision from
   measured accuracy and throughput. Bitwise equality across floating-point reduction orders is not the acceptance goal.

Set dimensional pass/fail tolerances from the physics and reference accuracy, and state them before reporting results.
Avoid broad tolerances that conceal zero reactions, incorrect moments, or large damage variation. PCG residual tolerance
does not directly bound maximum-stress error. Validate the actual damage observable as the linear tolerance is tightened.

For an episode that ends at failure, compare the first threshold crossing and pre-failure loading history. Peaks collected
after failure and topology changes represent a different question. Document the mechanical criterion, stress measure,
surface layer, units, and normal/friction contributions. Preserve bending effects for brittle hollow shells.

## Lightweight implementation and throughput

Keep simulation computations in native Genesis/Quadrants kernels and batched state on the device. CPU code is useful for
initialization and small diagnostic references. Follow the existing engine APIs, error handling, and repository tests.

The damage-only path should use fixed connectivity, evaluate its failure observation before any fracture mutation,
and allocate only the required vertex slots. Existing optional fracture behavior can remain a separate capability.
The optimized contact path must support changing contact locations, asymmetric grasps, arbitrary rigid motion, and
friction. Rest-geometry sharing is valid. Shared deformed operators need a demonstrated invariant.

Profile candidate generation, distance/normal evaluation, contact construction, elasticity assembly, matrix products,
preconditioning/coarse updates, reductions, coupling iterations, damage reduction, history/reset, dispatch, transfers,
and synchronization. Record solver iteration distributions and the small population of difficult environments.

Use the profile to decide among device-side convergence loops, kernel fusion, environment-local reductions, active-env
compaction, collision broad phase, cached fixed-topology data, and validated history reuse. Measure overhead as well as
saved work. Keep a shared reference preconditioner separate from each environment's actual deformed operator.

Use a reproducible headless Franka grasp of a low-resolution egg-shaped hollow shell, initially around the reported
150-vertex scale or the nearest useful mesh. Include off-center/asymmetric grasps and tangential motion. This remains
a deformable shell scene. Shell rest shape and material are fixed, while contact state varies.

Compare the source implementation and the new implementation at matched accuracy, controller inputs, time step, mesh,
and damage mode. Also report a rigid baseline with equivalent robot/control/observation costs. Warm up compilation,
measure synchronized elapsed time, and report both batched scene-steps/s and total environment-steps/s. If actions span
several scene steps, also report policy transitions/s. State GPU model, precision, batch size, substeps, memory use,
timing boundaries, and whether rendering and compilation are included.

Scale batches to the available hardware and report throughput saturation. The collaborators' 28,672-env/96-GB case is a
reference workload, not a required initial allocation. CPU-only measurements cannot substantiate a GPU throughput claim.
Large spatial mesh refinement remains deferred until contact correctness, low-mesh consistency, and profiling are done.

## Deliverables and source pointers

Extend the existing shell/coupling feature tests with focused observables that fail for the confirmed defects. Include
CPU coverage and GPU runs when available. Keep detailed internal traces in diagnostics rather than implementation-spying
regression assertions. Save complete test logs and reproducible benchmark commands with before/after settings.

Provide a runnable Franka/egg example and benchmark, plus a report of root causes, fixes, accuracy, throughput, memory,
supported geometry, remaining limitations, and which results were actually measured. Explain any temporal or material
model change. Check relevant existing shell and rigid-coupling behavior, then commit and push the completed changes.

| Area | Initial source files |
| --- | --- |
| Shell elasticity, PCG, fracture, reset | `genesis/engine/solvers/shell_solver.py` |
| Shell geometry and split capacity | `genesis/engine/entities/shell_entity.py` |
| Contact geometry and velocity response | `genesis/engine/couplers/legacy_coupler.py` |
| Solver/coupler scheduling | `genesis/engine/simulator.py` |
| Rigid integration | `genesis/engine/solvers/rigid/rigid_solver.py` |
| Coupling wrench consumption | `genesis/engine/solvers/rigid/abd/forward_dynamics.py` |
| Coupling force/torque accumulation | `genesis/engine/solvers/rigid/abd/misc.py` |
| Buffers and static configuration | `genesis/utils/array_class.py` |
| Shell options and material parameters | `genesis/options/solvers.py`, `genesis/engine/materials/shell.py` |
| Existing shell feature coverage | `tests/deformable/test_shell.py` |
| Existing Franka timing example | `examples/speed_benchmark/franka.py` |
