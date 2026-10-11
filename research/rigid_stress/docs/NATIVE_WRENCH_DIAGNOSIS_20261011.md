# Feasible-fit repair and continuous wrench incompatibility

The active nonconvex goal remains incomplete. This checkpoint follows
`ef388a7e` and the [exact first-failure diagnosis](NATIVE_SCHEDULING_DIAGNOSIS_20261011.md).
It distinguishes the feasible step517 solver failure from a different,
continuously incompatible original component at step602. Neither failure is
resolved by relaxing final acceptance or reporting only accepted environments.

## Native fitting repair

The native scalar fitter tries the unregularized Newton direction first. It
retains the original trace-scaled `1e-12` ridge when factorization or descent
is invalid, and retains the existing gradient fallback. A trust radius
`2 max(1, ||dual||)` limits the proposed dual step before backtracking. It
does not clip forces, change friction, alter the physical disk or change the
specified traction objective. Internal termination remains9e-9 and the
evaluation limit remains128.

Fit verification and applied sample evaluation use the same subtraction order,
`force - dual_force - cross(dual_moment, relative_position)`. The first adaptive
candidate passed50 CUDA cases but failed two CPU sliding cases: its dual check
and applied sample check used different floating-point operation orders.
Large dual components made that difference observable at the original wrench
budget. The trust radius and consistent evaluation repair that mechanism.
Those failed controls are retained rather than called harmless noise.

The new1036-byte step517 fixture is extracted from the exact first failed frame,
not from the later step520 capture. Its regression independently reconstructs
the CPU geometry/profile, fits physical tractions, checks signed P2 nodal loads,
original-point wrench and local cones, then compares complete equilibrium,
full stress and global peak. Restoring the mandatory ridge makes this new
regression fail at its unchanged native SUCCESS assertion.

Final original-component force and moment budgets remain1e-8. Complete
equilibrium, including six gauge rows, remains
`max(1e-11 N, 1e-8 ||rhs||)`. Stress remains
`rtol=1e-4, atol=1e-3 Pa`. No high-resolution physical mesh or integration
convergence experiment was run.

The complete combined regression passes50 CPU cases in404.67s and50 CUDA
cases in571.07s. That frozen epoch uses a shared expression helper. The final
code inlines the identical expression to follow repository helper conventions;
that intermediate epoch passes seven affected CPU/CUDA cases in62.96/138.16s.
The final direction retry uses a runtime two-attempt loop so the matrix helper
is inlined once, with the unregularized and ridge fallback branches unchanged.
Its seven affected CPU/CUDA cases pass in62.76/137.89s, retained as a separate
source epoch. The full suite covers nonconvex geometry,
source association, vector fitting, finite footprints, all saved surface-fan
contacts, shared support, lifecycle/reset and continuous force diagnostics.
Each allocated run verifies unchanged source hashes before and after execution.

## Exact step602: continuous incompatibility, not iteration exhaustion

An experimental adaptive cooperative fitter passes the former step517 failure,
then stops the B1024/seed623201 trajectory after602 completed steps. The exact
current failed input is env66, original component1, with physical radius
4.691972138963162mm and actual combined mu0.447395471683579. Both component
witnesses bind to their actual material vertices; geom24/pair22 is retained.
Force-only feasibility does not imply six-component wrench feasibility.

Independent offline FP64 circular-cone SOCPs use the exact1195 eligible samples
of the declared shared support. Component0 is feasible, with normalized
force/moment errors6.34e-16/1.00e-15. Component1 is infeasible. We independently
check its separating ray against every circular cone, rather than infer
incompatibility merely from solver failure or solver status.

| Certificate | Normalized target margin | Conservative continuous polar upper bound |
|---|---:|---:|
| Independent SOCP ray | 1.390666669e-3 | -4.069345775e-4 |
| Independently searched bounded ray | 2.045878843e-3 | -1.203060927e-4 |

For a ray `(r_F,r_M)`, the local polar value is
`n dot v + mu ||v - (n dot v)n||`, with
`v = r_F + cross(r_M,(x-anchor)/radius)`. It is convex in position on a face.
A negative upper bound on all allowed positions, together with positive target
margin, separates the target from every admissible finite traction distribution.

The continuous domain includes every actual material-boundary triangle
intersecting either unchanged Euclidean radius ball. It therefore contains
the intrinsic footprints, including branches that the intrinsic law excludes.
The independent CPU bound clips triangles to plane disks and bounds circle arcs
analytically between angular samples. This is a geometric certificate, not
a physical quadrature convergence study. The sampled and continuous results
prove that increasing integration samples or solver iterations cannot recover
this original component under the current declared support and friction law.

The independent SOCP uses Clarabel0.11.1 with the
[official circular-cone interface](https://clarabel.org/stable/python/getting_started_py/).
It is an offline diagnostic dependency installed into an isolated data-side
directory; production recovery does not use it.

## Native certificate candidates

A Quadrants candidate searches a bounded six-dimensional separating ray with
uniform sample-constraint penalties. A strict polar correction changes the
certificate, not the input load. A separate native triangle subdivision bound
then checks the continuous support superset. CPU and CUDA both produce a ray
and a valid continuous certificate for component1; neither certifies the
feasible component0.

| Native backend | Search evaluations, incompatible component | Target margin | Continuous polar bound | Visited subtriangles |
|---|---:|---:|---:|---:|
| CPU | 79 | 2.006010628e-3 | -2.095700773e-4 | 52 |
| CUDA | 42 | 2.045878843e-3 | -1.203060943e-4 | 52 |

A depth limit2 deliberately leaves the certificate unresolved. Limits6,10,14
resolve this case. Exhausted depth, queue or work budgets return unresolved;
they never establish physical incompatibility. These kernels remain
experimental: production integration, failure compaction, broader negative
controls and fallback profiling are still required. Their evidence does not
establish general feasibility detection for arbitrary local normal fields.

## Aggregate-wrench modeling decision remains open

The same collision pair's total original force and moment are feasible on the
same exact geometry, samples and circular cones: the independent SOCP returns
normalized force/moment errors1.11e-15/1.20e-14. This is a feasibility witness,
not the selected profile optimum, a full elastic reference or production
acceptance. Preserving that pair total would require an explicit aggregate
partition law and would change the current per-original-component contract.

The user has been asked to choose between explicit pair-total partitioning and
continued per-component acceptance with certified physical failures counted.
No answer is presumed. The current implementation keeps the existing original
component wrench checks. A normal grasp cannot be silently discarded, clipped,
assigned a larger radius or called unsupported to complete the goal.

## Cooperative scheduling and remaining work

Experimental cooperative v7 passes16 CUDA regression cases and measures a
matched B32 intrinsic pipeline of9.00573ms. A unified v8 implementation passes
the same16 CPU and16 CUDA cases; a trust-radius v9 candidate passes seven
targeted cases on each backend. These are diagnostic scheduling profiles and
correctness checks, not live four-scope throughput or a plateau. Production
still uses scalar fit/commit/scatter.

The portable actual-producer check is
`bash research/rigid_stress/scripts/check_nonconvex_producers.sh cuda`, with an
explicit new `RIGID_STRESS_RUN_DIR` inside an allocated node. It covers
primitive Box, authored U decomposition, U SDF and closed-cavity SDF, three
environments, rotations, actual mu0.3/0.7/1.2, slip/release and partial reset.
Its independent CPU elasticity receives native assembled loads; the saved
fixture regressions supply independent load fitting. An initial invocation
from the wrong directory failed before numerical execution and is retained.
The four-case portable CUDA run completes all192 environment transitions and
192 CPU full-field comparisons with unchanged source hashes.

Required work still includes the modeling decision above, production continuous
diagnostics, wider actual source/refinement handling, measured source/history
reuse and scheduling selection, followed by repeated matched rigid-only,
recovery-only, live rigid+stress and policy+rigid+stress measurements at1024,
2048 and larger batches. Numerical failures, grasp outcomes, resets, CPU
captures, tails, memory and required fallback cost must be included. No
large-batch acceptance or throughput plateau is claimed here.

GPU checks use RTX PRO6000 Blackwell Server Edition,97,887MiB, driver595.71.05,
Xeon Platinum8562Y+, Genesis1.4.3, Quadrants1.3.3 and container1_26. Final CPU
checks use AMD EPYC9454. Test durations include isolated compilation and are
not performance measurements.

The [publication manifest](../evidence/20261011-nonconvex/native-wrench-diagnosis-checkpoint-v1/manifest.json)
preserves the full source epochs, exact step602 input, independent feasibility
witnesses/certificates, CPU fields, successful and failed regression logs,
native certificate candidates and launchers. Dependency caches and isolated
Clarabel binaries remain data-side.
All1179 publication artifacts (13,547,466 bytes) and their data-side originals
pass SHA256 verification on an allocated CPU node. Manifest SHA256 is
`ba944e405154c98cf2b15794982bd9eb24de93a0a03559febc9def25898c5e8d`.

Two offline continuous-bound runs recorded script/output hashes after
execution; they do not have contemporaneous before/after source records. The
offline bounded search, inspection and extraction likewise archive their
source/output without such a closure claim. Closed native CPU/CUDA ray-search
and continuous-bound runs independently reproduce separation from the exact
input. The first publisher rejected the offline records' missing after files;
its unchanged program and failure log are preserved. The corrected manifest
explicitly identifies each evidence scope instead of manufacturing closure.
