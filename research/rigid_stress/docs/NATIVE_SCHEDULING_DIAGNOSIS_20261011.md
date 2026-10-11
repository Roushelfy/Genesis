# Native source coverage, scheduling and first-failure diagnosis

Source base is `9db33909a53563791719e7cfe93759e29eb48aaa`, following the
[declared pair-support implementation](NATIVE_PAIR_SUPPORT_PROGRESS_20261011.md).
The active nonconvex goal remains incomplete. Scheduling candidates below are
experimental; complete large-batch acceptance and four-scope measurements are
still required. Final numerical budgets and the finite-footprint law remain
unchanged throughout these comparisons.

## Actual native collision producers

Four short native scenes complete sixteen steps in three environments each:
primitive Box, an exact authored three-Box decomposition of connected U material,
nonconvex U SDF, and a closed-cavity SDF. The material mesh has six, forty-two or
156 tetrahedra, with arbitrary boundary triangle winding. The decomposition
test covers authored pieces; it does not establish general automatic convex
decomposition acceptance or permit internal piece faces as material boundaries.

Each scene varies actual combined contact friction through `0.3,0.7,1.2`, moves
and slides the contact, releases env0 on steps4/5/12, partially resets env0 on
step8, and applies distinct rotations and translations. Each step checks every
original component force/moment, the assembled nodal wrench and all three
environments' complete residual, full tensor/von Mises and global peak against
independent CPU FP64 elasticity. The CPU oracle receives native assembled nodal
loads here; independent CPU geometry/profile/load fitting remains covered by
the separate saved-fixture regressions.

| Actual producer | CPU full fields | Maximum normalized F / M error | Maximum complete residual, N |
|---|---:|---|---:|
| Box | 48 | 6.082731e-9 / 7.334946e-9 | 6.231112e-18 |
| Authored U decomposition | 48 | 8.280183e-9 / 7.033588e-9 | 1.111635e-14 |
| U SDF | 48 | 8.632173e-9 / 6.563046e-9 | 6.492052e-15 |
| Cavity SDF | 48 | 5.442209e-9 / 3.660852e-9 | 6.495574e-16 |

All192 environment transitions and CPU fields pass. These small source-coverage
fixtures use Q10, squared fans and s2 paths with explicitly declared6mm disks
and1mm projection bounds. They are neither throughput workloads nor physical
mesh/integration convergence experiments. The egg continues to use s8 paths.

Failed controls are preserved: the first launcher omitted a newline; the
second used unsupported fixed target roots; the third incorrectly compared
a global-origin moment using only the original-point moment budget. The
corrected global check propagates the independently accepted component errors
as `sum(delta_M_p + |p| delta_F)` plus operand-based FP64 summation roundoff.
Every original-point `1e-8` force/moment check remains unchanged.

## Matched experimental scheduling

The same B32/seed623101 condition bank, 600 warmup steps, twenty synchronized
snapshot repetitions, FP64, Q10 squared fans, s8, mesh64, GJK patches and
explicit pair support are used across candidates.

| Candidate | Intrinsic pipeline, ms | Fit kernel, ms | Component commit, ms | Nodal scatter, ms |
|---|---:|---:|---:|---:|
| Native scalar | 157.95 | 134.245 | 9.904 | 9.981 |
| Fit warp v2 | 29.25 | 5.707 | 9.895 | 9.993 |
| Fit v2 plus sample-parallel commit/scatter | 10.01 | 5.722 | 0.265 | 0.265 |
| Symmetric Hessian and single-lane direction v4 | 9.04 | 4.824 | 0.263 | 0.265 |

The sample-parallel candidate and symmetric v4 each pass sixteen CUDA cases.
The former also completes the two original B16/B32 1280-step validation
trajectories:61,440 environment transitions,432 independent CPU full fields,
and56 scheduled saved-state resets. Snapshot nodal differences are at most
`2.94e-15 N`; peak differences are at most `7.55e-9 Pa`. These measurements
establish synchronized scheduling profiles and intrusive correctness, not
live throughput or a measured plateau. No candidate is selected in production
by this checkpoint.

Symmetric v3 aborted in the Quadrants compiler when broadcasting a boolean;
v4 explicitly broadcasts a `u32` predicate and passes. The failed source and
compiler output remain available.

## Exact first B1024 failure

B1024/seed623201 warmup fails. A snapshot taken when the scene reports the
error at step520 has no current failed contact, and scalar replay of that
snapshot passes. That later replay does **not** establish first-frame
feasibility or a warp-specific bug. Kernel-boundary tracing instead locates
the first failure at step517, env624, original contact0: source association
and integration succeed, but fitting reaches its128-evaluation budget.

Replaying the exact saved first frame gives the following normalized force
errors. Original forces, points, radii, friction and Q10 samples are identical.

| Exact first-frame fit | Evaluations | Force error | Result |
|---|---:|---:|---|
| Scalar, fixed1e-12 ridge | 128 | 3.954825e-7 | Numerical failure |
| Warp v2, fixed1e-12 ridge | 128 | 3.954604e-7 | Numerical failure |
| Scalar, fixed1e-12 ridge | 512 | 2.531552e-8 | Numerical failure |
| Warp v2, fixed1e-12 ridge | 512 | 2.531893e-8 | Numerical failure |
| Warp v5,1e-16 ridge | 35 | 2.274691e-9 | Accepted |
| Warp v6, unregularized Newton | 34 | 6.346916e-11 | Accepted |
| Independent CPU convex dual/root polish | see raw source | 1.107795e-12 | Accepted |

CPU moment error is `5.645137e-12`, minimum compression zero, and maximum
local cone excess `3.39e-21 N`. This establishes feasibility of the exact366
sample cloud and diagnoses numerical stagnation, rather than sampling or
continuous-model incompatibility. Native unregularized moment error is
`7.17e-12`. An adaptive candidate keeps the unregularized step when its
factorization and descent direction are valid, retaining regularization for
singular/invalid systems. Its wider validation remains pending here.
The internal tolerance remains9e-9; no final acceptance is relaxed.

## Immutable evidence and remaining work

The [publication manifest](../evidence/20261011-nonconvex/native-scheduling-diagnosis-checkpoint-v1/manifest.json)
contains653 artifacts (20,358,707 bytes), including successful and failed
controls, exact first-failure fixture, source snapshots, logs, condition banks,
CPU fields and profiling. All published artifacts and data-side originals pass
SHA256 verification on an allocated CPU node. Manifest SHA256 is
`e9211b5f831534397a69be536c2852e11dc60027606745b1a7614111a669cda4`.
The89MB later-frame snapshot remains data-side with its SHA256 in the manifest;
the small exact first-failure capture is published in full.

Hardware remains RTX PRO6000 Blackwell Server Edition,97,887MiB, driver595.71.05,
Xeon Platinum8562Y+, container1_26, Genesis1.4.3 and Quadrants1.3.3. Closed
scheduling/diagnosis runs have matching source hashes before/after execution.
The early failed launcher controls precede a closed numerical run.

Next validate and integrate the fitting repair, then retain cooperative
scheduling only after matched end-to-end evidence. Port actual producer
coverage into the normal test path, extend source/refinement and continuous
failure diagnostics, and complete repeated four-scope1024/2048/larger-batch
measurements with outcomes, tail work and memory. Applicable high-return
profiling candidates remain; no completion or plateau is claimed.
