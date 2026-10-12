# Native shared-feature stability and failed-fan continuation

This checkpoint preserves the Quadrants production path, physical contact
inputs, public defaults and final acceptance budgets. It fixes repeat recovery
at a shared surface edge and a numerical stopping failure on an unchanged
refined fan. It also records wider rollout results with explicit failed-episode
accounting. The active nonconvex goal remains incomplete.

## Shared-edge repeat stability

The B2048/623102 profile preparation on core `688edabe` initially failed its
same-state comparison: env1747's peak changed from62112.032131 to61931.916738 Pa,
a180.115393-Pa difference. The original local witness/application point differed
by at most6.94e-18 m and frame quaternion by2.22e-16. Source5 selected face9 in
the original recovery and face6 after repeat transformation. These faces share
the same boundary edge, but the different port seed changed the finite profile;
the port distance changed by9.428695e-6 m and nodal load by0.000935005 N.
Both fits had small wrench errors. Looser fitting thresholds would not fix it.

Closest-face fallback now retains the first ascending-face representative
when the distance improvement and projected-point difference fit the existing
world-transform roundoff budget **and** topology proves the same shared edge
or vertex. Different sheets remain distinct. Known surface provenance and the
separate ambiguity checks are retained. No new geometric tolerance, radius,
friction, force or moment change is introduced.

The corrected B2048 preparation has nodal difference1.943e-16 N and peak
difference9.881e-9 Pa. The small exact before/after input is a regression asset,
SHA256 `113242d4e853c826cb9d0ea79547da36f0104d3d70b16db7c2736f25fa46afbd`.
Its tests compare physical integration weights by shared sample identity;
parallel packing order and negligible weights from roundoff-sized collapsed
fans do not define physical equality. The full-field test independently solves
three actual friction variants on each backend.

The corrected snapshot profile, before the continuation change, measures:

| Stage | Wall ms at B2048 |
|---|---:|
| Association and source capture | 0.45523 |
| Intrinsic contact pipeline | 39.05508 |
| Inertia relief and RHS | 0.25489 |
| Linear solve | 1.56502 |
| Complete residual | 0.65989 |
| Global peak | 0.45817 |

There is no pressure-subdivision timing in this run. This is a snapshot
profile, not an end-to-end optimization plateau. The contact pipeline is the
main actionable cost; a current-source baseline and finer subdivision remain
required.

## Continue genuinely numerical failures

Explicitly resetting the continuously certified B4096 env3092 failure lets
the original live trajectory reach another input at call1498/tick897/env645.
The two-source refined fan has1102 positive samples, with minimum normalized
weight2.838e-13. Its128-evaluation cold fit fails, while512/2048 cold budgets
both converge in155 evaluations. An inner polygon LP is infeasible and an
outer polygon LP is feasible; neither establishes true-cone infeasibility.
The recovered true-cone wrench proves this is a numerical stopping failure.

The native fitter now adds a bounded128-evaluation continuation only when a
refined fan still has `NUMERICAL_FAILURE`. It retains a finite existing dual
iterate, recomputes the same energy/gradient and counts all prior evaluations.
Successful constraints, sampled-infeasible constraints and certified physical
failures are not retried by this mode. The exact input completes in157 total
evaluations in the focused cold/continuation comparison. The full production
pipeline also counts its earlier cooperative/scalar/integration work.

No load-model, quadrature, precision or acceptance change is made by this
continuation. Internal fit tolerance remains9e-9; final normalized force/moment
acceptance remains1e-8, full residual including six gauge rows remains
`max(1e-11 N,1e-8 ||rhs||)`, and full tensor/von Mises/peak consistency remains
rtol1e-4/atol1e-3 Pa.

An independent CPU primal SOCP followed by true-cone dual root correction
qualifies the reference without native multipliers or recovered forces.
CVXOPT's primal status was `unknown` and SciPy's root success flag was false;
these flags are preserved. Direct final checks give force error7.783e-14 and
moment error3.059e-15, nodal native/reference difference3.289e-12 N, complete
residual5.349e-12 N (0.004592 of budget), and peak difference1.190e-6 Pa.
Different dual vectors reflect the near-degenerate dual; accepted primal fields
agree. Production imports neither CVXOPT nor SciPy for recovery.

The regression asset contains exact captured inputs and the independently
published SOCP/root initializer, SHA256
`ee2ba186c3e0c1debdbd034cff1bd1f7718b9359499475b46d4453a452a6c57d`.
The CPU reference first attempts its independent cold fit, then polishes that
independent initializer only if needed, and verifies the actual gradient.

## Qualification and wider rollout

Current targeted suites pass **15 CPU cases in232.96 s and15 CUDA cases
in364.53 s**. Eight field cases with three actual friction variants per
backend give **48 independent CPU FP64 full-field comparisons**. Largest
normalized wrench error is7.387e-10, largest complete residual budget fraction
is0.045857, and largest peak difference is8.130e-6 Pa. Lifecycle, source changes,
partial resets, isolation and strict physical failure semantics pass.

Earlier complete canonical suites are retained:64 passed/2 failed on CPU and
65 passed/1 failed on CUDA. All pre-existing cases passed. The new-test failures
were a CPU zero-copy oracle view modified by partitioning and comparisons of
raw packed order/exact sample count. Copying the reference inputs and comparing
physical sample weights fixed those tests; final acceptance budgets did not
change. The current15-case suites are reported separately from those earlier
full matrices.

The application driver checks errno after every completed live/policy step
inside timing. Only `INVALID_STRESS_LOAD` with continuous certificate status35
or36 becomes an explicitly failed episode and selected-environment reset.
Numerical, sampled and geometry failures remain fatal. The driver retains the
first diagnostics, verifies peer state isolation and reset stress, subtracts
invalid transitions from effective throughput, and counts the failed reset.
Native public strict error semantics remain unchanged. This response does not
turn an incompatible contact into successful recovery.

| B4096/623103 scope and source | Effective env-step/s | Batch step/s | Invalid env-steps/repetition | Resets/repetition |
|---|---:|---:|---:|---:|
| Live, core688 plus canonical/continuation | 41152.364 | 10.046966 | 1 | 4644 |
| Policy, core688 before these repairs | 42903.298 | 10.474438 | 0 | 4643 |

Both use600 warmup steps and three1280-step varied repetitions, low complete
hollow shell480 P2 tets/810 P2 nodes/2430 DOFs, Q10/squared fans/eight edge
subdivisions, FP64, dt0.01/one substep, GJK patch on, explicit body source
priors and native face frontier. All required updates, retries, observations,
controller/policy work, history, post-step checks and resets are timed.
The live invalid event is env3092/tick834 in each repetition; env645 now
recovers. Repeat live rates are41112.427/41101.810/41243.155. Native allocation
is12144479838 B; free device memory80586735616 B. Warmup full residual consumes
at most0.008765 of budget. Actual hardware is RTX PRO6000 Blackwell Server
Edition,97887 MiB, driver595.71.05, container1_26, Quadrants1.3.3, Genesis1.4.3.

The policy is26→128→128→7, FP32, seed99173 and residual action scale1e-4 rad.
Its warmup FP64 control-reference error is1.708e-11 rad within1e-9. It measures
inference rollout, not RL training. The two rows use different source epochs
and trajectories and do not define a policy overhead or rigid slowdown ratio.
The policy trajectory has no certified failures; the live trajectory's
certificate means legal-grasp robustness still needs the producer review below.

## Unresolved producer input

Offline transforms verify the original collider force/normal/witness mapping
and stress-side sign handling. At env3092, geom14/egg24 has material/collider
normal dot−0.999876 and negative material compression, although compression in
the collider frame is positive. Projecting the exact convex vertex sets onto
the reported normal gives46.436 mm separation travel in its force direction
and0.124003 mm in the opposite direction, against reported penetration0.116441
mm. Geom15/egg24 has0.095296 mm travel matching reported penetration.
An exact captured-pose detection reproduces the anomaly with patch both off
and on. The trace shows it already exists in the EPA face and original
witness pair before patching. This does not yet identify a qualified generic
rigid fix. No normal, original force, radius or friction is changed here to
make the stress certificate disappear.

## Evidence and next work

The [archive](../evidence/20261011-nonconvex/native-stability-continuation-checkpoint-v1/manifest.json)
contains1508 entries,279 unique blobs and37630220 stored bytes. SHA256:
`e9a8de3405a5d650e00713424d3a3bd18882d8e00fe6dd9d87f137fb67318a05`.
Allocated CPU publication verifies every original/stored/decompressed hash,
all48 fields and both rate/accounting summaries. Failed oracle, test and
diagnostic attempts, immutable source closures, launchers and captures are kept.

Next work is the generic producer diagnosis, broader legal contact/seed
qualification, current matched four-scope baselines, grasp outcomes and
per-environment tail work, detailed contact profiling and further measured
optimization. No plateau or goal completion is claimed. No high physical mesh
or physical integration convergence experiment was run.
