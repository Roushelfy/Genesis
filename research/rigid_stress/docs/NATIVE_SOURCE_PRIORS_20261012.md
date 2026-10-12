# Source-prior aggregate model and policy807 audit

This is an **unselected offline Quadrants model experiment**, based on
`cad0c46c`. Production modules and componentwise acceptance remain unchanged.
It addresses a limitation of the earlier aggregate experiment: total force
and moment can vanish while local source pressures still produce stress.

## Declared alternative load law

Keep every original force application point, source force/moment, physical
footprint, radius and actual destination friction. For source i and its
positive native area/profile weights α_iq, define

$$P_i=\sum_q\alpha_{iq},\qquad P=\sum_iP_i,\qquad
w_{iq}=\alpha_{iq}/P,\qquad g_{iq}=\alpha_{iq}F_i/P_i.$$

Fit local Coulomb tractions by minimizing

$$\sum_{iq}\frac{\|f_{iq}-g_{iq}\|^2}{2w_{iq}},$$

subject to the **complete aggregate original wrench**, expressed about a
declared anchor, and every actual local material-normal/friction cone.
This explicitly permits redistribution between a pair's sources and changes
the conserved object from individual point components to the aggregate.
It does not silently substitute for the existing componentwise law.
Original forces/moments are immutable priors and targets; every physical
source remains in the bound support even when only one working aggregate
constraint is solved. The isolated working original point is retained and
its moment offset is included in the final original-point wrench check.

The dual uses source density prior h_i=F_i P/P_i. Original source magnitude,
rather than total force alone, supplies its scale and error normalization;
nonzero self-balanced loads therefore survive a zero aggregate wrench.
Source profile integrals/forces are snapshotted into typed Quadrants scratch
before fitting, avoiding reads from a fit workspace that is being overwritten.
The prototype generalizes the group reduction from two sources to arbitrary
group size. All fitting, source-prior reduction, signed P2 scatter and native
elastic recovery execute in Quadrants. CPU numerical work is an independent
offline oracle.

## Same-cloud and full-field evidence

Four inexpensive low-full-shell cases, each at three different actual
friction settings, pass CPU and CUDA: **24 full-field comparisons**.

| Input | Actual μ values | Approximate peak at original μ |
|---|---|---:|
| Saved step602,2 sources |0.447395/0.492135/0.536875|20,064.164 Pa|
| Saved policy876,2 sources |0.519663/0.571629/0.623595|16,792.564 Pa|
| Self-balanced compression,2 sources |0.800/0.880/0.960|12,957.784 Pa|
| Captured B1024 policy807,3 sources |0.572925/0.630217/0.687509|7,479.680 Pa|

The compression case has exactly zero total force and anchor moment.
It retains two opposing0.05 N source resultants and nonzero stress. The
newly captured policy case contains three physical sources; none is discarded.
Both original physically incompatible point-component fixtures become
feasible under this explicitly different aggregate conservation model.

The CPU fit oracle implements an independent convex-dual trust-region/root
solve and uses no native fitted forces or duals. It shares the native
geometry/profile cloud, which is stated rather than treated as an independent
footprint audit. CPU elasticity assembly, full stress tensors, von Mises,
global unaveraged peaks and complete equilibrium including all six gauge rows
are independent. Maximum normalized original-point aggregate wrench error
is7.318e-10, complete residual budget fraction0.045836 and native/oracle
peak difference3.759e-7 Pa. Local compression and actual friction constraints
pass. Final1e-8 wrench, complete residual and stress budgets are unchanged.

This evidence does not establish a live model, full lifecycle/reset matrix,
policy rollout, arbitrary enabled-moment matrix or throughput benefit.
Those checks and explicit production/API model selection remain necessary.

## CPU/GPU sample-count differences

The native surface-fan counts still differ near certain source positions:
step602's second source has483/578 CPU/GPU points, policy876556/710.
Independent face/source cloud matching at1e-12 m shows the GPU points without
a matching CPU point carry respectively2.147e-16/3.669e-16 of source profile
mass, and8.473e-19/1.260e-18 N of fitted sample-force magnitude. Per-face
profile-integral L1 differences are at most7.947e-16 relative in the three
audited clouds. Every point is retained. The exact floating-point branch
creating the count difference is not established, and no numerical cutoff
or production sample deletion is introduced. Same-cloud full fields and
cross-backend peaks pass.

## Original B1024 policy failure and stage correction

An untimed `cad0c46c` policy run on RTX PRO6000 Blackwell Server Edition
preserves environment6 and all32 copied failures, index6 modulo32, at
call1,408/tick807:600 warmup steps plus808 steps in the first timed repetition.
Unsupported external/coupling-load predicates are false. The failed raw
contact2 is part of a three-source group, with original R0.006905288846 m
and actual μ0.5729245005. Native continuous proof uses164 search evaluations
and329 visited nodes, bound−2.817002e-4 and target margin1.881115e-3.
An independent clipped-boundary/arc-Lipschitz CPU audit over every selected
triangle intersecting any unchanged source disk gives upper bound−5.499414e-4.
No native subdivision or finite search witnesses enter that CPU proof.

This strengthens the diagnosis of the existing point-component law; it is
not legal-grasp recovery or an unsupported-grasp classification. The
source-prior aggregate field experiment on the same preserved inputs passes
separately. No policy-throughput number is obtained from either failed run.

**Correction:** the preceding checkpoint described policy failure as a
warmup abort. Its original traceback points into the timed repetition;
the new capture identifies the exact phase above. The underlying failure,
absence of valid policy throughput and original archived hashes are unchanged.
The older manifest's phase wording is preserved as evidence and superseded
by this correction.

## Reproducible evidence and next work

The [source-prior manifest](../evidence/20261011-nonconvex/source-prior-checkpoint-v1/manifest.json)
maps418 artifacts to123 unique content-addressed blobs/3,340,101 bytes,
SHA256 `0981aa9a13f12fde0898520120dbe7ec80e636b1f64205c0dcf9aaf0d19dd0b9`.
Each entry supplies its original run path, blob path, size and hash, so the
frozen source/launcher/fixture/results tree can be reconstructed without
duplicating identical source files. Allocated CPU publication verifies every
source closure, original and archived artifact. Failed capture-template and
missing-generator launch attempts are retained alongside corrected epochs.
Hardware, exact native numerical source, isolated candidates, independent
oracles, actual friction changes and complete field/cloud outputs are retained.
No high mesh or physical integration-convergence study ran.

Required next work includes a declared production model, legal changing
contact/reset/release/rotation acceptance, general boundary integration,
complete live and policy measurements, and applicable performance candidates.
Fresh B1024 profiling identifies surface-fan profile generation as the dominant
cost, so launch pruning/shared geometry work is being evaluated. This
checkpoint does not complete the active goal or establish a throughput plateau.
