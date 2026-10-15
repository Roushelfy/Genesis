# Moving references and original ground-supported inputs

The [complete physical-validation goal](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. Corrected moving holds now complete independent time-step,
feedback and native-replay checks. The actual original control history and
ground support are also captured and an initial0.1s prefix is independently
audited. Full approach/grasp/lift, physical contact/geometry/spatial tails,
remaining model sensitivity and verified physical accuracy/cost remain required.

## Wide-coordinate reference and independent qualification

The earlier extended-force-only producer715190 accepts58 states, then exceeds
its Newton budget at step59. Increasing force-contraction precision alone does
not complete the moving history. `wide_coordinate_franka.py` also keeps body
iterates in long double and assembles the final body residual directly from
nonlinear bulk, contact, gravity and consistent inertia. Constitutive law,
quadrature, material, geometry, friction and controls remain fixed. Energy,
tangent, IPC geometry and stress postprocessing retain their original FP64
paths. The internal Newton target is tightened to1e-10; independent external
body/joint/fresh-step budgets remain1e-9.

Jobs715396_0/1/2 reconstruct all three archived rejected time-step states.
Each recovers in one correction, with body residuals2.656e-11,6.483e-12 and
4.160e-11N. Complete tangent differences at the smallest perturbation are
2.994e-8,3.785e-9 and2.384e-7 relative; fresh robot-step checks also pass.

A separately tested initial guess uses the previous velocity to predict body
and robot positions. It is accepted only after collision-free endpoint-chord
and intersection checks. This changes the Newton starting point, rather than
the previous physical state, inertial predictor, equations or acceptance gates.
All280 tested predictions use scale1. Numerical qualification715594 compares
every original accepted state:22 coarse,48 middle and105 fine states, including
every saved array and all corner/quadrature tensors. The maximum all-corner
tensor difference is3.164e-11 relative. Rounding the wide coordinates to FP64
changes sampled raw corner peaks by at most2.826e-8Pa in those comparisons.

The complete80-step wide middle baseline715411_1 and predicted candidate715506_1
are additionally compared by715623. All80 states pass; maximum corner-tensor
difference is1.039e-12. Their recorded Newton correction counts are1190 and163,
respectively. These are reference-algorithm observations, rather than warmed
native throughput or a physical accuracy/cost range. The subsequently completed
three-level wide baseline passes its own independent280-state audit715424.
Qualification715624 compares all40/80/160 original-initial-guess states with the
predicted histories: maximum corner and quadrature tensor differences are
4.094e-12/5.885e-12 relative. All saved state arrays are retained and compared.

## Moving time-step and feedback evidence

The three predicted producers715506_0/1/2 reuse array-identical closures and
initial-velocity arrays, with the corrected captured COM twist and all nine
joint velocities. Independent audit715592 reconstructs all280 states at
2.5/1.25/0.625ms through0.1s, including body balances, contact pullback, all
joint efforts, fresh MuJoCo steps, objective tensors and raw extrema.

| Maximum adjacent change over common times | 2.5→1.25ms | 1.25→0.625ms |
|---|---:|---:|
| Full objective tensor |0.0110380%|0.00507289%|
| Elastic energy |0.0213616%|0.00907866%|
| Raw global peak |0.0114263%|0.00480007%|
| COM position, m |2.64887e-8|1.31651e-8|
| COM velocity, m/s |2.51935e-6|1.37299e-6|
| Angular velocity, rad/s |2.39162e-4|1.19976e-4|

This is a temporal diagnostic at one unresolved coarse spatial/contact setting.
It does not prove physical contact accuracy or original-history equivalence.

Corrected feedback audit715619 reconstructs all160 corresponding moving
rigid/deformable states. Its maximum resolved-load tensor change is0.0191970%,
peak change0.0189808%, energy change0.0381147%, per-patch force change0.0193966%
and moment change0.0295489%. Linear replay versus nonlinear body tensors differs
by at most0.00480821%. The deformable angular-momentum balance residual is at
most1.463e-12Nm; the rigid value is1.119e-11Nm. Final y angular velocities are
0.0954931/0.0954932rad/s. The original incorrect-reference-point pilots remain
excluded as described in [the preceding checkpoint](PHYSICAL_CAPTURED_MOTION_20261015.md).

Actual CUDA replay715621 checks both full-inertia and rigid-relief inputs at
steps1/40/80 for both body models:12 native solves. Complete raw native tensors
agree with independent FEM to1.056e-10 relative; maximum original residual-budget
fraction is0.02236. This verifies nodal replay, rather than finite-footprint
mapping or spatial/contact convergence.

Initial-motion comparison715630 uses the same closure arrays and physical
controls, comparing zero and captured initial velocities. Maximum tensor
differences are0.0318906% rigid and0.0318934% deformable, while orientation
differences reach0.00802732rad and angular-velocity differences0.0861214rad/s.
Small coarse stress differences therefore do not establish stationary motion.

## Original controller and ground-support provenance

Observer715620 adds read-only records of all690 macro steps:330 original
approach/grip/lift steps and360 added stationary-hold steps, at10ms with32
rigid substeps. Runtime control targets, caps and native radii are checked
against the archived condition bank and authored schedule. The original Torch
scalar/native-buffer FP32 grip and cap values are preserved explicitly.
All576 original captures and16173 NPY payloads across1165 NPZ archives remain
byte-identical to the parent capture.

The actual initial egg origin is`[0.65,0,0.031]`m and its COM is
`[0.65,4.51809e-19,0.0293639758]`m. Body and robot initial velocities are zero.
There are171 retained last-substep ground-contact records, first at macro1 and
last at macro180; observed opposing links are ground0 and fingers10/11, all
with effective friction0.6. Reported macro states are after integration;
listed source contacts are the last preintegration constraint evaluation.
Those time conventions are retained when comparing corresponding references.

## Original-prefix mechanics and the actual robot step

The new initial geometry retains all12 actual finger geoms, original full egg,
mass/material, initial robot pose and direct recorded control inputs. A large
rigid triangle supplies the original horizontal plane locally. Accepted ground
points must lie strictly inside it, and unmodeled non-body contact pairs are
rejected. No six-row held-body initializer replaces the actual initial support.

First attempt715639 accepts two steps, then the fresh robot-step check rejects
step3. The zero-effort affine response reports7.976e-15 while the actual step
has7.54715e-4 effort-equivalent residual. The original gate correctly detects
the inconsistency. `GroundedOriginalContact` now uses the actual MuJoCo step in
its robot residual and differentiates that response at the current applied
effort. The original inverse response only defines its effort-unit scale.
Same-state diagnostic715643 passes complete tangent checks to3.239e-13 relative
and recovers a fresh-step residual2.810e-15. Startup failure715642 is retained
separately; its copied wrapper initially passed an undeclared MJCF argument.

Producer715645 completes80 steps of the original first0.1s, including the
initial drop and real ground contact. Independent audit715654 reconstructs all
80 controls, ground/finger geometry, contact forces, body inertia/bulk balance
and actual robot steps. Maximum body residual is4.830e-11N and actual robot
effort residual3.857e-14. Ground activation begins at0.01375s. Final ground
reaction is approximately`[−3.70449e-11,7.18375e-12,−0.06405781095]`N; raw
global lower/upper stress bounds are22887.917910/22887.918238Pa. Same-macro
native COM difference is2.85541um at0.1s, an observed difference between these
declared contact/integration models, rather than a physical error bound.

Job715656 attempts all330 original macro controls,3.3s of approach/grasp/lift,
from the same initial state. It accepts865 states through1.08125s and rejects
step866 at1.0825s with body/joint residuals1.28074N/0.127709. Independent audit
715690 verifies every one of those865 saved states, controls and actual robot
steps: maximum body and robot residuals are9.884e-11/9.911e-11. It explicitly
records an incomplete requested history. Same-state retry715689 fails; further
small-step weak-mode tangent probes715702 show material step-size sensitivity:
the maximum body-block relative difference falls from0.05719 at probe scale
1e-10 to0.001155 at3e-12, then grows as arithmetic noise becomes relevant. The random
complete-field check alone does not resolve this transition. Complete grasp/
lift remains unverified and is not replaced by the successful prefix.

## Retained failures and open spatial requirements

Narrow-band local level15, job715083_2, accepts25 dynamic states and rejects
step26 at1.0114686e-9N above the original1e-9N budget. Its full mesh, held
initialization, partial states and failed iterate are retained. Level18 remains
live. Wide-coordinate same-state spatial diagnostic715582 does not converge
below the complete original gate; its failure report also exposes a long-double
JSON serialization error. It has not qualified a completed finer reference.
Diagnostics715674/715682 retain the8433-tet same-state body residual5.121e-10N
but complete reduced residual1.01144e-9N. The internal bulk resultant/moment
remain near1e-16N/1e-17Nm, while one normal stencil has a measured
point-to-plane separation9.145e-10m.
One3.46945e-18m global-coordinate ULP changes a stencil force by1.019e-9N.
Evaluating translated precise stencil coordinates changes a force by up to
4.687e-10N without changing physical inputs. These measure contact-arithmetic
sensitivity; they do not prove the sole failure cause or a converged patch.
The earlier3577-tet
two-node concentration and280.5MPa raw peak therefore remain unresolved.

Feedback audit715593 fails only when serializing a long-double report after
its physical checks;715619 preserves raw NPZ fields and converts report scalars.
Initial-motion comparison715622 uses incompatible11-point linear and15-point
nonlinear quadrature weights;715630 correctly uses the pinned nonlinear
quadrature. Neither failed report counts as an accepted result.

Reproduce with `scripts/run_wide_articulated_franka.sh`,
`scripts/run_predicted_wide_franka.sh`, `scripts/audit_wide_temporal.sh`,
`scripts/audit_wide_articulated_motion.sh`, `scripts/qualify_articulated_precision.sh`,
`scripts/capture_original_protocol.sh`, `scripts/run_original_franka_prefix.sh`
and `scripts/audit_original_franka_prefix.sh`. Runtime configurations, exact
Slurm commands, package/hardware identities and before/after source hashes
remain under the existing physical-validation data root. Source snapshots
distinguish input code from patched executable code. Live work is excluded
from immutable checkpoints. The complete original A/B/C, geometry, model,
feedback, original-history and physically verified cost scope is retained.

The immutable `moving-reference-checkpoint-v1` contains26 terminal runs,
4790 file entries and1102 source-hash records. Every entry is independently
verified against its complete original runtime bytes. Manifest SHA256:
`f56f66c73f8b221412e7d078dd80030dd03aa95b0881c36310eaa0625e957742`.
The later three complete wide-reference/equivalence audits are published in
`moving-reference-equivalence-checkpoint-v1`:399 entries, manifest SHA256
`3dad9cc7da1b9b4945c7fea5882297dc6a71b8584407c7174b0ed20064d37feb`.
The seven subsequently finished original-history/spatial diagnostic runs are
preserved in `moving-reference-late-checkpoint-v1`:2182 entries, manifest SHA256
`0a016221b3b99534f2fa7d89377594a3850ad4b0b112e2a3493ac680d5ba4459`.
Its separate workflow journal records cancellation of the impossible whole-array
audits715100/715102 after level15 fails; the live level18 producer is preserved.
Other live spatial/U/egg jobs are excluded. These publications retain successful
prefixes and natural failures without claiming complete physical acceptance.
