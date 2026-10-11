# Contact-patch input, finite support and cooperative fitting

This continues the [initial source/profile checkpoint](NATIVE_SOURCE_PROFILE_PROGRESS_20261011.md),
source `1fe4f804d19425772cdb6c6faf53326f2fbf2a00`. The active nonconvex goal
remains incomplete. The evidence below separates a defective collision normal,
longer legal-contact integration failures and a promising scheduling candidate.
It establishes neither large-batch throughput nor an optimization plateau.

## Collision input control

A capture immediately before the ordinary recovery call at tick 591 verifies
the reference/collision vertices and transformed witness to approximately
`6.43e-17 m` and `1.94e-18 m`. The generated normal is a supporting direction
for the finger, but its egg witness lies `37.9108626341 micrometres` from the
nearest support plane of the actual 42-vertex collision mesh. This supports
the diagnosis that recomputing the perturbed contact after unrotating its
portal does not preserve the actual supporting facet. GJK and a tighter MPR
portal tolerance alone did not repair that input.

An explicit GJK direct contact-patch control uses the existing collider path
instead of perturbed multi-contact backmapping. The example and online probe
now expose `--collision-method gjk --contact-patch on`; `auto` preserves the
original collider option. This is a declared collision-input/model comparison.
Rigid-only and stress-enabled measurements must select the same producer.
Recovery does not change rigid forces or select a different collider at runtime.

The direct-patch B=16, seed 623001, 640-step control passes every checked
contact wrench and five independent CPU FP64 elasticity captures, including
80 complete stress fields and all gauge rows. It includes partial resets and
changing combined friction. Longer public-example runs reveal additional
integration failures and are retained in full:

| Trajectory | Requested steps | Completed steps | Failure |
|---|---:|---:|---|
| B=16, seed 623001, initial direct-patch control | 640 | 640 | None |
| B=16, seed 623001, public CLI | 1280 | 1076 | env 13/contact 1; R=0.004862864397824547 m, mu=0.6054935315460935 |
| B=32, seed 623101, public CLI | 1280 | 594 | env 18/contact 1; R=0.0050082488050716654 m, mu=0.7900954592846008 |

The two partial trajectories pass seven and four preceding CPU captures,
respectively; they are not completed acceptance or throughput measurements.

## Distinguishing support and solver failure

Replaying the exact point, anchor, force, radius and friction at ticks 594 and
1076 shows that increasing the evaluation budget from 128 to 2048 leaves
normalized force errors near `0.001652` and `0.003918`. Scalar and cooperative
fit schedules exhibit the same plateau. Increasing the intrinsic edge-port
subdivision count from eight to sixteen does not repair tick 594.

Independent inner/outer polygon-cone LPs reveal CPU/CUDA support differences.
Some LP witnesses put almost all force on samples with area/profile weights
around `1e-25`; their force-to-weight ratio is around `1e24`. Consequently,
an LP's feasible status on that cloud is insufficient evidence of a well
conditioned numerical fit or resolved finite integration. Reconstructing a
closest edge point in Cartesian coordinates and taking another cross product
can turn an exactly zero-area fan into a positive-area sliver. The next source
state stores the closest barycentric coordinates directly and derives the
subtriangle area from its opposite coordinate. This removes that extra
round-trip, but very small positive coordinates from geometric projection
remain visible and are not silently thresholded away.

An independent six-component continuum audit bounds each real face by its
Euclidean R-ball section and circumscribed Coulomb cones. This is a superset
of the intrinsic footprint. Neither failed fixture has a separating certificate
in that outer model. Inscribed disk/cone LPs, including a one-percent contraction
toward each polygon's interior, find constructive wrenches with errors below
`2e-16` and true cone violations below `2e-17`. Those inner results concern
Euclidean sections; they do not by themselves certify intrinsic paths.

Local integration reparameterization supplies constructive native fits on
verified intrinsic support. Replacing the two unit-square Gaussian coordinates
by their squares, with both Jacobians included in positive finite-area weights,
preserves the physical radius, profile, force and friction. Q10, eight edge
subdivisions, FP64 and the `9e-9` internal stopping budget are retained:

| Saved input | Evaluations | Normalized force error | Normalized moment error |
|---|---:|---:|---:|
| tick 1076, squared coordinates | 17 | 4.49e-12 | 3.11e-13 |
| tick 594, squared coordinates | 20 | 5.02e-9 | 2.43e-9 |
| tick 1076, cubed coordinates | 31 | 1.45e-10 | 1.81e-11 |
| tick 594, cubed coordinates | 26 | 2.87e-15 | 1.86e-15 |

Both scalar and warp candidates pass these saved wrench fits. This is a
local finite-contact integration experiment, with no high-mesh or material
quadrature convergence study. The squared-coordinate source state has entered
full CPU/CUDA stress/scatter regressions and a longer B=32 trajectory. Those
checks remain pending in this checkpoint; final wrench, complete equilibrium
and stress acceptance have not been relaxed. A new matched baseline is
required after selecting any integration change.

## Cooperative fitting candidate

A pure Quadrants CUDA warp candidate evaluates each contact's samples and
reduces its six-dimensional resultants and Hessian cooperatively. It retains
the original FP64 law, Coulomb projection, stopping budget and final assertions.
Seven CUDA regression cases pass, including independent CPU FP64 stress and
complete residual checks. It is still an experimental scheduler.

An initial matched B=16 profile uses the saved condition bank, GJK direct
patches, the original unsquared Q10 fan rule, dt=0.01 s, one substep, contact
block size four, max output, no history, 180 warmup steps and 20 repetitions
on the same allocated GPU. Snapshot preparation passes before both timings:

| Intrusive stage | Scalar fit | Warp fit |
|---|---:|---:|
| Complete intrinsic contact pipeline | 125.830 ms | 32.371 ms |
| Inverse application | 0.182 ms | 0.174 ms |
| Complete residual | 0.065 ms | 0.062 ms |
| Global peak | 0.070 ms | 0.066 ms |

The contact-stage ratio is approximately 3.887. This is a single frozen-stage
comparison, not repeated live or policy throughput evidence. The warp kernel
profile assigns 42.15% to nodal scatter, 35.54% to wrench commit and 12.59%
to fitting. Their per-block mean costs are approximately 6.504, 5.485 and
1.943 ms. Scatter/reduction, commit reuse/fusion and large-batch layout remain
actionable; the candidate is not selected solely from this profile.

## Reproducible evidence and remaining work

The [complete capsule manifest](../evidence/20261011-nonconvex/online-checkpoint-v5/manifest.json)
contains 535 artifacts (15,285,943 bytes), including source snapshots,
before/after hashes, all partial trajectories, original failure inputs,
LP diagnoses, source geometry audits, candidate code, commands and profiler
logs. Every original and copied artifact passes SHA-256 verification. Manifest
SHA-256 is `ae66e6e6e70ca95fbaf622f603d6dcffa39d5fc47621dafe54c53bfcbdd15166`.
The initial archive metadata assertion also remains recorded: the scalar
profile generates/saves the bank and has a null input-file hash; the warp
profile loads it. Verification checks the save/load path, actual bank hash
and equal declared configurations rather than equating those two metadata
fields.

Hardware is RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver
595.71.05, Quadrants 1.3.3 and Genesis image `1_26`; CPU oracles run on allocated
compute. The unchanged low full shell has 42 collision vertices, 810 P2 nodes,
2430 displacement DOFs and 480 tetrahedra.

Required next work includes complete validation of the revised finite
integration, longer changing contacts and reset isolation, general source
coverage/refinement, applicable measured contact optimizations and repeated
four-scope B=1024/2048/larger throughput. No large-batch rate or completion is
claimed by this checkpoint.
