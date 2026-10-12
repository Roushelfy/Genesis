# Native wrench-line fan integration repair

This repairs the two single-source wider-seed failures recorded in
[the combined baseline](NATIVE_BODY_BASELINE_20261012.md). Production
calculation remains Quadrants. Successful existing contact fits retain their
samples and forces. Native public options/defaults and final force, moment,
full-equilibrium and stress budgets are unchanged.

## Integration and lifecycle

After the existing verified-disk retry, a still unresolved surface-fan group
is repartitioned using each physical source's original transverse wrench
line. For anchor a, force F and anchor moment M, the line is
`a + cross(F,M)/dot(F,F) + t F`. Its intersection with a candidate face
selects the fan's interior chart center; a closest-triangle projection keeps
that center on the same face. Pure moments retain the ordinary chart.
Parallel moment components remain in the full six-component fit.

This changes integration charts, not original force application points,
physical centers/radii, local friction, normals or surface topology. Each
triangle is still partitioned into three area-weighted Q10 fans with the
same squared coordinates and analytic profile. Candidate pruning continues
to use the original anchor; every accepted point still passes the original
certified intrinsic path/radius test. No distant sheet or point-load
replacement is introduced.

All constraints sharing a changed support are refitted, including previously
successful member constraints. Body source priors retain original forces and
moments and refresh only their quadrature mass normalization. Typed
`fan_refined` scratch is cleared by ordinary profile construction, recorded
in native state/allocation introspection and propagated to integration retry
counts. Evaluation counts include the earlier failed work. The general
failure semantics and continuous incompatibility certificates are retained.

The benchmark load-model tag adds `failed_wrench_fan_partition`: the
discrete integration path changed. Subsequent optimization comparisons need
this repaired baseline; the old model's aborted runs are not speed baselines.

## Numerical qualification

The exact captured single-source inputs are small normal regression assets:

| Fixture | Radius, m | Actual mu | Repartitioned samples | Native cold evaluations |
|---|---:|---:|---:|---:|
| Varied B2048/623102 env1914 | 0.004095322555 | 0.564764758 | 633 | 11 |
| Varied B4096/623103 env2174 | 0.004086578833 | 0.709940432 | 594 | 16 |

Independent true-cone inner LPs are feasible after the repair. Independent
CPU convex dual fits agree; native diagnostic original-point force/moment
errors are 1.137e-13/9.036e-14 and 4.193e-12/5.900e-12, respectively.
Production retains128 evaluations and internal9e-9; increasing the old
iteration budget to2048 had not repaired these original clouds.

The native suite passes **64 CPU cases in571.67 s and64 CUDA cases in884.44 s**.
It includes both new recovered fixtures, old contact/provenance/nonconvex
checks, continuous physical failure semantics and contact/reset/isolation
lifecycle. Six field fixtures with three actual mu variants on each backend
give36 independent CPU FP64 complete-field comparisons. The largest recorded
normalized force/moment error is7.387e-10; full residual including six gauge
rows consumes at most0.045857 of its unchanged budget; maximum global peak
difference is3.694e-7 Pa. Tensor/von Mises fields, cones, signed P2 scatter,
original points, retained local priors and self-balanced compression pass.

## Actual wider-seed live workload

The original independent B2048/seed623102 workload now finishes three
1280-step live repetitions, each with2323 actual per-environment resets.
All2048 failure counters are zero, with no invalid environment steps or
scatter overflow. Rates are36212.520/36302.547/36270.835 env-step/s;
aggregate **36261.929 valid env-step/s,17.706020 batch step/s**.
Warmup full residual budget fraction is at most0.009643469.

Hardware is RTX PRO6000 Blackwell Server Edition, UUID
`5c0d0cca-f892-19d7-d4f3-b68ab73a520b`, container1_26, FP64, dt0.01,
one substep, low full hollow shell480 P2 tets/810 P2 nodes/2430 DOFs,
Q10/squared fan/eight edge subdivisions. Native allocation is6108796510 B;
reported free device memory is89579323392 B. Updates, all contacts, fitting
retries, solve, residual, peak, controller and resets are within live timing.
This is one independent seed and one live scope, not a plateau or complete
large-batch four-scope acceptance.

## New B4096 limitation

B4096 passes its old single-source failure, then aborts on a different
two-source input in its first timed repetition. Untimed pre-integration
capture: call1435, workload tick834, env3092, status36, actual mu0.424924446,
two unchanged4.549869559-mm disks. Unsupported-input predicates are empty.
No B4096 throughput is valid.

An independent CPU outer proof covers every true planar boundary face
intersecting either Euclidean disk, a superset of the intrinsic footprint.
Circumscribed spatial and Coulomb polygons give an infeasible outer LP.
Its independently found unit separator has positive target margin0.241411
and continuous true-cone polar upper bound−4.967e-7 at128 sides. The native
unit ray is independently verified more strongly: target margin0.177199,
continuous polar upper bound−0.053762. Convexity of the polar on each affine
face makes the outer polygon vertex maximum a continuous bound; this is not
an inference from finite sampled infeasibility alone. The raw input and
source geometry/provenance/rigid physics still require review. Radius,
friction, force and acceptance are not altered to eliminate this diagnostic.

## Evidence and remaining work

[The archive](../evidence/20261011-nonconvex/native-wrench-fan-checkpoint-v1/manifest.json)
contains458 entries,146 unique blobs and9002251 stored bytes. Manifest
SHA256: `dc6fd849bb22424b38fe7527b4c55053a869579edba631e1fe9087f8bcfa92d1`.
All runs have unchanged source closures. Allocated CPU publication verifies
every original/stored/decompressed hash and all36 recorded fields. It keeps
both the successful B2048 run and the B4096 abort/capture/independent proof.
Archived drivers reproduce the native64-case CPU/CUDA suite, the integration
prototype, original fixture publication and the two live attempts.

The active goal is not complete. Explicit failed-episode accounting and
grasp outcomes, broader live/policy seeds, remaining valid four-scope scales,
tail work/memory, new profiling and remaining performance candidates need
completion. The contact pipeline still dominates the preceding baseline;
no measured plateau is claimed. No high physical mesh or physical
convergence experiment was run.
