# Native contact sampling ablations

These candidates use the low complete hollow shell, FP64/Q10/s8,
`body_source_prior`, qualified EPA correction and selected current-step
anchor-port reuse. Physical geometry, footprint law, friction and final
acceptance budgets stay fixed within each paired experiment.

## First-triangle loop bounds: rejected

The isolated candidate prechecks the original first-triangle plane and
endpoint containment. A plane-rejected segment executes zero adjacency
iterations. A contained endpoint executes the original first iteration. All
remaining cases retain the original bounded topology walk. No path or mask
definition changes.

CPU23 cases (299.13s) and CUDA23 cases (419.186s) pass. They cover source/prior
and lifecycle checks, U/cavity gap and hole exclusion, and intrinsic footprint
recovery. The48 recorded independent source-prior full fields have maximum
normalized wrench error7.386601836e-10, complete residual budget fraction
0.045856225 and peak difference7.964510587e-6Pa. Additional footprint tests
compare their complete fields with independent P2Shell oracles.

The B2048/623102 profile pair runs sequentially on the same RTX PRO6000
Blackwell GPU03ec8bc3 (97887MiB, driver595.71.05, container1_26,
Genesis1.4.3/Quadrants1.3.3). Each route warms up600 varied steps and profiles128
snapshot passes. Contact wall is35.043776ms original versus35.306725ms bounded.
Fan-profile device time is20.138 versus20.131ms. Memory is6,111,942,238 native
bytes in both routes. Snapshot nodal repeat difference is at most1.942890293e-16N
and peak difference1.092121238e-8Pa.

This profile provides no gain expectation for another live ablation. The
original simpler walk remains selected and this candidate is rejected. Its
full/live/policy rates are unmeasured. These data establish the decision for
this candidate, without attributing all remaining sampler time to loop control.

Reproduce from the archived source patch, `baseline_surface_segment_v1.py`,
`probe_segment_bounds_v1.py`, profile-pair launcher and regression wrapper.
The candidate's private base is90b2964e plus selected port-seed reuse. Both
measurement routes close all collider/stress sources and use the same inputs.

Archive:
`evidence/20261011-nonconvex/native-segment-bounds-rejected-v1/manifest.json`.
An allocated CPU publisher verifies343 original/stored/decompressed entries,
114 unique blobs/15,573,303 stored bytes, immutable source closures, both test
reports,48 recorded fields and both profile/snapshot results.
Manifest SHA256:
`0a25f532fe31d3a9c75f9a914c554d4593c4c809a1d8434d07d7275e833edcb3`.

## Current candidates

| Candidate | Purpose | Current evidence/decision |
|---|---|---|
| Sample-major frontier launch | Keep neighboring sample lanes on one contact and reduce divergent geometry work | [Selected](NATIVE_FAN_SAMPLE_LAYOUT_20261012.md):23 CUDA cases, unchanged CPU route,48 recorded oracle fields. Same-GPU live40955.914→55763.374 env-step/s (1.361546320x),zero failures in all three1280-step repeats/2323 resets. |
| Nonnegative port-distance bound | Skip a final Euclidean segment norm whenever its accumulated path already exceeds the best complete distance | Mathematical lower bound retains the same minimum. Isolated23 CPU cases pass and same-GPU profile contact wall33.176090→26.676988ms. Its increment on top of sample-major scheduling and combined CUDA qualification are under evaluation. |

The active goal still needs selected end-to-end benefits, new current-source
four-scope measurements, wider batches/seeds, timed grasp outcomes, per-environment
tail work and a measured plateau. No physical mesh/quadrature convergence or
RL training throughput is claimed.
