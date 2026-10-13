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
| Nonnegative port-distance bound | Skip a final Euclidean segment norm whenever its accumulated path already exceeds the best complete distance | [Selected](NATIVE_PORT_BOUNDS_20261012.md):23 CPU/23 CUDA cases,48 recorded fields. With sample-major scheduling, same-GPU live52414.728→54119.673 env-step/s (1.032527971x),zero failures in all three1280-step repeats/2323 resets. |
| Exact workspace growth | Allocate the actual observed contact requirement when a complete aggregate outgrows its workspace | [Retained for memory](NATIVE_EXACT_CONTACT_CAPACITY_20261012.md):CPU23/CUDA23 pass; same-GPU live ratio0.995473616 provides no speed gain, but native memory drops614,551,552 bytes atB2048. Current pruning/metrics combination passes anotherCUDA23/24 recorded fields and three1280-step live repeats with zero failures. More frequent growth costs remain to be counted at wider batches. |
| Final traction projection fusion | Reuse final projected traction for original-point wrench, cones and signed P2 scatter | [Selected](NATIVE_FUSED_TRACTION_ACCEPTANCE_20261012.md):CPU23/CUDA23 plus6 cached componentwise cases per backend pass. Matched live57355.126→58219.869 valid env-step/s (1.015076987x),zero failures, with six wrench components/cone validity cached per constraint. |
| Device frontier count | Remove a per-step host maximum read before fan sampling | [Not selected](NATIVE_DEVICE_FRONTIER_COUNT_20261012.md):CUDA23/24 independent fields pass; matched live ratio1.001762273 is within repeat scatter. |
| Contact graphs | Reduce per-kernel dispatch and capture the complete aggregate contact pipeline | [Not selected](NATIVE_CONTACT_GRAPHS_20261012.md):CUDA23/24 independent fields pass. Snapshot gains do not carry into three-repeat changing-contact live:per-kernel graph+fastcache0.972735x,complete graph0.994586x,complete graph+fastcache0.997816x. |
| Fastcache without graphs | Cache native contact dispatch signatures | [Not selected](NATIVE_CONTACT_FASTCACHE_20261012.md):same-GPU original/fastcache/original live ratio0.998677432 versus pooled originals,zero failures in all9 repeats. |
| Masked cold cooperative retry | Reduce the measured long serial fit tail while retaining complete serial fallback | [Selected](NATIVE_COOPERATIVE_RETRY_20261012.md):same-GPU live1.036986834x,23 CPU/23 CUDA production cases and320 captured independent fields pass. Successful constraints remain bitwise unchanged. Refinement tails remain actionable. |
| Eight-sample-lane retry | Replace the remaining first serial retry with a different parallel reduction order | [Selected](NATIVE_SAMPLE_LANE_RETRY_20261012.md):same-GPU live1.044883994x,23 CUDA production cases and160 independent dynamic fields pass. Refinement tails remain. |
| Actual-sample storage | Replace full-surface profile/index capacity with counted complete sample clouds | [Qualified prototype](NATIVE_COMPACT_TRACTION_PROFILES_20261012.md):31 CPU/31 CUDA cases and96 fields pass; native savings3.930GB atB2048/15.715GB atB8192, but rate ratios0.968963487/0.989900656. Fused capacity has31 CUDA/48 fields passing and a separate B2048 ratio0.977539074; still unselected. |
| Refined-fan scheduling/fitting | Reduce the remaining approximately107ms fan refinement and76–89ms serial refit calls | [Parallel sampling qualifies](NATIVE_CONTACT_TAIL_STORAGE_20261012.md) with23 CPU/23 CUDA,48 fields and same-GPU1.017149258x live.160 dynamic fields pass and fan hard calls fall to4.3–4.4ms. Cooperative refit qualifies23 CUDA/24 fields but its1.002938634x effect is within repeat variation and is unselected. Combined production qualification is pending. |
| Sample geometry reuse | Reduce repeated chart/position reconstruction inside dual iterations | [Qualification](NATIVE_CACHED_FAN_POSITIONS_20261012.md):31 CPU/31 CUDA and96 fields; eight native CUDA fixtures have exact cached/direct positions. [Combined full-storage comparisons](NATIVE_CONTACT_TAIL_STORAGE_20261012.md) establish1.022885722x B2048/1.027915774x separated-seedB8192 and14.116GB native savings atB8192. Selected for integration with parallel fans; production cleanup/qualification is pending. |
| Sample-contiguous scalar layout | Coalesce warp profile/index loads | [Unselected](NATIVE_CONTACT_TAIL_STORAGE_20261012.md):31 CUDA/48 fields pass, but matched B2048 live1.003031333x is within repeat variation. Vector position layout is unchanged. |
| Stress raw-contact packing | Reduce1200 scene-wide source slots despite observed complete aggregates around7 | Potential remaining memory opportunity after sample compaction; original contact identities, API indexing and lifecycle must be preserved. Benefit/cost has not yet been established. |
| Coalesced cached position coordinates | Coalesce all physical coordinates along the cooperative sample lanes | [Qualified and selected for integration](NATIVE_COALESCED_POSITIONS_20261012.md):31 CPU/31 CUDA cases,96 fields/eight exact cached-direct checks; same-GPU ratios1.017995104x B16384 and1.015946367x separated-seedB8192, no byte or fidelity change. This extends the earlier scalar profile/index-only experiment. |
| Valid-source-only initialization | Avoid writing inactive raw provenance on every step | [Qualified and selected for integration](NATIVE_VALID_SOURCE_CAPTURE_20261012.md):31 CPU/31 CUDA/96 fields, same-GPU ratios1.009451355x B8192 and1.011652788x separated-seedB16384; valid capture is unchanged and native bytes are identical. |
| Previous/current diagnostic clearing | Remove inactive diagnostic/certificate writes while clearing disappeared contacts |31CPU/31CUDA/96fields pass; selected with ratios1.009038163x B8192/1.010380685x separated-seedB16384. Previous complete indices are cleared before collecting new sources, and current sources receive full initialization. |
| FP32 warm Hessian | Reduce nonlinear search-direction cost while retaining FP64 physical loads, gradients, cold fallback and final acceptance |31CPU/31CUDA/96fields pass; selected with ratios1.033683688x B16384/1.020015526x separated-seedB8192. Final physical/numerical budgets are unchanged. |
| Diagnostic/Hessian replicated decision | Select the two independently qualified candidates for clean integration |[Qualified](NATIVE_WARM_HESSIAN_DIAGNOSTICS_20261012.md): diagnostic ratios1.009038163x/1.010380685x, warm Hessian1.033683688x/1.020015526x on separated B8192/B16384 cohorts; each31CPU/31CUDA/96 fields, unchanged final budgets and all outcomes valid. Clean four-candidate combination remains under qualification. |
| Squared remaining-path gate | Skip square roots unable to improve the complete minimum |[Not selected](NATIVE_WARM_HESSIAN_DIAGNOSTICS_20261012.md):31CPU/31CUDA and96 fields pass, B8192 changing live ratio0.998846865 with overlapping repeat ranges, no stable end-to-end gain. Physical radius, adjacent unfolding, plane checks and load law are unchanged. |
| Inactive surface-walk exit | Stop a monotone-inactive path walk before its fixed face-count iteration limit |31CPU/31CUDA pass; same-GPU B8192/623105 ratio1.000350925 with overlapping repeat ranges, all outcomes valid. Relative to the four clean optimizations, unchanged geometry/law. No useful measured gain; not selected. |
| Clean four-candidate combination | Qualify the selected layout/capture/precision/clearing implementation together |[Complete B16384 dynamic pair](NATIVE_COMBINED_QUALIFICATION_20261013.md) ratio1.073180556, zero invalid/failed outcomes, exact rigid counters/positions. First CUDA matrix32passes/1strict arithmetic-repeat assertion failure retained. Same-GPU FP64 diagnostic identifies repeat precision; physical-budget controls add CPU K/gauge and original wrench checks and pass. Expanded35CPU/35CUDA, dynamic oracles and final-source scopes pending. |
| FP32 Gaussian only | Assess whether Gaussian precision can materially reduce the remaining sampler |Standalone Quadrants off/on/off cost probe at captured mean/max sample counts saves0.244/0.285ms,0.153%/0.178% of the fenced B16384 mean step. Contiguous FP64 inputs/output, FP32 only exponential argument/result; CPU profile error≤1.961e-7. This is not a whole-kernel bound or stress/rollout qualification. No production precision change is selected: anticipated whole-step benefit is below0.2%, while geometry/distance dominate. Full stress consistency for this change is unmeasured. Raw first-signature failure and corrected source/results are retained in `native-gaussian-cost-checkpoint-v1`. |

The active goal still needs wider current-source four-scope batches/seeds,
dynamic/tail cost profiling, remaining applicable candidates and a measured
plateau. [B1024 four scopes](NATIVE_MATCHED_SCOPES_20261012.md) now include timed
grasp outcomes and environment work with a matched condition bank. No physical mesh/quadrature convergence or
RL training throughput is claimed.

The [clean production combination](NATIVE_SELECTED_STORAGE_FANS_20261012.md)
now passes31 CPU/31 CUDA and96 fields. Its B2048 same-GPU ratio over counted
positions with serial fans is1.003100208, within repeat variation, and all
condition banks/rigid outcomes agree. This does not multiply the earlier
isolated gains. Parallel fans retain their measured hard-tail reduction.
[B16384/B32768 live probes](NATIVE_WIDE_COUNTED_POSITIONS_20261012.md) establish
resident feasibility and roughly107642/104754 valid env-step/s for the earlier
counted-position prototype, without establishing a final-source optimum.
