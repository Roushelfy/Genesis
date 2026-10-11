# Native final measurement tables

Version: `v8`. This file is generated from the complete
summary `native-final-comprehensive-complete-v8-summary.json` (SHA256
`cacf23eccefbd1f40b19788817f2c7da39fae2254bc343a27a85a100139f7f73`). Renderer SHA256:
`139153e2a0e206dbc0ce6d0b0e3cc23cbf6e1f39402178d375aaf768db889e38`.

Measured source: `f4965c70-native-production-indices-v8`.
Per-case revisions and exact hashes remain in the summary. If a
recovery-version override is recorded, it uses the benchmark's explicit
standalone JIT/capture preparation fix; all engine, controller and
option source hashes match the other scopes. Preparation cost is
recorded separately. Original cold-path timings remain archived.
Every rate uses 900 warmup steps and three repeats of at least 2400
changing-contact steps and ten seconds. Both seeds are included. The
aggregate is total valid transitions divided by total wall time; the
six-repeat spread combines workload and temporal variation. All resets,
contact parameters, controller and required observation/inference work
belong to their stated scopes. Frozen recovery is a separate snapshot
diagnostic. Policy measures inference and rollout, not RL training.

## Four scopes and memory

Sampled memory is whole-card peak for each recorded GPU UUID, including
setup/JIT; it is separate from native buffer and Torch inventories.

| B | Scope | Valid env-step/s | Batch step/s | Six-repeat min–max | Std env-step/s | Peak MiB | Invalid |
|---|---|---|---|---|---|---|---|
| 1024 | rigid | 115094.76 | 112.40 | 114242.14–115770.57 | 533.60 | 3267 | 0 |
| 1024 | recovery | 289730.51 | 282.94 | 289525.07–289952.48 | 211.93 | 3525 | 0 |
| 1024 | live | 81657.09 | 79.74 | 80958.61–81980.82 | 386.04 | 3525 | 0 |
| 1024 | policy | 80896.80 | 79.00 | 79683.74–81597.23 | 752.01 | 3535 | 0 |
| 2048 | rigid | 189444.08 | 92.50 | 187881.01–190674.52 | 1067.63 | 4199 | 0 |
| 2048 | recovery | 365117.99 | 178.28 | 363505.84–366630.15 | 1510.81 | 4713 | 0 |
| 2048 | live | 123670.88 | 60.39 | 123319.33–123829.09 | 211.37 | 4713 | 0 |
| 2048 | policy | 123136.27 | 60.13 | 122756.59–123848.68 | 393.34 | 4725 | 0 |
| 32768 | rigid | 763399.21 | 23.30 | 761326.98–765063.95 | 1547.37 | 32709 | 0 |
| 32768 | recovery | 513718.08 | 15.68 | 508481.69–516371.69 | 3056.19 | 39847 | 0 |
| 32768 | live | 309434.72 | 9.44 | 308338.53–310679.05 | 998.34 | 39847 | 0 |
| 32768 | policy | 309128.16 | 9.43 | 307108.40–310576.95 | 1264.99 | 39971 | 0 |
| 49152 | rigid | 843798.56 | 17.17 | 841960.89–845425.92 | 1694.95 | 47889 | 0 |
| 49152 | recovery | 516618.47 | 10.51 | 513895.02–517901.09 | 1932.27 | 58609 | 0 |
| 49152 | live | 320216.92 | 6.51 | 319321.67–321075.33 | 763.37 | 58611 | 0 |
| 49152 | policy | 320239.86 | 6.52 | 319440.39–321586.73 | 842.92 | 58763 | 0 |
| 61440 | rigid | 861027.87 | 14.01 | 853455.19–868293.23 | 7665.17 | 59329 | 0 |
| 61440 | recovery | 519458.92 | 8.45 | 517550.52–520388.23 | 1154.00 | 72675 | 0 |
| 61440 | live | 328541.38 | 5.35 | 327838.40–329298.74 | 549.32 | 72675 | 0 |
| 61440 | policy | 328456.33 | 5.35 | 327891.64–329181.80 | 568.13 | 72877 | 0 |

## Matched rigid overhead

Step-time increase and throughput decrease are different quantities.

| B | Seed | Scope | Step time +% | Throughput −% |
|---|---|---|---|---|
| 1024 | 510000 | live | 41.85 | 29.50 |
| 1024 | 510000 | policy | 43.80 | 30.46 |
| 1024 | 623001 | live | 40.06 | 28.60 |
| 1024 | 623001 | policy | 40.75 | 28.95 |
| 2048 | 510000 | live | 53.88 | 35.01 |
| 2048 | 510000 | policy | 54.40 | 35.23 |
| 2048 | 623001 | live | 52.49 | 34.42 |
| 2048 | 623001 | policy | 53.31 | 34.77 |
| 32768 | 510000 | live | 146.94 | 59.50 |
| 32768 | 510000 | policy | 147.15 | 59.54 |
| 32768 | 623001 | live | 146.48 | 59.43 |
| 32768 | 623001 | policy | 146.75 | 59.47 |
| 49152 | 510000 | live | 163.59 | 62.06 |
| 49152 | 510000 | policy | 163.57 | 62.06 |
| 49152 | 623001 | live | 163.43 | 62.04 |
| 49152 | 623001 | policy | 163.41 | 62.04 |
| 61440 | 510000 | live | 159.83 | 61.51 |
| 61440 | 510000 | policy | 159.71 | 61.49 |
| 61440 | 623001 | live | 164.35 | 62.17 |
| 61440 | 623001 | policy | 164.62 | 62.21 |

## Same-GPU full-field output cost

Each pair uses the same physical GPU and source, with identical final budgets.

| B | Seed | Scope | Max env-step/s | Full env-step/s | Full step time +% | Additional native bytes |
|---|---|---|---|---|---|---|
| 1024 | 510000 | live | 82504.14 | 82003.53 | 0.610 | 110100480 |
| 1024 | 510000 | policy | 81923.71 | 81311.87 | 0.752 | 110100480 |
| 1024 | 623001 | live | 83541.68 | 82993.36 | 0.661 | 110100480 |
| 1024 | 623001 | policy | 82921.22 | 82506.15 | 0.503 | 110100480 |
| 32768 | 510000 | live | 304292.45 | 297723.47 | 2.206 | 3523215360 |
| 32768 | 510000 | policy | 305314.97 | 298607.52 | 2.246 | 3523215360 |
| 32768 | 623001 | live | 315692.57 | 309257.60 | 2.081 | 3523215360 |
| 32768 | 623001 | policy | 315165.76 | 308921.72 | 2.021 | 3523215360 |

## Every-environment episode outcomes

These intrusive untimed audits cover 900 warmup and 2400 trajectory
steps in every environment. Outcome totals below subtract warmup.
Started, unfinished and restart-aborted episodes remain visible.
Failure reasons overlap. Weak-grip height loss is a drop proxy, not
a measured tangential slip distance or an additional episode-success
criterion. Exact lift/hold/release definitions remain in every artifact.

| B | Seed | Scope | Started | Completed | Success | Failed | Unfinished | Aborted | Weak >1cm drop | Numeric steps failed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 510000 | live | 4096 | 3074 | 3074 | 0 | 1022 | 0 | 2672 | 0 |
| 1024 | 510000 | policy | 4096 | 3074 | 3074 | 0 | 1022 | 0 | 2655 | 0 |
| 1024 | 623001 | live | 4096 | 3074 | 3074 | 0 | 1022 | 0 | 2674 | 0 |
| 1024 | 623001 | policy | 4096 | 3074 | 3074 | 0 | 1022 | 0 | 2689 | 0 |
| 2048 | 510000 | live | 8192 | 6148 | 6148 | 0 | 2044 | 0 | 5305 | 0 |
| 2048 | 510000 | policy | 8192 | 6148 | 6148 | 0 | 2044 | 0 | 5284 | 0 |
| 2048 | 623001 | live | 8192 | 6148 | 6148 | 0 | 2044 | 0 | 5268 | 0 |
| 2048 | 623001 | policy | 8192 | 6148 | 6148 | 0 | 2044 | 0 | 5286 | 0 |
| 32768 | 510000 | live | 131072 | 98359 | 98359 | 0 | 32713 | 0 | 85385 | 0 |
| 32768 | 510000 | policy | 131072 | 98359 | 98359 | 0 | 32713 | 0 | 85386 | 0 |
| 32768 | 623001 | live | 131072 | 98359 | 98359 | 0 | 32713 | 0 | 85371 | 0 |
| 32768 | 623001 | policy | 131072 | 98359 | 98359 | 0 | 32713 | 0 | 85346 | 0 |
| 61440 | 510000 | live | 245760 | 184423 | 184423 | 0 | 61337 | 0 | 160499 | 0 |
| 61440 | 510000 | policy | 245760 | 184423 | 184423 | 0 | 61337 | 0 | 160528 | 0 |
| 61440 | 623001 | live | 245760 | 184423 | 184423 | 0 | 61337 | 0 | 160144 | 0 |
| 61440 | 623001 | policy | 245760 | 184423 | 184423 | 0 | 61337 | 0 | 160118 | 0 |

## Numerical and work tails

The following are worst values across environments, not individual
environment GPU latencies. Audit work totals span warmup and trajectory.
Complete balance includes all six gauge rows. Force and moment budgets
remain 1e-8; the full residual budget remains max(1e-11 N, 1e-8 times
the RHS norm). No slow or failed environment is removed.

| B | Seed | Scope | Full residual / budget max | Wrench force rel max | Wrench moment rel max | Newton eval max | Retries max | Corrections total | Factor fallback total | Packed columns max | Face tasks max | Zero RHS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 510000 | live | 0.00915349 | 7.84407e-10 | 6.85061e-10 | 6 | 0 | 0 | 0 | 79 | 120 | unrecorded |
| 1024 | 510000 | policy | 0.00872766 | 7.58894e-10 | 6.94155e-10 | 6 | 0 | 0 | 0 | 79 | 114 | unrecorded |
| 1024 | 623001 | live | 0.00882838 | 7.376e-10 | 6.61127e-10 | 5 | 0 | 0 | 0 | 77 | 130 | unrecorded |
| 1024 | 623001 | policy | 0.00967209 | 7.19324e-10 | 6.54502e-10 | 6 | 0 | 0 | 0 | 79 | 129 | unrecorded |
| 2048 | 510000 | live | 0.00951939 | 7.84406e-10 | 6.85061e-10 | 9 | 0 | 0 | 0 | 79 | 120 | unrecorded |
| 2048 | 510000 | policy | 0.00919439 | 7.73928e-10 | 6.94156e-10 | 6 | 0 | 0 | 0 | 79 | 121 | unrecorded |
| 2048 | 623001 | live | 0.00995149 | 7.376e-10 | 6.80153e-10 | 5 | 0 | 0 | 0 | 77 | 141 | unrecorded |
| 2048 | 623001 | policy | 0.0110723 | 7.19325e-10 | 6.63326e-10 | 6 | 0 | 0 | 0 | 79 | 134 | unrecorded |
| 32768 | 510000 | live | 0.00940047 | 7.8837e-10 | 6.98039e-10 | 94 | 3 | 0 | 0 | 79 | 158 | unrecorded |
| 32768 | 510000 | policy | 0.00943229 | 7.75188e-10 | 7.04139e-10 | 91 | 3 | 0 | 0 | 79 | 156 | unrecorded |
| 32768 | 623001 | live | 0.00876795 | 7.67415e-10 | 6.99284e-10 | 109 | 3 | 0 | 0 | 79 | 141 | unrecorded |
| 32768 | 623001 | policy | 0.0105567 | 7.76695e-10 | 6.98545e-10 | 90 | 5 | 0 | 0 | 79 | 141 | unrecorded |
| 61440 | 510000 | live | 0.00932319 | 7.8837e-10 | 7.02899e-10 | 94 | 4 | 0 | 0 | 79 | 158 | unrecorded |
| 61440 | 510000 | policy | 0.0100697 | 7.84414e-10 | 7.04139e-10 | 91 | 3 | 0 | 0 | 79 | 156 | unrecorded |
| 61440 | 623001 | live | 0.010362 | 7.67415e-10 | 7.00467e-10 | 109 | 4 | 0 | 0 | 79 | 147 | unrecorded |
| 61440 | 623001 | policy | 0.0105993 | 7.87724e-10 | 7.01339e-10 | 90 | 5 | 0 | 0 | 79 | 154 | unrecorded |

## Frozen-snapshot split profile

Separate synchronized stage passes are diagnostic; they are not added to
derive live throughput. Pressure subdivision is another separate pass.

| Snapshot | Stage | Wall ms |
|---|---|---|
| native-final-snapshot-profile-b1024-v8.json | begin_step | 0.018041 |
| native-final-snapshot-profile-b1024-v8.json | association_transform | 0.102092 |
| native-final-snapshot-profile-b1024-v8.json | anchor_search | 0.123858 |
| native-final-snapshot-profile-b1024-v8.json | candidate_weights_constraints | 1.119691 |
| native-final-snapshot-profile-b1024-v8.json | nodal_scatter | 0.582509 |
| native-final-snapshot-profile-b1024-v8.json | inertia_relief_centrifugal | 0.120912 |
| native-final-snapshot-profile-b1024-v8.json | rhs_reduction | 0.041299 |
| native-final-snapshot-profile-b1024-v8.json | linear_solve | 0.799405 |
| native-final-snapshot-profile-b1024-v8.json | complete_residual | 0.343858 |
| native-final-snapshot-profile-b1024-v8.json | global_peak | 0.248008 |
| native-final-snapshot-profile-b1024-v8.json | accept | 0.027444 |
| native-final-snapshot-profile-b32768-v8.json | begin_step | 0.017673 |
| native-final-snapshot-profile-b32768-v8.json | association_transform | 0.125966 |
| native-final-snapshot-profile-b32768-v8.json | anchor_search_candidate_weights_constraints_nodal_scatter | 23.898660 |
| native-final-snapshot-profile-b32768-v8.json | inertia_relief_centrifugal | 1.564723 |
| native-final-snapshot-profile-b32768-v8.json | rhs_reduction | 0.532315 |
| native-final-snapshot-profile-b32768-v8.json | linear_solve | 22.617046 |
| native-final-snapshot-profile-b32768-v8.json | complete_residual | 10.377403 |
| native-final-snapshot-profile-b32768-v8.json | global_peak | 6.916318 |
| native-final-snapshot-profile-b32768-v8.json | accept | 0.057277 |

## Changing-contact stream profiles

Nested CUDA-stream intervals overlap and must not be summed. They include
host preparation/dispatch idle gaps. Intrusive event instrumentation changes
timing; ordinary benchmark rates above remain the throughput evidence.

| Profile | Stage | Calls | CUDA interval mean ms | CUDA p99 ms | CUDA max ms | Host submission mean ms |
|---|---|---|---|---|---|---|
| native-final-rollout-profile-live-max-b1024-v8.json | contact_parameter_update | 1200 | 0.0458 | 0.1068 | 0.2554 | 0.0633 |
| native-final-rollout-profile-live-max-b1024-v8.json | controller_and_scene_step | 1200 | 12.1410 | 16.1185 | 17.6129 | 12.0695 |
| native-final-rollout-profile-live-max-b1024-v8.json | native_stress_recovery | 1200 | 3.6054 | 4.2565 | 4.5782 | 0.4667 |
| native-final-rollout-profile-live-max-b1024-v8.json | partial_reset | 600 | 1.1600 | 1.8044 | 2.4306 | 4.3687 |
| native-final-rollout-profile-live-max-b1024-v8.json | rigid_contact_force_postprocess | 1200 | 0.0748 | 0.1265 | 0.2258 | 4.7070 |
| native-final-rollout-profile-live-max-b1024-v8.json | rigid_integration_after_stress | 1200 | 0.3513 | 0.3750 | 0.3772 | 0.2261 |
| native-final-rollout-profile-live-max-b1024-v8.json | rigid_substep_including_recovery | 1200 | 11.1437 | 13.8726 | 15.3462 | 8.9353 |
| native-final-rollout-profile-live-max-b1024-v8.json | scene_step | 1200 | 11.2609 | 14.0614 | 15.5543 | 9.2567 |
| native-final-rollout-profile-live-full-b1024-v8.json | contact_parameter_update | 1200 | 0.0389 | 0.0800 | 0.1370 | 0.0539 |
| native-final-rollout-profile-live-full-b1024-v8.json | controller_and_scene_step | 1200 | 11.8766 | 15.5427 | 17.1008 | 11.8046 |
| native-final-rollout-profile-live-full-b1024-v8.json | native_stress_recovery | 1200 | 3.5346 | 4.1241 | 4.2859 | 0.3192 |
| native-final-rollout-profile-live-full-b1024-v8.json | partial_reset | 600 | 0.9716 | 1.3532 | 2.1518 | 4.3929 |
| native-final-rollout-profile-live-full-b1024-v8.json | rigid_contact_force_postprocess | 1200 | 0.0715 | 0.0778 | 0.1115 | 4.7039 |
| native-final-rollout-profile-live-full-b1024-v8.json | rigid_integration_after_stress | 1200 | 0.3538 | 0.3778 | 0.3792 | 0.1819 |
| native-final-rollout-profile-live-full-b1024-v8.json | rigid_substep_including_recovery | 1200 | 11.0477 | 13.7564 | 15.1550 | 8.7982 |
| native-final-rollout-profile-live-full-b1024-v8.json | scene_step | 1200 | 11.1363 | 13.8677 | 15.3156 | 9.0814 |
| native-final-rollout-profile-policy-max-b1024-v8.json | contact_parameter_update | 1200 | 0.0396 | 0.0816 | 0.2651 | 0.0555 |
| native-final-rollout-profile-policy-max-b1024-v8.json | controller_and_scene_step | 1200 | 11.7715 | 15.4218 | 16.2435 | 11.3272 |
| native-final-rollout-profile-policy-max-b1024-v8.json | native_stress_recovery | 1200 | 3.4599 | 4.0494 | 4.3119 | 0.3303 |
| native-final-rollout-profile-policy-max-b1024-v8.json | observation | 1200 | 0.0217 | 0.2458 | 0.5437 | 0.2295 |
| native-final-rollout-profile-policy-max-b1024-v8.json | partial_reset | 600 | 0.9764 | 1.4683 | 3.2571 | 3.8990 |
| native-final-rollout-profile-policy-max-b1024-v8.json | policy_inference | 1200 | 0.0359 | 0.1410 | 0.1730 | 0.1337 |
| native-final-rollout-profile-policy-max-b1024-v8.json | rigid_contact_force_postprocess | 1200 | 0.0721 | 0.0788 | 0.2453 | 4.6590 |
| native-final-rollout-profile-policy-max-b1024-v8.json | rigid_integration_after_stress | 1200 | 0.3517 | 0.3751 | 0.3775 | 0.1932 |
| native-final-rollout-profile-policy-max-b1024-v8.json | rigid_substep_including_recovery | 1200 | 10.9348 | 13.6420 | 14.5432 | 8.5549 |
| native-final-rollout-profile-policy-max-b1024-v8.json | scene_step | 1200 | 11.0256 | 13.8077 | 14.7084 | 8.8177 |
| native-final-rollout-profile-policy-full-b1024-v8.json | contact_parameter_update | 1200 | 0.0397 | 0.0862 | 0.2827 | 0.0557 |
| native-final-rollout-profile-policy-full-b1024-v8.json | controller_and_scene_step | 1200 | 11.8755 | 15.4867 | 16.3960 | 11.4340 |
| native-final-rollout-profile-policy-full-b1024-v8.json | native_stress_recovery | 1200 | 3.5519 | 4.1560 | 4.5612 | 0.3360 |
| native-final-rollout-profile-policy-full-b1024-v8.json | observation | 1200 | 0.0217 | 0.2435 | 0.6753 | 0.2282 |
| native-final-rollout-profile-policy-full-b1024-v8.json | partial_reset | 600 | 0.9860 | 1.4528 | 3.0480 | 4.0074 |
| native-final-rollout-profile-policy-full-b1024-v8.json | policy_inference | 1200 | 0.0359 | 0.1385 | 0.1565 | 0.1323 |
| native-final-rollout-profile-policy-full-b1024-v8.json | rigid_contact_force_postprocess | 1200 | 0.0722 | 0.0830 | 0.2200 | 4.6615 |
| native-final-rollout-profile-policy-full-b1024-v8.json | rigid_integration_after_stress | 1200 | 0.3541 | 0.3781 | 0.3804 | 0.1922 |
| native-final-rollout-profile-policy-full-b1024-v8.json | rigid_substep_including_recovery | 1200 | 11.0305 | 13.7999 | 14.4697 | 8.5990 |
| native-final-rollout-profile-policy-full-b1024-v8.json | scene_step | 1200 | 11.1222 | 13.9624 | 14.6324 | 8.8689 |
| native-final-rollout-profile-live-max-b32768-v8.json | contact_parameter_update | 1200 | 0.1641 | 0.2113 | 0.2430 | 0.0604 |
| native-final-rollout-profile-live-max-b32768-v8.json | controller_and_scene_step | 1200 | 102.1660 | 115.4601 | 135.7725 | 101.1249 |
| native-final-rollout-profile-live-max-b32768-v8.json | native_stress_recovery | 1200 | 61.7484 | 67.5931 | 90.1451 | 0.4189 |
| native-final-rollout-profile-live-max-b32768-v8.json | partial_reset | 600 | 1.4457 | 1.9863 | 2.7517 | 69.1381 |
| native-final-rollout-profile-live-max-b32768-v8.json | rigid_contact_force_postprocess | 1200 | 0.1168 | 0.1301 | 0.1676 | 18.2261 |
| native-final-rollout-profile-live-max-b32768-v8.json | rigid_integration_after_stress | 1200 | 2.4430 | 2.4880 | 2.4928 | 0.2050 |
| native-final-rollout-profile-live-max-b32768-v8.json | rigid_substep_including_recovery | 1200 | 100.9493 | 113.1658 | 133.4421 | 62.8333 |
| native-final-rollout-profile-live-max-b32768-v8.json | scene_step | 1200 | 101.0429 | 113.3178 | 133.5916 | 65.8282 |
| native-final-rollout-profile-live-full-b32768-v8.json | contact_parameter_update | 1200 | 0.1686 | 0.2206 | 0.2801 | 0.0629 |
| native-final-rollout-profile-live-full-b32768-v8.json | controller_and_scene_step | 1200 | 104.3744 | 117.5389 | 138.4483 | 103.2989 |
| native-final-rollout-profile-live-full-b32768-v8.json | native_stress_recovery | 1200 | 63.9259 | 69.9027 | 92.4777 | 0.4379 |
| native-final-rollout-profile-live-full-b32768-v8.json | partial_reset | 600 | 1.4765 | 2.0518 | 2.7541 | 71.4052 |
| native-final-rollout-profile-live-full-b32768-v8.json | rigid_contact_force_postprocess | 1200 | 0.1181 | 0.1334 | 0.2032 | 18.2290 |
| native-final-rollout-profile-live-full-b32768-v8.json | rigid_integration_after_stress | 1200 | 2.4373 | 2.4792 | 2.4825 | 0.2206 |
| native-final-rollout-profile-live-full-b32768-v8.json | rigid_substep_including_recovery | 1200 | 103.1274 | 115.1908 | 136.1895 | 63.7279 |
| native-final-rollout-profile-live-full-b32768-v8.json | scene_step | 1200 | 103.2261 | 115.3648 | 136.3380 | 66.8321 |
| native-final-rollout-profile-policy-max-b32768-v8.json | contact_parameter_update | 1200 | 0.1636 | 0.2074 | 0.3020 | 0.0629 |
| native-final-rollout-profile-policy-max-b32768-v8.json | controller_and_scene_step | 1200 | 102.1611 | 115.5822 | 124.5851 | 100.7634 |
| native-final-rollout-profile-policy-max-b32768-v8.json | native_stress_recovery | 1200 | 61.7560 | 67.5958 | 77.0163 | 0.4628 |
| native-final-rollout-profile-policy-max-b32768-v8.json | observation | 1200 | 0.0547 | 0.3378 | 0.6729 | 0.2740 |
| native-final-rollout-profile-policy-max-b32768-v8.json | partial_reset | 600 | 1.4126 | 1.9340 | 2.7741 | 68.6992 |
| native-final-rollout-profile-policy-max-b32768-v8.json | policy_inference | 1200 | 0.0956 | 0.1721 | 0.2121 | 0.1509 |
| native-final-rollout-profile-policy-max-b32768-v8.json | rigid_contact_force_postprocess | 1200 | 0.1183 | 0.1352 | 0.3497 | 18.2095 |
| native-final-rollout-profile-policy-max-b32768-v8.json | rigid_integration_after_stress | 1200 | 2.4440 | 2.4885 | 2.4949 | 0.2429 |
| native-final-rollout-profile-policy-max-b32768-v8.json | rigid_substep_including_recovery | 1200 | 100.9597 | 113.0407 | 122.4309 | 62.6506 |
| native-final-rollout-profile-policy-max-b32768-v8.json | scene_step | 1200 | 101.0543 | 113.2223 | 122.5702 | 65.6291 |
| native-final-rollout-profile-policy-full-b32768-v8.json | contact_parameter_update | 1200 | 0.1689 | 0.2173 | 0.3014 | 0.0638 |
| native-final-rollout-profile-policy-full-b32768-v8.json | controller_and_scene_step | 1200 | 104.3643 | 117.7521 | 126.9085 | 102.9289 |
| native-final-rollout-profile-policy-full-b32768-v8.json | native_stress_recovery | 1200 | 63.9094 | 70.0052 | 79.0992 | 0.4640 |
| native-final-rollout-profile-policy-full-b32768-v8.json | observation | 1200 | 0.0549 | 0.3021 | 0.6346 | 0.2767 |
| native-final-rollout-profile-policy-full-b32768-v8.json | partial_reset | 600 | 1.4939 | 2.0099 | 4.0061 | 70.9603 |
| native-final-rollout-profile-policy-full-b32768-v8.json | policy_inference | 1200 | 0.0957 | 0.1748 | 0.2093 | 0.1512 |
| native-final-rollout-profile-policy-full-b32768-v8.json | rigid_contact_force_postprocess | 1200 | 0.1182 | 0.1321 | 0.1392 | 18.2075 |
| native-final-rollout-profile-policy-full-b32768-v8.json | rigid_integration_after_stress | 1200 | 2.4380 | 2.4802 | 2.4866 | 0.2438 |
| native-final-rollout-profile-policy-full-b32768-v8.json | rigid_substep_including_recovery | 1200 | 103.1081 | 115.3698 | 124.6365 | 63.5615 |
| native-final-rollout-profile-policy-full-b32768-v8.json | scene_step | 1200 | 103.2048 | 115.5261 | 124.7847 | 66.6462 |

All per-environment p50/p90/p99/max work distributions, exact coefficients,
reset counts, cache hit/miss/eviction counts, byte inventories, source hashes
and original paths are preserved in the summary and its hashed artifacts.
