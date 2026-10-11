# Native final measurement tables

Version: `v9`. This file is generated from the complete
summary `native-final-comprehensive-complete-v14-v9-v10-summary.json` (SHA256
`b44198c77caf8f3c4aaea93f08a16a4f313114d638a91f97403330b787962100`). Renderer SHA256:
`a6c02d4984fb5399d4ab7f513e27bdfb10010866cede1b39e412b430f46e9646`.

Measured source: `f3e92c6267357143b859b24a27a3a67731586667`.
Per-case revisions and exact hashes remain in the summary. If a
recovery-version override is recorded, it uses the benchmark's explicit
standalone JIT/capture preparation fix; all engine, controller and
option source hashes match the other scopes. Preparation cost is
recorded separately. Original cold-path timings remain archived.
Every rate uses 900 warmup steps and three repeats of at least 2400
benchmark steps and ten seconds. Live/policy contacts change; recovery
repeats its frozen actual snapshot. Both seeds are included. The
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
| 1024 | rigid | 114955.85 | 112.26 | 114289.76–115607.11 | 477.99 | 3267 | 0 |
| 1024 | recovery | 287497.30 | 280.76 | 286604.90–288892.34 | 860.81 | 3545 | 0 |
| 1024 | live | 83206.34 | 81.26 | 82770.70–83548.23 | 318.79 | 3525 | 0 |
| 1024 | policy | 82701.39 | 80.76 | 82310.40–83175.68 | 392.63 | 3535 | 0 |
| 2048 | rigid | 188927.84 | 92.25 | 188247.47–189656.45 | 525.84 | 4217 | 0 |
| 2048 | recovery | 367027.94 | 179.21 | 366306.47–367781.40 | 709.66 | 4733 | 0 |
| 2048 | live | 125710.14 | 61.38 | 125317.02–125960.65 | 258.63 | 4731 | 0 |
| 2048 | policy | 124977.50 | 61.02 | 124152.71–125521.53 | 544.79 | 4743 | 0 |
| 32768 | rigid | 772715.25 | 23.58 | 761540.08–783472.28 | 11332.90 | 32691 | 0 |
| 32768 | recovery | 493423.87 | 15.06 | 491362.57–495522.76 | 2212.34 | 39829 | 0 |
| 32768 | live | 310412.27 | 9.47 | 307384.69–313301.68 | 2571.52 | 39829 | 0 |
| 32768 | policy | 310609.92 | 9.48 | 306999.95–313294.72 | 2830.43 | 39953 | 0 |
| 49152 | rigid | 845095.93 | 17.19 | 841038.64–849676.75 | 3940.59 | 47889 | 0 |
| 49152 | recovery | 507383.96 | 10.32 | 498504.25–517367.26 | 9911.66 | 58627 | 0 |
| 49152 | live | 324477.61 | 6.60 | 319295.39–329698.93 | 5121.00 | 58611 | 0 |
| 49152 | policy | 324233.09 | 6.60 | 319498.01–329424.97 | 5130.88 | 58763 | 0 |
| 61440 | rigid | 872767.76 | 14.21 | 864968.98–881202.17 | 8196.70 | 59347 | 0 |
| 61440 | recovery | 519383.15 | 8.45 | 517178.55–520334.96 | 1324.04 | 72675 | 0 |
| 61440 | live | 329512.88 | 5.36 | 326797.48–332076.47 | 2632.34 | 72693 | 0 |
| 61440 | policy | 329178.40 | 5.36 | 326435.18–331810.02 | 2778.00 | 72895 | 0 |

## Standalone recovery preparation

This untimed call compiles/captures the graph actually measured,
after restoring the pre-integration snapshot. Its time includes
snapshot restoration, JIT, synchronization and the errno check;
timed calls are unchanged.

| B | Seed | Preparation calls | Preparation seconds including JIT |
|---|---|---|---|
| 1024 | 510000 | 1 | 5.512774 |
| 1024 | 623001 | 1 | 5.335154 |
| 2048 | 510000 | 1 | 5.003910 |
| 2048 | 623001 | 1 | 5.309199 |
| 32768 | 510000 | 1 | 0.162711 |
| 32768 | 623001 | 1 | 0.168624 |
| 49152 | 510000 | 1 | 0.203768 |
| 49152 | 623001 | 1 | 0.208428 |
| 61440 | 510000 | 1 | 0.226283 |
| 61440 | 623001 | 1 | 0.224358 |

## Matched rigid overhead

Step-time increase and throughput decrease are different quantities.

| B | Seed | Scope | Step time +% | Throughput −% |
|---|---|---|---|---|
| 1024 | 510000 | live | 38.15 | 27.61 |
| 1024 | 510000 | policy | 38.87 | 27.99 |
| 1024 | 623001 | live | 38.17 | 27.62 |
| 1024 | 623001 | policy | 39.13 | 28.12 |
| 2048 | 510000 | live | 50.03 | 33.35 |
| 2048 | 510000 | policy | 50.83 | 33.70 |
| 2048 | 623001 | live | 50.55 | 33.58 |
| 2048 | 623001 | policy | 51.51 | 34.00 |
| 32768 | 510000 | live | 147.45 | 59.59 |
| 32768 | 510000 | policy | 147.48 | 59.59 |
| 32768 | 623001 | live | 150.46 | 60.07 |
| 32768 | 623001 | policy | 150.10 | 60.02 |
| 49152 | 510000 | live | 163.08 | 61.99 |
| 49152 | 510000 | policy | 163.31 | 62.02 |
| 49152 | 623001 | live | 157.80 | 61.21 |
| 49152 | 623001 | policy | 157.96 | 61.23 |
| 61440 | 510000 | live | 164.54 | 62.20 |
| 61440 | 510000 | policy | 164.92 | 62.25 |
| 61440 | 623001 | live | 165.20 | 62.29 |
| 61440 | 623001 | policy | 165.36 | 62.31 |

## Same-GPU full-field output cost

Each pair uses the same physical GPU and source, with identical final budgets.

| B | Seed | Scope | Max env-step/s | Full env-step/s | Full step time +% | Additional native bytes | Max peak MiB | Full peak MiB |
|---|---|---|---|---|---|---|---|---|
| 1024 | 510000 | live | 85769.20 | 85057.17 | 0.837 | 110100480 | 3525 | 3653 |
| 1024 | 510000 | policy | 85465.13 | 84732.52 | 0.865 | 110100480 | 3535 | 3663 |
| 1024 | 623001 | live | 84538.58 | 83864.51 | 0.804 | 110100480 | 3525 | 3653 |
| 1024 | 623001 | policy | 84132.60 | 83501.01 | 0.756 | 110100480 | 3535 | 3663 |
| 32768 | 510000 | live | 308646.47 | 303392.07 | 1.732 | 3523215360 | 39829 | 43189 |
| 32768 | 510000 | policy | 309872.73 | 303871.04 | 1.975 | 3523215360 | 39953 | 43313 |
| 32768 | 623001 | live | 307773.25 | 302956.47 | 1.590 | 3523215360 | 39829 | 43189 |
| 32768 | 623001 | policy | 309022.14 | 303406.25 | 1.851 | 3523215360 | 39953 | 43313 |

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
| 61440 | 623001 | policy | 245760 | 184423 | 184423 | 0 | 61337 | 0 | 160117 | 0 |

## Numerical and work tails

The following are worst values across environments, not individual
environment GPU latencies. Audit work totals span warmup and trajectory.
Complete balance includes all six gauge rows. Force and moment budgets
remain 1e-8; the full residual budget remains max(1e-11 N, 1e-8 times
the RHS norm). No slow or failed environment is removed.

| B | Seed | Scope | Full residual / budget max | Wrench force rel max | Wrench moment rel max | Newton eval max | Retries max | Corrections total | Factor fallback total | Packed columns max | Face tasks max | Zero RHS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 510000 | live | 0.00893132 | 7.84407e-10 | 6.85061e-10 | 6 | 0 | 0 | 0 | 79 | 120 | 0.4118 |
| 1024 | 510000 | policy | 0.00959931 | 7.58893e-10 | 6.94155e-10 | 6 | 0 | 0 | 0 | 79 | 114 | 0.4118 |
| 1024 | 623001 | live | 0.00961436 | 7.37599e-10 | 6.61127e-10 | 5 | 0 | 0 | 0 | 77 | 130 | 0.4211 |
| 1024 | 623001 | policy | 0.0096259 | 7.19324e-10 | 6.54502e-10 | 6 | 0 | 0 | 0 | 79 | 129 | 0.4211 |
| 2048 | 510000 | live | 0.0099658 | 7.84406e-10 | 6.85061e-10 | 9 | 0 | 0 | 0 | 79 | 120 | 0.4161 |
| 2048 | 510000 | policy | 0.00903617 | 7.73928e-10 | 6.94156e-10 | 6 | 0 | 0 | 0 | 79 | 121 | 0.4161 |
| 2048 | 623001 | live | 0.00984014 | 7.376e-10 | 6.80153e-10 | 5 | 0 | 0 | 0 | 77 | 141 | 0.4181 |
| 2048 | 623001 | policy | 0.0099515 | 7.19324e-10 | 6.63326e-10 | 6 | 0 | 0 | 0 | 79 | 134 | 0.4181 |
| 32768 | 510000 | live | 0.00959565 | 7.8837e-10 | 6.98039e-10 | 94 | 3 | 0 | 0 | 79 | 158 | 0.4198 |
| 32768 | 510000 | policy | 0.00962625 | 7.75188e-10 | 7.04139e-10 | 91 | 3 | 0 | 0 | 79 | 156 | 0.4198 |
| 32768 | 623001 | live | 0.00948542 | 7.67415e-10 | 6.99284e-10 | 109 | 3 | 0 | 0 | 79 | 141 | 0.4191 |
| 32768 | 623001 | policy | 0.010928 | 7.76695e-10 | 6.98545e-10 | 90 | 5 | 0 | 0 | 79 | 141 | 0.4191 |
| 61440 | 510000 | live | 0.0102756 | 7.8837e-10 | 7.02899e-10 | 94 | 4 | 0 | 0 | 79 | 158 | 0.4195 |
| 61440 | 510000 | policy | 0.00952671 | 7.84414e-10 | 7.04139e-10 | 91 | 3 | 0 | 0 | 79 | 156 | 0.4195 |
| 61440 | 623001 | live | 0.0105997 | 7.67415e-10 | 7.00467e-10 | 109 | 4 | 0 | 0 | 79 | 147 | 0.4196 |
| 61440 | 623001 | policy | 0.0109887 | 7.87724e-10 | 7.01339e-10 | 90 | 5 | 0 | 0 | 79 | 154 | 0.4196 |

## Frozen-snapshot split profile

Separate synchronized stage passes are diagnostic; they are not added to
derive live throughput. Pressure subdivision is another separate pass.

| Snapshot | Stage | Wall ms |
|---|---|---|
| native-final-snapshot-profile-b1024-v9.json | begin_step | 0.017153 |
| native-final-snapshot-profile-b1024-v9.json | association_transform | 0.100387 |
| native-final-snapshot-profile-b1024-v9.json | anchor_search | 0.123429 |
| native-final-snapshot-profile-b1024-v9.json | candidate_weights_constraints | 1.114519 |
| native-final-snapshot-profile-b1024-v9.json | nodal_scatter | 0.582315 |
| native-final-snapshot-profile-b1024-v9.json | inertia_relief_centrifugal | 0.121827 |
| native-final-snapshot-profile-b1024-v9.json | rhs_reduction | 0.041439 |
| native-final-snapshot-profile-b1024-v9.json | linear_solve | 0.803825 |
| native-final-snapshot-profile-b1024-v9.json | complete_residual | 0.343771 |
| native-final-snapshot-profile-b1024-v9.json | global_peak | 0.247860 |
| native-final-snapshot-profile-b1024-v9.json | accept | 0.027524 |
| native-final-snapshot-profile-b32768-v9.json | begin_step | 0.017093 |
| native-final-snapshot-profile-b32768-v9.json | association_transform | 0.126136 |
| native-final-snapshot-profile-b32768-v9.json | anchor_search_candidate_weights_constraints_nodal_scatter | 23.829006 |
| native-final-snapshot-profile-b32768-v9.json | inertia_relief_centrifugal | 1.555575 |
| native-final-snapshot-profile-b32768-v9.json | rhs_reduction | 0.530703 |
| native-final-snapshot-profile-b32768-v9.json | linear_solve | 22.462729 |
| native-final-snapshot-profile-b32768-v9.json | complete_residual | 10.303036 |
| native-final-snapshot-profile-b32768-v9.json | global_peak | 6.869763 |
| native-final-snapshot-profile-b32768-v9.json | accept | 0.057365 |

## Changing-contact stream profiles

Nested CUDA-stream intervals overlap and must not be summed. They include
host preparation/dispatch idle gaps. Intrusive event instrumentation changes
timing; ordinary benchmark rates above remain the throughput evidence.

| Profile | Stage | Calls | CUDA interval mean ms | CUDA p99 ms | CUDA max ms | Host submission mean ms |
|---|---|---|---|---|---|---|
| native-final-rollout-profile-live-max-b1024-v9.json | contact_parameter_update | 1200 | 0.0402 | 0.0886 | 0.1367 | 0.0545 |
| native-final-rollout-profile-live-max-b1024-v9.json | controller_and_scene_step | 1200 | 11.6434 | 15.2375 | 173.7312 | 11.5682 |
| native-final-rollout-profile-live-max-b1024-v9.json | native_stress_and_rigid_post | 1200 | 3.7379 | 4.2097 | 166.2955 | 4.8606 |
| native-final-rollout-profile-live-max-b1024-v9.json | partial_reset | 600 | 0.9628 | 1.3750 | 1.6586 | 4.5221 |
| native-final-rollout-profile-live-max-b1024-v9.json | rigid_substep_including_recovery | 1200 | 10.8120 | 13.3685 | 173.6828 | 8.4729 |
| native-final-rollout-profile-live-max-b1024-v9.json | scene_step | 1200 | 10.9019 | 13.5334 | 173.6912 | 8.7720 |
| native-final-rollout-profile-live-full-b1024-v9.json | contact_parameter_update | 1200 | 0.0387 | 0.0786 | 0.2295 | 0.0541 |
| native-final-rollout-profile-live-full-b1024-v9.json | controller_and_scene_step | 1200 | 11.7143 | 15.2513 | 175.5744 | 11.6381 |
| native-final-rollout-profile-live-full-b1024-v9.json | native_stress_and_rigid_post | 1200 | 3.8302 | 4.2998 | 169.0122 | 4.8582 |
| native-final-rollout-profile-live-full-b1024-v9.json | partial_reset | 600 | 0.9551 | 1.3359 | 2.6209 | 4.6239 |
| native-final-rollout-profile-live-full-b1024-v9.json | rigid_substep_including_recovery | 1200 | 10.8989 | 13.4897 | 175.3972 | 8.5058 |
| native-final-rollout-profile-live-full-b1024-v9.json | scene_step | 1200 | 10.9861 | 13.6329 | 175.5335 | 8.8046 |
| native-final-rollout-profile-policy-max-b1024-v9.json | contact_parameter_update | 1200 | 0.0390 | 0.0796 | 0.1415 | 0.0549 |
| native-final-rollout-profile-policy-max-b1024-v9.json | controller_and_scene_step | 1200 | 11.5917 | 15.0274 | 176.5782 | 11.1459 |
| native-final-rollout-profile-policy-max-b1024-v9.json | native_stress_and_rigid_post | 1200 | 3.7349 | 4.2005 | 167.2038 | 4.8121 |
| native-final-rollout-profile-policy-max-b1024-v9.json | observation | 1200 | 0.0223 | 0.2446 | 0.5672 | 0.2289 |
| native-final-rollout-profile-policy-max-b1024-v9.json | partial_reset | 600 | 0.9559 | 1.3270 | 1.7228 | 4.1345 |
| native-final-rollout-profile-policy-max-b1024-v9.json | policy_inference | 1200 | 0.0360 | 0.1407 | 0.1554 | 0.1307 |
| native-final-rollout-profile-policy-max-b1024-v9.json | rigid_substep_including_recovery | 1200 | 10.7685 | 13.3967 | 174.9466 | 8.2445 |
| native-final-rollout-profile-policy-max-b1024-v9.json | scene_step | 1200 | 10.8575 | 13.5537 | 175.1002 | 8.5224 |
| native-final-rollout-profile-policy-full-b1024-v9.json | contact_parameter_update | 1200 | 0.0383 | 0.0792 | 0.2641 | 0.0540 |
| native-final-rollout-profile-policy-full-b1024-v9.json | controller_and_scene_step | 1200 | 11.6921 | 15.2327 | 175.5379 | 11.2476 |
| native-final-rollout-profile-policy-full-b1024-v9.json | native_stress_and_rigid_post | 1200 | 3.8257 | 4.3018 | 166.2628 | 4.8116 |
| native-final-rollout-profile-policy-full-b1024-v9.json | observation | 1200 | 0.0226 | 0.2402 | 0.5462 | 0.2281 |
| native-final-rollout-profile-policy-full-b1024-v9.json | partial_reset | 600 | 0.9808 | 2.1895 | 2.6534 | 4.2553 |
| native-final-rollout-profile-policy-full-b1024-v9.json | policy_inference | 1200 | 0.0360 | 0.1412 | 0.1650 | 0.1303 |
| native-final-rollout-profile-policy-full-b1024-v9.json | rigid_substep_including_recovery | 1200 | 10.8579 | 13.4523 | 173.9018 | 8.2880 |
| native-final-rollout-profile-policy-full-b1024-v9.json | scene_step | 1200 | 10.9477 | 13.6253 | 174.0548 | 8.5721 |
| native-final-rollout-profile-live-max-b32768-v9.json | contact_parameter_update | 1200 | 0.1626 | 0.2058 | 0.3059 | 0.0594 |
| native-final-rollout-profile-live-max-b32768-v9.json | controller_and_scene_step | 1200 | 101.4470 | 114.3838 | 135.2349 | 100.4126 |
| native-final-rollout-profile-live-max-b32768-v9.json | native_stress_recovery | 1200 | 61.3085 | 67.0980 | 89.4406 | 0.3891 |
| native-final-rollout-profile-live-max-b32768-v9.json | partial_reset | 600 | 1.3530 | 1.8936 | 2.7409 | 68.6289 |
| native-final-rollout-profile-live-max-b32768-v9.json | rigid_contact_force_postprocess | 1200 | 0.1160 | 0.1300 | 0.2050 | 18.1311 |
| native-final-rollout-profile-live-max-b32768-v9.json | rigid_integration_after_stress | 1200 | 2.4291 | 2.4727 | 2.4773 | 0.1960 |
| native-final-rollout-profile-live-max-b32768-v9.json | rigid_substep_including_recovery | 1200 | 100.2866 | 112.1274 | 132.7421 | 62.4176 |
| native-final-rollout-profile-live-max-b32768-v9.json | scene_step | 1200 | 100.3780 | 112.2996 | 132.8928 | 65.3929 |
| native-final-rollout-profile-live-full-b32768-v9.json | contact_parameter_update | 1200 | 0.1689 | 0.2335 | 0.2803 | 0.0630 |
| native-final-rollout-profile-live-full-b32768-v9.json | controller_and_scene_step | 1200 | 103.6908 | 117.1319 | 137.2307 | 102.6220 |
| native-final-rollout-profile-live-full-b32768-v9.json | native_stress_recovery | 1200 | 63.4693 | 69.5759 | 91.7774 | 0.4251 |
| native-final-rollout-profile-live-full-b32768-v9.json | partial_reset | 600 | 1.4935 | 2.3420 | 3.0841 | 70.9245 |
| native-final-rollout-profile-live-full-b32768-v9.json | rigid_contact_force_postprocess | 1200 | 0.1189 | 0.2014 | 0.2920 | 18.1313 |
| native-final-rollout-profile-live-full-b32768-v9.json | rigid_integration_after_stress | 1200 | 2.4238 | 2.4686 | 2.4750 | 0.2153 |
| native-final-rollout-profile-live-full-b32768-v9.json | rigid_substep_including_recovery | 1200 | 102.4404 | 114.5995 | 134.8985 | 63.3304 |
| native-final-rollout-profile-live-full-b32768-v9.json | scene_step | 1200 | 102.5369 | 114.7625 | 135.0477 | 66.4140 |
| native-final-rollout-profile-policy-max-b32768-v9.json | contact_parameter_update | 1200 | 0.1641 | 0.2111 | 0.2339 | 0.0639 |
| native-final-rollout-profile-policy-max-b32768-v9.json | controller_and_scene_step | 1200 | 101.6311 | 114.9332 | 124.2829 | 100.1899 |
| native-final-rollout-profile-policy-max-b32768-v9.json | native_stress_recovery | 1200 | 61.3657 | 67.2785 | 76.4893 | 0.4824 |
| native-final-rollout-profile-policy-max-b32768-v9.json | observation | 1200 | 0.0546 | 0.3278 | 0.6602 | 0.2957 |
| native-final-rollout-profile-policy-max-b32768-v9.json | partial_reset | 600 | 1.5289 | 2.1744 | 3.3205 | 68.2315 |
| native-final-rollout-profile-policy-max-b32768-v9.json | policy_inference | 1200 | 0.0953 | 0.1932 | 0.2257 | 0.1741 |
| native-final-rollout-profile-policy-max-b32768-v9.json | rigid_contact_force_postprocess | 1200 | 0.1182 | 0.1362 | 0.2095 | 18.1141 |
| native-final-rollout-profile-policy-max-b32768-v9.json | rigid_integration_after_stress | 1200 | 2.4290 | 2.4725 | 2.4815 | 0.2539 |
| native-final-rollout-profile-policy-max-b32768-v9.json | rigid_substep_including_recovery | 1200 | 100.3575 | 112.3946 | 121.6652 | 62.2973 |
| native-final-rollout-profile-policy-max-b32768-v9.json | scene_step | 1200 | 100.4574 | 112.5805 | 121.8330 | 65.2634 |
| native-final-rollout-profile-policy-full-b32768-v9.json | contact_parameter_update | 1200 | 0.1698 | 0.2199 | 0.2788 | 0.0649 |
| native-final-rollout-profile-policy-full-b32768-v9.json | controller_and_scene_step | 1200 | 103.8359 | 117.4160 | 126.9054 | 102.3655 |
| native-final-rollout-profile-policy-full-b32768-v9.json | native_stress_recovery | 1200 | 63.4973 | 69.5783 | 79.0196 | 0.4819 |
| native-final-rollout-profile-policy-full-b32768-v9.json | observation | 1200 | 0.0554 | 0.3248 | 0.7128 | 0.2934 |
| native-final-rollout-profile-policy-full-b32768-v9.json | partial_reset | 600 | 1.6459 | 2.2574 | 2.9530 | 70.4940 |
| native-final-rollout-profile-policy-full-b32768-v9.json | policy_inference | 1200 | 0.0957 | 0.1967 | 0.2283 | 0.1741 |
| native-final-rollout-profile-policy-full-b32768-v9.json | rigid_contact_force_postprocess | 1200 | 0.1181 | 0.1372 | 0.2049 | 18.1187 |
| native-final-rollout-profile-policy-full-b32768-v9.json | rigid_integration_after_stress | 1200 | 2.4263 | 2.4682 | 2.4774 | 0.2494 |
| native-final-rollout-profile-policy-full-b32768-v9.json | rigid_substep_including_recovery | 1200 | 102.4903 | 114.7248 | 124.1032 | 63.2373 |
| native-final-rollout-profile-policy-full-b32768-v9.json | scene_step | 1200 | 102.5925 | 114.8807 | 124.2764 | 66.3099 |

All per-environment p50/p90/p99/max work distributions, exact coefficients,
reset counts, cache hit/miss/eviction counts, byte inventories, source hashes
and original paths are preserved in the summary and its hashed artifacts.
