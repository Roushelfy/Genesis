# Automatic DPR throughput optimization

The [remote throughput goal](DPR_AUTOMATIC_RADIUS_GOAL_20261023.md#immediate-goal-maximize-native-automatic-dpr-throughput)
is delivered by production source `db84114d1`. On the same RTX PRO 6000 Blackwell,
B64 full-scene Hertz DPR falls from 4.5836 to 3.1653 ms: **30.94% lower latency,
44.81% higher throughput**. The original pressure support, supplied forces and
friction, prefactorized FEM, complete max/full outputs and numerical gates remain.
The useful measured batch is B16384: 364,495 max / 356,895 full environment-steps/s.
This is a performance and numerical-consistency result, not physical sharp-edge
peak qualification. The [original physical limitations](AUTOMATIC_DPR_IMPLEMENTATION_20261023.md)
and failed exact-bitwise audit remain unchanged.

## Configuration and reproduction

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="elastic.npz",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",  # Explicit footprint approximation at sharp box edges.
        quadrature=16,
        output_mode="full",
    )
)
```

Moment reuse now defaults on for automatic-radius Hertz pressure on eligible
CUDA paths at every batch size. Keep the existing packed surface inverse,
FP64 arithmetic/storage and default scheduling. Choose `max` when full fields
are unnecessary. B4/B64 suit low latency; B16384 suits throughput on the tested
bank/device. B8192 uses less memory with about 9% lower full-output throughput.
The ordinary `examples.rigid.franka_egg_stress` entrypoint accepts
`--radius-model hertz --pressure-profile hertz --box-edge-policy planar --quadrature 16`.
No research callback is required.

The [evidence packet](../evidence/20261023-automatic-dpr-throughput-v1/) includes
a portable B64 example, both condition fixtures and the exact Slurm reproduction
command. It uses the established data directory, image `1_26`, the existing
data-side environment and private node caches. Full fields stay on the GPU
during primary timing. No new environment, data migration or home-data fallback
was needed.

## Matched complete-scene measurements

Production source is `db84114d1`; the direct old-Hertz control is `40775c1d0`.
The elastic mesh has 480 tetrahedra and 2,430 DOFs. All cases use FP64, q16,
0.01 s steps, one substep and the same rigid geometry, materials, controller,
source forces and update cadence within each condition bank. Manual DPR keeps
the bank's authored radii; radius-only changes sizing while retaining the DPR
pressure profile. Hertz additionally changes the pressure profile. Their costs
therefore compare declared model configurations, not identical stress fields.

Each row below shares one GPU allocation. Values are median ms/batch-step;
paired entries are max / full. B4/B64 use five 130-frame repeats, B16384 uses
three, and the diverse B8 bank uses five 160-frame repeats. Matching warmups
precede timing; each repeat restarts the saved initial state. Profiling and
research exports are disabled.

| Bank and batch | Rigid | Manual DPR | Radius-only DPR | Automatic Hertz DPR | Full Hertz added vs rigid / manual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four conditions, B4 | 1.6700 | 2.5730 / 2.5786 | 5.1975 / 5.2322 | 3.0854 / 3.0916 | 1.4216 / 0.5130 |
| Four conditions, B64 | 1.7016 | 2.6482 / 2.6476 | 5.2932 / 5.2873 | 3.1673 / 3.1653 | 1.4637 / 0.5177 |
| Four conditions, B16384 | 10.8343 | 32.0433 / 33.1829 | 59.9249 / 60.9599 | 44.9499 / 45.9070 | 35.0727 / 12.7241 |
| Eight diverse conditions, B8 | 2.1550 | 3.1666 / 3.1821 | 7.6132 / 7.6227 | 5.0482 / 5.0721 | 2.9171 / 1.8900 |

The same-allocation old B64 full Hertz control is 4.5836 ms, with a
4.5754–4.5903 ms repeat range. Final full Hertz repeat ranges are
3.0828–3.1124 ms (B4), 3.1611–3.1962 ms (B64), 45.8924–45.9321 ms (B16384),
and 5.0697–5.0740 ms (diverse B8). All repeats, GPU identities, source hashes,
settings, memory and invalid-step counts are in `timing.json`.

All 33 final primary configurations, covering 77,072,920 timed environment-steps,
have zero invalid steps. This count includes many tiled repeats, not that many
independent physical conditions. Final rigid positions match exactly within
each paired group. The diverse manual baseline invokes the complete native
workspace-overflow fallback 37 times per 160-step repeat in both output modes;
its cost is included and no load is truncated. All other final primary cases,
including optimized Hertz, have zero fallback calls.

The diverse bank contains eight independently seeded positions, yaw angles,
friction values, authored manual radii and staggered delays, with fresh inverse
kinematics. No case is filtered by admission. Its higher added Hertz cost shows
why the four-condition tiled result is not a universal workload claim.

## Useful batch and memory

The first sweep covered B4, B64, B256, B1024 and B4096. The final source confirms
the large-batch plateau with B8192, B16384 and B32768. The table uses the same
four-condition bank; max / full throughput is in environment-steps/s.

| Batch | Hertz max / full ms | Max / full throughput | Full native stress storage, GB | GPU used at end, GB |
| --- | ---: | ---: | ---: | ---: |
| 4 | 3.0854 / 3.0916 | 1,296 / 1,294 | 0.0690 | 2.55 |
| 64 | 3.1673 / 3.1653 | 20,206 / 20,219 | 0.0893 | 2.62 |
| 8192 | 24.8199 / 25.3411 | 330,058 / 323,269 | 2.8318 | 13.73 |
| 16384 | 44.9499 / 45.9070 | 364,495 / 356,895 | 5.5958 | 24.88 |
| 32768 | 90.2665 / 92.3522 | 363,014 / 354,815 | 11.1240 | 47.26 |

B32768 brings no throughput gain over B16384 and roughly doubles latency and
total stress storage. The measured useful optimum is therefore B16384.
This choice is specific to the bank, low-resolution mesh and GPU. Native stress
storage counts unique stress allocations; at B16384, max-only uses 3.8342 GB.
GPU-used values are total minus free memory at the end, including the runtime
and allocator, not process peak memory. Torch's separate peak allocation is
retained in the packet and is not presented as total GPU memory.

## Native changes and numerical checks

- Contained planar Hertz sections use exact Gram and P2/affine-pressure load
  moments. For section radius R, central covariance is
  `R**2 / 5 * (I - outer(n, n))` and mass is `2*pi*R**3/(3*radius)`; odd central
  moments vanish. Clipped and multiface supports retain their numerical rule.
- Each update packs intersecting faces, rejects exact misses and reuses refined
  pressure/load moments. Accepted affine pressures avoid a second integration
  during scatter; constrained fits and capacity overflow retain complete paths.
  Scratch is rebuilt each update and does not assume contact history.
- Face/ball sections are shared across samples. Each clipped fan ray reuses
  its support clipping across radial samples, and exactly zero-area fans are
  skipped. The quadrature nodes, weights, physical support and force direction
  remain unchanged; reduction order can differ.
- Converged Newton/line-search loops stop coherently, retaining their original
  limits and criteria. An already-refined failed fit stays failed. No wrench,
  FEM residual, normal/friction or stress comparison threshold is relaxed.

`automatic-dpr-opt-controls-v6` passes the original and new selected regressions:
**eight CPU and eight GPU tests**, including moving asymmetric contacts on
nonconvex inner walls, contact-slot changes, changing loads/friction, complete
release and recontact. Independent disk/sector P2 loads, same-mesh direct FEM
stress, explicit sharp-box statuses, cached moments and forced-overflow controls
pass. `automatic-dpr-opt-controls-q16-v1` repeats the independent controls at the
recommended q16 setting. Independent off-plane disk integration agrees with the
exact moments to about 2.91e-14 relative load error in the CPU control.

`automatic-dpr-opt-scenes-v4` passes the final default Hertz path at 130 frames,
B4 through the 600-frame controller cycle, B8 over the diverse 160-frame bank,
partial reset and unfused execution. Saved full-field comparison against the
original engine has maximum relative stress difference 8.22e-14, absolute
component difference 1.48e-8 Pa, and per-environment relative nodal-load
difference 2.91e-16. Rigid positions and source contact fields match exactly.
The cycle covers approach, closing, lifting, translation and opening commands;
it establishes numerical admission, not successful grasping or physical peaks.
The radius-only comparison also passes in `automatic-dpr-opt-scenes-v3`.

The cheaper-rule study `automatic-dpr-opt-quadrature-v1` uses `c3671fe62`.
q8's independent DPR edge-load error is 1.55e-9 against the unchanged 1e-9 gate;
q10/q12 primitive controls pass. All three Hertz scene runs reject the step-108
fit. These failures remain recorded, alongside the earlier original q10 failure.
q16 is the validated recommendation, not a claim that no future cheaper rule
can work. The original approximately 86% coarse-FEM physical discrepancy and
failed exact-bitwise audit remain in the starting evidence.

## Profiles, evaluated alternatives and stopping reasons

The following final-source profiles use the loaded frame-129 snapshot. They
are intrusive wall times in ms, not additive complete-scene costs. Radius-only
is part of association, and the pressure/scatter subdivisions overlap the
combined row.

| Stage | B64 | B16384 |
| --- | ---: | ---: |
| Association, including radius | 0.3960 | 0.8610 |
| Radius estimator alone | 0.2017 | 0.7511 |
| Anchor + pressure/fit + scatter | 0.4511 | 25.6861 |
| Pressure integration/fit subdivision | 0.2921 | 23.5278 |
| Scatter subdivision | 0.0755 | 1.8447 |
| FEM solve | 0.1975 | 8.7922 |
| Complete residual | 0.0739 | 5.1915 |
| Global peak/full output | 0.0857 | 4.2208 |

The isolated complete B64 full module is 1.0438 ms, compared with 1.0686 ms
before ray reuse. The B64 snapshot has 16 contained Hertz faces, 560 clipped
faces and 224 exactly zero-area fans. All 160 contacts have zero nonlinear-fit
evaluations there; integration dominates that snapshot's pressure work.

Quadrants' profiler omits captured CUDA graphs. Historical `stage_device_ms`
zeros mean unrecorded work, not zero device cost. New keys explicitly say
`stage_recorded_kernel_ms`. The final secondary `QD_GRAPH=0` profile records
B64 pressure/scatter kernel totals of 0.3980/0.0962 ms, versus synchronized wall
0.7087/0.2220 ms. It exposes 32 pressure/scatter offloaded kernels plus two
anchor kernels per separately called iteration; association has four, solve
five, residual five and peak five. Graph capture groups launches in production.
These instrumented, graph-disabled times change scheduling and cannot be
subtracted from production graph time to claim pure launch overhead.

Measured decisions are retained in the packet:

| Candidate | Observation and decision |
| --- | --- |
| Early exit alone | B64 full 4.6232 to 4.6186 ms: no material isolated gain. |
| Analytical/refined moment reuse | Same allocation: 4.6232 to 3.3577 ms. Main improvement. |
| Face-section reuse | Paired B64 full 3.3662 to 3.2630 ms; pressure subdivision 0.439 to 0.312 ms. Retained. |
| Ray reuse and zero fans | Paired B64 full 3.2799 to 3.2384 ms; B16384 full 50.3637 to 46.2823 ms. Retained. |
| Dense surface application | B64 full module 1.3928 vs packed 1.0443 ms. Declined. |
| Full inverse with FP32 storage | B64 full module 11.1408 ms. It keeps scene-precision arithmetic and loses the sparse surface/fused route; declined. |
| Packed solve blocks 512 / 256 / 0 | B16384 full 46.0564 / 46.0901 / 46.1501 ms. No material gain; default retained. |
| q8/q10/q12 | Original gates reject the cheaper scene rules. No tolerance change. |
| B32768 | No final-source throughput improvement over B16384; unnecessary memory/latency. |

The remaining large-batch bottleneck is clipped-support integration, followed
by the original FEM/residual/full-output work. Further material reduction would
need a better validated clipped integration rule or more extensive arithmetic
changes; the attempted cheaper rules fail. Fixed-shape curvature and collision
witnesses already avoid unnecessary geometry discovery, and radius estimation
is now a small fraction of the loaded large-batch profile. Adding cross-step
geometry history would require identity/invalidation complexity for limited
measured benefit. The sparse packed FP64 FEM and existing fused native graph
win the tested alternatives. No required contact, residual or global-field
work is dropped to obtain the reported result.

## Evidence and preservation

The packet retains 31 campaign records, all original setup/compiler/numerical
failures, final primary repeats and independent controls. The first publication
validator incorrectly required zero workspace fallbacks even for the diverse
manual baseline. Its failed log is retained; the corrected validator explicitly
requires the observed 37 complete fallback calls there and zero elsewhere,
while still requiring zero invalid steps in every final primary repeat.
This corrects metadata interpretation and changes no numerical gate or timing.

All 55 historical, 13 earlier automatic-DPR and eight optimization frozen
engines, and all 12 original unfinished user files, are preserved. Production
files match the tested `db84114d1` snapshot; later commits add audits/docs.
Runtime arrays and logs stay under the established
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/`
in `automatic-dpr-opt-*`. Small fixtures, concise results and receipts stay in
Git; full arrays and source inventories are identified by hashes and data paths.
