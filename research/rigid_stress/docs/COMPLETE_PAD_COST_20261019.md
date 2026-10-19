# Complete normal-pad cost matrix and admitted phase profiles

The normal0.1N B1/B4 every-step/sparse full/max cost matrix and separate
process-memory matrix now pass their own admission and provenance checks.
This completes this matched cost matrix, not the
[physical ranking goal](QUALIFIED_STRESS_RANKING_GOAL.md). Fine nonlinear
friction-reference qualification and final cross-case conclusions remain open.
The [packet](../evidence/20261019-qualified-ranking/complete-pad-cost-v1/README.md)
contains the executed aggregation, B4 primary receipts and independent phase
profiles; preceding packets retain B1 primary and all memory source records.

## Primary whole-workflow timing

`pad-primary-batch4-v1` (728283) completes all four groups and12 method/mode
runs; each has its own passing `audit_pad_batch`. The B1 runs and audits are
retained in the [preceding cost report](NORMAL_SPATIAL_COST_20261019.md).
`pad-cost-matrix-full-v2` (728458) rehashes591 audit inputs, verifies exact
start/result/cost/journal binding, physical configuration agreement and matched
GPU model/driver/capacity, and reconstructs all24 timing and12 memory rows.
Hardware is NVIDIA RTX PRO6000 Blackwell Server Edition, driver595.71.05,
97887MiB visible capacity. B4 uses four actual evolving environments; native
pad mechanics is serial per environment and included in the measurements.

All primary runs contain800 continuous steps at dt1ms: one200-step warmup
level and three200-step measured blocks at0.1N. These are consecutive warm
blocks, not independent-process statistical replicates. Each wall interval
includes native/pad or Shell integration, requested stress recovery, host
output, all per-step mechanical checks, endpoint fields/certificates and
step-journal I/O. Per-phase timers and intrusive memory sampling are disabled.
Every primary run passes its own complete history, source/cadence and endpoint
field/peak conformance checks; a separate profile never admits it.

Latency below is median [minimum, maximum] in milliseconds per **batch step**.
Sparse means one observation every20 physical steps, with endpoint output.

| Method / environments | Every-step full | Every-step max | Every20th full | Every20th max |
| --- | ---: | ---: | ---: | ---: |
| DPR / B1 | 76.916 [76.801, 77.142] | 68.331 [68.240, 68.569] | 16.629 [16.614, 16.767] | 16.331 [16.276, 16.341] |
| SCR / B1 | 1400.682 [1400.353, 1403.240] | 1384.356 [1384.193, 1385.525] | 82.728 [82.631, 82.825] | 82.368 [82.330, 82.581] |
| SHELL / B1 | 314.464 [255.251, 462.881] | 309.917 [249.458, 456.545] | 311.257 [251.709, 458.375] | 312.108 [251.301, 458.774] |
| DPR / B4 | 129.058 [129.049, 130.353] | 99.192 [97.814, 99.920] | 42.450 [42.347, 42.602] | 39.391 [39.349, 39.610] |
| SCR / B4 | 1530.621 [1529.523, 1530.635] | 1461.544 [1461.543, 1462.091] | 113.149 [113.004, 113.162] | 107.482 [107.080, 107.902] |
| SHELL / B4 | 501.180 [428.077, 689.822] | 477.271 [401.731, 665.945] | 484.149 [407.879, 666.571] | 478.937 [403.302, 663.590] |

Aggregate env-step/s equals B divided by batch-step seconds. For every-step
full output, the B1 medians are13.001/0.714/3.180 and the B4 medians are
30.994/2.613/7.981 for DPR/SCR/Shell. Sparse full-output B4 reaches
94.228/35.352/8.262 env-step/s. The complete CSV also retains every setup time,
observation byte count and endpoint peak-conformance measurement.

These numbers compare the declared research workflow, not isolated production
kernels or arbitrary-contact application throughput. Shell's substantial
within-run variation is visible; small Shell full/max or cadence differences
are not established speedups. Its contact/elastic solve still runs each step.
Native sparse recovery defers substantial work. The cost order changes with
observation cadence, so there is no cadence-independent method speed rank.
Accuracy conclusions remain those of the fixed configurations in the
[normal reference report](NORMAL_PAD_QUALIFICATION_20261019.md), including
uncertainty-supported ties and the subsequently measured Shell mesh sensitivity.

## Separate synchronized phase profiles

`pad-phase-profile-v1` (728400) completes all three400-step B1 full-output
profiles. Each has its own complete-history admission and field conformance:
relative complete-field differences from the admitted baselines are
3.31e-11 DPR,1.11e-9 SCR and5.70e-8 Shell. All three explicitly mark timing
non-primary. Process-local wrappers are restored after execution; scientific
solver source/arithmetic and acceptance gates are unchanged.

The first200-step level is warmup, and the second is profiled. Values below
are measured warm-phase totals divided by200, in ms per physical step.
Inclusive phases can contain other listed phases; **do not sum the rows** or
substitute them for primary wall times. Exact call counts, inclusive/exclusive
totals, preprocessing and native model build timings are retained in
`phases.json`/`phases.csv`. Exclusive totals subtract measured nested calls,
not every uninstrumented operation.

| Measured phase | DPR | SCR | Shell250um |
| --- | ---: | ---: | ---: |
| Physical-pad contact solve | 4.5783 | 4.5007 | part of coupled solve |
| Intrinsic contact reconstruction | not used | 1385.8510 | not used |
| Elastic stress reconstruction | 2.0075 | 2.0392 | part of coupled solve |
| Native observation inclusive / Shell observer update | 65.2322 | 1398.7218 | 0.7233 |
| Motion host reads | 0.5303 | 0.4631 | 0.9730 |
| Driver host reads | 0.1826 | 0.1633 | 0.0725 |
| Stress host reads inside observation | 1.0005 | 0.8422 | not separately wrapped |
| Shell coarse assembly | not used | not used | 1.0381 |
| Shell coarse factorization | not used | not used | 2.1849 |
| Shell contact detection | not used | not used | 0.3939 |
| Shell coupled contact/elastic solve | not used | not used | 428.2265 |

DPR's inclusive observation additionally contains anchor21.8616ms,
pressure9.4954ms and scatter20.4854ms, plus reconstruction, acceptance,
transfers and uninstrumented host work. SCR's intrinsic contact reconstruction
is the dominant measured phase in this all-source configuration. Shell's
observer row measures `FullField.update`; it does not include all the native
observer wrapper's host work, and the phase boundaries are not identical.
Shell contact and elastic correction are one coupled nonlinear solve; no separate
elastic/contact solve costs are invented. This one profiled warm block is
not a repeated performance estimate. The corresponding primary rows retain
three warm blocks and their dispersion.

## Memory, failures and reproduction

The [preceding memory checkpoint](FRICTION_RESOLUTION_MEMORY_20261019.md)
contains all12 separately admitted process-memory profiles, including exact
sample times, CPU max RSS and the3.85–6.53s maximum sampling gaps. This
aggregation verifies their receipts and configuration match again. Sampled
process peaks can miss short transients; they are not hard capacity limits
or sparse-cadence memory measurements.

The first aggregation728451 and fine-reference analysis728452 were submitted
with an environment-variable dependency that the scheduler did not adopt.
Their captured scheduler records show `Dependency=(null)`. They started early:
the cost analysis failed on a missing final Shell audit, and the reference
analysis retained a normal-only diagnostic while its shear analysis failed on
a missing frame. These records are preserved. Neither analysis admits its
upstream simulation. Original simulations were unaffected. The successful
cost aggregation728458 was submitted only after all four parent groups and
all12 own-run audits had completed successfully. Fine-reference analysis
requires a fresh post-completion run; its earlier partial output is not a
qualified friction comparison.

Rebuild the table using `pad_cost_matrix --data-root RUN_ROOT --batch-sizes
1 4 --output FRESH_DATA_PATH`. Reproduce a phase profile using
`pad_phase_profile --task DRIVER_TASK_JSON --data-root RUN_ROOT --output
FRESH_DATA_PATH`; task JSON specifies one of the captured B1 drivers and
includes `--primary-cost`, while the wrapper adds the explicit non-primary
phase flag. Use the retained task/source captures and the existing rtx-mid /
GENESIS_IMAGE_VER=1_26 workflow. Check upstream terminal results before
submitting dependent analyses; an unverified requested dependency is not enough.

This packet contains222 size/SHA256-verified payloads totaling2,345,974 bytes.
Large runtime arrays/journals and ongoing fine StVK references remain under
the established data-side root. No active writer or environment is moved.
