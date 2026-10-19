# Native contact consistency and traced SCR cost, 2026-10-19

A minimal MPR scratch change removes the strict B1/B8/B32 motion discrepancy while
preserving the original stress trajectories/fields within their declared
numerical targets. All349 combined motion/output/baseline checks pass, including
bitwise-equal complete mechanics against the repaired pure-rigid path.
Three focused MPR/GJK/patch witness and lifecycle regressions pass. The fix is
in task-branch commit58bbcdd5b; its isolated engine source is1ff39b077.
Canonical-source timing receipts also pass; larger-batch conformance and total
process-memory qualification remain independent work.

## Root localization and repaired behavior

The original complete15-case audit retained32 failed strict egg-motion checks
at B8/B32. A fresh rigid repeat was bitwise identical; manual/unrequested stress
and fused-post-off paths reproduced the original stress trajectory, so recovery
execution was not necessary. The1200-step controller-only check also found
bitwise-identical joint targets, phase, grip and force limits.

The corrected construction probe compares346 retained arrays.340 are identical;
the remaining six concern inactive contact pair indices and optional contact
witness storage shapes. Static configurations and rigid options are identical;
both paths already use deterministic mode with solver preference1. This excludes
the proposed timing-autotuner explanation for these measured runs.

At step108, all active sorted contacts and observed dynamics agree. At109,
one upstream contact-position component differs1.110223e-16m; contact force and
integrated dynamics then diverge. The next step also changes normal/penetration.
A384-portal local control independently reproduces witness-dependent MPR
position differences:81/80/81 of128 cases at geometry scales0.001/1/1000,
with maxima1.084e-19/2.220e-16/2.274e-13m. A shared-reciprocal candidate fails
to remove the discrepancy and was not promoted.

The accepted variant always retains MPR's two portal surface points in its
per-worker scratch. Their calculation already participates in constructing
the physical midpoint; making the layout independent of an auxiliary witness
consumer keeps its fast-math expression consistent. Optional long-lived contact
witness arrays remain conditional. The extra pure-rigid scratch is two3-component
vectors per actual MPR worker, not per maximum candidate contact. Stress-enabled
MPR already held these vectors. Exact total process-memory effects still require
measurement; logical/Torch buffers are not total GPU process peaks.

The factory no longer accepts an unused witness flag. Internal callers are
updated, and the MPR test strengthens active contact positions to exact equality.
The older tetrahedron fixture passes both before and after; it is not the
standalone reproducer of the amplified rollout issue. The complete old B8
cohort's16 failed strict checks and the repaired89-check cohort provide that
integration contrast. Original MPR/GJK lifecycle assertions are preserved.

## Verified B1/B8/B32 scope and source identities

The repaired cohort covers pure rigid, DPR max/full and SCR max/full over1200
steps after900 warmup, using the same eight-condition bank, dt0.01/substeps1,
level1 mesh64 geometry, MPR/patch-off, history0 and all original runtime checks.
Complete motion is bitwise identical within this cohort. Relative to original
stress runs, maximum complete-tensor differences are9.027e-7Pa for DPR and
4.509e-5Pa for SCR, passing the original rtol1e-8/atol1e-3Pa comparison. Full
finite fields, independent Mises, max reduction and every-step validity pass.
This is numerical conformance, not physical collider-load accuracy.

The separately completed B1/B32 controls join the original repaired B8 cohort
through an assembly manifest pointing to immutable terminal directories. Audit
729201 passes349 checks over all15 cases, including condition-matched batching
and comparisons to the original stress trajectories. The original failed
259-check audit and the initial89-check B8 repair remain separate evidence.

The engine fix is promoted only to the task branch and isolated production
source. The analysis worktree's baseline engine remains unchanged while long
reference writers use it; their research modules were captured at submission.
The packet includes the complete variant patch against92692aa61, so its exact
engine/test source can be reconstructed without relying on a mutable checkout.
No active source or reference output is hot-edited.

## Trace evidence and bounded workspace tuning

The changing-contact60-step trace measures7.403ms inclusive host time per SCR
contact pipeline and3.627ms in its three scalar readbacks per step, including
two profile-capacity checks. These are nested wall scopes; readbacks can wait
for prior GPU work and their duration is not a removable-overhead estimate.
The largest observed GPU kernels are traction fitting3.949ms/step, cached
traction scatter1.732ms and source resolution0.699ms. This identifies both
reconstruction work and synchronization to investigate. No speedup is inferred
by summing overlapping observer times or subtracting them from primary timing.

The raw trace was fully exported after runtime checks, but its final metadata
write failed on a NumPy boolean. The independent parser retains that failure
and counts60 CPU step annotations; the first parser also incorrectly included
120 GPU annotations and its rejection is retained. NumPy scalar conversion is
now confined to the JSON boundary in the collectors.

Contact-block1/4 variants grow to the actual nine-source workspace and preserve
all contacts/complete mechanics. Their full tensor-array contrasts against
block16 are6.853e-11/8.213e-11. Median module rates are930.44/963.92 env-updates/s,
versus980.56 in the earlier block16 allocation. This is not an isolated same-GPU
speed comparison and provides no demonstrated improvement to recommend.
No source is truncated and all numerical gates remain enabled.

## Evidence and next validation

Canonical repaired-source jobs729194/729236 complete and audit all15 primary
timings. Their exact source/configuration matches the separately admitted
B1/B8/B32 conformance cohorts. Each rate below is the median of three warm
1200-step windows, with zero runtime invalid-step counters:

| Batch | Rigid | DPR max | DPR full | SCR max | SCR full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 541.84 | 327.98 | 326.67 | 108.53 | 110.52 |
| 8 | 2885.44 | 2155.58 | 2145.17 | 656.64 | 654.74 |
| 32 | 11430.27 | 8434.60 | 8435.38 | 2603.79 | 2604.72 |

Units are aggregate env-scene-step/s at dt0.01 and substeps1. These are
independent measurements of the repaired engine, not adjusted old rates or a
controlled before/after speedup. The [canonical timing packet](../evidence/20261019-qualified-ranking/native-canonical-production-v1/README.md)
retains terminal runs, CSV, exact inputs, source hashes and audit receipts.
Pure-rigid/max/full within each batch use the same GPU. These are warm repeats
within one process per mode, not independent process repetitions.

The [localization packet](../evidence/20261019-qualified-ranking/native-contact-consistency-v1/README.md)
retains307 verified payloads totaling3,290,067 bytes, including all failed
metadata/local candidates, the89-check repair, lifecycle results and immutable
follow-up inputs. Full physical snapshots, replay inputs and the18MB raw trace
remain on data with exact hashes.

The [follow-up packet](../evidence/20261019-qualified-ranking/native-contact-conformance-scaling-v1/README.md)
retains terminal B1/B32, the349-check combined audit, B128/B512 timing receipts
and immutable inputs for canonical timing, larger conformance and independently
generated condition banks. It does not replace any localization payload.

Audit729202 verifies all ten B128/B512 timing receipts, source hashes and zero
runtime invalid-step counters. These repeat the frozen eight-condition bank;
they do not establish diverse-input capacity. Median env-scene-step/s are:

| Batch | Rigid | DPR max | DPR full | SCR max | SCR full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 44163 | 31817 | 31843 | 10704 | 10791 |
| 512 | 171798 | 103694 | 101303 | 35752 | 37315 |

Each five-mode batch shares a GPU and uses three warm1200-step windows.
Different batches may use different physical GPUs. Full output stays on GPU;
there is no policy inference or full-field host-output cost in these rates.
Larger complete motion/output conformance and total process-memory measurements
remain pending, so these are measured timing baselines, not a capacity recommendation.
Collider6mm source errors
remain pending and cannot inherit the qualified2mm full-centroid pad errors.
The original pad every-step friction failures remain distinct and unchanged.
The full goal remains active until fine friction qualification, matched delivery
and remaining completion requirements are verified.
