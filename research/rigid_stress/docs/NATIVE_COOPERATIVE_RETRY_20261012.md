# Cold cooperative traction retry

The selected CUDA route retries unresolved numerical or sampled-wrench fits
from a cold six-dimensional dual using the existing cooperative reduction.
Successful constraints retain every dual, scale, weight, status, error,
evaluation and certificate field. Evaluation counts include both attempts.
The serial fallback remains available for unresolved constraints. The CPU
engine route, physical footprint, Q10 sampling, local friction and final
wrench/residual/stress budgets are unchanged.

The same-GPU B2048/seed623102 ablation uses the complete low hollow shell,
FP64/Q10/s8, `body_source_prior`, max output, mesh64/GJK patch-on and
dt0.01/substep1. RTX PRO6000 Blackwell Server Edition has 97887MiB,
driver595.71.05, container1_26, Genesis1.4.3 and Quadrants1.3.3. Each route
warms up600 steps and measures three1280-step changing trajectories. Timed
control, contact updates, observations, metrics, resets and all fallback work
remain included.

| Route | Effective env-step/s | Batch step/s |
|---|---:|---:|
| Original retry |56621.634184|27.647282|
| Cold cooperative retry |58715.889184|28.669868|

The selected route is1.036986834x faster. All six repeats have zero invalid
steps,2323 selected-environment resets,3142 successful/zero failed completed
lift cycles and154804 passing/zero failing lift-window observations. Rigid
positions agree within1e-12m and every per-environment rigid grasp/contact/
descent counter agrees. Extra work is disclosed: maximum counted evaluations
increase270→398; first-repeat summed evaluations increase5178607→5179000.
Native stress allocation remains5498093150 bytes.

The ablation implementation passes23 CUDA cases in441.40s and24 independent
CPU FP64 full-field comparisons: maximum normalized wrench error
7.386596753e-10, complete residual budget fraction0.0452974583 and global
peak difference8.158705896e-6Pa. The production form removes the experiment
flag and passes23 CPU cases in303.03s and23 CUDA cases in436.96s. These tests
also verify that retry leaves all nine successful-constraint arrays bitwise
unchanged.

The production form has a separate two-pass1280-step dynamic latency audit.
Independent CPU stiffness assembly, gauge/factor and tensor/von Mises/global
peak calculations pass on320 captured GPU fields. Maximum complete residual
budget fraction is0.0060352792; maximum global peak difference is
2.314598532e-6Pa. Rigid positions between the two diagnostic passes agree
exactly; numerical failures are zero. The oracle receives captured balanced
RHS; it does not independently reconstruct dynamic contact fitting or relief.
Those have separate contact/full-field fixture oracles.

At hard tick934 the first serial retry drops from the earlier diagnostic
1.49s to0.156ms; the new cold cooperative attempt takes46.016ms. Whole-step
instrumented latency is123.175ms. Diagnostic fences and different allocations
prevent treating cross-run latency changes as a matched throughput comparison;
the same-GPU live ablation above establishes the selected speed benefit.
Three other refinement frames still have approximately0.74s serial retries
and approximately0.1s fan refinement. They remain included and actionable.

The v1 archive has248 entries/115 unique blobs/16029969 stored bytes;
manifest SHA256:
`8876ef02772a9603906bd0ba5c825c45721bade7254ec9562e9b289b5fac7834`.
The production-form v2 archive has299 entries/136 unique blobs/38152307
stored bytes; manifest SHA256:
`b023efb6c527b28760183527e4df0b1d8cf25bac3b16404d6f287e670cef8e17`.
Both publishers verify original/stored/decompressed hashes and source
closures. Their source patches, launchers, raw trajectories and test reports
are under `evidence/20261011-nonconvex/native-cooperative-retry-*`.
The private base90b2964 source closures contain current6a861556 math plus this
retry. The selected three source files are copied unchanged into the task
branch. Broader final-source batches/seeds, compact sample storage and the
remaining tail decisions are pending; this checkpoint is not a plateau.
