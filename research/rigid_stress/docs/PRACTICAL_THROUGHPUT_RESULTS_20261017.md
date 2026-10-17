# Practical native rigid stress throughput, 2026-10-17

**Measurements and operating-point selection are closed; final figure exports
and terminal completion audit remain open.** See the
[completion audit](PRACTICAL_THROUGHPUT_COMPLETION_AUDIT_20261017.md).

The selected robust workload gives a **50.845% same-GPU code improvement** at
B8192 with the original history4 setting. A separate, independently qualified
history4/history0 comparison gives **34.494% configuration improvement**.
The recommended changing-contact setting is **history0, B16384**: 116,912 live
and 116,695 policy valid env-step/s, with sampled process peaks about40.53GiB.
It reaches98.30%/98.41% of the highest repeated live/policy rates at B32768,
using about half the observed memory and batch latency.

## Workload and comparison scope

The workload retains the480 P2-tet/2430-DOF full hollow egg, original material
and fixed mass, native Franka controller, eight-condition bank and seed623016,
nominal6mm radius schedule, quadrature10, ports8, intrinsic body_source_prior,
mesh64 collider, MPR/patch-off, dt0.01 and one substep. Production stress
arithmetic remains Quadrants FP64. The policy scope adds the frozen
26/128/128/7 inference network and rollout work; it measures inference rather
than full RL training.

Normal rates use900 warmup steps and three2400-step repeats. Memory, work and
profile observations are separate intrusive runs. Accepted measurements close
actual program/wrapper exits, successful terminal Slurm accounting, all899
source hashes, source inventory and the frozen condition-bank hash. Independent
metadata audits reread raw rates/options and reject sampler/observer timings.

## Matched improvement and final scaling

| Same-GPU experiment | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| B8192 original/new code, fixed history4, A/B/B/A |52,719.54|79,524.92|1.50845|
| B8192 same selected code, history4/history0 |78,768.50|105,939.20|1.34494|
| B32 cache-only/new recovery code, A/B/B/A |2,256.89|2,317.37|1.02680|

Rates are median valid env-step/s. A/B/B/A sides each pool six raw repeats;
the configuration comparison has three per setting. These are distinct
experiments; multiplying their ratios would not constitute one combined
same-GPU measurement. The B32 source label explicitly denotes the cache-only
ancestor. The earlier unmatched low B32 observation remains in the ledger.

The closed history0 four-scope curve is:

| Batch | Rigid | Rigid + stress | Policy + rigid | Policy + rigid + stress |
| ---: | ---: | ---: | ---: | ---: |
|8,192|773,385.49|109,293.43|730,858.57|108,062.87|
|16,384|907,392.82|116,912.11|899,990.35|116,695.21|
|32,768|998,087.88|118,929.30|991,613.96|118,579.16|

| Batch | Live / policy batch latency, ms | Incremental stress live / policy, ms per batch | Sampled peak live / policy, GiB |
| ---: | ---: | ---: | ---: |
|8,192|74.954 /75.808|64.362 /64.599|19.045 /21.418|
|16,384|140.139 /140.400|122.083 /122.195|40.527 /40.529|
|32,768|275.525 /276.339|242.694 /243.293|78.746 /78.898|

Incremental cost subtracts the matched rigid-only scope's median batch latency.
These curves use comparable allocations, with the same physical workload and
one source manifest. The paired experiments above isolate code/configuration
changes on one GPU. Aggregate env-step/s is distinct from an individual
simulation's frame rate.

The original history4 workload has a B8192–B24576 shoulder around53k env-step/s.
The selected history4 curve closes six widths through B32768: B16384 gives
83,587.26 live/83,251.21 policy and B32768 gives85,841.03/85,651.18. Doubling
batch adds only2.70%/2.88%. For history0, B8192→B16384 adds6.97%/7.99%;
B16384→B32768 adds only1.73%/1.61%. This establishes the measured practical
shoulder and supports B16384 when memory/latency matter. B32768 supplies the
highest repeated aggregate rates in the tested normal curve.
The [warmup contact-mix check](PRACTICAL_THROUGHPUT_CONTACT_MIX_20261017.json)
finds identical normalized contact-count distributions at all seven original,
six selected history4 and three selected history0 normal widths (mean3.281944,
maximum9). These every10-step warmup counts are separate from complete timed
observer fractions; no batch-dependent contact-mix change is observed here.

Sources: [B8192](PRACTICAL_THROUGHPUT_HISTORY0_OPERATING_POINT_20261017.json),
[B16384 and final memory](PRACTICAL_THROUGHPUT_HISTORY0_WIDE_20261017.json),
[B32768](PRACTICAL_THROUGHPUT_HISTORY0_FINAL_SCALING_20261017.json),
[history4 wide points](PRACTICAL_THROUGHPUT_PACKED_HISTORY_FINAL_SCALING_20261017.json)
and the [experiment ledger](PRACTICAL_THROUGHPUT_EXPERIMENTS_20261017.md).
Final figure previews/exports and their independent complete metadata audits
are being assembled; the stage remains open until these artifacts close.

## What changed and the remaining bottlenecks

The geometry cache stores current fan samples as two barycentric coordinates
and reuses geometry/P2 shapes across mapping, fit and scatter passes. Contact
changes update the samples without changing physical footprints. Its isolated
B2048 timing change was effectively flat; it is retained for usable capacity:
at unchanged initial workspace2, original code OOMs at B40960 while the cache
completes the full untruncated contact workload.

History misses now use the existing exact boundary-force/centrifugal response
instead of the full dense inverse. Valid history predictions remain intact;
complete residual/gauge checks, inverse corrections, factor fallback and raw
global peak/full-output rules remain. In separate600-step moving B8192 profiles,
cache-only→selected history4 elastic recovery falls58.768→5.281ms/step and
CPU step annotation159.717→108.024ms. History still costs26.050ms with only
24.519% measured hits. Contact-witness and two-finger fractions remain
97.464% and56.292%, with zero observed retries, corrections, fallbacks or
invalid steps; a valid witness can carry zero force.

History0 removes prediction/maintenance and its preliminary prediction check.
Accepted output still passes complete equilibrium and all gauge rows every
step at the original tolerances. Its moving profile has CPU step77.541ms,
traction fitting19.567ms, surface sampling18.435ms, load assembly8.129ms,
elastic recovery5.324ms and residual checks2.665ms. These overlapping profile
categories do not add into independent wall-time components. CUDA memcpy API
spans overlap kernels by99.812% in the selected history4 trace; their duration
is not separate CPU overhead.

Sampling and constrained fitting are the largest remaining measured costs.
Bounded fit experiments give paired ratios1.003437 for one-warp scheduling,
1.000022 for fixed6x6 loop unrolling and0.964993 for lazy Hessian construction;
leader-only scheduling is effectively flat. Their sources, qualifications and
failed cases remain in the ledger. More efficient sampling/fitting remains an
opportunity requiring the original footprint/friction/wrench gates.

Cold setup in the first selected B8192 diagnostic is110.03s before policy
creation and warmup. Nested build/JIT subsets are0.311s topology,1.740s
operators,14.114s factor,0.611s inverse and0.329s surface inverse; these must
not be added again to setup. The900-step warmup includes physical work and
runtime compilation. Cold and profiled costs are excluded from headline rates.

## Capacity and recommended settings

The measured RTX PRO6000 Blackwell allocations expose101975851008 CUDA device
bytes (94.97GiB). The table's process peaks are independent100ms NVML sampled
lower bounds including setup/growth. They are distinct from native-buffer
counts. Different-process peak differences do not isolate policy memory.

With initial workspace2 growing to all9 actual contacts, selected history4
passes B40960 and OOMs at B43008; history0 passes B47104 and OOMs at B49152.
The history0 successful pass records113049600 valid transitions and a sampled
peak98098479104 bytes (91.36GiB). These are tested brackets for these settings,
with one intrusive timing pass; they are excluded from repeated headline rates
and do not establish an exact universal capacity limit.

Use intrinsic body_source_prior, nominal6mm/Q10/ports8, history0 and max output
for this changing-contact low-mesh workload. Choose B16384 for the measured
throughput/memory/latency balance, or B32768 for the highest demonstrated
repeated aggregate rate with about79GiB observed process memory. The normal
curves use initial workspace16. Workspace2 is separately labeled allocation
tuning with complete growth, useful when pursuing capacity. History4 remains
available for workloads with useful reuse. Full output remains available and
qualified at B16384; its throughput/memory was not measured by these max-mode
curves. Pressure remains a simplified comparator with the documented local
friction/admissibility defect.

## Numerical/application qualification and reproduction

CPU/GPU independent FP64 checks retain complete equilibrium including all six
gauge rows, full tensors/raw global maxima and correction/fallback behavior.
B16384 lifecycle checks cover checkpoint/reset isolation, the following live
step, a complete controller period and frozen complete release/recontact.
History0/history4 moving-field comparison passes all1370880 tensor components
at the original1e-4/1e-3Pa bounds, with maximum difference0.000156370Pa.
Both original force families retain three complete1200-step episodes.

The [current qualification recheck](PRACTICAL_THROUGHPUT_QUALIFICATION_RECHECK_20261017.json)
closes15 original scientific jobs, their raw JUnit/comparison/lifecycle records,
72 comparison-input hashes and all899 selected-source hashes/inventory.
All841 current engine files match the qualified source. One historical test
overlay differs in a corrected two-line comment with an unchanged AST; the
executed native recovery tests themselves match current tests.

The original0.2/4N model/radius ensemble retains7/8 above-band lift comparisons
and1/8 whole-episode comparisons. Load reconstruction bias and reference
uncertainty remain. Solver consistency does not calibrate physical damage;
force choices, phase windows and numerical tolerances were not retuned.
Full RL training, experimental fracture calibration and a new universal
high-mesh convergence campaign remain outside this stage.

The [running guide](RUNNING_PRACTICAL_THROUGHPUT.md),
[exact source/input recipes](../evidence/20261017-throughput/source-recipes-v1/README.md)
and [baseline figure packet](../evidence/20261017-throughput/baseline-scaling-v2/README.md)
provide commands, exact committed-source reconstruction and inspected baseline
exports. Final-source figure packets and the terminal completion decision
remain open.
