# Native DPR GPU throughput follow-up, 2026-10-25

This report accompanies the [active specification](GPU_THROUGHPUT_FOLLOWUP_GOAL_20261025.md)
and [execution record](GPU_THROUGHPUT_PROGRESS_20261025.md). The follow-up is complete within the numerical and performance scope below.
The [requirement audit](../evidence/20261025-gpu-throughput-v1/metadata/requirements-final.json)
and [evidence packet](../evidence/20261025-gpu-throughput-v1/README.md) retain all
40 campaigns, 201 task outcomes and 15 failed attempts/rejected configurations.

Measurements use NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (about
95 GiB), Quadrants 1.3.3, PyTorch 2.8.0, Python 3.10.12 and the established
Slurm image `1_26`. GPU identity, source hashes and job settings are retained
per run. Hardware peak specifications are not used as performance evidence.

## Integrated changes and recommended configuration

The winning pressure path builds a conservative per-contact shell face list
once in shared memory, reuses it throughout the constrained pressure fit, and
prepares each fixed physical Q4 projection once. A footprint touching more than
128 candidate faces uses the original exhaustive walk. Geometry, pressure law,
quadrature samples, nonlinear fit and acceptance thresholds remain the same.
There is no temporal contact cache or cross-environment scheduling readback.

Packed CUDA exact fallback at B>=1024 now compacts failed environment indices
on device and writes their solutions back to their original slots. A full-batch
masked route remains at small batches. The scratch list is rebuilt before use;
it adds 4*B+4 bytes per state and does not become checkpoint history. The
history-free fused production route retains its existing schedule.

Use FP64, quadrature 16, automatic native response selection and history_size=0
for the measured default path. Choose `output_mode="max"` before scene build
when only a sampled maximum is needed, or `"full"` for tensor/resultant output.
Both volume and shell remain available. These optimizations add no public knob.
For direct API use, initialize Genesis with `precision="64"`; setting only
`inverse_precision="64"` does not turn the entire simulation into FP64.
Shell output remains in-plane membrane/bending stress on its top/bottom sample
surfaces; it does not certify the local three-dimensional contact peak.

The gripper collision model retains ideal sharp box edges and the existing
explicit planar Hertz surrogate. Neither this speed study nor the surrogate
resolves a continuum sharp-edge singularity. A physical fillet is not assumed.

## Paired incremental evidence

The strongest pressure comparison runs v0, face-list v1 and projection v2
sequentially in each matched allocation, with the same four-condition bank.
All latency values below are synchronized milliseconds per batch step.

| Workload | B | v0 | v1 | v2 | v0/v2 speedup |
|---|---:|---:|---:|---:|---:|
| Complete shell/full, 600 evolving steps | 64 | 7.274 | 6.714 | 6.461 | 1.126x |
| Frozen-input complete stress module | 64 | 4.464 | 4.103 | 3.828 | 1.166x |
| Complete shell/full, 600 evolving steps | 8192 | 85.890 | 78.283 | 70.049 | 1.226x |
| Frozen-input complete stress module | 8192 | 65.480 | 62.005 | 55.086 | 1.189x |

Each live configuration has three full-cycle repeats, each frozen module five
repeats. All have zero invalid environment-steps. This is admitted simulation
throughput; optional host exports and policy inference are excluded. The
simulation uses dt=0.01 s, one physical substep and every-step stress updates.
Environment-steps/s is not a whole reinforcement-learning training rate.

Secondary device profiling disables CUDA graphs and is not added to headline
wall time. Pressure integration/correction changes from 2.039 to 1.449 ms at
B64 and 46.714 to 36.571 ms at B8192. Scatter changes from 0.569 to 0.520 and
5.339 to 4.987 ms. Pressure dominates the useful optimization opportunity.
Prepared projection adds 34,560 shared bytes on this shell mesh. The face list
uses temporary shared memory; no persistent face-list allocation is added.

The optional-history fallback comparison shows a modest complete-scene gain:
shell B8192 85.080 to 84.685 ms, volume 69.801 to 69.327 ms. Repeats remain
separate in the receipts. History itself is slower and fails a separate stress
budget, so these timings do not recommend enabling history. At a controlled
1% failure fraction, shell exact fallback plus original residual checking falls
from 1.261 to 0.658 ms. At 100% failures it is essentially unchanged.

## Numerical evidence and unchanged contracts

Budgets were declared before tuning in the execution record. Exact load
scheduling requires complete load relative error <1e-10, complete tensor
relative error <1e-8 and unchanged force/moment, original-operator residual and
lifecycle checks. The reference residual tolerances remain 1e-8 relative and
1e-11 absolute. They are numerical acceptance checks, not stress-error bounds.

The pressure candidates pass independent CPU/native flat, warped, egg and
concave-valley tests through both regular and reused-moment routes. Controls
include both material sides, finite moving patches, eccentric friction,
asymmetric contacts, contact-slot permutation, release and recontact. Complete
600-step trajectories retain all sampled peaks and seven complete load/tensor/
N/M/Q field frames. The exact and 64-condition heterogeneous comparisons agree
to about 2.1e-15 in worst per-step relative peak difference. Rigid trajectories
are bitwise equal. Radius differences of one ulp are separately reported and
pass the established relative 1e-12 radius contract.

The first added audit incorrectly required bitwise radius equality and failed;
that failed receipt remains. Its corrected contract matches the established
baseline assertion, while still reporting bitwise disagreement. Existing
baseline assertions were not relaxed. Candidate rejections below are retained.

## Candidate ledger

| Candidate | Outcome and evidence |
|---|---|
| Device-resident contacts, collision provenance and radius reuse | Already present in the baseline. No per-environment Python scheduling or host decision loop was introduced. |
| Fixed geometry, analytical contained support, physical surface inversion | Existing contained-support integration retained. Shared face preparation and prepared Q4 projection are measured and integrated winners. |
| Cheaper quadrature | q8 fails the evolving pressure fit at frame 108. q12 passes native admission but has 6.901e-4 peak error versus q16, above its 1e-4 budget. Keep q16. |
| Per-environment source aggregation and batched response | Already present for the single native mesh. Controlled tiles 128/256/512 and failed-environment fractions were measured; large sparse fallback compaction is integrated. |
| Few feature meshes | CPU 3/5-library, two-width controls implement per-contact weighting and merge loads by mesh, with full footprint fallback. Three-library spatial controls pass; 19/80 five-library cases fail the 10% field budget. |
| Nearest and top-two selection | Discontinuous at a three-region junction: nearest jumps up to 18.9%, top-two 9.4% in the bounded control. Not production defaults. |
| Continuous overlap | Allows all contributors near multiple-region junctions. Smooth transitions are comparable to physical motion in the fixture; finite-footprint coverage may require the mother route. |
| Common-mother recovery | Validated nested planar identity; actual GPU B1024 recovery improves from 0.987 to 0.545 ms (3 meshes), 1.393 to 0.773 ms (5 meshes), excluding load and solves. Native curved meshes lack a validated embedding and common discrete inertia relief; prototype is retained, not silently substituted. |
| Coarse plus local bubbles and both Schur directions | Exact hierarchical coupling and general distributed RHS retained in CPU controls. Relative displacement/residual errors about 1e-12. Both directions take roughly twice direct CPU solve time and add dense Schur/harmonic storage. No GPU win is claimed. |
| Stress/max fusion | Existing native fused recovery, local-frame max and max-only allocation remain the baseline. Extra shell frame-transform sharing passes numerics but slows the measured stage, so it was reverted. |
| Full H response | Pure Quadrants construction/application includes all force/moment and six spin channels, relief once, and full shell N/M/Q. FP64 H needs 320.36 MiB shell or 43.24 MiB volume on these coarse meshes. Large-batch sparse H loses to retained packed displacement recovery. |
| Dense versus sparse H, tile tuning | Tiles 128/256, dense/sparse and max/full are measured separately. Dense products lose on tested fixtures. H-only timings exclude the displacement and original residual that production still needs. |
| FP32 storage with FP64 arithmetic | Isolated packed response needs one correction and is slower: shell B8192 10.993 to 37.533 ms. Mixed H conversion is also slower. Keep FP64 storage. |
| FP32 H accumulation | Same-load combined field error about 3e-7 in the control, with faster small-batch H application. A complete replacement still needs the original acceptance path; no production default change. Aggregate field norms alone do not certify tiny individual resultant channels. |
| TF32 | No explicit supported entrypoint was found in the installed Quadrants Python API probe. Not executed or claimed as FP32 validation. |
| Global max screening | Includes every stress response column and every sample block. Bounds, selection and exact evaluation are measured together. Small batches lose; B1024 H screening wins over exhaustive H, but needs additional q/residual work in production. Keep exhaustive native max. |
| Four-column history | Native admission passes but a peak error of 4.548e-5 exceeds its 1e-5 stress budget. Complete scenes are slower. No default promotion. |
| Two-frame extrapolation | All accepted predictions feed later history; 600-step full fields pass the 1e-5 budget including a partial reset. Only 442/2350 loaded environment-steps predict successfully; separate module timing evaluates its cost. |
| History rotation | At B8192, rotation is slower than copying: shell 0.419/0.360 ms, volume 0.697/0.591 ms. Keep existing maintenance. |
| Actual cached Kq | Frozen-q cache matches direct residual exactly. Refresh plus checking is slower: shell 1.260/1.104 ms, volume 2.877/2.625 ms. No stale cache is admitted. |
| Global POD and fixed load-shape unit reuse | No improved basis or unmet performance need justifies reopening POD. Force-dependent Hertz radius changes the load shape, so a fixed patch response cannot replace arbitrary current-load construction. |

The common-mother GPU fixture uses CPU-built reference operators and solutions
as benchmark inputs; all measured products, per-contact selection/grouping and
max reductions run in Quadrants. It is a research control, not an exception to
the requirement that production numerical preparation/runtime remain Quadrants.
On B1024, selection/grouping add roughly 0.205 ms for 3 meshes and 0.326 ms for
5 meshes. The mother workspace is approximately 28.55/60.70 MiB respectively.
The five-mesh spatial failures and mapping limitations prevent promoting this
recovery-only improvement to an arbitrary native curved-egg atlas.

Screening bounds include a Frobenius-norm component-rounding allowance in
addition to relative upward margins, so almost hydrostatic columns are not
protected only by a relative von-Mises margin. These finite-FP64 fixture checks
do not establish a general quantized or FP32 screening implementation.

## Reproduction and retained artifacts

Engine sources are pinned in `gpu-throughput-v0` through `gpu-throughput-v6`.
The final engine is v6 (`5c44042a4`; production code equals v5), with source
inventories captured by each submission. Failed experiments and reverted v3/v4
remain reproducible. New data, raw traces, full fields, stdout/stderr and
analysis-source captures are under the project data run described in
[DATA_LOCATION.md](../../../DATA_LOCATION.md). The twelve unrelated original
files and 142 pre-existing worktree heads pass the preservation audit.

A direct steady-state benchmark, inside the established GPU job environment:

```bash
python -m examples.speed_benchmark.rigid_stress \
  --scope live --stress-backend shell --output-mode full --envs 8192 \
  --conditions /path/to/conditions.npz --radius-model hertz \
  --pressure-profile hertz --box-edge-policy planar --quadrature 16 \
  --warmup 600 --steps 600 --repetitions 3 --minimum-seconds 0 \
  --source-revision 5c44042a4 --output /data/path/result.json
```

Use `--scope recovery --warmup 130 --steps 100 --repetitions 5` for the
frozen loaded module, or `--scope rigid` for the matched rigid baseline.
The recorded task JSONs are the complete runnable command lists for all
candidates, including reference source overrides. Submit with the existing
`research.rigid_stress.shell_comparison.submit` helper, explicit data output,
`--node-local-cache`, the data-side environment and image `1_26`.

## Launch/copy observations and state maintenance

A separate 10-step loaded trace records 239 shell device-kernel events per
step, while exposing 58 driver kernel launches and 13 runtime launches. Volume
records 233/239 device kernels at B64/B8192, versus 30/50 driver launches plus
13 runtime launches. These are distinct profiler categories. Captured graph
submissions are not fully represented by the host launch events, so kernel
count must not be reported as CPU launch count.

The trace records 15 shell and 7/9 volume HtoD events per step. There is one
DtoH and one explicit device synchronization over each entire ten-step trace,
corresponding to the final observer/acceptance readback. Raw events and sizes
remain available; these observations do not imply per-environment scheduling
readbacks. Intrusive trace times are excluded from all throughput medians.

Native B8192 poisoned-displacement controls exercise failure fractions 0%,
about 1%, 10%, 50% and 100% for shell and volume. All recovered displacements
and full tensor fields match the unmodified exact solution exactly in these
fixtures; untouched environments remain bitwise equal. The device failure list
contains precisely the selected environments. The overflow pressure fixture
covers 256 candidate faces on each side, requires five constrained-fit
evaluations, and matches baseline generalized loads to 2.83e-16 relative.
Its first attempt mistakenly invoked the unrefined integration pass; that
non-exercising failed control is preserved separately from the passing run.

The two-frame trajectory module comparison (matched current loads, exact
recovery versus predictor/gate/fallback/output/append) averages 0.743 versus
0.705 ms at shell B64. At B8192 it regresses: shell 9.842 versus 10.877 ms;
volume 9.450 versus 11.808 ms. This is a module-only, per-frame synchronized
comparison with the first three compilation frames excluded, not a claimed
complete-scene improvement. The small-batch difference does not justify a new
public predictor path while useful large-batch throughput regresses.

The heterogeneous volume trajectory is **not admitted**: both the original v0
and final v6 reject frame 221, environment 1, contact 0, status 3. The
nonnegative sampled pressure fit remains unresolved at radius 3.09950899e-5 m
and friction coefficient 0.424854643. Both retain 221 accepted frames and the
same failing contact. This does not establish physical infeasibility. Frozen
module measurements at frame 130 remain useful within that input scope, but
must not be reported as a successful heterogeneous full-cycle volume rate.
The paired shell heterogeneous cycles complete with no invalid steps.

A second common-mother GPU control explicitly skips inactive mesh contributions
on both sides of the comparison. The B1024 nearest-case recovery changes from
0.891 to 0.501 ms for three meshes, and 1.192 to 0.668 ms for five meshes.
At B64 the extra indirect activity checks can cost more than the zero work
removed. The shared-recovery benefit survives the active-only comparison; it
remains conditional on the validated planar embedding and excluded solve/load
work. These results are retained separately from the first regular scan.

The separate cold-to-steady memory pass samples the entire visible allocated
GPU. At B8192 its maxima are 11,973/13,733 MiB for shell max/full and
12,229/13,093 MiB for volume max/full. Initial device-used memory is zero in
these allocations. Observed maximum sample gaps are 0.279–0.419 s, so these are
sampled peaks, not certified instantaneous upper bounds. They include context,
allocator and rigid-scene memory. PyTorch peak allocation alone excludes
Quadrants and must not substitute for these measurements. These instrumented
runs are excluded from headline latency measurements.


## Final engine throughput and allocation inventory

Final v6 results use FP64/q16/history-off and the same tiled four-condition bank.
Independent allocations produce the table below; use the paired incremental
table above for causal speedup claims. Complete scene includes approach,
closure, frictional motion and release. All 36 final timing tasks pass with
zero invalid environment-steps. Full repeat distributions remain in the packet.

| B | Rigid | Volume max | Volume full | Shell max | Shell full |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.280 | 4.901 | 5.034 | 6.271 | 6.355 |
| 64 | 2.342 | 5.164 | 5.320 | 6.496 | 6.493 |
| 1024 | 2.929 | 9.440 | 9.301 | 13.072 | 13.536 |
| 8192 | 9.600 | 46.385 | 46.510 | 69.879 | 72.272 |

Latencies are ms/batch-step. Final B8192 admitted environment-steps/s is volume/max 176,609, volume/full 176,135, shell/max 117,231, shell/full 113,350.

Frozen loaded module, ms/batch-step:

| B | Volume max | Volume full | Shell max | Shell full |
|---:|---:|---:|---:|---:|
| 4 | 1.037 | 1.077 | 3.852 | 3.843 |
| 64 | 1.030 | 1.067 | 3.846 | 3.853 |
| 1024 | 3.667 | 3.708 | 8.926 | 9.017 |
| 8192 | 22.806 | 23.130 | 53.744 | 55.850 |

The 64 distinct heterogeneous conditions vary pose, friction and phase.
B1024 repeats that bank 16 times, rather than claiming 1024 unique conditions.
Paired shell complete scene / frozen module, ms/batch-step:

| B / output | v0 scene | v6 scene | v0 module | v6 module |
|---|---:|---:|---:|---:|
| 64 / max | 11.511 | 8.808 | 7.202 | 4.486 |
| 64 / full | 11.488 | 8.954 | 7.284 | 4.486 |
| 1024 / max | 17.849 | 14.210 | 11.543 | 8.397 |
| 1024 / full | 18.146 | 14.617 | 11.666 | 8.497 |

DPR-owned persistent allocation at B8192 (aliases counted once):

| Backend/output | Shared MiB | Other bytes per environment, KiB | Total GiB |
|---|---:|---:|---:|
| volume/max | 64.557 | 224.507 | 1.817 |
| volume/full | 64.557 | 329.507 | 2.637 |
| shell/max | 42.212 | 195.202 | 1.566 |
| shell/full | 42.212 | 412.702 | 3.265 |

Shared corresponds to immutable native buffers; per-environment totals include
mutable load, recovery, acceptance and requested output storage. This inventory
is for the stress feature, not the entire rigid scene. The separate sampled
device peaks above include the whole process/context and temporary preparation.

Cold B64/full scene construction including build JIT was volume 146.45 s, shell 103.98 s.
The native operator/factor/response breakdown is recorded separately from
first-step warmup and steady-state timing. No cold preparation cost is hidden
inside or amortized into the headline steady-state rates.

B64 is useful for lower batch latency; B1024 and B8192 increase aggregate
throughput at higher memory and latency. B8192 is the largest batch tested,
not a claim of a global hardware optimum. Backend timings are not an accuracy
ranking between shell in-plane and volume stress.
