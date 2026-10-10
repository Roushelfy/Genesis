# Fine-mesh performance and actual grasp checkpoint, 2026-10-09

The full goal remains active. These are completed measurements, not the final
chosen live throughput profile. Raw artifacts are in
[`evidence/20261009-performance`](../evidence/20261009-performance/); large
contact histories remain at the hashed paths in its runtime manifest.

The detailed [acceptance audit](VALIDATION_RESULTS.md) links the mathematical,
physical and actual-scene checks and marks final timing/oracle items pending.

## Physics acceptance and the contact-normal fix

The complete level-6 shell has 2,457,630 DOFs and 491,520 affine P2 tetrahedra,
two wall layers, 0.5 mm wall thickness, E=10 GPa, nu=0.3 and density=2000 kg/m3.
The collision surface keeps all 81,920 exterior triangles. Surface integration
uses the unchanged positive order-10 rule. The full-model convergence evidence
and its representative-load limitations remain in the previous GPU checkpoint.

A real frame-283 contact exposed a rigid constraint-frame defect: a narrowphase
normal had length 1.000005055531268. The tangent-frame constructor assumes unit
length. Its reconstructed physical force exceeded the circular Coulomb cone by
3.4547720595412557e-7 N, versus a roundoff allowance of 1.03754e-15 N. Normalizing
the normal **before storing the contact and solving constraints** fixes that
inconsistency. The stress mapper still rejects inadmissible physical forces;
it neither clips them nor increases friction. The actual offending data and
CPU/GPU regression logs are retained. Both backend regression tests pass.

With that fix, the 32 independently phased level-6 seeds 610000--610031 complete
all 1,200 steps and 32 resets. There are 335 egg-as-A and 68,593 egg-as-B contacts;
maximum actual tangential force is 1.52299 N. GPU pressure diagnostics report
maximum moment mismatch 9.13e-19 N m. Newton--Euler force/torque discrepancies
are 6.34e-4 N and 1.47e-5 N m; the corresponding public constraint gradients
agree to 7.01e-16 N and 4.33e-17 N m. Those nonzero acceleration discrepancies
belong to the rigid constraint convergence/integration profile, and are not
removed by changing contact loads in recovery.

The scripted controller's held-out keep-hold success is **0/32**. A nominal
level-6 seed succeeds **1/1**, lifts and holds the egg, and encounters a maximum
tangential force of 0.23455 N. Its real 600-frame, 100 fps, 640x480 video is
[`nominal.mp4`](../evidence/20261009-performance/visualization/nominal.mp4).
This is an untrained scripted controller, not an assertion of robust grasping.
The record-only fine runs do not claim a completed CPU peak comparison.

An additional four-environment nominal level-2 run with four physical substeps
per 10 ms scene step passes CPU FP64 comparisons at every substep, including a
partial reset at step 107. Maximum peak relative error is 2.314e-11 and maximum
moment mismatch is 7.67e-18 N m; all four nominal grasps succeed. This validates
sampling/reset integration, while its coarse mechanical peaks are separate
from the physically converged level-6 profile.

## Measured recovery ablations

These are **assembled-RHS diagnostics**, not environment transitions/s. Each
variant has three synchronized wall/event measurements of at least ten seconds.
Inputs vary as explicitly synthetic combinations of six complete recorded live
RHS snapshots. Those calibration contacts originate in the earlier level-2
collision rollout; recovery always uses the complete level-6 elastic operator.
No contact-space basis or restricted displacement space is used for correctness.

Hardware is NVIDIA RTX PRO 6000 Blackwell Server Edition with 101,975,851,008
bytes of VRAM. Native cuDSS 0.8 uses one FP64 factor with 2,322,190,270 nonzeros,
approximately 14.5 s analysis and 9.6 s factorization. Shared-factor history
sweeps retain the same immutable factor rather than rebuilding it per variant.

| B=8 variant | Recovery transitions/s | Sampled device memory, GB |
|---|---:|---:|
| FP64 direct | 62.25 | 30.89 |
| Prediction only, h=0 | 41.35 | 35.63 |
| Compact history h=4 | 53.13 | 38.68 |
| Compact history h=8 | 52.80 | 40.41 |
| Padded correction h=4 | 27.75 | 38.52 |
| Density-adaptive correction h=4 | 53.29 | 38.68 |
| Rebuilt Gram h=4 | 7.77 | 38.68 |
| DOF-major history storage | 52.70 | 37.90 |
| All-failed abrupt direct | 62.66 | 31.20 |
| All-failed abrupt history | 27.57 | 38.68 |
| Raw FP32, four refinements and FP64 fallback | 27.25 | 53.33 |
| Scaled FP32, four refinements and FP64 fallback | 27.24 | 53.33 |
| Scaled FP32, immediate FP64 fallback | 51.98 | 53.17 |

History reduces failed columns to about 3% on these smooth combinations, but
its own bandwidth/reduction cost still makes direct recovery faster. Both
four-refinement FP32 runs fall back on every initially corrected environment
in their timed spans. This is measured accepted fallback, not accepted raw
FP32 accuracy or Tensor Core utilization.

Direct batch diagnostics measure B=1/8/32/128 rates of approximately
25.47/61.76/69.31/91.95 recovery transitions/s. B=512 actually exhausts the
95 GiB device allocation; B=2048/8192 are explicitly skipped after that
measured limit. These buffers include the six diagnostic RHS snapshots, so
this does not establish the memory limit of the different live scope.

The native direct graph experiment on level 2 passes changing-RHS validation
with maximum peak error 1.685e-13 and measures 5,830 recovery transitions/s.
cuDSS requires graph-compatible allocations; the experiment uses its public
device allocator callback and cudaMallocAsync/freeAsync, plus a retained
isolated CuPy capture pool. The isolated pool prevents eager validation from
reusing still-live graph temporary addresses. These requirements follow the
[cuDSS graph documentation](https://docs.nvidia.com/cuda/cudss/general.html).
Fine graph/eager and temporary-tensor peak comparisons remain in progress.

## Actual rigid-only baselines and mapping diagnostics

All rigid baselines below use the full level-6 collision surface, dt=10 ms,
one substep, FP64 rigid state, convex elliptic friction, 100 constraint
iterations and 1e-12 solver tolerance. The effective contact time constant is
20 ms, including the engine's minimum of twice the substep duration. Each
repeat includes complete independently delayed episodes and resets, has at
least ten seconds wall time, and excludes setup and visualization.

| Environments | Rigid env transitions/s | Policy + rigid env transitions/s |
|---|---:|---:|
| 8 | 1,816.75 | 1,636.06 |
| 32 | 5,752.63 | 5,288.21 |
| 128 | 17,426.33 | 16,538.00 |

The policy is the fixed untrained FP64 26->64 tanh->64 tanh->7 tanh MLP with
1e-4 rad residual actions. The final stress-augmented loop uses the same
weights and architecture; its stress observation can change residual actions.
This measures policy inference/rollouts, not learning or PPO training.

Stratified B=8 mapping-only CUDA spans average 35.38 ms for 3 mm grid cells
and atomic scatter, versus 41.12 ms with warp face-segment scatter. At 6 mm
the corresponding spans are 43.76 and 53.03 ms. Every physical quadrature point
inside the footprint is retained. Neither the wider cells nor warp scatter
is selected merely because it sounds faster. The mapping diagnostic's wall
scope also includes independent GPU wrench checks, and is not simulator FPS.

The original B=8 full pressure/body/recovery replay measures approximately
10.16 recovery transitions/s. Equation recovery alone is approximately 62/s;
the body-load/projection stages therefore need explicit profiling rather than
attributing the whole difference to contact integration.

## Exact geometry-field candidates and remaining work

For each fixed reference coordinate r, centrifugal acceleration is
omega x (omega x r). It is exactly a sum of six fields weighted by the three
squared and three cross products of angular velocity components. Multiplying
those six fields by the full consistent mass matrix offline preserves the
unchanged nodal inertial load. This cache does not restrict any contact or
elastic displacement. A separate exact nodal force/moment reduction and
fixed-field multiplication avoid poorly shaped dense products.

Both quadratic-cache and fused-field candidates pass CPU FP64/GPU tests at
B=1/3/8/17, including changing angular velocity, every complete RHS entry,
zero load, freefall, centrifugal stress and global corner peaks. Their new
full replay/live measurements are active; no gain is assumed before completion.
The original sparse-mass/cuBLAS paths remain available as ablation baselines.

The residual acceptance is explicitly max(1e-11 N, rtol*||b||), with strict
rtol=1e-6 and throughput rtol=1e-3. Near-zero rows are not claimed to satisfy
a relative guarantee: reports retain absolute residual, relative residual
including that region, and the maximum above the absolute floor separately.
CPU peak checks separately publish absolute Pa error below 1 Pa.

Remaining acceptance work: finish the fine performance candidates; choose and
freeze on calibration data; complete saved identical-RHS CPU/mapping checks
on the held-out fine contacts; report all fine stress/live/matched-policy
repetitions, stages and memory; publish the selected strict/throughput profiles
and final full acceptance audit. This checkpoint is not completion of GOAL.md.

## Completed follow-up measurements at 22:00 UTC

The exact fused-field path completes three full 1,200-step B=8 stress replays
at 46.94, 47.08 and 46.83 recovery transitions/s (aggregate 46.9483), versus
10.1571 for the original path. A separate stage pass measures approximately
0.077 ms selection/reset, 41.58 ms warp pressure mapping, 2.74 ms body/inertia
assembly and 127.58 ms recovery per batch. These contact inputs remain the
declared earlier calibration rollout; they do not establish fine-collision
held-out throughput. Quadratic caching alone still uses the slow dense
products, so the fused field and nodal wrench operations matter separately.

The native level-6 assembled-RHS graph experiment completes three repeats at
70.17 recovery transitions/s, versus 62.24 for eager execution with the same
asynchronous allocator. The maximum graph/eager peak error is 1.053e-13.
Temporary-tensor peak evaluation measures 51.56/s. Natural native ordering
measures 21.82/s; the faster automatic native ordering remains selected.

Row-major cuSPARSE stiffness multiplication is exposed as an optional layout,
with public descriptors and CSR_ALG2. The native factor's multi-RHS layout is
unchanged. Complete CPU comparisons at B=1/3/8/17 pass. Fine assembled-RHS
rates are F/C=61.60/59.95 at B=8, 69.16/74.90 at B=32, and 93.84/107.67 at
B=128. Its benefit depends on batch size. No measured microbenchmark rate is
relabeled as live simulator throughput.

Full fine-collision live B=32 eager calibration runs finish at 60.4951,
60.5106 and 60.5114 environment transitions/s (aggregate 60.5057). Each repeat
contains 1,193 steps, 38,176 transitions and 32 resets. All full residuals pass;
maximum above the absolute floor is 1.719e-10, absolute residual is 5.969e-12 N,
and the near-zero relative maximum is 8.750e-6. Sampled total device memory is
35.824 GB. The seeds are calibration seeds 510000--510031. Fine graph/layout
and larger live batches still need all three repeats before selection.

The reusable captured recovery now works on both the default live stream and
nondefault consumer streams. Ninety-six changing-RHS CPU-oracle cases pass
across F/C layouts, including zero-load environments and angular motion.
Contact mapping and complete body loads stay outside the graph and are counted
in live timing. Every recovery inside the graph retains the full residual and
global peak. A separate actual one/four-substep observation regression rejects
an invalid footprint as a NaN device observation for that environment, retains
the independent valid environment and accepts the next restored step. The
four-substep case would miss 385,592 Pa if only its final substep were sampled.

The B=512 live row-major graph candidate actually exhausts device memory when
allocating the first 10.066 GB complete mapped load, after successful setup and
factorization. This establishes a limit for that graph pipeline, not all other
pipelines. The separate eager B=512 live probe also exhausts memory during
complete centrifugal/body assembly. Both raw failure logs are retained. The
CPU held-out oracle is still running; saved complete RHS comparisons are not
declared complete here.

The B=32 fine live graph run now completes all three repeats at 69.8753,
69.8512 and 69.8095 environment transitions/s (aggregate 69.8453), versus
60.5057 eager. All repeats pass with 39.198 GB sampled total device memory.
Matched B=256 rigid/policy-rigid baselines complete at 31,533.66/30,245.54
environment transitions/s. The allocated CPU is Intel Xeon Platinum 8562Y+;
Slurm grants eight CPUs and the numerical thread settings are one. Driver
595.71.05, CUDA runtime API 12090 and cuSPARSE integer version 12510 are
independently recorded, alongside all installed numerical package versions.

The one-command acceptance runner completes all thirteen selected checks in
fresh subprocesses, including CPU-only startup without CUDA imports. Its
actual coarse varied Panda suite repeats 5/32 keep-hold success and maximum
CPU peak error 1.732e-6. Four nominal environments with four substeps and a
partial reset all succeed, with maximum CPU peak error 4.556e-13. These are
integration checks, separate from fine collision/elastic convergence.
The earlier clean-process CPU reference import failure is fixed at its
reference-only boundary. Setup, tests, demo, timing and oracle instructions
are in [RUNNING.md](RUNNING.md).

Final timing entrypoints also audit the complete warmup's actual contact
counts, radii/friction, tangential forces and explicitly defined hold success.
This quality instrumentation is outside steady-state timing. The coarse
graph-policy regression passes and its contact histogram sums to every
warmup environment step. `benchmark_suite` records a frozen configuration
hash and runs scopes as explicit subprocesses; its rigid dispatch passes.
Neither wrapper changes the mechanical or recovery kernels. Selection and
final held-out timing remain active.

## Completed follow-up measurements at 23:20 UTC

The full fused-body level-6 history-h=4 replay completes three 1,200-frame
repeats at aggregate 24.0630 recovery transitions/s. Each repeat corrects
9,520 of 9,600 environment columns (99.17%); direct on the same calibration
load path measures 46.9483/s. Maximum warmup peak relative error above 1 Pa
is 8.512e-7 and maximum near-zero absolute error is 3.526e-7 Pa. The real
load changes therefore provide little successful history reuse, and history
remains slower even after body assembly is fused. The raw
[JSON](../evidence/20261009-performance/raw/20261009-fine-calibration/history4-fused-unit.json)
and CSV retain all corrections and complete residuals.

The older nonfused history replay and the B=128 eager live allocation time
out before three complete repeats. Their completed rows and Slurm statuses
are [retained](../evidence/20261009-performance/raw/20261009-live-calibration/timeouts.json),
and neither is an eligible primary three-repeat result. Fine graph/layout
and B=256 candidates are still running; single completed rows do not choose
the winner.

An additional untimed native GPU [gauge/pose test](../evidence/20261009-performance/raw/20261009-invariance/result.json)
passes five random world poses in each F/C layout with seven environments,
varying contact counts and egg-as-A/B signs. Alternate gauge rows agree with
CPU canonical displacement to 2.51e-10 relative; maximum complete RHS entry
difference is 2.89e-15 N. The exact small QA script is retained without
changing the frozen numerical runtime code.

## Replay validation memory fix at 23:49 UTC

The B=256 graph live point uses about 101.37 GB sampled device memory.
Untimed replay validation initially fails with duplicated full-batch oracle
operators; a 32-column oracle still does not fit. Sharing the existing
immutable operators with the ungraphed direct baseline avoids those copies,
but retained eager displacement and previous replay RHS references also
occupy a full 5.033 GB array each. The harness releases eager displacement
after validation and the replay releases its previous RHS before the next
map. These lifetime fixes preserve every current RHS, physical load, sparse
solve, complete residual and global peak. The live calibration path is
unchanged; complete source hashes record the later replay/harness revision.

The corrected actual CLI regression completes 1,200 warmup frames and three
ten-second level-2 stress repeats. Graph/eager maximum peak error above 1 Pa
is 3.710e-13 and near-zero absolute error is 7.647e-15 Pa. Its roughly
9,355 recovery transitions/s is a coarse regression, not final fine FPS.
Ninety-six changing-RHS graph and ungraphed direct cases also pass CPU oracle
comparisons with default/nondefault streams and F/C layouts.

The complete level-6 B=256 memory probe then passes six changing recorded
calibration frames, maximum graph/eager peak error 1.550e-13, with device
used memory 98.558 GB after the probe. This is an untimed memory/accuracy
check, not a six-frame throughput result or the final 1,200-frame acceptance.
All failed probe logs and the successful script/report are retained in
[`raw/20261009-graph-replay`](../evidence/20261009-performance/raw/20261009-graph-replay/).

The earlier full B=128 row-major graph calibration replay completes all three
1,200-frame repeats at aggregate 100.5030 recovery transitions/s, with every
warmup frame compared against eager direct recovery. Maximum/p95/mean peak
relative errors above 1 Pa are 2.372e-13 / 8.302e-14 / 3.081e-14; near-zero
maximum absolute error is 1.017e-14 Pa. Sampled memory is 74.638 GB. These
inputs remain the earlier calibration contacts; final fine held-out rates
remain pending. The artifact manifest distinguishes large original trajectory
records retained on data from their checked-in summaries and hashes each
independently.

## Held-out CPU oracle completed at 00:18 UTC on 2026-10-10

The level-6 CPU mapping/direct job completes all sixty stride-20 frames,
32 columns each, in 03:00:25. Its 1,884 reference peaks above 1 Pa have
maximum/p95/mean relative errors 8.670e-12 / 2.095e-12 / 7.480e-13. The 36
near-zero cases have maximum absolute error 2.161e-9 Pa. Complete RHS entry
difference is at most 3.990e-17 N, full absolute residual 2.865e-12 N and
relative residual above the absolute floor 1.557e-10. Both declared profiles
pass on these evaluated columns. This is sixty CPU frame comparisons, not
1,200 CPU frame comparisons or a universal output-error bound.

The [raw report](../evidence/20261009-performance/raw/20261009-heldout-oracle/strict-cpu.json),
summary and per-column CSV are committed alongside a sixty-file RHS manifest.
Full arrays remain on data: 35.543 GB compressed, 37.749 GB logical RHS.
Original node rtx-209-201 hardware metadata is collected in a later same-node
allocation, explicitly marked as such: Intel Xeon Platinum 8562Y+, eight
allocated CPUs and one numerical thread. The contact input is separately
published as a 6.27 MB exact NPZ plus its phase/reset companion under
[`evidence/20261010-final/inputs`](../evidence/20261010-final/inputs/), so a
replay command has its actual source available with the repository.

The final selection and full five-scope held-out timing still remain active.
