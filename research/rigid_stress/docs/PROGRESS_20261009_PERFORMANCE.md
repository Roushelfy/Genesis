# Fine-mesh performance and actual grasp checkpoint, 2026-10-09

The full goal remains active. These are completed measurements, not the final
chosen live throughput profile. Raw artifacts are in
[`evidence/20261009-performance`](../evidence/20261009-performance/); large
contact histories remain at the hashed paths in its runtime manifest.

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
