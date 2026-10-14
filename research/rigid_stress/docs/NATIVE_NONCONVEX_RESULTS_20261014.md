# Native nonconvex stress recovery: validation and performance

Numerical engine `76537c2fb64cd469e83ba9921b0724b5c82a5f36` implements the
2026-10-11 nonconvex extension inside the existing rigid solver. The normal
Franka example, maximum/full-field API and benchmark use production Quadrants
kernels. The final source/data audit and publication verification pass.
[The closeout audit](NATIVE_NONCONVEX_CLOSEOUT_20261014.md) maps the requirements
to implementation, tests and retained evidence. Deliverables are on
`Roushelfy/Genesis:rigid-stress-recovery`.

## Model, API and supported domain

The complete low hollow shell has 126 linear geometry vertices, 810 P2 nodes,
2430 displacement DOFs, 480 tetrahedra and 80 exterior triangles. Its collision
shell has 42 authored vertices/80 faces. E=10 GPa, nu=.3, rho=2000 kg/m³ and
rigid mass 0.006529848210401324 kg are fixed operator inputs. Rigid geometry
and dynamics remain rigid; recovered elastic displacement is auxiliary.

Witnesses retain collision geom/pair identity, a reference boundary face and
barycentric coordinates. Tetrahedron ownership orients material normals,
including cavity surfaces. Intrinsic footprints follow actual topology;
nearby separate sheets and holes do not create adjacency. Verified planar
segments and adjacent-face unfolding supplement the s8 shared-edge graph.
Its distances can overestimate continuous distance. Q10 finite-area integration
uses squared fan coordinates and a Gaussian width 0.45R with squared compact
taper. Local disk/fan refinement preserves the physical radius and inputs.

Original application points and free moments survive projection and signed
P2 scatter. Local vector tractions obey compression and actual Coulomb cones.
The measured `body_source_prior` law keeps every source as a local traction
prior and conserves its complete aggregate original-point wrench. Individual
disk resultants may differ from their priors; separate islands retain their
own geometry, normals, radius and friction. Self-balanced compression remains
nonzero when the net wrench vanishes. This explicit partition law is documented
in [BODY_SOURCE_PRIOR_MODEL.md](BODY_SOURCE_PRIOR_MODEL.md). The alternative
componentwise/pair-union modes retain their own declared constraints.

All production assembly, fitting, scatter, balance, solve, residual and peak
work is native Quadrants. FP64 physical geometry/loads/operators/residuals/
stress are retained; only warm Newton Hessians use FP32, with FP64 gradients,
cold retry and unchanged final acceptance. Immutable operators are shared;
loads, fitting state, history and observations remain independent by link/env.

`output_mode="max"` allocates no complete field. `"full"` exposes all four
unaveraged affine-P2 corner tensors/von Mises values in Pa, authored-link
frame, while preserving the global maximum. The latest-substep field and
scene-step temporal maximum have distinct documented lifetimes. Ordinary
release/reset/restoration invalidates and recomputes affected observations.
See [NATIVE_API.md](NATIVE_API.md) and [OUTPUT_MODES.md](OUTPUT_MODES.md).

Valid connected material boundaries, resolved finite footprints and feasible
declared wrenches are the supported domain. Independent free roots use the
standard rigid contact coupler. Unsupported autodiff, joint support,
undeclared applied/coupling loads, dynamic mass/inertia changes and enabled
rolling/torsional contact moments raise explicitly. Ambiguous provenance,
unresolved integration, numerical failure and certified physical
incompatibility have distinct retained diagnostics; no source is truncated,
force/friction clipped or radius silently enlarged.

## Numerical and lifecycle acceptance

Final budgets remain original-wrench 1e-8, complete equilibrium including all
six gauge rows `max(1e-11 N, 1e-8 ||rhs||)`, stress RTOL 1e-4/ATOL 1e-3 Pa.

- Current-source coverage is 135 CUDA cases, with no skips, and 114 CPU passes/
  21 explicitly CUDA-only skips. The complete initial suite is retained;
  all 10 cases in the sole changed test module pass again on CPU/CUDA.
  Every other recorded source is byte-identical. This is combined current
  coverage, not a claim that the original CUDA run had zero failures.
- Its two original saved-edge failures compared `K u_native` against loads
  from a separate CPU contact fit. Contact-map error was therefore counted
  as elastic residual. The corrected test independently mass-balances the
  actual native nodal loads for complete equilibrium, while retaining the
  independent CPU contact-fit/full tensor/von Mises/global-peak comparisons
  and all original wrench/cone/stress gates. On the problematic 147 cases,
  CUDA complete residual budget fractions are at most 0.002480; independent
  fitted-RHS differences are recorded separately. No engine tolerance changed.
- The clean sampler qualification includes 35 inherited CPU/35 clean CUDA
  cases and 48 fields per route. Worst normalized original wrench error is
  7.394051316e-10, full residual budget fraction0.04585622523 and peak
  difference7.967217243e-6Pa.
- Final-source 464 captured dynamic complete fields pass independent CPU FP64
  K/gauge/tensor/von Mises/global-peak checks: residual fraction≤0.005638003,
  peak difference≤1.860404154e-6 Pa. Another 160 original-point/free-moment
  samples use independent mass/centrifugal/inertia relief: wrench error
  ≤8.627857791e-9, residual fraction≤0.005517647 and peak difference
  ≤1.540160156e-6Pa.

The cases cover U/cavity/narrow-gap/hole geometry, arbitrary surface winding,
rotated/offset/friction/moment inputs, source ambiguity, changing contacts,
workspace growth/replay, multiple links/environments, release/slip, partial
resets/checkpoints, full/max/unbatched/substep observations and invalid loads.
The [requirement map](NATIVE_NONCONVEX_CLOSEOUT_20261014.md) identifies their
normal test paths and archived independent evidence.

## Complete measured scopes and scaling

RTX PRO 6000 Blackwell Server Edition, 97887 MiB, driver 595.71.05,
container 1_26, Genesis 1.4.3/Quadrants 1.3.3. Each width runs four scopes on
one assigned GPU; widths use separate allocations. Original baseline banks,
FP64/Q10/s8/body-source-prior law, mesh64/GJK patch-on,dt.01/substep1 and max
output stay fixed. Each scope has 600 warmup and three 1280-step repeats.
Every environment, observation, parameter update, reset, metric, history and
fallback is counted. No hard environment or tail frame is removed.

| Batch/seed | Rigid env-step/s | Frozen recovery env-step/s | Live env-step/s | Policy env-step/s |
|---|---:|---:|---:|---:|
|1024/623101|125859.551|97589.574|45399.359|42502.831|
|2048/623102|211411.647|135534.656|68509.262|65377.457|
|4096/623103|353558.373|159786.742|91945.117|94228.290|
|8192/623104|505233.279|178368.978|117496.253|116118.497|
|16384/991909|658136.833|187299.691|131919.205|132785.500|
|32768/623102|823471.248|191316.287|145256.946|145431.227|

All attempted steps are valid, with zero overflows/failed lifts. Rigid/live
positions and per-environment rigid/contact/grasp/descent counts are exact.
Recovery freezes an actual solved contact snapshot in its matching
pre-integration frame; it is not dynamic rollout throughput. Policy is a
26→128→128→7 FP32 residual network on the scripted controller. It consumes
current observations, so changed contact work/repeat variation affects rates;
the comparison is inference rollout, not pure policy overhead or RL training.

A separate one-GPU curve tiles the same 2048-condition bank at four widths:

| Batch | Live env-step/s | Sampled setup/workload peak MiB |
|---|---:|---:|
|8192|120536.835|21413|
|16384|139124.948|40849|
|24576|146295.829|64185|
|32768|150133.237|84017|

The final 33.3% batch increase gains 2.623% aggregate rate. B36864 edge trials
on a separate GPU complete at 154343.359 live/154176.223 policy, retaining
56556 successful/zero failed lift cycles and 41814 resets per repeat. Sampled
peaks are 92607/94633 MiB. The edge/curve ratio is not a controlled speedup.
Every repeat/source/bank/work record is preserved.

B40960 is rejected during build by the unchanged 3678×15 Jacobian signed-i32
addressing guard; its exact limit is 38924. B38912, the last convenient complete
2048-condition multiple, fails during warmup's complete sample-position growth
with CUDA_ERROR_OUT_OF_MEMORY. Both failures are preserved without rates.
B36864 is the largest successfully measured complete bank, not proof of the
largest feasible individual integer batch. The remaining index margin is
5.59% in environment count at unchanged batch time; it is not a wall-time
bound. No collision capacity, physical sample or guard was reduced to fit.

## Output costs and measured bottlenecks

Each output pair uses one GPU/bank/configuration in max/full/max order, with
three complete repeats per route. The pooled max denominator includes both
max controls. Same-state numerical accuracy uses the tests/oracles above.

| Batch | Scope | Pooled max env-step/s | Full env-step/s | Measured rate cost | Additional field bytes |
|---|---|---:|---:|---:|---:|
|1024|Live|44614.176|44606.583|0.017%|110100480|
|1024|Policy|42682.843|42353.412|0.772%|110100480|
|8192|Live|115474.673|114621.652|0.739%|880803840|
|8192|Policy|113804.266|112955.938|0.745%|880803840|

Live positions are exact. Policy keeps identical banks/configuration/network
and per-environment outcomes, but independent CUDA/closed-loop trajectories
are not bit-identical. At B8192 full/max final-position difference is223µm,
with65µm in the same-mode control; peak differences are9.496Pa versus199.072Pa
in that control. These compare different evolved loads, not same-state stress
error. The original overly strict cross-process audit failures are retained.
Policy rates are reported as rollout costs rather than isolated output-write
overheads. No physical or same-state numerical tolerance was relaxed.

A separate intrusive50-step trace measures peak/output kernels at
0.244007→0.324171ms (B1024) and1.750063→2.287547ms (B8192). Those intervals
and setup/JIT/device telemetry are excluded from headline throughput timing.
200ms assigned-device samples include setup/trace and are not exact transient
peaks. Full field storage is107520 bytes/env; max stores zero field bytes.

Final B16384 stage-fenced profile means, in ms/step: fitting25.026,
initial fan samples22.906, rigid constraints14.432, signed scatter10.310,
blocked inverse10.212, ports6.391, complete residual5.562, global peak3.458,
source resolution3.082, inertia relief1.640, raw packing/capture0.477/0.096.
Serialized whole-step mean is125.934ms, p99/max284.191/329.985ms.22 retry
frames average288.024ms; all others123.099ms. Maximum fit evaluations have
min/median/p90/p99/max5/11/16/146/401. Nested profile intervals overlap;
neither their sum nor different-GPU profile means defines a speedup.

The four clean layout/capture/Hessian/diagnostic changes have a complete
same-GPU1.073181x live qualification. Warp counters then qualify separately
at1.094391x B8192 and1.173599x B16384. These gains are not multiplied.
The [candidate ledger](NATIVE_CONTACT_SAMPLING_ABLATIONS_20261012.md) and
[sampler report](NATIVE_WARP_COUNTERS_20261013.md) retain selected/rejected
trials, exact source closures and failures. Shared immutable operators,
actual sample storage/current-step geometry reuse, native packing/scatter,
cooperative retries, full residual/peak scans and reset/observation paths have
all been profiled or ablated. Shared elastic response, linear/history choices,
complete residual/peak scans, selectors and observation/lifecycle placement
also retain their applicable [native low-shell decisions](OPTIMIZATION_COVERAGE_20261011.md).
Those earlier contact-law rates remain historical; the current numerical
checks, profile and matched scopes qualify the nonconvex load path.
Graphs/fastcache, scalar layout, inactive walks
and extra refitting do not yield a stable selected dynamic gain. FP32 Gaussian
cost is below0.2% of the instrumented step. An unconditional Gram prepass or
deferred full-scatter replay is not selected: first acceptance covers13.829%
of positive samples, replay86.171%, and additional interior evaluations3.799%
of sample-evaluation work. These are work proxies, not universal latency bounds.

The diminishing copied-width returns, measured resource endpoints and completed
material-bottleneck inventory establish a practical endpoint for this device,
low mesh, law and budget. Fitting/geometry, rigid work and complete inverse/
scatter remain the main costs. Raw-source packing/growth placement could
change memory behavior, but no material steady-state gain is established;
preserving scene-wide source identity/API/lifecycle is necessary. Further
selective algorithms or backend address/memory changes are separate work,
not a claim of mathematical global optimality.

## Reproduce and inspect evidence

From this worktree on an allocated container, set `RIGID_STRESS_DATA_ROOT`,
`RIGID_STRESS_ENV` and source `research/rigid_stress/scripts/cluster_env.sh`.
The environment sets `PYTHONPATH=$PWD` and data-side caches. Use the archived
condition bank for exact workload reproduction:

```bash
BANK="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/native-final-matched-scopes-v1/b8192-623104/conditions.npz"
NEW_TEST_RUN="$RIGID_STRESS_DATA_ROOT/runs/reproduce/stress-cuda"

"$RIGID_STRESS_ENV/bin/python" examples/rigid/franka_egg_stress.py \
  --envs 32 --steps 2000 --varied --contact-model intrinsic \
  --intrinsic-load-partition body_source_prior --egg-geometry mesh64 \
  --collision-method gjk --contact-patch on

"$RIGID_STRESS_ENV/bin/python" -m examples.speed_benchmark.rigid_stress \
  --scope live --envs 8192 --seed 623104 --warmup 600 --steps 1280 \
  --repetitions 3 --minimum-seconds 10 --varied --contact-model intrinsic \
  --intrinsic-load-partition body_source_prior --contact-block-size 4 \
  --egg-geometry mesh64 --collision-method gjk --contact-patch on \
  --trajectory-metrics --compact-log --conditions "$BANK" \
  --output "$RIGID_STRESS_DATA_ROOT/runs/reproduce/live8192.json"

bash research/rigid_stress/scripts/check_nonconvex_final.sh cuda "$NEW_TEST_RUN"

"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
  --manifest research/rigid_stress/evidence/20261011-nonconvex/*/manifest.json \
  --output "$RIGID_STRESS_DATA_ROOT/runs/reproduce/verification.json"
```

Archive verification works from the published Git files. Add `--originals`
and `--allow-appended-allocation-logs` when the original runtime mount is
available at its recorded paths. The second flag validates the one known
appended allocation log's exact archived prefix.

`measure_nonconvex_output_modes.sh` reproduces the paired output protocol.
`summarize_nonconvex_closeout.py` validates source/bank/outcomes, scopes,
current-case coverage, output/trace costs and diagnosed capacity limits.
`publish_nonconvex_checkpoint.py` preserves complete source/results with
ordered32MiB large-JSON byte ranges; `verify_nonconvex_publication.py` checks
stored/original/decompressed hashes and exact chunk reconstruction. Original
data/caches remain under the declared project data root. The final closeout
checkpoint is [native-nonconvex-closeout-20261014-v1](../evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1/manifest.json):
6997 entries, 6997 exact originals and two complete large-JSON reconstructions.
Manifest SHA256 is
`cf30881e3185fa62d05d0460dd67bb34d6de77a6ad52fb8760cbf8d7a762798e`.

This is same-mesh low-resolution numerical/performance acceptance. Physical
mesh/quadrature convergence, stress/fracture calibration and complete RL
training remain separate studies, as required by the active brief.
