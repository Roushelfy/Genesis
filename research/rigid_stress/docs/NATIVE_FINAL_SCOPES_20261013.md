# Current native nonconvex four-scope measurements

Numerical source `76537c2f` integrates the qualified coalesced-coordinate,
valid-source, warm-Hessian, sparse-diagnostic and warp-counter optimizations.
The extension remains active while wider and copied-cohort curves finish.
No unfinished width or throughput optimum is claimed.

## Workload and scopes

RTX PRO6000 Blackwell Server Edition,97887MiB,driver595.71.05,
container1_26,Genesis1.4.3/Quadrants1.3.3. Each width runs its four scopes
sequentially on one assigned GPU; different widths use different allocations.
All scopes load the original92ec3e0c baseline bank for that width/seed,
preserving contact parameters and controller targets byte-for-byte.

The complete low hollow shell has42authored collision vertices/80outer
triangles,126linear volume geometry nodes,810P2nodes/2430DOFs and480tetrahedra.
`mesh64` preserves coordinate precision; it does not mean64vertices.
The auxiliary mesh/material/operator computation is native Quadrants.
E=10GPa,nu=.3,rho=2000kg/m³; rigid mass0.006529848210401324kg.
FP64 physical loads/operators/residual/stress, Q10/s8 intrinsic finite
footprints and `body_source_prior`; only warm Newton Hessians useFP32.
All final wrench, complete K/gauge residual and stress budgets remain unchanged.
GJK/contact patch enabled,dt=.01/substep1,max output. Actual combined friction
varies because the other contact materials use0.1, rather than dominating
the egg's varied coefficient with1.0.

Every scope uses600warmup and three1280-step repeats. Live/rigid/policy
include observation, contact-radius updates, reset, metrics, history and
fallback. Recovery-only repeats a frozen actual contact snapshot in its
matching pre-integration frame; it is not dynamic-contact rollout throughput.
Policy includes observation/action interoperability and a FP32 network
26→128→128→7 producing arm residuals on the scripted controller. Finger
targets remain scripted. This measures inference rollout with rigid/stress,
not a trained-policy success claim or complete RL training throughput.

## Verified current rates

Cells show effective env-step/s, with batch step/s in parentheses.
All attempted steps are valid. The full per-environment records remain in
the archives; no difficult environment is removed.

| Batch /seed | Rigid-only | Frozen recovery-only | Live rigid＋stress | Policy＋rigid＋stress |
|---|---:|---:|---:|---:|
|1024 /623101|125859.551 (122.9097)|97589.574 (95.3023)|45399.359 (44.3353)|42502.831 (41.5067)|
|2048 /623102|211411.647 (103.2283)|135534.656 (66.1790)|68509.262 (33.4518)|65377.457 (31.9226)|
|4096 /623103|353558.373 (86.3180)|159786.742 (39.0104)|91945.117 (22.4475)|94228.290 (23.0050)|
|8192 /623104|505233.279 (61.6740)|178368.978 (21.7736)|117496.253 (14.3428)|116118.497 (14.1746)|

Every repeat has zero invalid environment steps, scatter overflows, failed
lift windows and failed lift cycles. Rigid/live final positions and every
per-environment rigid/contact/grasp/descent counter match exactly. Policy
actions differ, so its rate is not a controlled estimate of policy overhead;
changed contact work and repeat variation can make it slightly faster.
Frozen snapshot repeat loads, original wrench and peak checks also pass.

The native and sampled assigned-GPU memory records, setup/JIT costs, repeat
durations, resets, actual grasp outcomes and per-environment recovery tails
are preserved in each `native-final-matched-scopes-b<B>-<seed>-checkpoint-v1`
manifest.200ms memory samples include setup and are not exact transient peaks.
All original/stored/decompressed hashes and large-JSON reconstruction verify.

## Remaining measurements

B16384/991909 andB32768/623102 four-scope runs continue. A separate same-GPU
curve tiles the same2048-condition bank atB8192/16384/24576/32768. Those
measurements and the final plateau/cost summary remain required.
[Sampler qualification and detailed profiling](NATIVE_WARP_COUNTERS_20261013.md)
include two controlled live improvements and624final-source dynamic fields
checked independently on CPU FP64. The highest current controlled live rate
is135495.178env-step/s atB16384/734001; this is not yet a final optimum claim.

Reproduce with the exact archived source and
`run_native_final_matched_scopes_v1.sh <batch> <seed>`, or the public
`examples.speed_benchmark.rigid_stress` command and saved conditions captured
by that launcher. Outputs/caches belong under `RIGID_STRESS_DATA_ROOT`.
Physical high-mesh or quadrature convergence is outside this phase.
