# Native surface sampler warp counters

This follow-up preserves the qualified low-shell FP64/Q10/s8 intrinsic
footprint and `body_source_prior` vector-traction law. It changes CUDA
scheduling and counters; reference geometry, physical radius, Gaussian
profile, friction, original-point wrench and final accuracy budgets remain
unchanged. The nonconvex extension is still active.

## Implementation and qualification

Each complete32-lane group processes one source. Eligible lanes retain every
sample, including metadata for a profile that underflows after the original
eligibility test. Warp reductions accumulate candidate/accepted counts and
the FP64 maximum distance. A lane prefix assigns every accepted index to
one complete reserved interval. Exact observed-capacity growth and replay
remain active; no sample or contact is truncated.

Three CUDA hard/growth smoke cases pass157.13s. The complete35-case prototype
CUDA matrix passes808.30s, including source-prior full-field CPU oracles,
U/cavity/narrow-gap/hole checks, rotated/offset friction loads, changing
contacts, streaming, max/full output, state restoration and partial resets.
Final wrench1e-8, complete residual including6gauge
`max(1e-11N,1e-8||rhs||)`, stressRTOL1e-4/ATOL1e-3Pa are unchanged.

Production numerical commit `76537c2f` removes the unused original
CUDA kernel, private experiment flag and host dispatch. Its CUDA kernel
arguments/body and frontier kernel are AST-identical to the prototype.
The CPU route uses the unchanged footprint implementation and does not
allocate the CUDA frontier. The previous35CPU qualification therefore
continues to cover that route. A fresh35CUDA clean-source matrix passes810.81s.
The published qualification includes48CPU/inherited,48CUDA/prototype and
48CUDA/clean independent fixture fields. Their worst normalized wrench
error is7.394051316e-10, full residual budget fraction0.04585622523 and
global peak difference7.967217243e-6Pa. CPU asset inspection verifies126
linear geometry nodes,810P2nodes/2430DOFs,480tetrahedra and80outer faces.

## Complete B8192 dynamic pair

Both routes run on RTX PRO6000 Blackwell Server Edition GPU
`GPU-35160c09-d879-c0fe-c36b-aec37590eb6a`,97887MiB,
driver595.71.05/container1_26. Genesis1.4.3/Quadrants1.3.3,
B8192/seed623105,mesh64/GJK patch-on,dt.01/substep1,max output.
Each route uses600warmup and three1280-step varied trajectories. The saved
condition banks are identical. Observation, contact-parameter updates,
resets, metrics, history, fallback and every environment are included.

| Route | Valid env-step/s | Batch step/s | Repeat seconds | Native bytes | Sampled peak MiB |
|---|---:|---:|---|---:|---:|
|Qualified four-optimization baseline|107515.913|13.124501|97.704219 /97.550371 /97.327960|7658839626|21895|
|Warp counters|117664.460|14.363337|89.142193 /89.116178 /89.088966|7658839626|21477|

The ratio is1.094391115; repeat ranges do not overlap. Final rigid positions
and all per-environment rigid/contact/grasp/descent counters agree exactly.
Each repeat retains9287resets and12562successful lift cycles, with zero
invalid stress steps, scatter overflows, failed lift windows or failed cycles.
Weak-grip and release descents remain in the reported workload. The200ms
assigned-device memory samples include setup and are not exact transient
peaks. This is a complete live result; gains from the four earlier isolated
optimizations are not added to it.

## Independent B16384 pair

B16384/seed734001 uses the same complete protocol on GPU
`GPU-1cab8d77-cbae-8609-6ffb-46e684ef34dc`.

| Route | Valid env-step/s | Batch step/s | Repeat seconds | Native bytes | Sampled peak MiB |
|---|---:|---:|---|---:|---:|
|Qualified four-optimization baseline|115452.687|7.046673|181.762680 /181.460198 /181.715154|16129154634|46503|
|Warp counters|135495.178|8.269969|154.742754 /154.927646 /154.660220|16129154634|44017|

The ratio is1.173599173 with nonoverlapping repeat ranges. Every repeat
retains18571resets and25124successful lift cycles. Invalid steps, overflows
and failed lifts remain zero; banks, positions and every per-environment
rigid/contact/grasp/descent counter agree exactly. This separated GPU/seed
cohort is a replicated decision, not an additive gain estimate.

The complete qualification archive is
`native-sampler-warp-qualification-checkpoint-v1`, verified manifest SHA256
`f7f30330a21851a75b8a5f1236b04c8a4407dae71b23d20a9fca147db2bf2bdb`.
All original/stored/decompressed hashes and complete large-JSON chunk
reconstruction pass. Exact prototype/clean source equivalence, CPU route
inheritance, both35CUDA reports and all complete live outcomes are retained.

## Final-source profiling and CPU audit

The final B16384/623102 complete1280-step profile retains all18571resets,
25124successful lift cycles and22retry frames, with zero invalid/failed
outcomes. Instrumented mean is125.933700ms; step minimum/median/p90/p99/max
is24.024565/134.952510/148.715007/284.190540/329.985027ms. Retry mean is
288.023924ms versus123.099054ms without retry. Per-environment maximum fit
evaluation minimum/median/p90/p99/max remains5/11/16/146/401.

| Stage | Mean ms per instrumented step |
|---|---:|
|Initial sampler|22.906279|
|Warm cooperative fit|25.026422|
|Signed P2 scatter|10.310457|
|Blocked inverse application|10.211866|
|Surface ports|6.391099|
|Complete residual|5.562350|
|Global peak|3.457928|
|Source resolution|3.081685|
|Contact packing|0.476584|
|Source capture|0.095575|

Nested wrapper/chart/sample timings overlap. Fences and exports perturb
scheduling; these are diagnostic latencies. Different GPU allocations
preclude treating the125.934/147.439ms profile means as a paired speedup.
The two uninstrumented complete pairs above establish end-to-end gain.

Independent CPU FP64 checks464captured complete fields: full residual budget
fraction≤0.00563800285, global peak difference≤1.860404154e-6Pa, all tensors
and von Mises fields within original gates. An additional160fields verify
original contact/free-moment wrench after signed scatter, independent mass,
centrifugal loads, inertia relief and the complete K/gauge system. Their
worst normalized wrench is8.627857791e-9, relief RHS difference6.325950689e-8
of its residual allowance, full residual fraction0.00551764678 and peak
difference1.540160156e-6Pa.

Verified manifests are
`native-final-sampler-dynamic-checkpoint-v1`
(`f0977e350798a5e3cfe60fae4723c95710dc766c54022123bdcbfa9b906538e8`)
and `native-final-sampler-wrench-oracle-checkpoint-v1`
(`a1201060018083030a4855aa71a9859f2eecfa564216c78eb693ca36e524ccf7`).

## Remaining evidence

Final-source B1024/2048/4096/8192 four scopes are verified and published;
see [the current scope tables](NATIVE_FINAL_SCOPES_20261013.md).
Wider four-scope measurements and copied-condition scaling are running.
A complete B2048/623102 applicability audit observes51.036% of loaded groups
with no projected samples, but only12.638% of sample-evaluation work. These
interior cases average1.42983 evaluations;72.928% of their constraints and
58.634% of their samples already finish at the first evaluation. Additional
interior evaluations are3.799% of total sample-evaluation work.

Across all loaded groups,40.612% finish at the first warm evaluation, covering
only13.829% of positive samples and5.657% of sample-evaluation work. A route
that first verifies a warm proposal through full scatter would replay86.171%
of samples. Current normalization/validation and physical verification remain
required. Combined with25.026ms fit and10.310ms scatter, these work counts give
no reasonable material-gain expectation for an unconditional Gram prepass or
deferred full-scatter/replay route. Both remain unimplemented on this concrete
cost evidence. These proxies are not latency bounds and do not rule out all
unknown selective algorithms.

The complete read-only Quadrants diagnostic retains all1280frames/2048envs,
actual outcomes and raw per-environment counters; production numerics and
all acceptance gates are unchanged. Verified archive
`native-fit-interior-cost-checkpoint-v2`, manifest
`10609e8f9a80a89d7e3b784949aaead3d2591637e62bc1e2e884b87cc3746e6e`.
The earlier audit is retained and referenced. Wider scope/copy publications,
the final profile/cost plateau report and final publication remain required.
No unfinished measurement is counted as a successful result.

The earlier standalone counter probe validates six synthetic count/index/max
configurations and motivates this candidate. It excludes geometry/profile
work and does not explain all measured rollout improvement or establish a
whole-step saving bound. Its failed versions are preserved in the verified
`native-sampler-counter-cost-checkpoint-v1` archive, manifest SHA256
`49b56b3d72aa55d756dd96f7cb17ff3c4b179f27f5090264e679ed3440386804`.

The completed B8192 pair is checked by
`check_native_final_memory_pairs_v6.py`, with exact before/after source
closures, assigned-GPU telemetry and all workload records. Its raw runs and
immutable launchers are under the task's declared data root and in the
completed two-cohort publication. Policy measurements describe inference rollout, not full RL
training. No high-resolution physical convergence is claimed.
