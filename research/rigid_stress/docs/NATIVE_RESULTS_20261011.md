# Repaired native low-shell validation and performance

Completed on 2026-10-11. Production numerical/lifecycle validation, the complete
measurement matrix and all original/publication hash checks pass. The numerical engine is
`f3e92c6267357143b859b24a27a3a67731586667`; benchmark `1f22f6a2` adds only
explicit untimed standalone recovery preparation. Aggregator `881ab76a`
checks every engine/controller/options hash and permits only the two pinned
benchmark hashes. Exact per-case revisions and GPU UUIDs remain in the evidence.

The native feature, normal Franka example, regression tests, benchmark and
public maximum/full-field API are runnable without a research numerical backend.
The [final generated tables](FINAL_MEASUREMENTS_20261011.md) contain all rates,
repeat spreads, matched overhead/output costs, episode outcomes, work tails
and profiles. The [v8 baseline](BASELINE_MEASUREMENTS_20261011.md) uses the
repaired load model before final graph selection; the
[2026-10-10 report](NATIVE_RESULTS_20261010.md) is the earlier coefficient-1
workload. Rates from separate hardware allocations are not causal ablations.

## Model and unchanged acceptance

The collision shell has **42 vertices and 80 faces**. The complete elastic
hollow shell has 126 P1 vertices, **810 P2 nodes**, 2430 displacement DOFs,
480 tetrahedra and 162 exterior P2 nodes. E=10 GPa, nu=0.3, density=2000
kg/m3 and authored mass=0.006529848210401324 kg, with compatible COM/inertia.
Geometry, material and mass identify shared immutable operators. All production
assembly, loads, solves, residuals and stress reductions use Quadrants FP64.
Recovery does not feed deformation or forces back into rigid dynamics.

Positive Q10x10 integration gives 8000 exterior samples. A failed sampled fit
retries Q10 on a seven-triangle partition of the anchor face, replacing that
face's original samples with 700 local samples. The original B2048 fixture
has 132 eligible samples and an LP-infeasible transverse hull gap of 10.9 um;
it and the other two captured actual contacts now recover successfully.
This is a declared local integration repair. Contact forces, radii and final
acceptance are preserved. Sampling inadequacy, numerical stagnation and
inadmissible inputs are distinguished in [contact repair](CONTACT_REPAIR_20261010.md).

The declared law uses the supplied pad normal, nonnegative scalar pressure
and a constant tangential/normal ratio within each footprint. That supplied
frame need not equal each faceted surface normal. Force-line anchoring
preserves the original point wrench. Radius is an explicit input, varying
here over 4.08–8.28 mm. Actual combined contact friction varies over 0.42–0.84;
the other bodies no longer mask it with coefficient 1.
See [the pad law](PAD_LAW.md) for assumptions and friction admission.

Internal fitting stops at normalized gradient **1e-9**. The independent CPU
fit retains 2e-12. Final acceptance remains:

| Check | Budget |
|---|---|
| Contact resultant force | Relative 1e-8 |
| Contact moment | Error / (force magnitude times original radius) <= 1e-8 |
| Complete equilibrium, including six gauge rows | max(1e-11 N, 1e-8 times RHS norm) |
| Independent CPU FP64 peak and complete field | rtol=1e-4, atol=1e-3 Pa |

The selected engine passes **52 GPU cases**, **31 CPU cases** (21 CUDA-only
CPU skips), **432 independent CPU FP64 full-field trajectory samples**, the
configuration-input oracle, batching/link isolation and partial/global reset
checks. Forced bounded-task overflow and four physical substeps are included.
Paired two-seed 2400-step trajectories retain exact equality of all six public
rigid state fields and controller phase. Nodal differences are at most
1.38778e-16 N and paired full-field differences 2.16962e-6 Pa.

Independent oracle samples use at most 0.62436% of the complete residual
budget; peak/tensor/von-Mises differences are below 6.085e-6 Pa. The larger
every-environment audits use at most 1.09887% of that budget, force relative
error 7.8837e-10 and moment relative error 7.04139e-10. No final threshold
is relaxed to accept the internal stopping change.

## Hardware and measured scopes

NVIDIA RTX PRO 6000 Blackwell Server Edition, 101975851008 device bytes,
188 SMs, compute capability 12.0 and 128 MiB L2; Intel Xeon Platinum 8562Y+
hosts. Allocated `rtx-mid`, `genesis:1_26`, Python 3.10.12, Genesis 1.4.3,
Quadrants 1.3.3 (`00423caa`) and Torch 2.8.0. Each case records hardware,
source hashes, complete arguments, job ID and GPU UUID.

dt=0.01 s, one physical substep, elliptic friction, 100 rigid constraint
iterations and rigid tolerance 1e-12. Seeds 510000 and 623001 independently
vary pose, actual coefficient, footprint and episode delay. Each scope uses
900 warmup and three repeats of at least 2400 steps and ten seconds. Reset,
bounded selector-cache maintenance, contact updates and required observation/
policy work are timed. Setup/JIT and intrusive audits/profiles are separate.

Rigid-only runs the matching scripted trajectory with recovery disabled.
Recovery-only repeats a frozen actual snapshot in its pre-integration frame.
Live includes changing contacts and complete scripted trajectories. Policy
adds 26-column device observations and seeded FP32 26->128->128->7 tanh MLP
residual inference (seed 99173, scale 1e-4 rad, FP64 shadow budget 1e-9 rad).
Policy results measure inference rollouts, not complete RL training.

## Final effective throughput

Rates below are both seeds' summed valid transitions divided by summed time.
The 40 selected measurements contain **8,436,997,120 attempted timed env-steps
and zero invalid env-steps**. Frozen recovery is a diagnostic scope, not live
trajectory throughput.

| B | Rigid-only env-step/s | Frozen recovery env-step/s | Live rigid + stress env-step/s | Policy + rigid + stress env-step/s |
|---:|---:|---:|---:|---:|
| 1024 | 114955.85 | 287497.30 | 83206.34 | 82701.39 |
| 2048 | 188927.84 | 367027.94 | 125710.14 | 124977.50 |
| 32768 | 772715.25 | 493423.87 | 310412.27 | 310609.92 |
| 49152 | 845095.93 | 507383.96 | 324477.61 | 324233.09 |
| 61440 | 872767.76 | 519383.15 | 329512.88 | 329178.40 |

The best measured aggregate is B61440. Its four batch step/s rates are
14.2052 / 8.4535 / 5.3632 / 5.3577. Live/policy six-repeat ranges are
326797.48–332076.47 / 326435.18–331810.02 env-step/s, with standard
deviations 2632.34 / 2778.00. These spreads include different seeds and
physical GPUs; each case retains its own three-repeat spread. Sampled
whole-card live/policy peaks are 72693 / 72895 MiB, including setup/JIT.
Native buffers, saved reset state, Torch allocation and selector bytes are
separately recorded. No slow or failed environment is removed from time.

Same-seed matched live step-time overhead over rigid-only is 38.15–38.17%
at B1024, 50.03–50.55% at B2048 and 164.54–165.20% at B61440. Corresponding
throughput decreases are 27.61–27.62%, 33.35–33.58% and 62.20–62.29%.
The aggregate B61440 live step takes 2.64866 times the rigid-only step.
Full per-case step-time and throughput comparisons remain in the tables.

All ten recovery cases explicitly prepare the standalone graph before timing.
Small-batch preparation takes 5.00–5.51 seconds including snapshot restoration,
JIT/capture, synchronization and errno check; large-batch preparation takes
0.16–0.23 seconds. Original small-batch v9 cold-path timings are preserved
and marked invalid timing. No numerical failure is inferred from that issue.

Full-field output writes all four corners of every affine-P2 tetrahedron.
It adds exactly **107520 bytes/env**: 110100480 bytes at B1024 and
3523215360 at B32768. Same-GPU full-output step-time costs are 0.756–0.865%
at B1024 and 1.590–1.975% at B32768. Both modes retain the same complete
load, solve, residual and global maximum; max-only allocates no full field.

## Contact outcomes and tail work

Sixteen untimed every-environment audits cover 900 warmup plus 2400 trajectory
steps at B1024/2048/32768/61440, both seeds and live/policy. Measured episode
totals subtract warmup: 1556480 started, 1168016 completed and meeting the
declared lift/hold/release criteria, zero failed completed episodes, 388464
unfinished and zero restart-aborted. **1014609 episodes have deliberately
weak-grip height loss above 1 cm.** Weak-phase drops remain disclosed and
are distinct from the declared strong-grip/hold/release success criteria.
Unfinished episodes are not counted as success. No numerical step fails.

Independent contact-motion audits measure pre-response tangential velocity
and friction-cone excess through support, eccentric grip, slip, contact
birth/death, release and rotation. Two seeded small-batch trajectories include
76219/97843 loaded-finger sliding samples and 2351/4399 weak-grip samples;
maximum pre-response tangential speeds are 0.669/0.613 m/s and cone excess
is at most 5.56e-17 N. Height loss is a drop proxy, not a tangential slip
distance. The diagnostic exposes the pre-integration frame with post fusion
disabled; paired lifecycle validation proves exact rigid-state equivalence.

Work tails include up to 109 counted Newton evaluations for one contact mapping,
five local-integration retries per environment over the audited trajectory,
79 packed boundary columns and 158 admitted face tasks. Audited inverse
correction and factor-fallback totals are zero; forced overflow/fallback
correctness is separately tested. Full p50/p90/p99/max distributions and
all failure/unfinished counts remain in the summary. Individual GPU latency
cannot be separated from shared SIMT kernels; work counts and batch intervals
are reported separately.

## Optimization decisions and practical plateau

The [coverage inventory](OPTIMIZATION_COVERAGE_20261011.md) and
[ledger](OPTIMIZATION_LEDGER_20261010.md) evaluate 22 candidate families.
Retained routes include the exact complete-exterior shared FP64 response,
all-nonzero load packing, face reduction and bounded tasks with complete
overflow, cached gradients, current-footprint reuse, coalesced residual,
serial stress graph, native configuration, bounded selectors and saved public
reset state. Mixed precision/correction, history prediction, extra layout/
tiling, failure compaction, alternate state copies, observation/policy graphs
and direct graph entry have measured losses, marginal gains or concrete costs.

The extra rigid-post graph has stable small-batch matched gains: B1024 live
1.79–1.94%, policy 1.60–1.65%; B2048 live 1.35–1.52%, policy 1.00–1.12%.
Ordinary large-batch gains have opposite signs (B32768 +1.034%/-0.299%,
B61440 +0.271%/-0.033%), so automatic fusion stays below B8192.

Adding 50% environments from B32768 to B49152 gains only 4.53% live and
4.39% policy; another 25% to B61440 gains 1.55% / 1.53%. The repaired v8
baseline independently shows only 2.60% live gain over that last increment.
The diminishing repeated aggregate returns, marginal last candidates and
completed cost inventory establish a **practical plateau for this device,
low mesh, workload and accuracy budget**. This is not a global optimum proof.

The remaining costs are native rigid work, complete inverse application and
contact integration/scatter. B1024 snapshot solve/scatter/pressure costs are
0.8038 / 0.5823 / 1.1145 ms. B32768 combined contact work is 23.8290 ms,
solve 22.4627, complete residual 10.3030 and global peak 6.8698. Changing-stream
profiles retain p99/max dispatch and reset spikes; nested intervals overlap
and are not summed to derive throughput.

The original rigid Jacobian guard limits this scene to B61497. B61440 is the
largest convenient multiple of 512; the remaining 57 environments offer only
0.093% ideal gain at unchanged batch time. B65536 exceeds signed-i32 indexing.
Scalar allocation, Vec3 packing and explicit-i64 boundary diagnostics do not
provide safe access: packed storage allocates but boundary markers raise CUDA
illegal address. The original guard stays intact. This is a backend address
capacity limit, not measured physical-memory exhaustion.

## Reproduction and limits

The complete final summary SHA256 is
`b44198c77caf8f3c4aaea93f08a16a4f313114d638a91f97403330b787962100`.
It contains all 40 rates, eight output-cost pairs, sixteen audits and ten
profiles, with no missing or invalid-timing selected case. The earlier source-
key aggregation failure is preserved; a 56-case validator checks the exact
engine and the two pinned benchmark hashes and rejects unreviewed changes.
Publication checkpoint v14 verifies all **2004 artifacts and 2004 originals**.
Its manifest SHA256 is
`07059dc5f7a23c35dd52cab335c576886d03d5dbc5905b87c2f41610456fbf2f`.
Commands, explicit
data/cache paths and preserved launch scripts are in [RUNNING.md](RUNNING.md)
and the [evidence manifest](../evidence/20261010-contact-repair/manifest.json).

This phase validates the low-resolution complete hollow shell on the same
mesh. Physical stress/fracture convergence remains a separate later study.
Independent free rigid roots are supported. Autodiff, joint supports,
external applied wrench and rolling/torsional contact moments need additional
declared load models and have explicit failure semantics. No high-resolution
physical mesh or global quadrature convergence experiment ran in this round.
