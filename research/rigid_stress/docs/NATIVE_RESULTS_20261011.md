# Repaired native low-shell validation and performance

Status at 2026-10-11 03:41 UTC: production source `1f22f6a2` is committed and
pushed. Numerical/lifecycle validation passes. Production-v8 measurements
cover all five batches, both seeds and four scopes. The production-v9 final
remeasurement and formal large-batch post-graph comparisons remain in progress;
this document does not yet declare goal completion or a throughput plateau.
Exact source hashes accompany every measurement. The older
[2026-10-10 report](NATIVE_RESULTS_20261010.md) is a historical fixed-friction
milestone, not the repaired workload's baseline.

## Model and fixed acceptance

The rigid collision shell has **42 vertices and 80 faces**. Its complete
elastic shell has 126 P1 vertices, **810 P2 nodes**, 2430 displacement DOFs,
480 tetrahedra and 162 exterior P2 nodes. E=10 GPa, nu=0.3, density=2000
kg/m3 and the declared hollow asset mass is 0.006529848210401324 kg, with
compatible authored COM/inertia. The rigid shape stays rigid. All production
assembly, loads, recovery, residuals and stress reductions use Quadrants.

Q10x10 positive quadrature gives 8000 exterior samples. A failed sampled fit
retries Q10 on a seven-triangle partition of the anchor face, replacing that
face's original samples with 700 local samples. This is a declared local
integration change, not a larger footprint or a discarded contact. The
original B=2048 132-sample LP-infeasible apex fixture now recovers correctly.
Sampling inadequacy, numerical stagnation and inadmissible input are distinct
diagnoses in [contact repair](CONTACT_REPAIR_20261010.md).

The pressure law uses the supplied pad normal and a nonnegative scalar
pressure with constant tangential/normal ratio within each footprint. That
frame need not equal every faceted reference-surface normal. Force-line
anchoring preserves the original point wrench. Radius is an explicit pad
input, varying here over 4.08–8.28 mm. Actual combined contact friction varies
over 0.42–0.84; the other bodies no longer mask that variation with mu=1.
See [the load law](PAD_LAW.md) for model assumptions and friction admission.

The internal normalized fit threshold is **1e-9**, selected by the actual
error budget. The following final checks remain unchanged:

| Check | Budget |
|---|---|
| Contact resultant force | Relative 1e-8 |
| Contact resultant moment | Error / (force magnitude times original radius) <= 1e-8 |
| Complete equilibrium, including all six gauge rows | max(1e-11 N, 1e-8 times RHS norm) |
| Independent CPU FP64 peak and complete field | rtol=1e-4, atol=1e-3 Pa |

The offline CPU pad oracle retains its stricter 2e-12 internal fit threshold.
Production v8 passes **48 GPU cases**, **27 CPU cases** (21 CUDA-only skips),
**432 independent full-field CPU FP64 trajectory samples**, the independent
configuration-input oracle and two-seed partial/whole reset equivalence.
The worst sampled full residual uses 0.6274% of its budget; maximum absolute
peak/tensor/von-Mises discrepancies are below 5.647e-6/4.102e-6/5.647e-6 Pa.
Forced bounded-task overflow and four physical substeps are included.

The additional serial rigid-post graph in `f3e92c62` passes 52 GPU cases,
31 CPU cases (21 CUDA-only skips), another 432 independent full-field CPU
FP64 samples and two-seed public-state/partial-reset equivalence. All six
rigid fields and controller phase agree exactly over paired 2400-step
trajectories. Full residual uses at most 0.62436% of its budget; peak,
tensor and von-Mises errors remain below 6.085e-6 Pa. Final warmup already
compiles live/policy graphs. Commit `1f22f6a2` separately primes standalone
recovery before its timing, recording JIT/capture preparation cost; original
small-batch v9 cold-path timings remain archived and flagged invalid timing.

## Workload and hardware

NVIDIA RTX PRO 6000 Blackwell Server Edition, 101975851008 device bytes,
188 SMs, compute capability 12.0 and 128 MiB L2; Intel Xeon Platinum 8562Y+
hosts. Runs use allocated `rtx-mid` GPUs and `genesis:1_26`, Python 3.10.12,
Genesis 1.4.3, Quadrants 1.3.3 (`00423caa`) and Torch 2.8.0. Each artifact
records its exact GPU UUID, job ID, source hashes and configuration.

dt=0.01 s, one physical substep, elliptic friction, 100 rigid constraint
iterations and rigid tolerance 1e-12. No contact pruning, hibernation,
rolling/torsional friction or stress feedback into rigid motion. Both seeds
510000 and 623001 independently vary pose, coefficient, footprint and
episode delays. Each rate uses 900 warmup steps and three measured repeats
of at least 2400 steps and 10 seconds. Reset, selector-cache maintenance,
contact parameter updates and required observation/policy work are timed.
Setup/JIT, intrusive audits and profiling are separate.

Recovery-only repeats a frozen actual contact snapshot in its corresponding
pre-integration frame. Live and policy run changing contacts and complete
scripted trajectories. Policy adds device observations and seeded FP32
26->128->128->7 tanh MLP inference; the FP64 shadow control budget is 1e-9
rad. These rates describe inference rollouts, **not full RL training**.

## Completed production-v8 small-batch rates

The complete v8 baseline covers all 40 measurements, 16 every-environment
episode audits and ten profiles. Its complete rates, same-GPU full-field
costs, overhead, outcomes, numerical/work tails and detailed split profiles
are in [the baseline tables](BASELINE_MEASUREMENTS_20261011.md). B61440
achieves 328541.38 live and 328456.33 policy valid env-step/s, with six-repeat
ranges 327838.40–329298.74 / 327891.64–329181.80 and zero invalid transitions.
Both rates are about 5.35 whole-batch steps/s. Maximum sampled live/policy
whole-card memory is 72675/72877 MiB. Rigid-only at that batch is 861027.87
env-step/s; frozen recovery is 519458.92. These are the completed v8 baseline,
before final post-graph selection and remeasurement.

The table uses both seeds' summed valid transitions / summed time. All
completed measurements below have zero invalid transitions. Six-repeat
ranges include seed and temporal variation.

| B | Scope | Valid env-step/s | Batch step/s | Six-repeat env-step/s range |
|---:|---|---:|---:|---:|
| 1024 | Rigid only | 115094.76 | 112.40 | 114242.14–115770.57 |
| 1024 | Frozen recovery only | 289730.51 | 282.94 | 289525.07–289952.48 |
| 1024 | Live rigid + stress | 81657.09 | 79.74 | 80958.61–81980.82 |
| 1024 | Policy + rigid + stress | 80896.80 | 79.00 | 79683.74–81597.23 |
| 2048 | Rigid only | 189444.08 | 92.50 | 187881.01–190674.52 |
| 2048 | Frozen recovery only | 365117.99 | 178.28 | 363505.84–366630.15 |
| 2048 | Live rigid + stress | 123670.88 | 60.39 | 123319.33–123829.09 |
| 2048 | Policy + rigid + stress | 123136.27 | 60.13 | 122756.59–123848.68 |

Relative to matched rigid-only, live step time increases 40.06–41.85% at
B=1024 and 52.49–53.88% at B=2048. A step-time increase differs from a
throughput decrease; both are preserved in the summary. Sampled whole-card
peak memory is 3525/3535 MiB for live/policy at B=1024 and 4713/4725 MiB at
B=2048. These include setup/JIT. Native buffers, saved public state, Torch
allocations and reset selector storage are separately inventoried.

At B=1024, same-GPU max/full live/policy pairs on both seeds show full-field
step-time costs of 0.503–0.752%. Full field adds exactly 107520 native bytes
per environment. Fields represent all four corners of every affine-P2
tetrahedron; maximum-only allocates none of that storage. Both modes use
the same full load, solve and acceptance. Larger output-cost pairs remain
in measurement.

## Contact trajectories, lifecycle and remaining measurements

Every-environment audits retain started, completed, unfinished and restart-
aborted episodes, overlapping failure reasons, weak-grip drop proxies and
all numerical failures. Passing lift/hold/release criteria does not imply
that an intentionally weakened grasp never drops the object. Actual
pre-response tangential contact velocities and cone excess are independently
audited; a height-drop proxy is not described as a measured slip distance.
Work tails include Newton evaluations, local retries, packed columns, face
tasks and fallback. Individual GPU latency cannot be separated from shared
SIMT kernels; work counts and whole-batch intervals are reported separately.

Production-v8 B=32768/49152/61440 four-scope measurements, larger max/full
pairs, profiles and every-environment audits are complete. The production-v9
remeasurement and v10 primed-recovery replacement are still running.
The original rigid flattened i32 buffer limit rejects B=65536; this is a
recorded capacity diagnostic, not an OOM result. The optimization inventory
and its decisions are in [coverage](OPTIMIZATION_COVERAGE_20261011.md) and
the [ledger](OPTIMIZATION_LEDGER_20261010.md).

Current publication checkpoint v11 verifies all 1835 publication artifacts
and all 1835 preserved originals. Its manifest SHA256 is
`a7f0faea456b50f2e1a0dd0906743829a9b2938cb6fa758151d05a7d0e48ee73`.
Later closed cases await the next checkpoint. Commands and data
locations are in [RUNNING.md](RUNNING.md) and the evidence manifest.

This phase uses only the low-resolution full hollow shell. It establishes
same-mesh numerical consistency and measured performance, not physically
converged egg stress or fracture accuracy. Independent free rigid roots are
supported; autodiff, joint supports, external applied wrench and rolling/
torsional contact moments require additional declared load models and are
diagnosed explicitly. Final goal acceptance remains open.
