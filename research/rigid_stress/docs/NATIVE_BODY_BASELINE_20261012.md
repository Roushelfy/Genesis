# Native body source-prior and sampling frontier baseline

This checkpoint measures the explicit `body_source_prior` law together with
the selected exact-sample face frontier. It is a new baseline for that law,
not a matched comparison against componentwise contact constraints. The
numerical source is `5e8f97d3`; run labels retain the pre-commit `cefc2780`
plus combined-model suffix. Exact source hashes, unchanged before/after
closures, commands and hardware identifiers are archived.

## Copied B=1024 results

NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU UUID
`779281f8-1132-b676-132d-571b2ccec62c`, Genesis container `1_26`,
Quadrants 1.3.3, FP64, dt=0.01 s, one substep. The same complete low hollow
shell has 480 P2 tetrahedra, 810 P2 nodes and 2430 DOFs; Q10, squared fan
coordinates and eight edge subdivisions are unchanged. The 32-condition bank
SHA256 is `156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.

| Scope | Effective env-step/s | Batch step/s | Repetition rates, env-step/s |
|---|---:|---:|---|
| Rigid-only | 229478.178 | 224.099783 | 229609.445 / 229020.330 / 229804.695 |
| Recovery-only | 46301.686 | 45.216490 | 46293.971 / 46300.967 / 46310.122 |
| Live rigid + stress | 34843.754 | 34.027104 | 34827.345 / 34840.876 / 34863.060 |
| Policy + rigid + stress | 34092.695 | 33.293648 | 34051.873 / 34176.309 / 34050.212 |

Recovery repeats the validated frozen snapshot. Live/policy each use three
1280-step repetitions with 1184 actual per-environment resets per repetition;
all 1024 failure counters are zero, with zero invalid environment steps or
scatter overflow. Necessary contact/radius updates, controller, reset,
observations and inference are timed within their scopes. Policy scope is
inference rollout, not complete RL training.

The rigid-only minimum duration extends its repetitions to 2243/2237/2245
steps and 2816/2816/2848 resets. Therefore the rigid/live rate ratio is not a
trajectory-matched slowdown comparison. Stress-enabled native allocations
are 3090946654 bytes; device and Torch memory counters are preserved in the
raw results. The maximum warmup complete residual budget fraction is
0.006811217 across stress scopes; final acceptance is unchanged.

## Profile and snapshot checks

The 128-step frozen-snapshot stage wall profile is:

| Stage | ms/batch |
|---|---:|
| Begin | 0.056662 |
| Association / transform | 0.191177 |
| Source capture | 0.169595 |
| Intrinsic contact pipeline | 22.305332 |
| Inertia relief / centrifugal | 0.121862 |
| RHS | 0.060125 |
| Linear solve | 0.841630 |
| Complete residual | 0.343734 |
| Global peak | 0.247954 |
| Accept | 0.027559 |

Captured Quadrants graphs are omitted by the kernel profiler; wall stages
include their execution. Both profiler output and stage timing are archived.
Intrinsic contact work remains the main actionable bottleneck. This is not
a plateau declaration.

Snapshot replay preserves original points exactly. Maximum nodal difference
is 1.943e-16 N; normalized aggregate force/moment errors are 8.923e-10 and
5.825e-9, respectively. Maximum native replay peak difference is
1.042e-8 Pa, and maximum complete residual is 1.469e-11 N. This replay is
separate from the independently solved full-field qualification of the
[native body source-prior feature](BODY_SOURCE_PRIOR_MODEL.md).

## Wider-seed failures and independent classification

Independent varied B2048/seed623102 and B4096/seed623103 live runs abort during
warmup. They have no valid throughput; their subsequent policy commands are
not reached. Untimed captures preserve the first failed environment before
rigid integration. The B2048 capture fails at call469/tick468, env1914,
status33; B4096 at call490/tick489, env2174, status34. Both contain one actual
source, and the unsupported-input predicates are empty. Original force,
application point, moment, radius, actual friction, charts and scratch are
retained. Different runs may reach the first failure at different ticks;
these are the exact capture ticks.

Both reconstructed clouds have 586 positive-profile samples on six faces.
An independent 64-ray inner Coulomb polygon LP gives B2048 a true-cone
feasible primal: normalized wrench error 3.681e-13 and cone excess
8.327e-17. Its original-point normalized force/moment errors are
3.682e-13 / 2.564e-13. The required sparse LP forces include samples with
normalized weights near 1e-21; minimum cloud weight is 4.466e-26.
Independent trust-exact/root optimization and native cold fitting stagnate
at force/moment errors around 0.00828/0.01817. Native diagnostic budgets
128,512,2048 do not repair it. Production still uses128 and unchanged final
budgets. The independent optimizer exceeds2000 iterations; its failed state
is retained, rather than presented as a completed optimal stress oracle.

For B4096, both inner and circumscribed outer Coulomb polygon LPs are
infeasible. The outer polygon contains the true cone, so this proves
sampled cone infeasibility. It does not prove continuous-footprint physical
incompatibility. Minimum cloud weight is 4.486e-26. Further local integration
and representation repair is required. Neither case is classified as an
unsupported ordinary grasp.

## Reproduction and archive

Use the archived `run_native_body_baseline_v1.sh` / five-scope launcher,
capture launcher and `audit_native_body_failure_v3.py`; the latter rebuilds
the exact captured cloud and uses offline SciPy LP/optimization, without
implementing production recovery. Validator v4 handles the deliberately
absent stress fields in rigid-only output. Earlier audit loading/schema
errors and their closures are retained separately from numerical findings.

[The archive](../evidence/20261011-nonconvex/native-body-baseline-checkpoint-v1/manifest.json)
contains776 entries,134 unique blobs and8545048 stored bytes. Manifest
SHA256: `4b5c7759fc2356ac0a128f832afd30918a734f05d6e6887784d8bf25724eeb10`.
Allocated CPU publication checks original, stored and decompressed hashes.

Only copied B1024 four-scope rates are valid here. Diverse-seed contact
robustness, B2048/larger valid four-scope measurements, grasp outcomes,
tail costs, further optimization and a measured plateau remain open. No
high physical mesh or physical convergence experiment was run.
