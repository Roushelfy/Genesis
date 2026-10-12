# Native traction initial guesses and precision rejection

The selected CUDA fitter now starts a newly validated traction problem from
the previous successful workspace dual when it is finite and bounded.
Every current footprint/profile, original wrench and actual local friction
is recomputed and checked. Scratch-slot reassignment may make this guess
unrelated to the current contact; it is still only an initial guess.
No source identity, previous acceptance or unchanged radius/normal/friction
is assumed. Numerical/sampled failures retain the canonical masked cold
retry, followed by the existing disk-refinement and diagnostic routes.

`warm_start_tractions=True` is the native default. Disable it with
`RigidStressOptions(warm_start_tractions=False)` or benchmark
`--cold-tractions`; the cold CUDA specialization reads no old dual.
CPU retains the canonical scalar numerical route. The option can save work
on smooth loads; abruptly changing problems can incur additional cold-retry
work. No persistent numerical buffer is added. This is local fitting reuse,
separate from displacement history and contact-geometry caching.

The internal original-point fit tolerance remains `9e-9`, final wrench
acceptance remains `1e-8`, full residual acceptance remains
`max(1e-11 N,1e-8||rhs||)` including all six gauge rows, and independent
stress acceptance remains `rtol=1e-4,atol=1e-3 Pa`. No load, radius,
coefficient, timestep or numerical acceptance budget changes.

## Numerical and lifecycle acceptance

The generic-guess prototype passes55 CUDA cases in628.46 s. B16/seed623001
and B32/seed623101 then each complete1280 actual steps with every contact
wrench checked, respectively19/37 saved-state resets and nine independent
CPU FP64 full-field captures per run: **432 full fields and56 resets**.
Those comparisons check the entire tensor/von Mises field, global maximum,
displacement and complete equilibrium including the gauge rows. Every-step
normalized force/moment maxima remain below9e-9; total fitting work reaches
57/82 evaluations, including retry work. Maximum independent complete
residual budget fractions are0.005886/0.006710. All source closures match.

The actual native option/dispatch/compiled specialization then passes
**55 CUDA cases in658.19 s**. These include the same independent CPU oracle
fixtures, release/source changes, partial reset, environment isolation,
state restore, shared-pair streaming, refined disks/torques and first-failure
persistence. The canonical CPU route is unchanged from `c560c5e4` and its
55-case CPU epoch; a new CPU suite is not claimed for this CUDA-only change.
The standalone CPU live oracle work above is excluded from throughput.

## Same-GPU current-law performance

The actual native cold/warm comparison runs on one RTX PRO6000 Blackwell
Server Edition GPU, with hardware UUID recorded in its manifest. Driver
595.71.05,97,887 MiB, container `genesis:1_26`, Quadrants1.3.3 and FP64 are
unchanged. B32 uses the full low shell with480 P2 tetrahedra,810 nodes,
2430 DOFs,80 selected contact faces, Q10 squared-coordinate surface fans,
s8 validated intrinsic paths, verified single-face disks and failed-disk
nine-panel refinement. Requested block4 resolves to8. GJK mesh64 collision,
contact patches, pair-union componentwise law, maximum-only output and
dt0.01/one substep are identical.

The varied condition bank SHA256 is
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
Each process uses600 warmup steps, then three1280-step repetitions with all
32 environments, changing contacts/parameters,37 resets per repetition,
the current input checks and all fallback work included.

| Native live scope | Repeated effective env-step/s | Aggregate env-step/s | Aggregate batch step/s |
|---|---|---:|---:|
| Cold cooperative | 2608.197,2638.893,2637.396 | 2628.086 | 82.128 |
| Warm cooperative | 2945.927,2983.449,2962.133 | 2963.757 | 92.617 |

Total counted steps divided by total wall time gives **1.12772x gain**.
Every repeat has zero invalid environment steps, zero overflow and zero
failures in every environment. Rigid final positions and all logged
physical contact/trajectory quantities match exactly. Warmup scalar peak
differences reach0.0001176 Pa and pass the unchanged stress budget; the
largest complete-residual budget fraction is0.0059574. Both variants use
167,466,930 native bytes. This is live rigid-plus-stress with the controller,
without policy; it is not full RL training or large-batch four-scope evidence.

Separate enabled kernel profiling uses128 fixed-snapshot repetitions after
the same600-step warmup. Warm fitting is0.466 ms versus about5.47 ms cold;
the warm intrinsic stage wall is5.506 ms. The replay's converged initial
guess is a special reuse case, so these snapshot gains do not replace the
measured12.8% live gain. Warm snapshot hotspots are profile generation
0.771 ms, source resolution0.727 ms and packing0.651 ms. Validation and
scatter are about0.271 ms each. Dispatch/synchronization remain included in
the wall times, which are intrusive when kernel profiling is enabled.

## Rejected precision candidate and further work

An experimental FP32 Hessian accumulation keeps FP64 tractions, objective,
gradient, direction solve and all final checks. Its complete CUDA epoch
finishes **53 passed,2 failed in624.45 s**. In the legal saved edge fixture
`[147-1]`, the independent CPU-reference-load full residual is
`3.7692e-9 N` against a `1.3722e-10 N` budget. A streamed partial-release
check also changes untouched-environment nodal loads by up to
`3.6856e-11 N`, above its lifecycle consistency budget. Accepted wrench
errors alone therefore do not establish same-mesh load/stress consistency.
This candidate is rejected; FP64 Hessian arithmetic remains selected.
No valid throughput gain is claimed for a numerically failed candidate.
This does not rule out every possible mixed-precision design.

Gradient-only trial evaluation, parallel raw metadata clearing, geometry
reuse, layout/packing/synchronization and the general contact/model work
remain active candidates. The actual B1024 step602 and policy tick876
component incompatibilities still require an explicit modeling resolution;
original component acceptance has not been silently replaced. Current-law
complete policy and1024/2048/larger acceptance, all four scopes and an
end-to-end performance plateau remain unestablished. No physical mesh or
quadrature-convergence study ran.

## Reproducible evidence

The [native warm manifest](../evidence/20261011-nonconvex/native-warm-traction-checkpoint-v1/manifest.json)
contains127 artifacts/4,822,895 bytes; SHA256 is
`0aabb86112ff4d1916432e71147ccfb27fca058e3966563bf24e63599c87cffc`.
The [full-field/precision manifest](../evidence/20261011-nonconvex/traction-guesses-checkpoint-v1/manifest.json)
contains195 artifacts/12,977,368 bytes; SHA256 is
`10dcd9a879e178de35bf7be2ecdcf6a68f67228e7f6b82e3c888246597a453cf`.
All originals, published files and source closures were verified on allocated
CPU compute. Failed precision logs and exact frozen source are retained.
The previous [cooperative checkpoint](NATIVE_COOPERATIVE_TRACTIONS_20261012.md)
records the separately qualified cold native scheduler and its15.322x gain
over the repaired scalar baseline; gains from different physical GPUs are
not multiplied into a claimed new matched end-to-end result.

Reuse the cooperative checkpoint's benchmark command and add
`--cold-tractions` only to the cold variant. The archived native launcher
provides exact source/cache/output paths and runs profiling separately from
live timing. Runtime data and caches stay under the established data root.
