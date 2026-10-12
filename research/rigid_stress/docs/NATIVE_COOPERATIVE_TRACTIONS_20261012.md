# Native cooperative contact recovery checkpoint

This continues the active nonconvex extension at source `a509a6e8` and the
[finite-disk refinement checkpoint](NATIVE_DISK_REFINEMENT_20261012.md).
The native scheduling change preserves the declared intrinsic distance,
Q10 integration, failed-disk nine-panel refinement, original component
wrenches, actual combined friction and final numerical budgets. It is a
performance checkpoint; general live contact-model acceptance and the
1024/2048/larger four-scope milestone remain unfinished.

## Native scheduling and numerical checks

The CUDA route assigns 32 lanes to each contact, with 128 threads per block.
Lanes partition the current support samples and reduce the dual objective,
six gradient entries and 21 Hessian entries. Lane zero solves the small
Newton system, then broadcasts the direction. The existing original-point
termination uses internal `9e-9`, followed by unchanged force/moment `1e-8`
acceptance. The complete elastic residual remains
`max(1e-11 N, 1e-8 ||rhs||)`, including all six gauge rows; independent CPU
field/maximum acceptance remains `rtol=1e-4, atol=1e-3 Pa`.

Numerical or sampled failures receive a device-masked cold retry through
the canonical native fitter. Successful contacts keep their state; total
evaluation counts include fast and retry work. This retains the checked
legal fixture for which an unguarded scheduling prototype failed. Verified
disk refinement and continuous-certificate logic remain in the normal
pipeline. Original-contact cone/wrench validation and signed P2 scatter
also partition samples across lanes. CPU and the explicit
`cooperative_tractions=False` baseline use scalar scheduling.

The complete current suite passes **55 CPU cases in 460.57 s and 55 CUDA
cases in 681.45 s**, including the independent same-mesh FP64 oracle,
nonconvex/gap/cavity/source checks, finite near-boundary disks and torques,
actual contact/friction fixtures, environment separation, partial reset,
state restore and first-failure persistence. Each source epoch has matching
before/after SHA256 records. These counts supersede the prior checkpoint's
54-case suite plus separate nine-case diagnostic run.

## Matched current-law profiles

Both variants use the same physical GPU, RTX PRO 6000 Blackwell Server
Edition, UUID `GPU-5c0d0cca-f892-19d7-d4f3-b68ab73a520b`, 97,887 MiB,
driver 595.71.05, container `genesis:1_26`, Quadrants 1.3.3. The host is an
AMD EPYC 9454. The full low hollow shell has 480 affine-P2 tetrahedra,
810 nodes, 2430 DOFs and 80 selected contact surface faces. These are
performance-development results, not physical mesh convergence.

The workload uses B=32, FP64, maximum-only output, dt=0.01 s, one substep,
GJK mesh64 collision, contact patches, `pair_union`, requested contact block
4/effective block8, s8 validated paths and Q10 squared-coordinate fans with
the selected disk refinement. The frozen varied condition bank has SHA256
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
Profiles use 600 live warmup steps. Both native allocation totals are
167,466,930 bytes; the change adds no persistent numerical workspace.

| Stage wall time, kernel profiling disabled | Scalar | Cooperative |
|---|---:|---:|
| Intrinsic contact pipeline | 198.887 ms | 8.707 ms |
| Association/frame transform | 0.102 ms | 0.102 ms |
| Elastic solve | 0.133 ms | 0.133 ms |
| Complete residual | 0.062 ms | 0.063 ms |
| Global maximum | 0.088 ms | 0.089 ms |

This is **22.84x snapshot-stage speedup**, not an end-to-end throughput
claim. `native-cooperative-profiles-v2` did not enable the Quadrants kernel
profiler; its wall timing is valid, but its log cannot establish kernel
means. A separate enabled, 128-repeat run on the same GPU provides them:

| Kernel mean, profiling enabled | Scalar | Cooperative |
|---|---:|---:|
| Dual fitting | 175.578 ms | 5.472 ms |
| Original-contact validation/commit | 10.349 ms | 0.272 ms |
| Signed P2 scatter | 10.274 ms | 0.272 ms |

The enabled profile's cooperative stage wall is 10.481 ms. Timing overhead
is why this run does not replace the uninstrumented timing. Other measured
cooperative kernels are profile generation 0.775 ms, source resolution
0.730 ms, contact packing 0.656 ms and intrinsic path work 0.133 ms.
Fitting remains the main measured hotspot. Kernel means do not sum to the
complete stage wall; dispatch and synchronization remain included there.

An allocated CPU audit checks identical inputs and all 3840 logged warmup
environment states across the two uninstrumented profiles. Rigid/contact
quantities match exactly. The maximum cross-scheduler peak difference is
`1.0242e-6 Pa`, at most `1.7328e-7` of its unchanged stress error budget.
The largest complete-residual budget fraction is `0.0064423`. Snapshot
original points do not move; the normalized force/moment maxima are
`6.5770e-9` and `2.0091e-9`. This comparison supplements the independent
oracle tests; agreement between two device routes is not itself an oracle.

The matched native live comparison is closed on a separate physical GPU,
UUID `GPU-d74daa1e-6b6e-b4da-5f5f-def9418b9092`, of the same model. Each variant
uses600 warmup steps and three1280-step repetitions, retaining all32
environments and counting controller work, changing radius/friction/contact
parameters,37 saved-state resets per repetition and all cold retries.

| Live scope | Repeated effective env-step/s | Aggregate env-step/s | Aggregate batch step/s |
|---|---|---:|---:|
| Scalar baseline | 170.047, 169.993, 169.989 | 170.010 | 5.313 |
| Native cooperative | 2601.064, 2607.840, 2605.740 | 2604.878 | 81.402 |

The aggregate uses total counted environment steps divided by total wall
time, giving **15.322x** live gain on the repaired current law. All three
repetitions have zero invalid environment steps, zero scatter overflow and
zero failures in every environment; no slow or failed environment is
excluded. Rigid final positions and warmup contact/trajectory quantities
match exactly. Full/max field consistency is covered by the numerical
suite. This is B32 live rigid-plus-stress, without policy; it does not
establish the remaining four-scope or large-batch plateau requirements.

## Experimental candidates and physical diagnosis

Previous workspace duals are being evaluated solely as generic initial
guesses for a newly validated problem. No contact identity or unchanged
geometry is assumed, no previous acceptance is reused, and the native cold
retry remains mandatory. This prototype passes 55 CUDA cases in 628.46 s.
A same-GPU B=32 live comparison records about2634 versus3003 env-step/s over
three complete repetitions, with zero invalid environment steps/overflow
and37 saved-state resets per repetition, giving1.1400x aggregate gain.
Independent live full-field checks now complete B16/seed623001 and
B32/seed623101 for1280 steps each, with432 CPU FP64 fields and56 saved-state
resets. Their every-step original-point force/moment errors remain below
`9e-9`; maximum total fit work is57/82 evaluations. These later closed raw
epochs are not included in this checkpoint's manifest and will receive a
separate follow-up publication. Initial guesses remain unselected native
history. A separate FP32 Hessian candidate retains FP64 tractions, objective,
gradient, solve and final checks; its qualification already exposes a failed
legal surface-fan case. The complete failed epoch is retained and further
timing/adoption depends on resolving its unchanged-budget numerical checks.

The tick876 policy component's strict continuous incompatibility is already
independently established. An exploratory native diagnostic cloud now
generates ray candidates from vertices, edge intersections and64 angular
witnesses per intersecting triangle/disk, including unreachable surface
branches as a conservative superset. It changes diagnostic search only,
not the physical load law or stress integration. The old depth8 continuous
checker remains unresolved; depth12 resolves the new ray on CPU and GPU,
with101 search evaluations,116 visited subtriangles, positive target margin
`1.37068e-3` and conservative polar upper bound `-4.57805e-4`. An independent
CPU boundary/arc upper-bound audit obtains `-4.57805e-4` for the CPU ray.
The other, independently feasible original component receives no false
continuous-incompatibility classification. The cloud is a diagnostic
prototype; production storage/scheduling integration is not selected yet.

An independent SOCP on the same1200 audited support samples also finds the
explicitly different aggregate-pair wrench model feasible in10 iterations,
with normalized force/moment errors `5.67e-16`/`4.48e-16` and strict local
cone slack. This is feasibility-only evidence. It does not establish the
weighted minimizer, native stress recovery or policy throughput and does
not authorize silently replacing original component acceptance.

No high-resolution physical mesh or quadrature-convergence run is included.
Continue general legal boundary-crossing integration/model qualification,
first-failure semantics, independent numerical/lifecycle checks, measured
fitting/layout/packing/history candidates and complete larger-batch
four-scope trajectories. A throughput plateau and goal completion are not
established by this checkpoint.

## Reproduction and publication

The [publication manifest](../evidence/20261011-nonconvex/native-cooperative-checkpoint-v1/manifest.json)
contains624 copied artifacts totaling17,854,454 bytes, with every original
and published SHA256 verified on allocated CPU compute. Its SHA256 is
`d55de54a58a3b395deda119c59d9e564ce3fb3f20e82683bfbdce5e542629cdc`.
It includes frozen source trees, condition bank, CPU/CUDA tests, enabled
kernel logs, uninstrumented profiles, live repeats, numerical validation,
experimental seed qualification/timing, continuous diagnostic prototypes,
the independent CPU ray audit and aggregate-only SOCP feasibility. Failed
construction/depth8 epochs remain labeled; no pending job is archived.
Runtime caches and outputs stay under the established data root.

Run `research/rigid_stress/scripts/check_nonconvex_contact.sh cpu` or `cuda`
inside an allocated compute job for the55-case native suite. The benchmark
uses `examples/speed_benchmark/rigid_stress.py`; compare the following current
law commands in separate processes on one allocated GPU, adding
`--serial-tractions` only to the scalar variant:

```bash
source research/rigid_stress/scripts/cluster_env.sh
python examples/speed_benchmark/rigid_stress.py --scope live --envs 32 \
  --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 10 \
  --seed 623101 --varied --contact-model intrinsic \
  --intrinsic-support pair_union --contact-block-size 4 \
  --egg-geometry mesh64 --collision-method gjk --contact-patch on \
  --conditions /path/to/frozen/conditions.npz --compact-log \
  --source-revision "$(git rev-parse HEAD)" --output /data/path/result.json
```

The archived launcher provides the exact data-side condition path, explicit
cache/output locations and source closure. For wall profiling, choose
`--scope profile`; for kernel subdivision additionally set
`QD_KERNEL_PROFILER=1`, keeping that intrusive run separate from throughput.
The initial profiler-construction failure was a missing example argument;
its complete failed epoch precedes the corrected constructor and successful
profiles and is not counted as a numerical pass.
