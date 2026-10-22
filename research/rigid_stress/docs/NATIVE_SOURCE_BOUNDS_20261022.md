# Shared bounds for witnessed source resolution

The native source resolver now reuses existing immutable triangle bounds before
calculating exact closest points. It retains canonical face order, source and
component identity, uncertainty/ambiguity checks, original force/moment and
all acceptance thresholds. There is no additional cache or runtime allocation.
Use experimental source
[`5a28cf837`](https://github.com/Roushelfy/Genesis/commit/5a28cf83734756020776ff885a39233ccd567913).
Its unmodified baseline is the preceding
[native normal route](NATIVE_LIVE_HERTZ_NORMAL_20261022.md), source `fc63cf32e`.

## Geometry and numerical conformance

Distance to a triangle's enclosing box is a lower bound for distance to the
triangle. The nearest-face pass omits faces whose lower bound exceeds the
current candidate distance plus the existing transform-roundoff allowance.
The ambiguity pass omits faces beyond the original candidate distance,
twice the declared source uncertainty and that same allowance. Every
remaining face uses the original exact projection and topology rules.

Both passes retain ascending face order and the existing shared-feature tie
rule. This avoids changing a material anchor merely because several incident
faces have nearly identical distance. A nearby disconnected sheet remains a
distinct source candidate. The existing shared bounds are prepared once by
`StressSurface`; reset/history semantics and buffer capacity are unchanged.

Seven existing source tests pass on each of CPU/CUDA. They cover shared-edge
roundoff, same-surface uncertainty, actual rigid witness/contact transitions,
nonconvex ambiguity and offset wrenches. Eight rotated actual contact steps
on four environments per backend retain all saved source/probe/corner fields,
identities, statuses and force/moment checks against the frozen baseline.
The complete saved-array inspector passes its original numerical scales.

## Actual scene measurements

Separate child processes compare the original and bounded sources on one GPU
allocation, each with private compiler caches. Each mode includes input motion,
two physical substeps, native acceptance, recovery and sampled reductions.
Every 24-step audit admits all environments; all max/full peaks and pure-rigid
trajectories pass their unchanged comparisons. Readback and research auditing
remain outside primary timing.

The first same-allocation run uses three eight-step warmed blocks per mode:

| Batch | Baseline max/full (ms) | Bounded max/full (ms) | Observed max/full reduction |
| ---: | ---: | ---: | ---: |
| 1 | 7.203 / 7.332 | 7.817 / 7.235 | -8.52% / 1.33% |
| 64 | 7.310 / 7.374 | 7.088 / 7.143 | 3.03% / 3.13% |
| 256 | 8.546 / 8.323 | 7.770 / 7.875 | 9.08% / 5.38% |
| 1024 | 9.220 / 9.274 | 8.741 / 8.859 | 5.20% / 4.47% |

The first B1 max result is slower and its repeat ranges separate. B1 full and
B64 max repeat ranges overlap. Larger-batch reductions have separate repeat
ranges. These observations do not establish confidence intervals or a general
speedup. The measured B1 regression motivates a reversed-order comparison,
with B4096 added to inspect larger useful batches; both attempts remain in
the terminal evidence.

The second allocation reverses source order and checks the same full audits:

| Batch | Baseline max/full (ms) | Bounded max/full (ms) | Observed max/full reduction |
| ---: | ---: | ---: | ---: |
| 1 | 7.562 / 7.729 | 7.049 / 7.082 | 6.78% / 8.37% |
| 1024 | 9.351 / 9.465 | 8.834 / 9.007 | 5.53% / 4.84% |
| 4096 | 19.054 / 19.457 | 18.358 / 18.725 | 3.66% / 3.76% |

B1's direction changes with the source order/allocation, so its speedup is
unresolved. B1024 improves in both orders; B4096 gives approximately
223,100/218,700 max/full environment scene-steps/s. Every B4096 audit admits
98,304/98,304 loaded observations per mode. Its pure-rigid medians are
3.809/3.729 ms baseline/bounded, making the observed added max/full costs
15.245/15.648 versus14.629/14.996 ms. Both the rigid baseline variation and
inclusive scene results are retained; device-task savings are not substituted
for scene gain. The bounded resolver is useful for the measured larger batches,
while arbitrary Franka contact layouts still need matched measurements.

Independent intrusive CUDA traces show the source-resolution device work falls
by approximately60% at B1/B1024. Other work, CPU dispatch and argument/input
copies limit the complete-scene gain. Those separately allocated traces are
diagnostic attribution, while the table is paired inclusive wall time.
Exposed stress-field storage is unchanged at 3.32/47.32 MB for B1/B1024,
including shared geometry/operators. Compiler and allocator overhead are
outside these logical counts.

## Reproduction and remaining work

The [terminal packet](../evidence/20261019-qualified-ranking/hard-native-source-bounds-and-friction-reference-v1/README.md)
retains actual scheduler states, source captures, exact historical replay,
regression sources, full-array hashes, the original sparse-checkout
prelaunch refusals and both measured source orders. Scientific work uses
Slurm `rtx-mid`, image `1_26`, the existing physical environment, explicit data
outputs and private node scratch. The resolver is inside Genesis's existing
rigid solver and all numerical work remains Quadrants.

Use captured `check_source_bounds`, `time_live_hertz_normal --reference-source
<frozen-source>` and `inspect_source_bounds` task lists. Optional
`--current-first` reverses source order; `--batches` selects the measured
batch list. Captured normal controls remain required for conformance.

The [independent vector-friction control](TANGENTIAL_CONTACT_REFERENCE_20261022.md)
runs alongside this performance work. Native tangential/history integration,
surface/global field qualification, useful curved/inner/hollow-egg regimes
and accepted evolving Franka throughput remain part of the complete goal.
