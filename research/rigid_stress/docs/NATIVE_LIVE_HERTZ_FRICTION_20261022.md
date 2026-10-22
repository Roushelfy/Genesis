# Native changing-force friction and physical memory

The experimental native route now advances radial vector friction memory from
actual solved rigid contacts and combines its tangential stress with normal
Hertz stress and the regular finite-domain elastic response. The frozen
uncompressed implementation is
[`795805290`](https://github.com/Roushelfy/Genesis/commit/7958052904398bfa31842ea6a2c40ecec1220de1).
The [independent friction study](TANGENTIAL_CONTACT_REFERENCE_20261022.md)
provides conditional physical evidence for this radial approximation; this
integration does not enlarge that study's physical scope. The complete
[hard-contact goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md) remains active.

## Actual solver behavior

The production path stays inside the existing rigid solver and Quadrants.
The normal law predicts the current ellipse. Sixteen radial intervals retain
vector history under changing normal/tangential force. Collision geom/pair,
material center/frame and an actual adjacent-face walk identify continuity;
a separate staging pass protects prior memory from collider-slot permutations.
Ambiguous or multiply consumed correspondence fails explicitly. Frame changes
transport the previous tangent by the minimal normal rotation.

The equivalent center lies on the force line's material-boundary intersection.
This preserves the original resultant and representable moment rather than
removing tangential load. The existing force/moment gates remain
`1e-8 |F|` and `1e-8 Fn a`; free couples remain unsupported. The history keeps
its `1e-9` normalized force gate and continuous cone bound `1+1e-10`.
The finite-domain integrated wrench gate stays `1e-4`.

Physical history advances on every physical substep, including unobserved
manual steps. Manual fields, peaks and the probe activity mask retain the last
requested observation. Selected geometry/velocity mutations clear only the
selected physical histories and observations. A checkpoint carrying physical
STATE restores the line, physical scales, correspondence and loss flags while
invalidating derived output. Lost active history stays lost until release or
an explicit reset. Release removes stale loads and memory.

The radial transport of a moving footprint, elliptical partial slip and remote
point-traction correction remain approximations. The present recovery retains
its locally planar patch, global half-space and four-major-radius noncoplanar
clearance checks. Legitimate unsupported curved/inner contacts are refusals;
they are not admitted Franka stress updates.

## Executed checks

All numerical execution uses Slurm `rtx-mid`, image `1_26`, the existing
data-side physical environment and private node-local compiler caches.

- CPU/GPU core controls cover 48 varying-normal/vector increments, frame
  covariance, slot permutation, selected reset and failure/release recovery.
  The maximum native/CPU line and traction differences are below the unchanged
  `1e-5` and `1e-4` gates. The tangential primitive retains 2,923 CPU/GPU tensor
  controls, including surface, edge, axis and near/far probes.
- An independent complex-step Cerruti control has maximum relative difference
  `3.31e-14`. The original 216 combined tensor cases differ from the independent
  supplied-traction implementation by at most `2.98e-13` of peak pressure.
- Actual force-driven sphere/flat-link scenes cover 18 steps, two physical
  substeps, four environments, changing vector/normal commands, full/max,
  every-step/manual output and a pure-rigid companion. Maximum observed
  `|Ft|/Fn` is about 0.0816. CPU/GPU and rotated scenes pass held output,
  physical-history equivalence, checkpoint continuation and selected reset.
- Material binding separately checks two moving regions, rotating tangent
  frames, arbitrary slot permutations and four environments against the CPU
  radial law. Maximum line difference is `1.45e-7`. Duplicate correspondence
  triggers an environment-local loss latch; repeated active input cannot
  clear it, and release does clear it.
- Three complete CPU scene snapshots, four environments each, agree with
  independent P2 derivatives/consistent-mass relief plus declared particulars:
  maximum probe/corner relative differences are `2.51e-9` / `4.53e-13`.
  This uses the captured regular RHS and the same prescribed traction model;
  it does not independently validate pressure, moving-footprint transport or
  the remote-boundary point approximation.
- Two existing public DPR lifecycle tests pass on each of CPU and CUDA,
  covering manual/full/B4 and every-step/max/B1, checkpoint restore and
  selected resets. The new physical-state restore notification preserves
  the tested original recovery behavior.

The first binding-control worker failed before execution because a vector
kernel argument had an unresolved dtype in the pinned Quadrants version.
A typed tensor argument repaired the worker; original logs/source remain.

## Inclusive scene cost and profiling

These are warmed force-driven **sphere/flat-link** scenes, float64, two physical
substeps per step, default surface order8 and all 89 moving probes. Primary
wall time includes command generation, native association/provenance, geometry,
laws/history, transfer, elastic solve, probes, reductions and required acceptance.
Research field exports and intrusive profiling are outside primary timing.
Every audited max/full environment-step was admitted and loaded. Raw timing
repetitions and matched pure-rigid trajectories remain in the receipts.

| Batch | Rigid ms/step | Max ms/step | Full ms/step | Full env-step/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.787 | 16.351 | 16.339 | 61.2 |
| 64 | 1.947 | 15.686 | 15.793 | 4,052 |
| 256 | 1.941 | 24.509 | 24.895 | 10,283 |
| 1024 | 2.065 | 48.126 | 48.141 | 21,271 |

These rows use source795805290. Separate CUDA traces of the original producer
show that combined-field evaluation dominates B1/B64 device work, followed by
remote traction. Friction iteration and material binding are smaller. Trace
sums overlap with host/dispatch timing and are not a scene wall-clock total.

## Far-field compression experiment

A separate frozen source
[`6ffdd399b`](https://github.com/Roushelfy/Genesis/commit/6ffdd399b8c49c7bcb27af570b5867f6cba541d1)
uses four radial interpolation rings instead of sixteen
beyond eight major radii, preserving the first four even radial moments and
zero-resultant residual fields. The first GPU control exposed a preexisting
switch-roundoff issue: uncompressed and compressed sources both differ by
`1.1517e-4` relatively at an exact eight-radius query, including a normal-only
case. CPU selected the opposite approximation branch. The original failure
and cancelled pending speed run are retained.

The repaired switch uses a squared-distance band of64 machine epsilons.
The same declared switch is used by the independent supplied-load reader.
CPU/GPU controls now include408 combined tensors and exact switch queries;
maximum difference from the uncompressed bank is `2.82e-11`, below the
predeclared `1e-6` relative gate. No stress tolerance was loosened. Both sources
also agree over18 complete force-driven frames and four environments on CPU
and GPU: maximum relative probe/corner differences are `3.58e-15` / `1.09e-20`,
with identical history and von-Mises peaks. Independent P2 inspection of the
compressed fields gives maximum probe/corner differences `2.73e-9` / `4.63e-13`.
These checks use float64; float32 switch behavior remains unmeasured.

The same-allocation paired run executes baseline first and compressed second,
each in a separate process with private compiler caches. Values below are
medians of three warmed eight-step repetitions. Both variants admit all24
audited loaded steps per environment for max and full output.

| Batch | Mode | Baseline ms/step | Compressed ms/step | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Rigid | 1.779 | 1.771 | 0.5% |
| 1 | Max | 16.078 | 15.364 | 4.4% |
| 1 | Full | 16.355 | 16.035 | 2.0% |
| 64 | Rigid | 1.939 | 1.930 | 0.5% |
| 64 | Max | 15.493 | 15.433 | 0.4% |
| 64 | Full | 15.718 | 15.501 | 1.4% |

Several repeat ranges overlap, and the small reductions do not establish a
general performance advantage. Retain the compressed source as a numerically
checked experiment. Use795805290 as the measured uncompressed reference for
the B256/B1024 scaling above. Neither optimization is a solution to the
remaining curved-shell applicability gap.

## Reproduction and remaining work

Use the captured module/argument lists in the
[companion evidence packet](../evidence/20261019-qualified-ranking/hard-native-friction-live-v1/README.md) with
`research/rigid_stress/shell_comparison/submit.py`. Select the stated immutable
source, an explicit fresh output under the established data root, the existing
physical Python environment and `--node-local-cache`. Relevant entrypoints are
`check_native_friction_history`, `check_native_friction_binding`,
`check_native_friction_stress`, `check_live_hertz_friction`,
`inspect_live_hertz_friction` and `time_live_hertz_normal --force-driven --friction 0.4`.
All are under `research.rigid_stress.qipc_comparison` and run on compute nodes.

The original 12 unfinished files and earlier source baselines remain unchanged.
Full fields, histories, mesh arrays and CUDA traces stay on data; compact
executed workers, patches, receipts and failures are publication evidence.
Curved/thin-shell recovery, nonconvex moving-contact physical error, accepted
Franka throughput and the complete practical estimator remain required work.
