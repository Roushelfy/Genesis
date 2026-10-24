# Native shell DPR: integration and tuning in progress

The optional native Q4 shell now runs through the existing rigid stress API.
This is an interim engineering record. The complete throughput matrix,
targeted physical checks and final configuration recommendations remain open.
The [active goal](SHELL_DPR_INTEGRATION_GOAL_20261024.md) remains in force.

The gripper retains the user's requested sharp box geometry. Automatic radius
uses the explicitly selected planar edge surrogate, whose distinct status and
physical limitations remain in the options documentation. Contact radii,
source application points and acceptance tolerances are unchanged.

## Implemented and checked

- Native curved Q4 shell preparation, six DOFs per node, membrane/bending/shear,
  rotary inertia, six-mode relief and centrifugal loading.
- Physical bilinear inner/outer shell boundaries with consistent eccentric
  force/rotation transfer through the existing support-ball pressure law.
- Selectable volume/shell backend, sampled in-plane max/full stress and
  separate membrane, bending and transverse resultants.
- Public lifecycle controls covering cadence, environment isolation, release,
  slot changes, partial resets and checkpoints.
- Independent CPU/native stiffness, rigid modes, load transfer and complete
  field controls, including warped and concave fixtures and rejected friction.

Frozen source `ccd2967cc` passed the full 600-frame four-environment Franka
cycle and a separate partial-reset control. All frames were retained. Its
seven captured fields agree with the independent CPU shell solve to a maximum
relative field difference of 1.50e-10. The independent complete residual is
at most 2.43e-13. Membrane, bending and transverse resultant differences are
at most 3.35e-12, 2.47e-10 and 4.89e-11 respectively. This checks implementation
consistency for the same model and captured loads.

The original six focused volume pressure tests passed after shell integration.
Shared pressure scheduling changes receive another volume regression run.

## Measured tuning evidence

RTX PRO 6000 Blackwell Server Edition, FP64, coarse egg, 64 environments,
full output, fixed condition bank, Hertz radius/profile and planar box-edge
surrogate. These are intrusive stage measurements at the same loaded frame,
not complete robot throughput and not additive speedup estimates.

| Change | Before | After | Measurement |
|---|---:|---:|---|
| Analytic projected Q4 inverse | 13.53 ms | 3.81 ms | Anchor stage |
| Parallel result invalidation | 14.15 ms | 0.051 ms | Acceptance stage |
| Conservative force-line bounds | 3.93 ms | 1.09 ms | Anchor stage |
| Explicit Q4 integration-moment reuse | 12.63 ms | 7.49 ms | Pressure integration/fit, separate measurements |
| Fixed bilinear coefficient preparation | 9.09 ms | 8.44 ms | Combined anchor/pressure/scatter |
| Constrained-fit geometry reuse | 6.85 ms | 4.11 ms | Pressure integration/fit |
| Four-warp Q4 face scheduling | 4.11 ms | 1.92 ms | Pressure integration/fit |

The Q4 inverse change preserves bounded CPU/GPU full fields to approximately
2e-15 relative difference. Force-line bounds, compact nodal scatter and fixed
coefficients also pass the independent load controls. Compact nodal scatter
reduces the measured scatter stage from 1.33 to 1.19 ms at 1024 environments;
fixed coefficients reduce it further to 1.02 ms.

The secondary profile disables CUDA graphs and exposes individual kernels.
It identifies constrained pressure correction as the largest kernel, about
6.75 ms at 64 environments. Graph-enabled profiler counters omit captured
work, so their zero device times are not used to estimate launch overhead.
Face/radial geometry reuse and four-warp face scheduling pass independent
contacts, the 600-step scene and volume regression. At 1024 environments,
pressure integration/fit decreases from 15.80 to 6.44 ms across these changes.
Shell recovery graph fusion reduces matched module medians from 6.656 to
6.426 ms at B4, 7.075 to 6.883 ms at B64, and 15.135 to 14.965 ms at B1024.
These pairs predate the final four-warp pressure change.

Diagonal equilibration and natural/minimum-degree ordering pass independent
spinning controls (complete residual at most 1.37e-12). Equilibration produces
no measured runtime benefit in the B64 response path. Natural ordering grows
the coarse egg factor from 12,466 to 63,606 block entries and its preparation
from about 1.47 to 17.70 seconds, with no response-application speed gain.
Defaults retain the unscaled minimum-degree factor.

An independent two-side analytic disk-moment fixture covers a support ball
intersecting both physical shell faces. Hertz and original DPR laws pass on
CPU/GPU, including moment reuse: generalized-load errors are below 1.8e-15
and complete-field errors below 7e-12. This supplements the small-radius
one-side warped, concave and inner/outer controls.

FP32 inverse storage with FP64 arithmetic passes the focused spinning controls
with one or two residual corrections and no factor fallback. Complete fields
remain within 7e-12 of the independent reference. Full FP32 arithmetic fails
the strict plate control: complete residual 5.32e-4, three loaded environments
rejected after two corrections and factor fallback. The rejected fields and
the initial nonfinite-JSON diagnostic failure are preserved. Neither result
establishes a final performance recommendation.

## Evidence and remaining work

Runtime evidence is under the existing data directory:
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261024-native-shell-dpr/`.
Each campaign contains source inventories, task arguments, Slurm receipts,
logs, return codes and complete artifacts. `meta/campaign-status-v4.json`
records a checkpoint of the campaign inventory. Frozen engine checkouts stay
under the source worktree directory and are never edited after submission.

Primary evidence includes `operators-v5`, `operators-spinning-v1`,
`contacts-v1` through `contacts-v6`, `shell-regression-v1` through
`shell-regression-v3`, `volume-regression-v1`, `scene-v2`, `scene-v3`,
`scene-audit-v1`, `scene-audit-v2`, `profile-v1` through `profile-v6`,
`profile-device-v1`, `geometry-audit-v1`, and `precision-v1`/`precision-v2`.
Earlier failed builds and tests remain available beside their successful
replays. Ongoing campaigns are not represented as completed evidence here.

Next: finish pressure/fusion tuning, compare factor/response/gauge/precision
choices and useful batch sizes, measure matched complete 600-step rigid,
volume and shell workflows in max/full modes, and record setup, memory,
launch/copy/synchronization costs. Then run targeted global membrane/bending
physical checks and publish final recommendations and reproduction commands.

All shell peaks remain maxima over the authored in-plane sampling locations.
The roughly 45-micrometre footprint in a 0.5-mm wall is unresolved, and these
results do not certify a physical global three-dimensional contact peak.
