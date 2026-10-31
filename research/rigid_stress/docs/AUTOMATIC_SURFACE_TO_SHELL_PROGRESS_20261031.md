# Automatic surface-to-shell progress

The [October 31 goal](AUTOMATIC_SURFACE_TO_SHELL_GOAL_20261031.md) is active and incomplete. The starting remote
revision is `7c7cdf6cd`; prototype geometry integration is `694be1506`. Original unrelated QIPC changes are preserved
in place; their starting hashes are recorded in the data-side `meta/start-state.json`.

The implementation splits each triangle into three conforming Q4s without moving the midsurface. It joins only exactly
coincident coordinates, checks material connectivity/manifoldness/intersections, normalizes directors consistently
with the native solver, constructs both thickness sides and open-edge caps, and rejects collapsed/inverted offsets.
Thickness participates in the cache key. Custom shell collision uses the material boundary without convexification
or watertight wrapping. Complete qualification remains in progress.

At `ac30844ea`, eleven focused geometry/cache/units/material/color checks pass. The first run caught OBJ coordinate
rounding (3.33e-9 m); export now retains 17 decimal places. All thirteen controls pass at `512c1c8ec`, including source-buffer, 1e-10 m close-gap, transformed-scene and
crossing-ribbon checks. A further annulus/hole control is pending. Open-edge caps are displayed in gray because the shell field
does not provide a cap-normal three-dimensional contact stress.

Four generated families passed independent CPU and RTX PRO 6000 Blackwell native operator, complete tensor/resultant,
peak and contact controls: 20-triangle egg (60 quads / 372 DOF), 80-triangle narrow neck (240 / 1452), open U (24 / 210)
and plate (12 / 114). Both-side normal/friction/eccentric contacts, slot permutation, release/recontact, unloaded
environment isolation, original force/moment/residual gates and spinning rigid-mode relief were checked. GPU
same-load full-tensor relative errors range from 9.2e-12 to 5.1e-10. These are discrete-model checks, not continuum accuracy.

Independent flat controls have six rigid null modes, membrane energy errors below 3.2e-15 and bending energy errors
below 8.6e-11. At 12/48/192 quads, unit-load cantilever compliance errors against the stated nu=0 beam/shear reference
are 0.8307% / 0.1978% / 0.04786%. This bounded test does not certify curved-shell or sharp-edge continuum peaks.

The public entrypoint completed 140 GPU steps on the small automatic egg (3,414 Pa terminal peak). That shape is too
narrow for the established finger closure, so this is a ground-support control, not a successful grasp. A dedicated
`grasp` recipe uses the established exterior as an explicitly declared midsurface. Its finite-thickness collision
boundary triggers unsupported Hertz curvature during the actual trajectory; k8/k16/k32 and viewer attempts all stop
at the original gate. These are retained negatives, not timing results. Contact diagnosis and an additional coarse
ellipsoid control are in progress.
The ground-support B4 max profile passes original acceptance and partial reset. Preparation took 46 ms cold / 4.7 ms
warm; the 173 s cold scene build includes JIT. Module/scene costs are about 99.85 / 101.82 ms per step, with 16,759,797
native buffer bytes. Pressure fit and validation dominate its GPU trace (about 89% and 9%). This is preliminary profiling,
not a throughput recommendation. Bounded k8/k16/k32 and matching automatic/explicit/authored comparisons are next.

Data receipts: `controls-cpu-v1`, `controls-gpu-v2`, `geometry-tests-v3.xml`, `entrypoint-v2`, `scene-profile-v1`
under the runtime root below. A wrapper diagnosis found that non-array sbatch calls can create a nested default-partition
job; affected smoke attempts are retained, and subsequent launches use arrays, including one-task arrays, in `rtx-mid`.

Explicit generated-asset reloads pass on both CPU and GPU for the 240-quad grasp midsurface and 12-quad plate.
Relative complete-tensor differences are <=1.4e-12; independent CPU full-tensor errors are <=1.5e-10.

Pending work: final geometry regressions; actual grasp and
viewer color demonstration; final cold/warm build profiles; small/large GPU module and complete-scene max/full timings,
memory and comparisons with explicit generated and authored egg assets; measured tuning; final usage/limits/report.

Runtime root: `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261031-automatic-surface-shell/`.
No experiment outputs are committed. All frozen worktrees use `/mnt/data/zhaofeng/projects/workspace/Genesis/worktrees/`.
Completed `automatic-shell-geometry-v1` and `v2` were removed; Git left filesystem residues, subsequently hash-verified
and removed, with receipts in `meta/worktree-cleanup-*-v1.json`. Session October 31 retained only
`automatic-shell-validation-v1` (`512c1c8ec`) after completed geometry-v3, scene-v1, import-v1 and scene-v2 were removed.
Cleanup receipt: `meta/worktree-cleanup-final-v2.json`. The remaining source is used by active diagnostic/public-entrypoint
jobs. Planned removal: once jobs and audits finish, by November 1.
