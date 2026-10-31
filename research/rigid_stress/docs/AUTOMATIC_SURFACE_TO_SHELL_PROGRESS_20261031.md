# Automatic surface-to-shell progress

The [October 31 goal](AUTOMATIC_SURFACE_TO_SHELL_GOAL_20261031.md) is complete within its recorded geometry,
contact and spatial scope. Read the [measured report](AUTOMATIC_SURFACE_TO_SHELL_RESULTS_20261031.md),
[compact path/hash/key-number manifest](AUTOMATIC_SURFACE_TO_SHELL_MEASUREMENTS_20261031.json) and
[running guide](RUNNING_AUTOMATIC_DPR.md).

Source progression: remote start `7c7cdf6cd`, geometry prototype `694be1506`, finalized import/material display
`6bbd768f5`, initial one-substep measurements `512c1c8ec`, fourteen geometry tests `28a253b6d`, and the measured
custom-shell four-substep public default `01d12d006`. Per-run source/helper hashes remain on data; final research
helpers retain exact 17-digit fixture export, the nonconvex notch negative, memory substep selection and quality audit.

The implementation prepares three conforming quads per triangle, applies scene transforms without small-gap
welding, validates finite-thickness offsets/material topology, derives material inertia, preserves nonconvex
collision, caches source/dependency/thickness identity, and displays both stress sides with gray unsampled caps.
Existing authored egg and volume defaults remain. Custom shell material collision defaults to four physical
substeps, with an explicit override. The 0.01 s outer step and all original stress acceptance gates remain.

Completed checks include fourteen geometry/cache/unit/material/hole/color tests; independent CPU/GPU full fields,
both-side normal/friction/eccentric loads and lifecycle controls for egg, narrow neck, U and plate; generated asset
reload comparisons; membrane/bending patches, six null modes and 12/48/192-quad cantilever refinement. Exact
17-digit grasp/plate controls give <=1.2e-12 auto/reload and <=6.1e-11 independent-CPU tensor error. The additional
notch prescribed-load control passes with <=1.6e-12 / <=5.5e-11 respectively. The independent Q4 quality audit
quantifies corner angles and edge ratios; conversion preserves poor input triangle quality rather than repairing it.

The real public egg, nonconvex neck and open dome complete 140 GPU steps. The authored shell viewer and the new
public converted grasp each complete 730 steps. A separate converted-grasp full-field run checks every physical
substep through 730 steps and passes original gates plus reset. Actual 800×600 viewer captures of automatic,
authored and converted grasp assets were inspected. The early wrong-thread screenshot failure remains recorded.

Two matched twelve-cell timing sets cover automatic/reloaded/authored assets, B4/B256 and max/full: a retained
one-substep ground-support control and the final four-substep grasp. All 24 pass original acceptance and partial
reset. Final grasp B256 complete-scene rates are 9,513 / 8,998 environment-steps/s; auto/reload mechanical inputs
match exactly and full tensors agree within 3.2e-13. A separate B256 full-mode process samples 4,934 MiB peak GPU
memory. Native shared/per-environment payload, viewer/readback cost and source/model differences are reported.

The initial profile precedes bounded k8/k16 tuning. Only about 2% speed benefit accompanies complete-tensor changes
of 2.39e-5 / 7.79e-6, so k32 remains recommended. Identical-source/cache small-egg full builds measure 166.64 s cold /
22.41 s warm. No universal throughput maximum, general mesh-quality guarantee or sharp-edge peak convergence is claimed.

Retained contact negatives include one-substep neck curvature at step 3, broad ellipsoid pressure fit at step 109,
plate pressure fit at step 3, U/dimple rejection, and one-substep converted-grasp inner-wall rejection at step 121.
Four substeps admit the neck and full grasp but still reject the notch's automatic curvature. Prescribed-load
numerical agreement does not override contact admission. A failed pressure fit is not proof of physical infeasibility.

Runtime root: `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261031-automatic-surface-shell/`.
All artifacts, failures and logs remain there. Eight unrelated QIPC files match `meta/start-state.json`, verified
in `meta/preservation-check-final.json`. Git receives only code, documentation and the compact manifest.

All campaign jobs are terminal. Nine frozen worktrees were created under
`/mnt/data/zhaofeng/projects/workspace/Genesis/worktrees/`: geometry-v1/v2/v3, scene-v1/v2, import-v1,
validation-v1/v2 and public-v1 (each prefixed `automatic-shell-`). All were removed after their jobs finished,
with clean-source/local-process checks and `git worktree prune`; see `meta/worktree-cleanup-*.json` through
`worktree-cleanup-final-v5.json`. No campaign frozen worktree remains. Source commits remain in task branch history.
Prior campaigns, their worktrees and unrelated running jobs were preserved; no Atlas/continuum campaign was reopened.
