# Rigid-stress data and artifact locations

The active automatic DPR fast-path/defaults phase from `d661349b1` uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261028-automatic-dpr-fast-path/`.
Exact rejected-contact replay, authored controls, full-field/trajectory qualification,
job logs, profiles, timing and memory outputs remain on data. Source-only frozen
worktrees stay in workspace. Reuse the existing data-side physical environment,
Slurm `rtx-mid` / image `1_26` and private node-local caches. Previous run directories,
twelve original unfinished files and all prior frozen sources remain in place.
The 2026-10-28 goal changes pressure/default semantics only after its recorded
qualification. No data or active-environment migration is performed.

The completed planar-loading DPR phase from `f8fad8ba6` uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261027-planar-loading-dpr/`.
Frozen CPU-oracle exports, native operator/load/field controls, trajectory
arrays, profiles, job logs and source receipts stay there. Small source and
project reports remain in the existing workspace; frozen engine worktrees
remain immutable. Scientific execution reuses Slurm `rtx-mid`, image `1_26`,
the existing data-side physical environment and private node-local caches.
The start inventory preserves twelve original unfinished files and thirty-three
prior unique frozen engines. No data, environment or compatibility link moved. The publication packet is
`research/rigid_stress/evidence/20261027-planar-loading-v1` (at most10MiB);
all large runtime records remain on data. Its independent verification receipts
are siblings of the packet. The31 new frozen sources also remain preserved.

The local-refinement atlas follow-up from revision `4bc4ada25` uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-local-refinement-atlas/`.
New control assets, operators/fields, profiles, job logs and source receipts
stay there. Existing source checkouts, compact reports, frozen experiments
and data-side environments are reused. Numerical execution uses Slurm
`rtx-mid`, image `1_26`, with explicit output and private cache paths.
No data or environment migration is performed.

The native candidate integration goal from revision `b8e63b195` uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261025-native-candidate-integration/`.
All generated numerical arrays, profiles, execution logs and source receipts
stay there. Existing environments, prior campaigns and frozen sources are
reused without migration. Scientific execution stays on Slurm `rtx-mid`,
image `1_26`, with private node caches and explicit output paths. The start
receipt preserves the twelve original unfinished files and existing worktrees.

The 2026-10-25 GPU throughput follow-up uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261025-gpu-throughput/`.
Fresh baseline timings, profiles, candidate results, source inventories and
Slurm logs stay there. Frozen source variants remain in workspace. The prior
native-shell campaign, environments and unrelated unfinished files are preserved.
The compact publication packet is `research/rigid_stress/evidence/20261025-gpu-throughput-v1`;
its artifact manifest references raw files by data path, size and SHA256.
Only small reports, receipts, summaries and condition fixtures are tracked.
No data migration or environment relocation was performed.

Native thin-shell DPR integration uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261024-native-shell-dpr/`.
Operator/control arrays, asset authoring outputs, Slurm logs, source inventories
and failed attempts remain there. Runs use Slurm `rtx-mid`, image `1_26`, the
existing data-side physical environment and private node caches. Small matched
shell assets and their geometry metadata are tracked with the examples; their
authoring receipts and complete results stay on data. Frozen source checkouts
remain in the existing workspace. No data or environment migration occurred.
The small tracked `research/rigid_stress/evidence/20261024-native-shell-dpr-v1`
packet contains final measurements, controls, requirement/source audits and a
condition fixture. Full arrays and traces stay on data, referenced by hashes.
The pinned optional pyfe3d reference is under the same run root at
`dependencies-v1/group0/pyfe3d/site`; it does not modify the active environment.
See the [native shell report](research/rigid_stress/docs/SHELL_DPR_NATIVE_20261024.md).

The throughput phase uses 31 `automatic-dpr-opt-*` campaigns under the same
20261019 ranking data root. All runs use explicit data outputs, the existing
data-side environment, Slurm `rtx-mid`/image `1_26` and private node caches;
no new home-data fallback or migration occurred. The compact
`research/rigid_stress/evidence/20261023-automatic-dpr-throughput-v1` packet
contains timings, controls, task receipts and two small condition fixtures.
Full fields, logs and source inventories stay on data. Metadata include
`meta/automatic-dpr-opt-launch-v8.json`, the final timing validation and
`meta/automatic-dpr-opt-preservation-v1.json`. All 76 frozen engines and 12
original unfinished files are preserved. See the
[throughput report](research/rigid_stress/docs/DPR_THROUGHPUT_20261023.md).

Automatic-radius DPR development uses `automatic-dpr-*` campaigns under the
established 20261019 ranking data root below. Generated control meshes, scene
arrays, compiler logs and submission/source receipts remain on data. Active
source/docs and frozen test checkouts stay in workspace. These runs reuse the
existing data-side environment, Slurm rtx-mid/image 1_26 and private node caches.
The 2026-10-23 goal correction is recorded in `meta/automatic-dpr-goal-transition-v1.json`;
previous analytic-field jobs retain their captured sources as historical research.
The small tracked `20261023-automatic-dpr-v1` publication packet contains compact
results, repeat timings, task/source receipts and the unchanged 3,754-byte
condition fixture. Full runtime arrays, logs, generated assets and complete
benchmark buffer reports stay under data. Existing environments, compatibility
links, frozen worktrees and unrelated unfinished files are preserved.

The exact-column and cell-aligned volume studies use edge-shell-exact-columns-*,
edge-shell-volume-window-* and edge-shell-integrals-inspection-* under the
established20261019 ranking data root. The sharp-edge-shell-integrals-v1 packet
retains426 compact payloads (2,906,225 bytes) and33 external receipts from ten
terminal campaigns, including failed column searches, a rejected radial rule
and two cancelled pending inspectors. Complete resultants and window tensors
stay on data; original pressure/displacement/field arrays are reused in place.

The same-GPU B1024/B2048 timing and separate fresh-process full memory run use
hard-batch-scaling2048-v1 and hard-batch-scaling2048-inspection-v1 at that data
root. The hard-batch2048-v1 packet retains93 compact payloads (636,120 bytes)
and11 external receipts. All campaigns are terminal. Scientific execution
used Slurm rtx-mid, image1_26, the existing data-side environment and private
node caches. All55 engine heads and12 original unfinished files remain
preserved. Source/docs and compact evidence stay in workspace; there is no
new home fallback, data migration, environment migration or engine change.

The complete-domain sharp-edge follow-up uses edge-shell-rectangle-controls-*,
edge-shell-sampling-v1, edge-shell-fields-*, edge-shell-locations-v1,
edge-shell-peaks-v1 and edge-shell-integral-diagnosis-v1 under the established
20261019 ranking data root. The sharp-edge-shell-fields-v1 packet retains457
compact payloads (3,309,232 bytes) and25 external receipts from ten terminal
campaigns, including two failed readouts and a cancelled pending inspector.
Full volume/column/corner tensors and sampling arrays stay on data. Repaired
readouts reuse the original completed particular arrays in place. CPU work
used Slurm rtx-mid, image1_26, the existing data-side environment and private
node caches. All55 engine heads and12 original unfinished files remain
preserved. There is no new home fallback, data migration or engine change.

The single-edge hollow-shell control uses edge-shell-pressure-pilot-v1,
edge-shell-pressure-refinement-v1 and edge-shell-pressure-inspection-v1
under the established20261019 ranking data root. The
sharp-edge-shell-pressure-v1 packet retains156 compact payloads
(1,134,755 bytes) and23 external receipts from three terminal campaigns.
Complete cell pressures, gaps and positive-depth fields stay on data;
existing meshes and normal-source response tables are reused in place.
CPU scientific work used Slurm rtx-mid, image1_26, the existing data-side
physical environment and private node caches. All55 engine heads and12
original unfinished files remain preserved. No engine changes, new home
fallback, data migration or runtime coupled solve were introduced.

The `hard-traction-vectors-v1` packet retains 290 compact payloads (1,921,637 bytes) and
352 external receipts from four terminal campaigns. Initial B1024 profiling, paired
normal/sharp recovery, B4/B1024 profiling/timing, full-field captures and frame428
replay remain in the data root below under `hard-large-profile-v1`,
`hard-traction-controls-v2`, `hard-traction-profile-v1` and `hard-traction-timing-v1`.
The unsubmitted `hard-traction-controls-v1` partial capture remains in place. Source
`hard_contact_traction_vectors` is pinned in workspace at e2979aed2. All55 source heads
and12 original unfinished files are preserved. Scientific work used Slurm `rtx-mid`,
image `1_26`, the existing data-side environment and source-private node caches.
Source/docs and compact evidence stay in workspace; no new home fallback output or
environment migration occurred.

The `hard-batch-scaling-v1` packet retains 158 compact payloads
(1,057,402 bytes) and 22 external receipts from two completed campaigns.
Primary B256/B1024 timings, fresh-process GPU polls, audits and logs remain in
the restored data root below under `hard-batch-scaling256-v1` and
`hard-batch-scaling1024-v1`. Scientific execution used Slurm `rtx-mid`, image
`1_26`, the existing data-side environment and source-private node caches.
No engine checkout moved or changed; all 54 frozen source heads and 12 original
unfinished files remain preserved. Source/docs and compact evidence stay in
workspace. No new home fallback output, data migration or environment change occurred.

The `hard-source-warps-v1` packet retains256 compact payloads
(1,674,747 bytes) and332 external receipts from four terminal campaigns.
Complete-source query arrays, CPU/GPU controls, paired profiles and B4/B64
scene timings, full-field captures and the frame428 replay use the restored
data root below in `hard-source-warps-cpu-v1`, `hard-source-warps-gpu-v1`,
`hard-source-warps-profile-v1` and `hard-source-warps-timing-v1`.
The isolated sparse source checkout `hard_contact_source_warps` remains in
workspace atf6c4ff2b0. All54 frozen source heads and12 original unfinished
files are preserved. Scientific work uses Slurm `rtx-mid`, image `1_26`,
the existing data-side environment and source-private compiler caches in
node scratch. Source/docs and compact publication evidence stay in workspace.
No new home fallback output, data migration or environment change occurred.

The `hard-edge-projection-v1` packet retains251 compact payloads
(1,698,372 bytes) and329 external receipts from four terminal campaigns.
Projection arrays, CPU/GPU controls, profiles, paired B4/B64 scene timings,
complete fields and the frame428 replay use the restored data root below in
`hard-edge-projection-cpu-v2`, `hard-edge-projection-gpu-v2`,
`hard-edge-projection-profile-v2` and `hard-edge-projection-timing-v1`.
Three initial v1 preflight directories remain explicitly unsubmitted; the
runner now validates task-file input before creating output directories.
Source `hard_contact_edge_projection` stays in a sparse workspace checkout
at1c8720391. All53 frozen source heads and12 original unfinished files are
preserved. Scientific work uses Slurm `rtx-mid`, image `1_26`, the existing
data-side environment and source-private compiler caches in node scratch.
Source/docs and compact evidence stay in workspace. No new home fallback
output, data migration or environment changes occurred.

The `hard-native-surface-basis-v1` packet retains302 compact payloads
(2,055,379 bytes) and332 external receipts from six terminal campaigns.
The surface-basis trials, corrected source tests, profiles, B4/B64 paired
timings and frame428 replay use the restored data root below in
`hard-native-surface-geometry-*`, `hard-native-surface-unroll-*` and
`hard-native-surface-basis-tests-v1`. Source inventories, logs, full fields
and generated test assets have explicit data-side destinations. Scientific
work uses Slurm `rtx-mid`, image `1_26`, the existing physical environment
and private node caches; paired timing now namespaces Genesis/Quadrants
caches by source path within each allocation. All52 source heads and12
original unfinished files are preserved. Three isolated source checkouts
stay in workspace, with the two later ones using sparse checkout. No new
home fallback output, data migration or environment changes occurred.

The `box-corner-sliding428-v1` packet retains243 compact payloads
(1,544,585 bytes) and50 external receipts from five terminal campaigns.
Contact-motion captures, prescribed-sliding controls, graded-grid fields and
surface reintegration use the restored data root below in
`box-corner-motion428-v1` and `box-corner-sliding428-v1/v2/v3/v4`.
All runtime arrays, stdout/stderr and captured execution metadata remain on
data. Scientific work uses Slurm `rtx-mid`, image `1_26`, the existing physical
environment and private node caches. All49 frozen engine heads and12 original
unfinished files are preserved. Source/docs and compact evidence stay in
workspace; no new home fallback output, data migration or environment change.

On2026-10-23 the exclusive-create/write/fsync probe succeeded again under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/`.
New planar profiling and cooperative-root validation use that restored data
root in `hard-native-planar-stages-*` and `hard-native-cooperative-roots-*`.
Outputs, stdout/stderr and source captures are explicit; dependency assets
reuse the data-side physical environment and compilation caches use private
node scratch. The existing bounded home fallback remains in place without
new runtime output. Source/docs and the isolated `hard_contact_cooperative_roots`
checkout remain in workspace. No data, environment or compatibility link moved.

The `hard-native-cooperative-roots-v1` packet retains247 compact payloads
(1,727,323 bytes) and286 external receipts from six terminal campaigns,
including all three original failed attempts. New runtime arrays, complete
fields, profiles and logs stay under data. Source03ee8b02a adds one isolated
engine to the48 preserved heads; all12 original unfinished files are intact.

The `box-corner-predict428-v1` packet retains136 compact payloads /985,989 bytes and11 external receipts. Four terminal CPU campaigns at the existing authorized fallback root `/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261023-box-corner/` retain independent profile predictions, the unsuccessful finite-face length correction, surface ordering and fixed-force displacement integration controls. They are `box-corner-predict428-v1/v2` and `box-corner-surface428-v1/v2`. Original reference arrays stay on data and are read in place. The data-side exclusive-write probe still failed errno122 with zero available capacity before this increment. The complete fallback now totals3,968,378,618 bytes after publication preparation, below the existing4,000,000,000-byte bound. All jobs explicitly record `--allow-home-output` and use Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private node caches. Source/docs and compact publication evidence remain in workspace;48 frozen engines,12 original unfinished files and existing compatibility links remain preserved. No data or environment moved.

The `box-corner-elastic428-v1` packet retains332 compact payloads /2,243,865 bytes and19 external receipts. Ten terminal campaigns retain frictionless two-face pressure controls, Hertz validation, finite-depth field/refinement diagnostics and the original data-quota initialization failure. The original arrays remain under the established data-side `20261019-qualified-ranking` root in `box-corner-elastic428-*`, `box-corner-fields428-*` and `vector-halfspace-hertz-v1`.

On2026-10-23 the shared data volume again reports800TB used, zero available and errno122 on exclusive creation. Job741762 failed before output initialization. The user's existing4GB home-output exception authorizes one bounded replacement campaign at `/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261023-box-corner/box-corner-outage428-v1`; original inputs are read in place. The complete existing fallback totals3,950,696,903 bytes after publication preparation, below4,000,000,000 bytes. New output/metadata use that disjoint fallback root, with `--allow-home-output`, Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private node caches. Its `meta/storage-incident-v1.json` records the failure and prior authorization; `meta/readonly-assembly-links-v1.json` lists publication aliases to the terminal data campaigns. These links move no data. All48 frozen engines and12 original unfinished files remain preserved. Source/docs and compact publication evidence remain in workspace; existing environments and compatibility links stay in place. Data remains the preferred destination when writable.

The `box-edge-wrench428-v1` packet retains85 compact payloads /513,653 bytes and6 external receipts. Five terminal campaigns retain the corrected contact-time opposing plane, independent raw witness checks, two-face continuous wrench-cone certificates, analytic support bounds and the original dtype/HiGHS failures. Full inputs and logs remain under `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking` in `edge-contact-frame428-v2` and `box-edge-wrench428-*`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the data-side environment and private node caches. All48 frozen engines and12 original unfinished files are preserved. Source/docs remain in workspace and no environment, data or compatibility link moved.

The `edge-shell-transition-v1` packet retains181 compact payloads /1,144,228 bytes and40 external receipts. Five terminal campaigns cover exact contact-frame transformations, admitted frame419/420/427 elastic captures, smooth shell corrections, the repeated frame428 refusal and finite-depth self-displacement quadrature through order384. Full arrays and logs remain under the existing `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking` root in `edge-contact-frame428-v1`, `edge-shell-transition-*` and `edge-self-displacement428-*`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the data-side environment and private node caches. All48 frozen engines and12 original unfinished files are preserved. Source/docs stay in workspace and no environment, data or compatibility link moved.

The `hard-region-moments428-v1` packet retains163 compact payloads /1,050,781 bytes and15 external receipts. Six terminal campaigns retain the frame428 repeat, cleared-scratch reader failures, independently reconstructed pressure scales, actual curved moment quadrature, collider application-point checks and continuous quadratic-surface traction bounds. Full arrays and logs stay under the existing `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking` root in `hard-region-moments428-*`, `hard-region-application428-v1` and `hard-region-cone428-v1`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the data-side environment and private node caches. The48 frozen engines and12 original unfinished files are preserved; source/docs remain in workspace and no environment, data or compatibility link moved.

The `hard-native-boundary-lanes-v1` packet retains523 compact payloads /3,063,847 bytes and1461 external receipts. Fourteen terminal campaigns cover boundary quadrature scheduling, its compaction ablation, CPU/GPU lifecycle controls, B4/B64 short-window scene costs, original full-cycle timeouts, three frame428 refusals and independent point-force wrench certificates. Full arrays, large journals, profiles and job logs remain under `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking` in `hard-native-geometry-cycle-v1`, `hard-native-active-boundary-*`, `hard-native-boundary-lanes-*` and `hard-native-allocation428-v1`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private node caches. All48 frozen engine heads and12 original unfinished files are preserved. Source/docs and compact evidence remain in workspace; no environment, data or compatibility link moved.

The `hard-native-geometry-search-v1` packet retains516 compact payloads /3,111,032 bytes and1294 external receipts. Sixteen terminal campaigns cover ordered geometry/source-vertex pruning, its retained threshold-boundary failure and repair, CPU/GPU disconnected/nonconvex controls, complete B4/B64 scene timings, raw field differences and an unchanged baseline repeat. Runtime arrays, profiles and logs remain under `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-geometry-search-*`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private node caches. All46 frozen engine heads and12 original unfinished files are preserved. Source/docs remain in workspace; no environment, data or compatibility link moved.

The `hard-native-friction-newton-v1` packet retains282 compact payloads /1,701,012 bytes and297 external receipts. Eight terminal campaigns retain the near-saturation friction repair, CPU/GPU history controls, admitted B4/B64 paired scene costs and a separate original-kernel profile. All runtime arrays, profiles and logs remain under the established `20261019-qualified-ranking` data root in `hard-native-friction-newton-*`, `hard-friction-newton-study-v1` and `hard-native-hard-friction110-penalty-v2`. Scientific execution used Slurm `rtx-mid`, image `1_26`, the existing data-side Python environment and private node caches. All43 frozen engine heads and12 original unfinished files remain preserved; no data, environment or compatibility link moved.

The `hard-native-hard-input-boundary-v1` packet retains640 compact payloads /3,508,302 bytes and303 external receipts. Seventeen terminal campaigns cover loaded-region input acceleration, source/face boundary diagnosis, repaired moving-boundary quadrature, CPU/GPU controls and the subsequent frame110 friction failure. Runtime arrays and logs remain under the established `20261019-qualified-ranking` data root in `hard-native-hard-*`. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private node caches. All42 frozen engine heads and12 original unfinished files are preserved; no environment, data directory or compatibility link moved.

The `hard-native-hard-scene-v1` packet retains 553 compact payloads / 3,010,939 bytes and 1203 external receipts. Thirteen terminal campaigns cover actual sharp-edge main-pipeline integration, CPU/GPU friction and lifecycle controls, exact numerical/geometry replays, and the still-rejected moving-frame boundary correction. Runtime arrays and logs remain under the existing `20261019-qualified-ranking` data root in `hard-native-hard-*`; no environment, existing data or compatibility link moved. Source/docs remain in workspace. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the established data-side environment and private node caches. All 37 frozen engine heads and 12 original unfinished files are preserved.

The `hard-edge-farfield-v1` packet retains304 compact payloads /1,861,974 bytes
and23 external receipts. Nine terminal campaigns cover remote pressure-moment
quadrature, paired64-bit component timings, the original32-bit cancellation
failure, corrected32-bit evaluation, complete-shell regression and constrained
all-contact load sensitivity. The unassigned ground-cache reader trial remains
labelled invalid geometry evidence. Full arrays and logs stay under the existing
`20261019-qualified-ranking` data root in `hard-edge-*` and
`hard-native-edge-farfield-*`. Engine `hard_contact_edge_farfield` is frozen at
`4a8125c4c`, with29 frozen heads and12 original unfinished files preserved.
Scientific execution uses Slurm `rtx-mid`, image `1_26`, the existing data-side
environment and private node caches. Source/docs and compact evidence stay in
workspace. No data, environment or compatibility link moved.

The `hard-edge-fields-v1` packet retains 293 compact payloads / 1,613,039 bytes
and 22 external receipts. Eight terminal campaigns cover continuous sharp-edge
fields, native CPU/GPU 32/64-bit evaluation and timings, original endpoint and
frame-orientation failures, curved-shell source-placement response and a smooth
shell-tilt diagnostic. Full arrays and job logs remain below the established
`20261019-qualified-ranking` data root in `hard-edge-fields-*`,
`hard-native-edge-fields-*`, `hard-edge-source-response-*` and
`hard-edge-shell-tilt-*`. Engine `hard_contact_edge_fields` is frozen at
`e64780805`, with 28 frozen heads and 12 original unfinished files preserved.
Source and compact publication evidence stay in workspace. Scientific execution
uses Slurm `rtx-mid`, image `1_26`, the existing data-side environment and
private node caches. No data, environment or compatibility link moved.

The `hard-edge-law-v1` packet retains 174 compact payloads / 771,900 bytes
and 34 external receipts. Five terminal campaigns cover independent
shear-coupled sharp-edge pressure, the unfitted slender-edge law, and native
CPU/GPU 32/64-bit controls and resident scale timings. Full pressure/stress
arrays and job logs remain below the established `20261019-qualified-ranking`
data root in `hard-edge-*` and `hard-native-edge-law-*`. Source and compact
publication evidence stay in workspace. Engine `hard_contact_edge_law` is
frozen at `88d9e891a`, with 27 frozen heads and 12 original unfinished files
preserved. Scientific execution uses Slurm `rtx-mid`, image `1_26`, the
existing data-side environment and private node caches. No data, environment
or compatibility link moved.

The `hard-force-line-v1` packet retains195 compact payloads/1,133,852bytes
and477 external receipts. Six terminal scientific campaigns cover the native
curved force-line repair, its frozen negative control, actual Franka replay,
field comparison and sharp-edge normal-only diagnostics. The manufacturer
document, rendered page, visual-mesh sections and isolated PDF renderer use
`assets/franka-fingertip-spec-v1` under the existing20261019 ranking data root;
original inspection failures remain retained. Runtime arrays, logs and assets
remain on data. Engine `hard_contact_force_line` is frozen at`601cde58f`, with26
frozen heads and12 original unfinished files preserved. Scientific work uses
Slurm`rtx-mid`, image`1_26`, the existing data-side environment and private node
caches. Source/docs stay in workspace. No active environment, data or link moved.

The `hard-curved-walk-v1` packet retains378 compact payloads/2,287,422bytes
and479 external receipts. Twelve terminal campaigns under `hard-curved-walk-*`
cover quadratic material paths, curved/affine friction binding, nonconvex
first exits, GPU component cost and original/refined Franka replay. Original
negative controls, the sign-label reader and near-zero relative-comparison
failure remain preserved. Full arrays, generated meshes, scene captures and
logs use the established20261019 ranking data root. Engine
`hard_contact_curved_walk` is frozen at`3025c965b`, with25 frozen heads and12
original unfinished files preserved. Source/docs remain in workspace;
scientific execution reuses Slurm`rtx-mid`, image`1_26`, the data-side environment
and private node caches. No data, environment or compatibility link moved.

The `hard-matched-collision-v1` packet retains304 compact payloads/1,690,286bytes
and819 external receipts. Collision meshes generated from the curved material
surface, original/repaired Franka captures, source-binding CPU/CUDA controls,
nonconvex regressions and saved-frame inspections remain under the established
20261019 ranking data root in `hard-matched-collision-*` and
`hard-curved-source-geometry-*`. All nine campaigns are terminal. Original failed
geometry, reader and exact-comparison attempts remain preserved. Source/docs
stay in workspace; runtime arrays, generated assets and logs stay on data.
Engine `hard_contact_matched_collision` is frozen at`ad6b71984`, with24 frozen
heads and12 original unfinished files preserved. Scientific work reuses Slurm
`rtx-mid`, image`1_26`, the data-side environment and private node caches.
No data, active environment or compatibility link was migrated.

The `hard-finite-edge-v1` packet retains213 compact payloads/1,343,627bytes
and240 external receipts. Same-substep face-boundary capture, CPU/CUDA controls,
independent edge-pressure trials and saved-frame/force-line inspections use
`hard-finite-edge-*` below the established20261019 ranking data root. Nine
submitted campaigns are terminal; the unsubmitted sparse-checkout preflight,
original failed readers and rejected pressure models remain preserved. Full
arrays, scene frames and generated meshes stay on data. Source/docs remain in
workspace. The new engine `hard_contact_finite_edge` is frozen at`660c85e98`,
with23 frozen heads and the original12 unfinished files preserved. Scientific
execution uses Slurm`rtx-mid`, image`1_26`, the existing data-side environment
and private node caches. No data, active environment or compatibility link moved.

The `hard-native-curved-source-v1` packet retains249 compact payloads/
1,407,781bytes and509 external receipts. Native curved-source CPU/CUDA fields,
actual Franka frames, the refused gripper contact and geometry diagnosis all
remain under the established `20261019-qualified-ranking` data root, in
`hard-native-curved-source-v1/v2`, `hard-live-curved-franka-v1`,
`hard-live-curved-franka-grasp-v1`, `hard-live-curved-franka-geometry-v1/v2`
and `hard-curved-franka-region-v1`. Full arrays, generated assets and large
per-step JSON histories stay on data. Compact source/log/summary evidence is in
Git. New engine `hard_contact_curved_source` is frozen at`ce57a7fb2` with22
frozen heads and the original12 unfinished files preserved. All scientific work
uses Slurm`rtx-mid`, image`1_26`, the established data-side environment and
private node caches. No data, active environment or compatibility link is moved.


The `hard-native-curved-material-v1` packet retains289 compact payloads/
1,669,564bytes and71 externally hashed fields and meshes. Native curved material
CPU/GPU checks, original compiler failures, affine/friction regressions, corner
precision tests and curved source-traction comparisons use the existing
`20261019-qualified-ranking` data root in `hard-native-curved-material-v1` through
`v3`, `hard-curved-material-corner-v1` and `hard-curved-hertz-residual-v1` through
`v3`. All seven campaigns are terminal. Runtime outputs and logs stay on data;
compiler caches are private and node-local. All21 frozen engine heads and12
original unrelated files remain preserved. Source/docs stay in workspace. The
new experimental engine is`aa9cc8298`; final material and residual workers are
captured in v3. No data or active environment was migrated.

The `hard-native-quadric-curvature-v1` packet retains345 compact payloads/
1,988,937bytes and121 externally hashed fields and generated meshes. Curved
rest-surface/gap studies, native curvature controls, actual Franka snapshots
and independent curved P2 checks use the existing`20261019-qualified-ranking`
data root under`hard-quadratic-rest-surface-*`, `hard-curved-contact-gap-*`,
`hard-vertex-curvature-*`, `hard-native-quadric-*` and`hard-curved-shell-operators-*`.
All12 campaigns are terminal; original failures remain intact. Runtime arrays
and logs stay on data, with private node-local compiler caches. All20 frozen
engine heads and12 original unrelated files are preserved. Source/docs stay
in workspace. The new native implementation is commit`e7e49b738`; successful
curvature and curved-operator workers are captured in their v3/v4 campaigns.

The `hard-native-parallel-boundary-v1` packet retains227 compact payloads/
1,306,754 bytes and100 externally hashed arrays. The geometry-gap control,
complete-shell module timings, parallel CPU/GPU checks, independent fields,
paired sphere/flat friction scenes and cost comparisons use the existing
`20261019-qualified-ranking` data root in the `hard-franka-contact-gap-v1`,
`hard-native-complete-shell-timing-v1` and `hard-native-parallel-boundary-*`
runs. Runtime arrays/logs stay on data; compiler caches were private node-local
directories. All19 frozen source heads and12 original unrelated files are
preserved. Source/docs remain in workspace. The formatting-only descendant
was made after every scientific reader terminated and has an unchanged AST.

The `hard-transfer-walk-exit-v1` packet retains 185 compact payloads/1,707,799
bytes for the declined projected-walk exit, its original failed GPU group and
later baseline/variant diagnostics. Its 138 external fields, histories,
trajectories and meshes remain hashed in `hard-scene-walk-stop-v1` and the
`hard-walk-*-diagnostic-v1` / `hard-walk-*-context-v1` runs on data. Subsequent
masked-sample and paired bandwidth jobs are separate and excluded.

The completed `hard-shell-residual-v1` packet retains 55 compact payloads/357,879
bytes for prescribed-pressure hollow-shell decomposition and read-only full-field
inspection. Six mesh/complete-field archives remain hashed under the data-side
`hard-shell-residual-v1` run; inspection logs/receipts use
`hard-shell-residual-inspection-v1`. Private node-local caches were removed by
the existing job trap. The separate live scene-walk experiment is excluded.

The `hard-scene-large-batch-v1` packet adds 60 compact payloads/658,695 bytes
for completed B256/B1024 paired max/full/rigid scenes and complete-history
comparisons. Its 72 external artifacts, including large compressed histories,
stay hashed under `20261019-qualified-ranking/hard-scene-large-batch-v1` on data.
The separate `hard-shell-residual-v1` study and its job logs also use this data
root; its live scientific outputs are excluded from the large-batch packet.
The adjacent `hard-scene-public-source-binding-v1` packet retains three compact
stdlib source-verification payloads, binding the isolated scientific checkouts
to public task commits without changing the original execution provenance.

The `hard-scene-transfer-v1` packet retains 326 compact payloads/3,209,088 bytes
for B4/B64 paired moving-scene max/full transfer and coarse elasticity, CPU/GPU
independent bridge controls and complete-history comparisons. Its 180 external
artifacts stay hashed under the established `20261019-qualified-ranking` data root
in `hard-scene-pilot-v1`, `hard-scene-check-v1`, `hard-scene-scaling-v1` and
`hard-scene-replay-capture-v1`/`hard-scene-replay-v1`, `hard-scene-stable-v3` and
`hard-scene-preparation-v1`. Original B64/regression failures and two unsubmitted
sparse-checkout capture failures remain retained beside the passing repair.
Full fields, trajectories, generated meshes and job logs use data; private
compiler caches were node-local. No active environment, existing data, source
checkout or compatibility link was migrated. Source/docs remain in workspace.

The `hard-hertz-tangential-v1` packet retains87 compact payloads/395,654 bytes
for independent elliptic tangential stress controls, a loading/reversal comparison
and native CPU/GPU accuracy/timing. Its five full tensor/cycle arrays remain
externally hashed under the established20261019-qualified-ranking data root in
`hard-tangential-v1` and `hard-native-tangential-v2`. The original native assertion
failure remains in `hard-native-tangential-v1`; no failed output was overwritten.
Private compiler caches were node-local. Source/docs remain in workspace;
existing data, environments and compatibility links were preserved. Supplied
traction-kernel timing does not include a friction-history law or scene output.

The `hard-normal-roots-and-edge-v1` packet retains96 compact payloads/677,074
bytes and67 externally hashed arrays/meshes for direct planar normal roots,
matched live Franka input/rigid timing and original/corrected straight-edge
pressure controls. New runtime data and logs are under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/`
in `hard-normal-roots-v1`, `hard-normal-live-v1` and `hard-edge-v1`/`v2`.
The original edge-domain clipping and quadrature limitations remain documented.
Exact executed source variants and compact histories are archived; private
compiler caches were node-local. No existing data, environment or compatibility
link was migrated. Native input timing and offline edge fitting remain distinct
from physically admitted shell-stress throughput.

The hard-hertz-inputs-v1 packet retains270 compact payloads/1,680,227 bytes
for native curvature, twelve CPU/GPU region/source controls, and same-substep
moving Franka input pilots with matched rigid baselines. Original option,
compilation and NaN-acceptance failures are preserved. Its182 externally hashed
artifacts include full arrays, trajectories and generated control meshes under
the existing20261019-qualified-ranking data root. Compact histories remain with
the receipts; private compiler caches were node-local. The isolated engine
checkout is in workspace. No existing output, environment or compatibility link
is moved; live region rejection is retained rather than counted as usable stress
throughput.

New automatic-contact support/pressure studies, paired independent P2 fields and
figure renders use the established data run root. The automatic-contact-support-v1
packet and its exact-size figure companion retain compact executed
sources/receipts, original failed readers and publication artwork; full runtime maps, quadrature and stress arrays remain
hashed on data. The style helper is copied into that run's meta directory and
private job caches are node-local. Existing native baselines, fine writers,
environments and compatibility links are preserved in place.

The solid-pad-forward-prefix-guard-v1 packet adds44 compact payloads/253,293
bytes for the actual old/new inherited-prefix counterexample,20 passing metadata
controls and unchanged restoration/physics-source inspection. Seven existing
checkpoint arrays remain externally hashed on data; mesh/source dependencies
retain their earlier packets and exact receipts. No FEM trajectory is rerun and
no live fine source/output is copied or migrated. New refusal/control outputs,
private job caches and logs use the existing data run root or node-local paths.

The independent-method-preparation-v1 packet adds12 compact payloads/58,999
bytes for scheduler-only dependency updates and final source inspections. It
contains no runtime arrays or live numerical output. All seven captured analysis
inventories and six published computation/acceptance modules remain unchanged.
The original receipt-reader failure is retained beside the completed correction.
Existing fine writers, their data-side outputs, environments and links stay in
place. New scheduler journals and inspection receipts use the same data run root;
only compact publication evidence is added to the source checkout.

The solid-pad-partial-step-recovery-v1 packet adds205 compact payloads/1,066,370
bytes for terminal step300/225 recovery, complete400-step controls, exact captured
worker variants and ancestry/negative checks. Twenty-four full state/field/column
arrays remain externally hashed under the established data run root. The tiny
damaged ZIP is an explicit regression fixture. New preparations, journals,
checkpoint arrays and Slurm logs all stay on data, with private node-local caches.
No live fine output/source, active environment or existing link is migrated or
rewritten. Only compact publication evidence remains in the source checkout.

The canonical125-resilient-recovery-v1 packet adds794 payloads/5,326,422 bytes
for terminal preemptions, exact saved-prefix preparations, completed periodic
state/control reinspection and immutable replacement launch/source receipts.
Eighteen arrays remain externally hashed on data. New full-state checkpoints,
continued histories, fields and logs use the established data run root; live
scientific outputs are excluded from the archive. New relative data-side assembly
links reuse two completed comparison groups without moving original files or
replacing tracked paths. Node exclusions are per-job options; existing shared
configuration, environments and compatibility links stay intact.

The contact125-pair-scheduling-v1 packet adds23 payloads/165,478 bytes for
immutable pending contact-order launch/source receipts and the exact per-element
scheduler update. It includes no live history, field or new scientific result.
New guard outputs/logs stay under the established data run root; its temporary
caches are private and node-local. No scientific data, environment or link moves.

The surface125-component-v1 packet adds104 payloads/457,365 bytes for terminal
q9/q25 two-wall250→125um comparisons,65 component checks, two rejected metadata
controls and the exact per-array-element scheduler update. Four complete column
arrays remain in their existing data-side outputs with hashes. No scientific
output is moved; active fine writers and the six still-pending field groups
remain in place. Temporary caches stay private and node-local.

The canonical125-surface-terminal-v1 packet adds64 payloads/341,005 bytes for
the actual terminal400-step125um/two-wall physical parent and its10-check strict
admission. Two complete endpoint fields and the mesh remain on data with external
hashes. Only terminal group0 receipts/source/logs are frozen; live group1 and the
other fine writers are excluded. No scientific data/environment/link is moved.

The independent-process-memory-v2 packet adds331 compact payloads/2,775,215
bytes for14 collector controls and24 source-matched fresh memory profiles.
Twenty full raw benchmark JSON files and eight complete replay arrays remain
under the established20261019-qualified-ranking data root with external hashes.
The independent child uses only private job TMPDIR for temporary query logs;
complete query records are retained in the explicit data-side profile results.
Active fine histories, environments and compatibility links stay in place.

The oversized generated publication draft was untracked, unpublished and had no
active writer after all associated jobs/publisher completed. It was copied with
copy2 metadata and symlink preservation,352 files verified for bytes/SHA-256,
mode and mtime, then only its verified home copy was removed:

- Old: /mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/independent-process-memory-v1
- New: /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/publication-drafts/independent-process-memory-v1

No compatibility link is needed for this unpublished draft. Exact checks and
inventories are in meta/independent-process-memory-draft-migration-v1.json at the
same run root and in the compact packet. No frozen manifest is rewritten, no
tracked directory is replaced and no live scientific output is migrated.

The native-performance-summary-v1 packet adds203 compact payloads/1,464,777
bytes for the61-row source-bound scene summary, four current-source B8 module
modes and four separate process-memory runs. It retains original reader failures,
exact worker/source/configuration receipts, B8 comment-only reconstruction and
actual sample gaps. Sixteen full histories/trajectories remain under the existing
20261019-qualified-ranking data root with hashes. Original sparse/large-batch
arrays, source checkouts, environments, active fine outputs and compatibility
links remain in place. New job outputs/logs are explicitly on data and private
caches node-local. Sampled process peaks are not maximum-capacity measurements.

The canonical-force/batch checkpoint adds `modal-canonical-force-v1`
(143 payloads,1,162,186 bytes) and `native-batch-memory-diversity-v1`
(413,3,964,315 bytes). They retain terminal controls, original failures,
portable small rejection inputs, deterministic gzip and both original/encoded
hashes. Full coarse/fine fields, trajectories and raw benchmark JSON stay in
the established `20261019-qualified-ranking` data root. New corrected125um
reference/time/contact and larger batch/memory jobs contribute immutable inputs
only. No live reference output, environment or compatibility link is migrated.

The contact/static-control checkpoint adds three compact Git evidence packets:
`native-contact-consistency-v1` (307 payloads,3,290,067 bytes),
`shell-pad-load-separation-v1` (175,1,248,122 bytes) and
`native-contact-conformance-scaling-v1` (212,7,522,851 bytes).
They preserve source/task identities, original failures, deterministic gzip and
both original/encoded hashes. The follow-up includes terminal B1/B8/B32
conformance, larger timing receipts and the rejected125um two-wall history;
active canonical timing/larger conformance contribute immutable inputs only.
Full arrays, raw18MB trace, large reference frames, logs and all new runs remain
under `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/`.
The combined native audit uses data-side symlinks plus an assembly manifest to
point to original terminal cohorts; it does not move or replace their paths.
The isolated fixed engine is a source checkout under workspace. Active analysis
engine/reference sources, environments, caches and compatibility links remain
in place. New task outputs are explicitly on data and private caches node-local.

The companion `native-canonical-production-v1` retains92 payloads totaling
1,416,760 bytes for all15 repaired-source B1/B8/B32 timings and independent
receipt audit. Separate memory jobs contribute immutable inputs only. Full
timing JSON, source captures and runtime outputs remain at the same data root.

The native-production-measurements-v1 packet retains254 payloads totaling
4,272,574 bytes; native-module-replay-v1 retains46 totaling501,678 bytes.
Large JSON and raw logs are losslessly gzip-encoded, with encoded and original
hashes/sizes in the manifest. Full scene trajectories and all24-buffer replay
histories remain under the established data run directory with exact identities.
The completed controller-only probe is retained with terminal receipts. No active
reference output, environment, cache or compatibility symlink is migrated.

The prescribed-method-ranking-v1 packet retains78 verified payloads totaling
1,199,808 bytes for terminal C2 comparisons. The friction-wall250-reference-v1
packet retains80 payloads totaling941,008 bytes for completed250um mechanics
and wall comparisons plus immutable fine-control inputs. Full NPZ arrays,
complete journals and active125um outputs remain on the existing data path.

The production-throughput-pilot-inputs-v1 packet contains immutable launch,
source and cancellation receipts only. The new isolated source checkout stays
under workspace; all benchmark outputs/logs and the cancelled invalid-provenance
patch stay under the existing20261019-qualified-ranking data directory. Private
runtime caches remain node-local. No active environment or compatibility link
is moved, and no pending throughput rate is published as admitted.

The `prescribed-reference-controls-v1` packet retains209 verified payloads
totaling2,776,889 bytes for the terminal C2 reference qualification, signed
tensor controls, exact allocation repair and original Shell replay OOM.
Full fields/columns remain on data. Active Shell replay/native decomposition
contribute immutable submitted inputs only; their results and active friction
writers are excluded. No runtime data, environment or link is migrated.

The companion `prescribed-shell250-terminal-v1` adds25 verified payloads
totaling455,148 bytes for the admitted static replay, all3 iteration records
and frame/rest identities. Full arrays stay on data; the follow-up three-field
comparison contributes immutable submitted inputs only.

The `normal-absolute-errors-v1` packet retains84 verified payloads totaling
1,472,625 bytes: terminal81-row dimensional normal errors,69 input identities,
CPU-only metadata controls and immutable submission/cancellation records.
All column/full-field arrays and active prescribed/friction outputs remain on
data. CPU-only tasks use the same rtx-mid/container path with explicit data
outputs and node-local caches. No runtime data or environment is migrated.

The `native-reconstruction-material-v1` packet retains362 verified payloads
totaling4,010,250 bytes for terminal native quadrature/core and normal material
controls, prescribed-C2 common sampling and retained diagnostic failures.
All original/replayed/independent fields, quadrature arrays and full journals
stay on data. The active finer prescribed reference contributes immutable
submission receipts only; its output and active friction writers are excluded.
No runtime data, environment or compatibility link is migrated.

The `friction-spatial250-v1` packet retains97 verified payloads totaling
1,088,827 bytes for terminal250um nonlinear mechanics,500-to250um spatial
comparisons and a500um plotting-data control. Full fields, complete journals,
prepared CSV/NPZ and active finer-reference/material outputs remain on data.
Only terminal receipts, source, logs and hashes are copied into the publication;
there is no runtime-data, environment or compatibility-link migration.

The `friction-common-controls-v1` packet retains88 verified payloads totaling
1,012,437 bytes for completed coarse nonlinear time/contact controls using
q9/q25 common sampling. Eight full endpoint fields remain on data with verified
hashes. Pending normal-load material and fine spatial/method controls contribute
only immutable submissions, targets or explicitly unexecuted task plans; their
active results are excluded. No data or environment is migrated.

The `friction-keyframe-reuse-v1` packet retains329 verified payloads totaling
6,028,562 bytes for terminal matched friction keyframe costs, original/captured
SCR every-step failures, actual source-cone inspection and validated reference
response reuse. The small before-call source packet is included; the rejected
full field and other large runtime files remain on data with manifest hashes.
Active250um/125um reference writers are excluded. No migration is performed.

The later `reference-preemption-v1` packet adds59 verified payloads totaling
473,402 bytes for preempted125um partial histories, scheduler duplicate records,
signal-exit controls and immutable retry submission receipts. New execution uses
`solid-pad-stvk-final125-reference-v2/` on data; the originalv1 outputs remain
unchanged. Active retry outputs are not copied into the publication.

The `friction-reference-observation-v1` packet retains205 verified payloads
totaling2,289,881 bytes from terminal nonlinear reference/field comparisons,
captured DPR observation rejection and admitted Shell friction cost. The
manifest identifies terminal leaves selected from an active cost array; its
SCR writer and fine-reference jobs remain untouched. Large fields and the
rejected source/field NPZ remain on data with hashes and paths retained.

The `complete-pad-cost-v1` packet adds222 verified payloads totaling2,345,974
bytes for terminal B4 primary costs, the audited complete B1/B4 cost/memory
aggregation and separate solver-phase profiles. Premature-analysis failure
records are retained; full fields/journals and fine-reference writers stay on
data. No live data, environment or compatibility link is moved.

The `friction250-cache-memory-v1` checkpoint retains257 verified small
payloads totaling6,571,125 bytes from terminal Shell friction/field, exact
StVK cache and process-memory controls. Full fields, journals and active fine
reference/cost/profile writers remain under the established data-side ranking
root. This bounded publication does not migrate runtime data or environments.

The follow-up `normal-spatial-cost-v1` publication adds175 verified payloads
totaling7,057,473 bytes: inspected normal field/traction figures, bounded
plotting data, exact source, sparse primary receipts and the CPU profile.
Larger plotting arrays and full trajectory data remain at the manifest's
data-side paths. Active B4/memory and fine-reference writers are unaffected.

The `common-peaks-batch-v1` selected publication retains897 verified small
payloads totaling10,609,874 bytes: terminal receipts, executed source, failure
records and editable response/peak figures. Its manifest explicitly identifies
completed leaves selected from ongoing cost/memory arrays; other active output
is not copied or claimed complete. Full fields, journals, process-memory staging
and ongoing fine StVK runs remain under the established data-side ranking root.
No live writer, environment or compatibility link is moved or changed.

## Qualified ranking stage, 2026-10-19

New runs, generated contact meshes, regression logs and source receipts use
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/`.
The exclusive-create/write/fsync probe succeeded before submission. Source
changes are isolated in the `stress_qualified_ranking` worktree under the
existing workspace; private compilation caches remain compute-node-local.
Existing home fallback evidence remains in place. No migration is performed.

The selected `native-pad-history-v1` publication under
`research/rigid_stress/evidence/20261019-qualified-ranking/` contains about3.5MB
of existing terminal receipts and executed research source, copied with size
and SHA256 verification. Full native/Shell/FEM fields, step journals, large
assets and active joint-refinement outputs remain under the data-side run root.
This small publication is not a runtime data destination or a migration.
The adjacent `full-source-factor-control-v1` adds about0.9MB of verified
terminal receipts, failed-factor logs and exact factor-control source. Its
CVXOPT adapter reuses an existing data-side environment; no new dependency
installation, active-environment edit or data migration is involved.

The optional independent QP audit uses Clarabel0.11.1 installed with
`--no-deps --target` at
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/assets/clarabel-0.11.1/`.
Existing environments are unchanged. Wheel/install receipts and logs are under
`runs/20261019-qualified-ranking/clarabel-install-v2/`; the earlier login-node
attempt failed before installation because that container's interpreter is
unavailable on the login host. Scientific execution and the successful install
use the existing rtx-mid / GENESIS_IMAGE_VER=1_26 workflow.

The optional independent CHOLMOD factor uses Cholespy2.2.0 installed with
`--no-deps --only-binary=:all: --target` at
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/assets/cholespy-2.2.0/`.
The5.6MB wheel, install report and log are recorded by Slurm727753 under
`runs/20261019-qualified-ranking/cholespy-install-v1/`. Active environments
are unchanged; factor controls and generated fields use that same run root.

The `transient-radius-controls-v1` selected publication adds159 verified
receipts/source/log files totaling3,178,729 bytes. Full fields, live transient
and joint-reference jobs, new logs and all dependency assets remain on data.
It contains retained failed attempts and explicitly partial reference controls,
not a migration or a runtime output destination in the source checkout.

The `normal-pad-reference-v1` checkpoint retains verified terminal normal
reference/uncertainty receipts, objective transient controls, preflight cost
and failure records, plus editable figures and executed plotting source.
Figure staging and full fields/journals remain on data, as do ongoing
StVK time/contact/spatial refinements and paired-output controls. No active
writer, environment or compatibility symlink is moved or changed.

## Parallel-plate benchmark storage, 2026-10-18

The updated goal is pinned at fork commit `914a558df`. A new exclusive-create,
write and fsync probe succeeded on data, whose filesystem reports available
capacity again. New plate inputs, jobs, logs and outputs use
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates/`.
Private per-job compilation caches remain node-local. Existing home fallback
results are preserved in place; this is not a migration or a change to the
previous 4GB exception. The goal/storage receipt is `meta/goal-and-storage.json`
under the new run root.

## Explicitly authorized temporary home output, 2026-10-18

After reviewing the storage requirement and the existing temporary outputs,
the user explicitly authorized continued writes to the existing home fallback
("允许继续写入用户"). This exception is scoped to the current task and the
existing4GB persistent-output bound; data remains the preferred destination.
At authorization the fallback held about2.79GB. An exclusive-create/fsync probe
still fails122 and the data mount has zero free bytes. Existing running/queued
experiments may continue; new submissions must explicitly record the exception
with `--allow-home-output` and retain private node-local caches. The default
submission path check still rejects runtime destinations resolving under home.
Existing compatibility links resolving into data remain usable. Large original
inputs, environments and builds remain on data; no migration or deletion has
occurred. Future consolidation requires writer/job checks, metadata/symlink-
preserving copy and content verification before removal or replacement.

At the sparse-keyframe checkpoint, successful terminal analysis726752 had
produced ten reproducible common-column alignment NPZs,129240196 bytes, under
the fallback's `prescribed-sparse-fields-v1/group0/stock/` and
`group1/volume-surface2/`. Their hashes matched the completed result receipts
and all referenced input receipts still matched. These derived intermediates
were retired to keep persistent runtime outputs within4GB; original complete
spatial simulator fields, rest meshes, metrics, source snapshots and inputs
were retained unchanged. Exact retired paths/hashes and the captured rebuild
recipe are in `meta/prescribed-sparse-fields-v1-derived-retirement.json`, also
published in the
[sparse checkpoint](research/rigid_stress/evidence/20261018-qipc-reference/sparse-keyframes-v1/README.md).
This is bounded removal of regenerable derived arrays after writers exited,
not migration of original observations or a compatibility-link change.

`runs/20261018-qipc-reference/` under the established task data root holds the
new three-method comparison's source/build/input receipts, native build/runtime,
generated CCCL assets, controls, trajectories and per-job caches/logs. The
original QIPC source checkout is an isolated sibling worktree. Large binaries,
dependencies and runtime arrays stay on data. Existing QIPC dependency/toolkit
caches are reused without modifying active environments or unrelated jobs.
See [execution status](research/rigid_stress/docs/QIPC_REFERENCE_PROGRESS.md).

On2026-10-18 the shared data mount reports800TB used and zero available;
exclusive file creation fails with errno122. The initial submission hold was
lifted only for the disclosed bounded writable fallback below. Existing results,
native checkpoints/builds, profiler traces, source receipts and Genesis
precomputations stay in place. Only private regenerable compilation caches of
successful terminal groups were removed, with exact paths/bytes in the
[incident journal](research/rigid_stress/docs/QIPC_STORAGE_INCIDENT_20261018.json).
No runtime data was moved during that incident cleanup. The small selected publication checkpoint
under `research/rigid_stress/evidence/20261018-qipc-reference/checkpoint-v1/`
contains existing derived JSON summaries and provenance for the working report;
it is not a runtime storage fallback or a completed-goal evidence closure.
The adjacent `checkpoint-figures-v1/` contains about6MB of existing inspected
SVG/PDF/PNG, plotted arrays and exact rendering source copied with byte/hash
verification from the original data-side jobs. Publishing these selected
deliverables uses byte copies without rerendering them.
The later `checkpoint-v2/` selects about9.4MB of already Slurm-derived controls,
cost/negative-result summaries, exact source/receipt captures and inspected
uncertainty SVG/PDF/PNG/CSV. Large memory-polling sample arrays remain in their
original hashed audit rather than being duplicated in this publication.

## Bounded storage-outage validation fallback

After a third exclusive-create probe still failed122 at12:31UTC, lightweight
observer/admission/audit validation uses the explicitly announced temporary
project data root
`/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/`.
The initial1GB persistent-output budget was revised to2GB at13:44UTC and
then explicitly to4GB for the selected complete case7 sensitivity study, before
exceeding either prior limit. The six9s controls have concurrency3,240-minute
per-group limits and actual0.5s native checkpoints. The original case7 uses
232.5MB; the existing fine6s prescribed history uses621.8MB, providing a
measured output estimate. Original large inputs, builds and environments remain
on data. No broad new sweep or environment is included. Source,
task capture, logs and small derived results remain there; compile/model/tmp
caches use private compute-node `/tmp` directories through `--node-local-cache`
and are removed when each allocation exits. This does not change the default
data-side cache behavior of other submissions or shared environments.

Original experiments, large inputs, native binaries, checkpoints and traces
stay on `/mnt/data` and are read in place. Two fresh coarse9s capped histories
reuse the existing solver/input configuration and start from the initial state;
their output estimate uses the existing232.5MB complete case7 and147.9MB/5.27s
case0 prefix. The completed validation currently uses about215MB. Actual public
checkpoints are retained every0.5s. No interrupted native state is reconstructed
from trajectory frames. No migration, compatibility symlink change, or new
environment is part of this fallback.
When data storage is writable again, any consolidation must check writers/jobs,
preserve metadata/symlinks, verify hashes and record paths before removing the
temporary source, following the normal storage policy. The fallback is a
disclosed response to the unwritable mount, not a new preferred runtime root.

# Completed shell PR22 comparison

`runs/20261018-shell-comparison/` under the established task data root holds
immutable analysis captures, exact PR API/source receipts, matched input meshes,
controlled fields/trajectories, Slurm logs and isolated per-allocation caches.
The external original and diagnostic source checkouts are isolated sibling
worktrees under the existing workspace. Scientific execution uses the existing
data-side `genesis-shell-contact` and `rigid-stress-physical` environments without
modifying either. Original accepted evidence and unrelated project outputs are
read without migration. Selected compact publication files are linked from
[the comparison report](research/rigid_stress/docs/SHELL_COMPARISON_RESULTS_20261018.md)
and stored in `research/rigid_stress/evidence/20261018-shell-comparison/comparison-v1/`.
Full raw traces, runtime arrays and isolated source/figure capture staging remain
on data, including `runs/publication-capture-shell-v2/`; no active writer was moved.

# Current pressure/intrinsic comparison

The pressure/configuration comparison uses
`runs/20261017-pressure-comparison/` under the established data root below.
It contains immutable current899-file native snapshots, small reference and
orchestration source captures, paired four-scope scaling, separately sampled
memory/capacity, moving traces, current R/L/M/convex/narrow-gap fields and
complete-grasp reward diagnostics. Each process has explicit data-side
output/tmp/cache/log paths. The frozen eight-condition bank and previous
independent reference packets are read without migration or rewriting.
Selected compact results/figure sources are linked from the comparison
report; full raw measurements and arrays remain under this data-side run root.
PDF export QA additionally uses the isolated data-side asset
`assets/pymupdf-1.26.5/`; its wheel/install receipt is under the comparison run's
`pdf-qa-install-v1/`. The active scientific environments are not modified.

# Current checkout and archived publications

The 2026-10-17 branch cleanup retains seven canonical current evidence/source
packets in the checkout. Historical publications and one-off research files
listed below are recoverable at Git revision
`1606ee3a9e9628a935ad5e446984f920af3adc66`; their former paths are historical
provenance. See [the archive index](research/rigid_stress/archive/README.md)
for exact retrieval and isolated sparse-worktree commands.
Canonical retained packet bytes and SHA256 manifests were preserved.
No runtime data, dependency environment, compatibility symlink or other
worktree was migrated. Git history is retained without rewriting.

Cleanup metadata and bounded verification outputs use
`runs/20261017-repository-organization/` under the established data root below.
New benchmark snapshots include the stable `NATIVE_API.md` contract in place
of the superseded application-stage goal. Exact historical899-file recipes
continue to restore their original files and identities unchanged.

# Runtime data locations

The unselected `research/rigid_stress/evidence/20261014-physical/five-level-thickness-v1`
publication contained redundant direct copies of the same ten large fields
already represented as exact byte chunks. Its publication process was terminal
before copying it with metadata and symlink preservation to
`runs/20261014-physical-validation/publication-attempt-five-level-thickness-v1-unbounded-extras`
under the data root below. All192 files (3312301696 bytes) were SHA256-verified
before removing only that untracked publication from the checkout. Original
scientific runs, field arrays and publication staging remain intact. The selected
`five-level-thickness-v2` avoids redundant direct extras and retains every original
field through exact reconstructed chunks. The complete old/new paths, active-job
check, per-file hashes and old publisher source are retained in
`runs/20261014-physical-validation/meta/thickness-publication-large-extra-correction-v1.json`.

The rigid stress recovery worktree uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery` for runs,
generated collision assets, video, numerical replays and caches. Its optional
environment is `/mnt/data/zhaofeng/venvs/rigid-stress-recovery`.

`research/rigid_stress/scripts/cluster_env.sh` sets these locations explicitly.
Override `RIGID_STRESS_DATA_ROOT` and `RIGID_STRESS_ENV` for another machine.
Dependencies reuse `/mnt/data/zhaofeng/pip_cache`.

The practical large-batch throughput stage uses
`runs/20261017-practical-throughput/` for immutable source snapshots, Slurm
submission records, matched scope measurements, intrusive traces, targeted
tests and metadata summaries. It reuses the eight-condition bank and the
qualified native application snapshot under
`runs/20261014-physical-validation/`; neither input is migrated or rewritten.
Per-job stdout/stderr, temporary files, CUDA/Genesis/Quadrants/Numba caches and
test artifacts are explicitly placed under the data root. The scientific
environment remains `/mnt/data/zhaofeng/venvs/rigid-stress-physical`.

The compact baseline scaling figure packet is tracked under
`research/rigid_stress/evidence/20261017-throughput/baseline-scaling-v1/`
and the expanded closed-width update `baseline-scaling-v2/` alongside it.
It contains editable/exported figures, source CSV, exact plotting inputs/source
and validation records. Full traces, raw candidate arrays, failed checker
versions and long-running capacity experiments remain in the data-side run
root above. No runtime output or dependency cache was moved into the checkout.

The final selected figure packets `selected-history4-scaling-v1`,
`selected-history0-scaling-v1` and `baseline-selected-comparison-v1` live beside
the baseline packets. Together their payloads are about2MB, including exact
compressed metadata inputs, executed source/style/orchestration, CSV, editable
SVG/PDF/PNG and accepted visual/source QA. Original final rendering and small
publication staging remain under the same data-side run root in
`final-throughput-figure-preview-v2`, `final-throughput-comparison-preview-v3`,
`final-throughput-pdf-v1` and `publication-final-figures-v1`. Large raw benchmark
JSON, traces, arrays, caches and failed launch/source versions remain on data.
No active directory was migrated and compatibility links were preserved.

Small source, configuration, documentation and selected published evidence
remain in this worktree. No existing runtime directories were migrated or
replaced, and compatibility symlinks remain in place. Raw run directories are
listed in `research/rigid_stress/docs/PROGRESS_20261009.md`.

The nonconvex extension uses `runs/20261011-nonconvex/` for its original
qualifications, matched scopes, profiles, scaling and capacity trials.
Final regression, output-mode comparisons, analysis, archive staging and
verification use `runs/20261014-closeout/`. These paths are relative to the
data root above. The selected immutable publication is
`research/rigid_stress/evidence/20261011-nonconvex/`, including
`native-nonconvex-closeout-20261014-v1`. Complete raw JSON and caches stay on
the data mount. Published large JSON uses ordered compressed byte ranges
with full-file reconstruction hashes.

The new load-model/physical-validation stage uses
`runs/20261014-physical-validation/` for independent FEM qualifications,
mesh assets, fine sweeps, reference setup and failed attempts. Its optional
reference environment is `/mnt/data/zhaofeng/venvs/rigid-stress-physical`,
configured through `PHYSICAL_VALIDATION_ENV`. It inherits the existing
validated base site's dependencies via a recorded `.pth` and adds pinned
reference packages without migrating or editing that base environment.

The load-model matrix, refined shell, friction controls and publication tools
remain under `runs/20261014-physical-validation/`. Selected immutable
deliverables are `load-model-checkpoint-v2`, `load-refinement-figures-v1` and
`egg-solver-comparison-v1` under `research/rigid_stress/evidence/20261014-physical/`.
Runtime caches and long job logs continue to use the data mount.

The unselected initial publication at
`research/rigid_stress/evidence/20261014-physical/load-model-checkpoint-5` was
copied with metadata/symlink preservation to
`runs/20261014-physical-validation/publication-attempt-load-model-checkpoint-5`
under the data root. Its writer job712467 was terminalCOMPLETED before copying.
All4321 files were SHA256-verified before removing the untracked source copy.
No tracked deliverable or active writer was moved. Exact old/new absolute paths
and validation are recorded in that run root's `publication-attempt-migration.json`.

The `native-capacity8192-v1` packet retains164 compact payloads totaling1,291,101 bytes for terminal B2048/B8192 timing, complete conformance and separate memory. Full arrays and large raw benchmark JSON stay on data. The public-source overlay packet adds3 payloads/45,790 bytes and reconstructs858 exact source files from a public commit without relying on a private Git object. Temporary source validation was removed after verification; no active source or data was moved.

The publication-index repair explicitly tracks169 previously ignored, manifest-listed tiny logs (24,922 bytes) in the20261019 qualified-ranking evidence. Their original bytes, existing manifests and hashes are preserved. Final verification reads all5,865 manifest-listed payloads directly from the Git tree, including encoded and recorded decoded hashes; raw datasets and active jobs remain on data. The repair inventory and portable reader are in `research/rigid_stress/evidence/20261019-qualified-ranking/publication-index-repair-v1/`.

Native observation-cadence pilots, paired primary/conformance runs and their
complete GPU-recorded control arrays use the same data-side ranking run root in
`native-sparse-observation-pilot-v1` and `native-observation-cadence-*`. Guarded
fine-field plans and explicit parent-admission records remain there as well;
only source, documentation and selected compact immutable receipts are published.
Cancelled pending analysis plans are retained in place, with no live source,
reference data, environment or cache migration.

The selected `native-observation-cadence-pilot-v1` and
`fine-parent-admission-guard-v1` publications retain66/84 exact compact payloads
(390,160/502,986 bytes). Two pilot trajectory NPZs remain on data with explicit
hashes; new fine and paired-cadence outputs are excluded while active.

The terminal `native-observation-cadence-v1` packet adds312 compact payloads
(879,107 bytes) for26 source-matched primary/control cases and the521-check
audit. All26 complete trajectory NPZs stay on data with exact hashes. The
spatial-parent dependency update is retained as an explicit scheduler journal;
no captured worker source, live reference output or previous archive is changed.

Friction-reference gate controls, explicit seven-parent qualification, subsequent
DPR/SCR/Shell method fields and force-figure data use fresh
`friction-reference-gate-controls-v1`, `canonical-friction-reference-qualification-v1`,
`canonical-stvk-method-fields-v1` and `canonical-friction-plot-data-v1` directories
under the ranking run root. CPU extraction validation uses
`friction-plot-data-cpu-control-v1`. Complete arrays and journals remain there;
only source, configuration and selected immutable evidence enter Git.

The `friction-qualification-pipeline-v1` publication retains92 compact payloads
(503,395 bytes). Eight CPU-control traction NPZs and one complete history CSV
remain on data with explicit hashes. Live fine-reference results and future
qualification, method fields and final figure data are excluded from this
immutable checkpoint.

Complete-array error-reader controls and candidate admission use
`stress-error-measures-controls-v1`, `pad-candidate-admission-controls-v1` and
`stvk-ranking-admission-negative-v1` under the ranking run root. Replacement
`canonical-stvk-method-fields-v2` retains full q9/q25 surface arrays as well as
resultants/column integrals. Final pending audit uses
`canonical-stvk-method-ranking-v1`. All full arrays remain on data; cancelled
pending v1 plans, actual failures and immutable source snapshots are preserved.

The `stvk-ranking-reader-controls-v1` publication retains83 compact payloads
(492,556 bytes). Full volume-column, resultant/surface, raw-field and legacy
control arrays remain on data with their exact hashes in the receipts. Live
fine and pending final ranking/figure outputs are excluded; no existing
source, array, environment, compatibility link or shared cache is migrated.


Renderer controls use `stvk-figure-render-controls-v1` through `v4` under the
20261019 ranking data root. The first three failures, final15-check/four-preview
success and negative `stvk-final-figure-admission-negative-v1` stay in place.
`canonical-stvk-figures-v1` captures the queued matching fine-reference renderer;
no live fine output is moved or frozen. Full plotted arrays, CSV and SVG/PDF/PNG
exports stay on data. Small selected previews, source, QA and exact external
hashes may be retained as repository deliverables; node-local private caches
and original engine sources/environments/compatibility links are preserved.


Matched normal error/cost consolidation uses `pad-cost-accuracy-controls-v1`
under the ranking data root (478 checks,24 configurations,120 applicable metric
associations); its negative friction input writes no CSV. Final guarded friction
work uses `pad-friction-cost-accuracy-v1`. Full original fields, histories,
source trees and sampled-memory data stay in place; compact CSV/metadata/source
receipts may be published. The read-only constructor provenance audit and denied
half-dt walltime-extension journal are under `meta/`. No active data, runtime,
environment or compatibility path is migrated.


Terminal-endpoint continuation uses `solid-pad-resume-preparation-v1`, the
actual `solid-pad-resume-live-negative-v1`, `solid-pad-resume-coarse-control-v1`,
`solid-pad-resume-input-controls-v2` and `solid-pad-resume-equivalence-v1` under
the ranking data root. Original parent fields/journals stay untouched. New
endpoint copies and new segments stay on data, with exact saved-prefix lineage;
only small source/proof/control metadata is selected for publication. No live
fine task is resumed, relocated or hot-edited. The new segment's elapsed time
is explicitly separated from inherited physical history.


Resolved-pressure native controls, the400-step trajectory and complete-column
comparisons use `distributed-pad-*` directories under the20261019 ranking data
root. Source-roundoff controls, both1000-step trajectories, independent audits
and wrench-attribution variants use `pad-source-roundoff-cpu-*`,
`pad-roundoff-every-step-v1`, `dpr-roundoff-admission-v1`,
`scr-roundoff-admission-v1` and `native-wrench-attribution-*`. Two new relative
data-side audit aliases preserve the existing primary auditor convention;
their paths are recorded in `meta/roundoff-admission-plans-v1.json`. No existing
outputs are relocated. Three terminal publication packets retain554 compact
payloads (3,638,756 bytes) and78 externally hashed arrays, including the completed
main/contact125um reference checkpoints. Earlier archives and captured sources
remain immutable; all full fields, trajectories, caches and logs stay on data.


The completed fixed-radius friction chain remains under its existing
`solid-pad-stvk-canonical125-halfdt-resume-v1`, resilient spatial/control/field,
qualification/ranking/figure and matching-cost directories in the20261019 ranking
run root. `qualified-friction-final-v1` retains519 compact/export payloads
(14,050,058 bytes) and152 externally hashed arrays. Selected final SVG/PDF/PNG
and CSV deliverables may remain in Git; all complete fields and original logs
stay on data. Visual review is a separate immutable companion receipt.
The new CPU compliance study uses `pad-compliance-study-v1` and explicit
`meta/pad-compliance-cases-v1.json`; it reuses the existing data-side operator
and environment and writes new operators/fields/logs only in that campaign.


The completed compliance study/control/figure campaigns use
`pad-compliance-study-v1`, `pad-compliance-controls-v1` and
`pad-compliance-figure-v1`. Their terminal packet retains91 selected payloads
(3,764,739 bytes), including reviewed pressure-map exports, and58 externally
hashed arrays. Actual undersized-search states stay alongside the accepted
expanded states. The cached834-node operator is reused in place; no original
data, environment or compatibility link is migrated.

The resolved-boundary reference comparison and source-local engineering study
use `resolved-pad-qualified-reference-v1`, `resolved-pad-ranking-v1` and the
explicit `nearby-scr-*` campaigns under the established20261019 ranking run root.
Their `engineering-load-sensitivity-v1` packet retains532 compact payloads
(2,909,373 bytes) and38 externally hashed arrays. Earlier failed controls,
underresolved support sampling and rejected optimizer results remain in place.
Successful finer cases are reused by explicit paths; no aliases hide a failed
parent. The corrected signed decomposition was excluded while live and is now
published separately with the cooperative-load study. Full stress/source
arrays, job logs and private node-local caches remain on data; no data or
active environment is migrated.

`cooperative-load-and-transfer-decomposition-v1` freezes the completed
`resolved-pad-decomposition-v2` and its summary, centroid/group and improved
port-path SCR budget variants, solver-accuracy and same-path local controls,
independent comparison and inspected figure campaigns. All originate under
the established20261019 ranking run root. Failed `AlmostSolved` attempts and
the comparison-v2 metadata-key failure remain retained alongside the successful
v3 reader. The historical moment-anchor metadata wording is corrected in the
reader/report without altering frozen observations. Compact evidence and
publication figures stay in Git:499 payloads total12,408,095 bytes. The98
externally hashed arrays remain on data.

The `native-resolved-boundary-v1` packet adds126 compact payloads
(525,480 bytes) and8 externally hashed arrays for the production
resolved-boundary transfer. The isolated engine source lives in
`.worktrees/resolved-boundary-production`; its original baseline remains
unchanged. GPU/CPU virtual-work and lifecycle tests, manual-pressure
regressions, actual normal/friction field comparisons and earlier failed test
setups are retained. Test fixtures and private caches stayed in each job's
node-local scratch; durable receipts/fields stay under the existing data run.
The separate trajectory was excluded while running. Its now-completed
`native-resolved-history-v1` companion packet retains70 compact payloads
(612,228 bytes) and11 externally hashed arrays for all400 observations,
independent mechanical/source admission and the endpoint/every-step comparison.
Intermediate correction magnitudes are recorded, but complete original source
arrays were captured only at endpoint keyframes. No new intermediate-event
stress bound is inferred from absent arrays.

The completed reduced pad, PCG forcing and independent comparison campaigns
use `reduced-pad-equilibrium-v1`, `reduced-pad-pcg-v1` and
`reduced-pad-comparison-v1` under the same data run root. Their
`reduced-pad-equilibrium-v1` packet keeps89 compact payloads (536,562 bytes)
and78 externally hashed arrays, including48 new complete fields and12
load/tilt control captures. Existing physical reference fields and the
834-node compliance are reused in place. Job logs and generated arrays stayed
on data; CPU process caches stayed node-local. No production environment,
original reference or live source was migrated or overwritten.

The native pressure-stage and physical thickness campaigns use
`native-pad-newton-{inputs,fields}-v1`, `native-pad-newton-v1`,
`pad-thickness-study-v1`, `native-pad-thickness-{inputs,fields}-v1`,
`native-pad-thickness-v1`, `pad-thickness-figure-draft-v1` and `pad-thickness-figure-v1`
outputs under the same ranking run root. Their
`native-pad-pressure-and-thickness-v1` packet retains301 selected payloads
(4,900,765 bytes) and120 externally hashed arrays. New meshes,834-node
compliances,48 native pressure/full-field checks and24 physical thickness
fields remain on data. Inspected SVG/PDF/PNG publication deliverables stay in
Git with their compact records. Original source checkouts, active environments,
historical arrays and compatibility links remain in place.

`native-pad-compliance-v1`, `native-pad-basis-v1` and
`native-pad-basis-fields-v1` add the native operator/factor/cold-start controls
under that same run root. The `native-pad-preparation-v1` packet retains140
compact payloads (481,066 bytes) and98 externally hashed arrays, including48
new full fields and the small frame/material operator controls. Large physical
compliances and source maps are reused in place. All new numerical arrays and
job logs stay on data, with private process caches on the compute nodes.

`native-pair-geometry-tests-v1/v2/v3` retain the current-geometry component
controls under the same run root. The `native-pair-geometry-v1` packet contains88
compact payloads (468,439 bytes): terminal CPU/GPU outcomes, both earlier setup
failures, exact engine/test sources and source-bound receipts. The source
checkout is isolated at `.worktrees/automatic-pad-geometry`. Durable job logs
stay on data; generated test assets and private compilation caches use
compute-node scratch. No existing environment, data or compatibility link
was moved.

`native-pair-accessible-baseline-v1` and `native-pair-accessible-tests-v1/v2`
retain the extended visibility controls under the same ranking run root.
Their `native-pair-accessible-v1` packet contains79 compact payloads
(486,468 bytes), including the actual baseline miss, the first mesh-candidate
failure and ten final passing CPU/GPU cases. The source remains isolated in
`.worktrees/automatic-pad-geometry`; test assets and private compiler caches
use node scratch, and durable outputs remain on data. No migration is performed.


Cheap hard-contact estimates use `hard-hertz-v1`, `hard-hertz-interaction-v1`
and `hard-hertz-figure-v1` under the established20261019 ranking data run root.
Complete BEM pressures, sampled tensors and native comparison arrays stay on
data. Private caches use node scratch. Only concise captured source/receipts
and inspected SVG/PDF/PNG publication figures enter the `hard-hertz-v1` packet.
The source checkouts and existing environments/symlinks remain in place; no
new soft-pad campaign or data migration is performed.


Finite Hertz footprint transfer uses `hard-hertz-transfer-v1` through `v4`
and `hard-hertz-wrench-v1`/`v2` under the same20261019 ranking data root. Original
import/geometry failures remain intact. Complete meshes, mapped samples and
full independent/native tensor arrays stay on data with exact hashes; compact
executed sources/receipts form the `hard-hertz-transfer-v1` publication packet.
The existing `rigid-stress-physical` environment is reused unchanged; private
caches remain node-local. No pad campaign, shared configuration or data migration
is introduced.

Normal near-field tensor studies use hard-hertz-nearfield-v1 through v3 and
hard-hertz-native-stress-v1 through v5 under the same20261019 ranking data root.
Full pressure-integral and native/reference tensor arrays stay on data. The
hard-hertz-stress-v1 packet retains compact captured sources, configuration,
original reference/initialization failures and terminal receipts. Existing
coarse-shell arrays are reused by hash, with no migration or duplication of
large data in home. Private caches remain compute-node local.

Finite-box particular/residual controls use `hard-hertz-residual-v1` through
`v3`, and native combined-field controls use `hard-hertz-box-v1` through `v4`
under the same20261019 ranking data root. Complete meshes, pressures, nodal
loads/displacements, volume-weighted tensors and peak probes remain on data.
The two initial8GB failures and native initialization/reference-comparison
failures are preserved beside final results. The compact `hard-hertz-residual-v1`
publication packet captures executed sources, logs and hashes; no runtime
dataset or environment is relocated into home. Private caches stay node-local.


Hard-contact input profiling and optimization use `hard-input-profile-v1`,
`hard-input-grouping-v1` and `hard-input-precomputed-v1` through `v3` beneath the existing
20261019 ranking data run. Complete trajectories, source snapshots and generated
control meshes remain there, with private compiler caches on compute-node scratch.
The isolated source checkout is `.worktrees/hard_contact_optimized`; the compact
`hard-input-optimization-v1` packet retains executed sources, terminal receipts,
compressed audit histories and the fixture/numerical failures. No live data,
existing environment or compatibility link was moved. Unfinished friction-history
sources/results remain preserved separately while the remote input-overhead
priority is executed.


Contact-input cache/scaling ablations use `hard-input-cache-v1/v2`,
`hard-input-trace-v1/v2`, `hard-input-scaling-v1`,
`hard-input-scaling-baseline-v1`, `hard-input-scaling-check-v1`,
`hard-input-bounds-v1`, `hard-input-bounds-trace-v1` and
`hard-input-bounds-paired-v1` under the established20261019 ranking run.
Complete trajectories, large compressed histories, expanded snapshot summaries,
raw CUDA traces and generated meshes stay there with external hashes. The
`hard-input-cache-and-scaling-v1` packet keeps compact terminal/source evidence,
including both initial trace-writer failures. Candidate source worktrees remain
under the existing workspace: `hard_contact_optimized` holds the cache experiment
and `hard_contact_bounds` holds the selected small change. Private compilation
caches are node-local; the data-side physical environment is reused. No existing
runtime data, active environment, compatibility link or preserved unfinished
friction-history source is migrated.

Inactive-sample bandwidth controls use `hard-scene-masked-samples-v1`,
`hard-scene-sample-bandwidth-v1` and
`hard-scene-sample-bandwidth-reinspection-v1` beneath the established20261019
ranking run. Complete trajectories, meshes, sampled fields and histories remain
on data. The `hard-transfer-sample-bandwidth-v1` packet retains194 compact
payloads (2,046,589 bytes) and204 external-array receipts, including the final
paired allocation's preempted state and the separate read-only full comparison.
No original outputs or returncode markers are rewritten. Private compilation
caches use node scratch; the preempted allocation's cleanup is not asserted.
Source, small receipts and documentation remain in the workspace. No existing
environment, live data or compatibility link is migrated.

Independent hollow-shell pressure work uses `hard-shell-contact-pressure-v1`
through `v4`, `hard-shell-contact-pressure-inputs-v1/v2`,
`hard-shell-contact-pressure-refinement-v1`,
`hard-shell-contact-pressure-grid97-v1`,
`hard-shell-contact-pressure-grid97-refinement-v1`,
`hard-shell-contact-pressure-inspection-v1` and `hard-shell-contact-kernel-v1`
under the same20261019 ranking run. Existing fixed meshes are reused in place.
Complete normal-response tables, pressures/forces/gaps and the large failed
coordinate diagnostic remain on data. The `hard-shell-contact-pressure-v1`
publication packet keeps279 compact payloads (1,641,121 bytes) with13 external
data/mesh receipts, including original failures and independent reductions.
All scientific work uses Slurm `rtx-mid`, image `1_26`, the existing data-side
physical environment and explicit output/private cache paths. No environment,
live dataset or compatibility link is moved; no runtime coupled solve is added.

Independent-pressure full stress work uses `hard-bilinear-stress-kernel-v1`,
`hard-shell-pressure-stress-v1`, its two saved-array inspections and
`hard-shell-pressure-stress-figure-v1` through `v4` beneath the same ranking
data root. Complete bilinear particular fields, P2 readouts, element corners and
queries remain on data. The `hard-shell-pressure-stress-v1` packet keeps175
compact/export payloads (3,929,274 bytes) and28 external-array receipts,
including both actual figure-QA failures and reviewed SVG/PDF/RGB PNG exports.
Completed normal-response tables and meshes are reused in place.

Cone-preserving friction work uses `hard-cone-history-v1` through `v5`,
`hard-native-cone-history-v1` through `v3`, `hard-native-cone-vectors-v1`,
`hard-cone-penalty-paired-v1`, `hard-cone-warp-paired-v1` through `v4`,
`hard-cone-warp-inspection-v1` and `hard-cone-baseline-final-v1` under that data
root. The `hard-cone-history-native-v1` packet keeps450 compact payloads
(2,725,219 bytes) and161 externally hashed history/benchmark/input arrays.
Failed solver/compile/test attempts remain alongside final controls. The
original four unfinished radial-history source files remain unchanged. All
scientific execution uses the existing Slurm/image/environment and private
node scratch. No live data, environment or compatibility link is migrated.

Native normal recovery uses `hard-native-hertz-recovery-v1` through `v5`,
the surface-depth/rotation inspections, `hard-native-hertz-farfield-controls-v3`
through `v5`, paired timing `v3` through `v5`, the far-kernel control, two final
saved-array inspections, component regressions and refusal `v1` through `v4`
under the same ranking data root. The `hard-native-hertz-recovery-v1` packet
keeps 798 compact payloads (4,066,408 bytes) and 182 external-array receipts.
It retains actual failed jobs, unsubmitted preparation failures, captured
workers and every historical source patch. Source replay verifies every
captured engine file; the final public experimental source is `67c9de0b2`.
Complete native/reference fields, meshes and per-frame benchmark audits stay
on data. Source checkouts `hard_contact_enrichment` and `hard_contact_farfield`
remain in the workspace. All scientific execution uses Slurm `rtx-mid`, image
`1_26`, the existing data-side environment and private node scratch. No existing
dataset, environment or compatibility link is moved.

Moving normal material probes use `hard-native-hertz-moving-probes-v1` through
`v5`, `hard-native-hertz-moving-roundoff-v1`, paired cost `v1/v2` and saved-array
inspection `v1/v2` under that data root. The `hard-native-hertz-moving-probes-v1`
packet retains 399 compact payloads (1,965,299 bytes) and 285 external-array
receipts. It preserves failed rotation/quadrature attempts and both strict
corner-accuracy inspection failures, separately recording the preexisting
engineering budgets. Complete moving fields, independent references, meshes
and benchmark source/peak/status audits remain on data. The fixed experimental
checkout is `.worktrees/hard_contact_moving_probes`, public source `a599b9c50`.
Focused test sources and results are hash-bound to that source. Scientific work
uses the established Slurm/image/environment and private node scratch; no live
data, existing environment or compatibility link is moved.

Actual native normal lifecycle work uses `hard-native-hertz-live-normal-v1`
through `v3`, live lifecycle `v1` through `v4`, saved-array inspection `v1/v2`,
failure inspection `v1/v2`, public/transfer regression `v1`, walk-roundoff
controls and CPU/GPU physical-clock controls under the same ranking data root.
Actual paired scene cost, final default checks and intrusive original-producer
traces use `hard-native-hertz-live-timing-v1`, `hard-native-hertz-live-default-v1`
and `hard-native-hertz-live-trace-v1`. The publication packet
`hard-native-hertz-live-normal-v1` retains 659 compact payloads (3,733,019 bytes)
and 253 externally hashed arrays/traces, including every original failed job
and primary/override source inventory. Native/reference fields, full benchmark
audits and CUDA traces remain on data. The workspace checkout
`hard_contact_live_normal` is pinned at public experimental source `fc63cf32e`;
`hard_contact_local_clock` preserves the clean negative baseline `69f89c064`.
Scientific work uses the existing data-side physical environment, Slurm
`rtx-mid`, image `1_26` and private node scratch. No runtime data, environment,
compatibility link or protected unfinished source is migrated.

Shared source-bound controls use `hard-source-bounds-controls-v2`, paired
source orders `hard-source-bounds-paired-v2/v3`, separate device tracing `v2`
and saved-array inspectors under the same ranking data root. Their unsubmitted
`v1` preparation directories remain intact. Independent vector-friction work
uses `hard-tangential-contact-pilot-v1/v2`, five matched studies in
`hard-tangential-contact-reference-v1` and failed/corrected read-only inspections.
The `hard-native-source-bounds-and-friction-reference-v1` publication packet
retains 445 compact payloads (2,303,663 bytes) and 154 external-array/trace
receipts. Full cell histories, independent/candidate stress fields, actual
scene audits and CUDA traces stay on data. The isolated source checkout
`hard_contact_bounded_sources` is pinned at public alias `5a28cf837`, with
the earlier `fc63cf32e` source unchanged. Scientific work uses the existing
Slurm/image/data-side environment and private node scratch; paired source
processes also use separate private compiler caches. No runtime dataset,
active environment, compatibility link or protected unfinished source is moved.

Native changing-force friction uses `hard-native-friction-*` campaigns under
the same20261019 ranking data run. The `hard-native-friction-live-v1` packet
retains623 compact payloads (3,756,502 bytes) and398 external array/trace
receipts. It includes failed worker and exact-switch controls, the cancelled
pending speed attempt, full-field checks and the completed source-paired timing.
Complete physical histories, scene fields, meshes and CUDA traces remain on
data. Workspace sources `hard_contact_friction_core`, `hard_contact_live_friction`
and `hard_contact_friction_moments` are frozen at07bf46ce8,795805290 and6ffdd399b.
Scientific execution reuses Slurm `rtx-mid`, image `1_26`, the data-side physical
environment and private node scratch. The original12 unfinished files and all
16 frozen source heads remain preserved. No data, environment or link is moved.

The actual Franka admission probe and separate level1 shell controls use
`hard-native-friction-franka-admission-v1` and
`hard-native-hertz-shell-admission-v1` through `v4` under the same data root.
Their `hard-native-hertz-shell-admission-v1` packet retains201 compact payloads
(985,120 bytes) and24 external-array receipts. Complete refused Franka frames,
shell fields and normal-component references stay on data. Source
`hard_contact_shell_recovery` is frozen atd26b5b3d4, with the original16 heads
and12 unfinished files preserved. The existing Slurm/image/environment and
private node caches are reused. No runtime data or environment is migrated.

Complete cavity-boundary controls use `hard-native-hertz-shell-quadrature-v1`
through `v3`, `hard-native-hertz-complete-boundary-v1/v2` and its inspection
under that same data root. The `hard-native-hertz-complete-boundary-v1` packet
retains193 compact payloads (1,169,923 bytes) and52 external-array receipts.
Complete independent/native loads, corner fields and probes remain on data.
The new isolated source `hard_contact_complete_boundary` is frozen at2e39f5f75.
The asset's selected outer faces remain unchanged in Git. Native Hertz appends
omitted material faces during preprocessing. All18 frozen source heads and
the original12 unfinished files remain preserved; scientific execution reuses
the established Slurm/image/environment and private node scratch.

Sharp-edge collision application provenance, division32 collision assets and
updated-support load studies use `hard-contact-application-*` under the same
20261019 ranking run. Their `hard-contact-application-v1` publication packet
retains compact captures, source inventories and terminal jobs, including all
three native Franka refusals. Full frame/contact/field arrays and generated
OBJ/URDF assets remain at hashed data paths. Source `hard_contact_edge_farfield`
stays frozen at4a8125c4c; no engine is changed by this study. All29 frozen source
heads and12 original unfinished files remain preserved. Scientific runs reuse
Slurm `rtx-mid`, image `1_26`, the existing data-side environment and private
node caches. No runtime dataset, environment or compatibility link is migrated.

Native edge geometry and constrained force-allocation controls use
`hard-native-edge-geometry-v1` and `hard-native-force-allocation-v1/v2` under
the same data run. The `hard-native-edge-geometry-v1` packet keeps compact
source/terminal records and external array identities. Rotated mesh controls,
complete numerical arrays and component timing outputs stay under data. The
isolated `hard_contact_edge_geometry` source is frozen at6c2e887be, with its
geometry-only ancestor f57245943 retained. All30 frozen heads and12 original
unfinished files are preserved; Slurm/image/environment/cache policy is unchanged.

Native hard-input binding and complete-field composition use
`hard-native-hard-binding-v1/v2/v3`, `hard-native-hard-recovery-v3/v4/v5` and
`hard-native-hard-stability-v1/v2` under the same data run. The
`hard-native-hard-composition-v1` packet keeps compact sources, terminal records,
failures and hashed external arrays. All full fields, raw captures and logs
remain under data. The isolated binding/recovery/stability sources are frozen
at58c18847e/edcffa58b/8324e7d53. All33 frozen heads and12 original unfinished
files are preserved. Slurm `rtx-mid`, image `1_26`, the existing data-side Python
environment and private node caches remain in use. There are no data migrations.

Matched CPU twelve-feature GPU parity work uses `20261026-cpu12-gpu-parity`
under the established rigid-stress-recovery data run directory. Frozen physical
inputs, independent CPU numerical oracles, complete GPU fields and paired timing
logs remain there; only source, schemas and concise evidence enter Git. The
initial receipt verifies the original twelve unfinished files and twenty-three
prior frozen experimental engines. Scientific execution uses Slurm `rtx-mid`,
image `1_26`, the existing data-side environment and private node caches. No
existing data, environment or compatibility link is migrated.

The completed CPU12 packet is `research/rigid_stress/evidence/20261026-cpu12-gpu-parity-v1/`
(10,476,640 bytes, under the 10 MiB publication limit). It retains compact
terminal/source receipts, raw paired timing rows and external array SHA256s for
24 campaigns; full fields, own-process memory traces, oracle matrices and logs
remain at the data run above. Final preservation checks verify the original
12 files, 23 old engines and 11 campaign source inventories. There were no
runtime-data, environment or compatibility-link migrations.
