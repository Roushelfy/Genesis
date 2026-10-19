# Current QIPC reference comparison

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
