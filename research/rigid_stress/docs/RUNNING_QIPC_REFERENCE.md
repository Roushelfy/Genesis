# Run the three-method QIPC reference comparison

The [goal](QIPC_REFERENCE_COMPARISON_GOAL.md) defines the required deliverable;
the [progress record](QIPC_REFERENCE_PROGRESS.md) distinguishes completed controls
from failed or unqualified reference attempts. This is an ongoing comparison.
The [working report](QIPC_REFERENCE_RESULTS_20261018.md) and
[metric contract](QIPC_REFERENCE_METRICS.md) retain the complete-history scores
and unresolved reference uncertainty.

**Storage incident,2026-10-18:** `/mnt/data` is currently unwritable with
`OSError122: Disk quota exceeded`. The explicitly disclosed bounded temporary
destination in [DATA_LOCATION.md](../../../DATA_LOCATION.md) passed exclusive
create/write/fsync; new selected campaigns use that root with node-local
compilation caches. Preserve interrupted campaigns and use fresh output
directories. Exact
affected runs/checkpoints are in the
[inventory](QIPC_STORAGE_AFFECTED_RUNS_20261018.json). Hash-verified native dumps
are not validated cross-process restarts; the grasp runner has no resume option.
Trajectory snapshots are not native contact/friction checkpoints.
The [validation tasks](QIPC_PENDING_VALIDATION_20261018.md) executed in
job726497 with all eight groups/nine tasks returning0. The
[fallback execution record](QIPC_REFERENCE_FALLBACK_20261018.md) separates those
passing observer/admission/audit controls from running and unqualified physics.

## Source, runtime and storage

Use a fresh run under the task's established data directory:

```bash
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery
task_run="$task_root/runs/20261018-qipc-reference"
```

Keep scientific execution, tests, analysis and rendering inside Slurm
`rtx-mid` with `GENESIS_IMAGE_VER=1_26 gs-srun`. The stdlib-only submitter captures
analysis sources and source/input identities, writes explicit data-side caches,
temporary files and logs, and submits GPU task groups. Do not edit a frozen
engine checkout while its jobs import it. Do not change shared environments.
For an allocation with sufficient node-local space, `--node-local-cache` creates
a private `/tmp/gs-shell-compare.XXXXXX` for compilation/model/tmp caches and
removes only that directory at allocation exit. Persistent logs, results and
source receipts stay under `--output`; default cache behavior is unchanged.

| Role | Frozen source or identity |
| --- | --- |
| Original QIPC main | `fe252bd30acc6c717507b74add31512812f48761`; sibling `qipc-stress-reference-original` |
| QIPC capped-robot integration | `641459b8826bb061e1bba01dfe9eba39793c3855`; sibling `qipc-stress-reference-integration` |
| Later capped integration | `40275f0a26f0318659000c00191760fe83d9414c`; sibling `qipc-stress-reference-current`; separate native receipt |
| Capped numerical diagnostic | Same40275 source plus [KKT roundoff patch](../qipc_comparison/kkt-roundoff-diagnostic.patch); sibling `qipc-stress-reference-capped-roundoff`; actual checkpointed and case7 sources |
| Original QIPC numerical diagnostic | Same original main plus [the recorded patch](../qipc_comparison/line-search-roundoff-diagnostic.patch); sibling `qipc-stress-reference-roundoff` |
| Native capped validation | `9bd1086ca`; sibling `rigid-stress-qipc-validation` |
| Native prescribed example | `ad6b3e2da`; sibling `rigid-stress-qipc-prescribed`; optional scene timestep, original default preserved |
| Shell original | `fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`; sibling `kashu-shell-pr22` |
| Shell read-only diagnostics | Same pinned shell plus [the diagnostic patch](../shell_comparison/pr22-diagnostics.patch); sibling `kashu-shell-pr22-diagnostics` |

QIPC uses the existing data-side `qipc-owned-coarse-tree-check01-20261015`
Python3.12 environment. Native and shell share `genesis-shell-contact`; independent
CPU elasticity and MuJoCo FK use `rigid-stress-physical`. Actual executable,
Python/Torch/CUDA versions, hostname, Slurm identity and visible GPU are recorded
in each new scientific worker. Do not infer the container Python version from
the host executable alone.

The source/build campaigns preserve exact submodule versions, compiler flags,
native hashes and import tests. Original main solver SHA256 is
`cebd257411c97a01e5d0c78ca8d5fb6271955c9b3b5c20ce3e53dab95e175cf1`;
the capped integration library is
`8763018e8ca49e2b7c156cc5cdcf42318b03dfd0363618c3b824daaca39e1213`.
These are binary hashes, not Git revisions. A shared build directory may later
advance; reuse the pinned artifact and its receipts, never a similarly named
current library without checking its identity.

## Task submission

From this task checkout, create a data-side JSON list of independent groups.
Each group is a sequential list of module/argument tasks, for example:

```json
[[{
  "name": "reference",
  "module": "research.rigid_stress.qipc_comparison.prescribed_reference",
  "args": ["--motion", "MOTION.npz", "--conditions", "CASE.npz",
           "--mjcf", "PANDA.xml", "--mesh", "HOLLOW-P1.npz",
           "--duration", "6", "--dt", "0.001",
           "--velocity-tolerance", "1e-9", "--linear-tolerance", "1e-8",
           "--dbc-stiffness", "1e7",
           "--keyframes", "1.33", "2.77", "4.02", "4.38", "5.52"]
}]]
```

Use actual absolute paths in place of the capitalized input names. Submit with:

```bash
python3 research/rigid_stress/shell_comparison/submit.py \
  --source /ABS/FROZEN/QIPC/CHECKOUT \
  --tasks "$task_run/meta/fresh-tasks.json" \
  --output "$task_run/fresh-campaign" \
  --python /mnt/data/zhaofeng/venvs/qipc-owned-coarse-tree-check01-20261015/bin/python3.12 \
  --minutes 120 --concurrency 2
```

The helper appends each task's `--output` directory. Existing outputs are never
overwritten. `--keep-going` retains per-task return-code files, runs independent
later configurations after a failure, and exits the campaign nonzero if any
task failed. A queued/running job and a finite accepted prefix are not completion.

The displayed1e-8 is the historical strict diagnostic, not a practical-tolerance
recommendation. Standard PCG `linear_tolerance` controls the ratio of
preconditioned squared residuals; its square root is the corresponding norm
ratio with fixed SPD preconditioning. Newton velocity tolerance controls the
mapped displacement criterion `velocity_tol * dt`. Qualify actual force/field
residuals and complete-history cap status before selecting either tolerance.
PCG1e-2 and1e-3 pass the prescribed short window but exhaust Newton during the
complete grasp;1e-4 is under separate validation. No full-history accelerated
reference setting is admitted yet.

Both `native_grasp` and `shell_grasp` accept the same explicit `--keyframes`
times, retaining continuous mechanics and computing/exporting full stress only
there. Native builds with manual update mode and requests after controller/
radius/state writes. Existing no-keyframe behavior remains available for
labeled controls. `shell_grasp --dense-observer-control --keyframes ...`
computes stress every step but transfers/exports only the selected fields;
its actual update counter distinguishes dense computation from sparse files.
This is an observation control, not the primary workload.

`compare_cadence --comparison-scope stress-sampling` retains its original
thresholds while checking all new keys. A failed Shell comparison is a failed
equivalence check, not a successful validation. `compare_fields
--skip-snapshot-payloads` retains all computed metrics without duplicating
large derived aligned arrays. Keep original simulator fields and the common
domain; render/rebuild derived arrays in node-local temporary storage as needed.

## Inputs and protocols

`controls-v1/group0/inputs/` contains fixed-facet hollow P1 meshes with independent
surface and wall refinements. Subdivision preserves the same base1 continuum;
base2 is a separately rounded geometry. `qualify-tight-v1/group0/evaluation/`
contains single-case condition banks and shell midpoint meshes on those fixed
facets. Source densities/materials are unchanged; the nominal area-times500um
shell has a6.38% mass mismatch relative to the solid, explicitly retained until
a separately identified geometric correction is qualified.

`prescribed-input-v1/group0/motion/motion.npz` is the frozen6s common input,
SHA256`b050060473fd7332b339b6a91f82955876962e04c3c044b8c3bfb068160c2d29`.
The native/Shell prescribed workers accept `--motion`, `--sim-dt 0.001`,
`--substeps 1`, `--steps 6000`, `--tracking-damping` and `--record-every 10`.
They override force caps and measure actual joint/link tracking. This protocol
is distinct from the original capped-controller workers, which retain10ms
targets/gains/caps, weak/release schedule, resets and1ms physical substeps.

Reference snapshots contain actual physical Cauchy stress from the unchanged
energy, nodal contact forces, measured inertial reactions, material/world
geometry and moving robot poses. Contact gradients are last-assembled solver
gradients; the residual against the exported accepted-state stress/acceleration
is an explicit measured closure check. Small relative residual against squeeze
forces does not imply a small error in the net egg wrench. Preserve raw peaks,
gap/CCD diagnostics, cap exhaustion and failed histories.

## Common fields and independent load replay

`build_metric` produces material-column integration points on the union of
reference/native tet intersections, with constant-stress, affine, rotation and
linear-bending controls. `common-metric-v2/group0/metric/columns.npz` is the
current40960-point domain. It is neither a full-volume quadrature nor a global
peak search.

Inside the same Slurm/container setup, run:

```bash
python -m research.rigid_stress.qipc_comparison.compare_fields \
  --reference /ABS/REFERENCE/HISTORY --dpr /ABS/DPR/HISTORY \
  --scr /ABS/SCR/HISTORY --shell /ABS/SHELL/HISTORY \
  --columns /ABS/columns.npz --output /ABS/FRESH/COMPARISON

python -m research.rigid_stress.qipc_comparison.reference_replay \
  --reference /ABS/REFERENCE/HISTORY --native-mesh /ABS/elastic.npz \
  --columns /ABS/columns.npz --output /ABS/FRESH/REPLAY
```

Use the independent CPU environment for replay. It solves measured spatial
contact minus acceleration/gravity loads with independent P1/P2 elasticity.
P2 transfer preserves virtual work, force and undeformed moment. The output
retains balance correction and deformed-coordinate moment differences. This
is the resolved-load elastic/discretization layer. Subsequent`contact_packets`
and`native_reference_replay`workers separately measure compression to the rigid
feature's point-wrench information:

```bash
python -m research.rigid_stress.qipc_comparison.contact_packets \
  --reference /ABS/ORIGINAL-MAIN-PRESCRIBED/HISTORY --output /ABS/PACKETS
python -m research.rigid_stress.qipc_comparison.native_reference_replay \
  --packets /ABS/PACKETS --oracle /ABS/SPATIAL/REPLAY --columns /ABS/columns.npz \
  --route dpr --body-loads measured --output /ABS/POINT/REPLAY
```

The first worker uses pinned QIPC; the second uses pinned native Genesis.
Run core/measured, DPR/SCR measured and DPR/SCR rigid-body inertia separately.
Retain rejected packets and inferred source provenance. Physical point-wrench
agreement does not make traction fields equivalent. `audit_replay`checks
physical compression and plane-gradient units without summing layer error norms.

`compare_fields` matches actual saved times, including the different filename
cadences of capped and prescribed workers. It records missing/off-grid reference
observations and compares compatible corotated tangent fields and membrane/
bending resultants. Qualification flags remain false until the full uncertainty
study supports each measured quantity.

## Timing and interpretation

Primary `full-field-native-timing-v1` and `full-field-shell-timing-v1` campaigns
useFP64,900 warmup and three900-control-tick repeats, without timed field exports,
memory polling or solver instrumentation. Native max-only is separate. Native
rigid/DPR/tuned-DPR/SCR and shell default/fine-tuned settings are kept distinct,
including substep counts and batch sizes. The duplicate case0 cohort matches
shell's scalar coupling friction; it is not a varied-policy workload.

The older-named `primary-native-cost-v1` and `primary-shell-cost-v1` enable memory
polling and therefore remain intrusive diagnostics. FP32 native rejected-contact
failures must stay in the tables. Do not silently relax force/moment checks,
reuse old rates as new results, or label numerical completion as physical
reference qualification. Detailed profiles, additional lifecycle checks and the
final figures/report are still required by the active goal.

## Lifecycle, raw peaks and activation

`qipc_comparison.lifecycle`uses the common rollout parser and original10ms
capped controller. It measures B1/B2, duplicate inputs, fresh processes, three
restores and partial-restore continuation with separate selected controller
clocks. Native immediate full-field invalidation is expected; evaluate the
continuation. `audit_lifecycle`retains all differences and separate wall inputs.

`qipc_lifecycle`uses actual QIPC Scene.dump/recover/reset plus explicit Python
servo state. Its bounded contact-onset scope is not a complete near-slip batch
qualification. Capped histories accept`--checkpoint-interval 0.5`and retain
native state, servo q/velocity, absolute next step and hashes under
`native-checkpoints`. A checkpoint/prefix is not completion. Slurm may forbid
extending a running allocation; longer independent repeats keep earlier
attempts and true process statuses intact.

The native physical-step observer keeps10ms capped targets/radius and1ms physics,
retaining interior peak fields and10ms averaged forces/physical moments.
`compare_cadence`must validate it against the original ten-substep scene. The
130-observation DPR/SCR pilot passes; Shell does not and stays a separate variant.
For actual Shell interior peaks, use PR22 plus[the combined read-only substep
patch](../shell_comparison/pr22-substep-peaks.patch)with
`--instrumented --substep-peaks`. It reads fields after unchanged integration
and requires every physical step. The200-step pilot passes. Moment-of-impulse
uses actual substep points. A final point times accumulated force is not a
moment measurement. `--bank-coupling-friction`materializes0.6 times the bank ratio
for one homogeneous cohort before validating it; use separate Shell cohorts
for different coefficients.

`cold_activation`uses a fresh task group/cache for each configuration and records
initialization, build and first/second steps. Imports are outside that timer;
no shared cache is reset. Warm full/max timing, memory polling and profiles keep
separate scopes in`audit_cost`and`tables`.

`figures`and`tables`consume measured receipts and common comparisons. Their
[contracts](QIPC_REFERENCE_FIGURE_SPEC.md)and[captions](QIPC_REFERENCE_FIGURE_CAPTIONS.md)
identify actual evidence and open qualifications. Use the existing
`shell_comparison.figure_qa`with the data-side PyMuPDF library, then inspect
rendered previews. Editable publication figures may stay in Git; large raw
histories, traces, caches and binaries remain in the established data directory.


## Corrected ET and complete-observation admission

Current reference workers record native ET-check enabled status, pair ids and
counts at every physical step. A nonzero count marks the step invalid, writes
the rejected frame/static topology/real native checkpoint, stops execution and
exits nonzero. This observes an existing native forced exit; it does not modify
the detector or contact law. Historical4k/16k adaptive pilot receipts missed
this condition and remain immutable; `history_audit` separately scans their
native warning logs and preserves unknown per-step counts. Never equate finite
completion or no PCG cap with an admitted nonlinear solve.

`et_failure_geometry --failure /ABS/REJECTED/HISTORY --output /ABS/FRESH/AUDIT`
checks the actually reported pair using the captured native topology/global
positions and independent80-digit Decimal plane crossing. It verifies the
single-FEM owner mapping excludes the floor, and checks saved FEM/active points
exactly. A false-positive detector report is distinguished from physical
intersection; the prematurely terminated solve still fails admission. No
exhaustive CCD search or continuation is inferred.

Common field/reaction/uncertainty workers now require every expected10ms
reference observation and every matched candidate observation before emitting
a whole-history norm. Sparse14-frame pilots retain per-time results only.
`compare_fields --snapshot-times 1.1 4.2 9.0` limits only duplicate derived
snapshot exports; an empty `--snapshot-times` exports none, with complete
metrics and source raw/peak fields unchanged. The default export behavior is
unchanged.

The storage-policy review and explicit user exception are recorded in
[DATA_LOCATION.md](../../../DATA_LOCATION.md). During the current unwritable-
data outage only this task's existing home fallback is authorized, within4GB
persistent output. Pass `--allow-home-output --node-local-cache` to the
submitter and record each task/source/return-code receipt. Without the explicit
flag it rejects runtime output resolving under home, while allowing existing
compatibility symlinks resolving into data. Original inputs, builds, environments
and results remain at their existing data paths.
