# Reproduce the shell comparison

Read the [results and limits](SHELL_COMPARISON_RESULTS_20261018.md) first.
Scientific work ran through Slurm `rtx-mid`, one GPU/task, image `1_26`.
Use a fresh output directory per submission. `submit.py` freezes analysis source,
captures native/external engine/assets inventories and patches, and isolates
data-side temporary/compile caches. It does not run scientific work on login.
Do not edit or reset source checkouts while jobs use them.

## Evidence and exact source

The [packet](../evidence/20261018-shell-comparison/comparison-v1/README.md) contains
raw receipts, full fields at each application peak, compact derived plotting
inputs, editable exports and `reproduction.tar.gz`. The archive contains **every
submitted campaign**, including failures, exact task argument lists, launchers,
source inventories/patches and a deduplicated closure of executed Python files.
`data-artifacts.json` identifies larger originals retained on data, including
GB-scale raw traces. Compact plotting NPZs aggregate contact impulses over
vertices; they are explicitly derived, not byte copies of raw trajectories.

```bash
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
task_data=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery
task_run="$task_data/runs/20261018-shell-comparison"
task_packet="$PWD/research/rigid_stress/evidence/20261018-shell-comparison/comparison-v1"
tar -tzf "$task_packet/reproduction.tar.gz"
python3 research/rigid_stress/shell_comparison/restore_snapshot.py \
  --archive "$task_packet/reproduction.tar.gz" --campaign validated-native-v2 \
  --output "$task_data/runs/reproduce-shell-native/analysis-source"
```

All engine imports must use the pinned checkout separately from restored
analysis source. Native revision is `7fe068267e09ff97c09ffac4904a2f47914596d9`;
external original and diagnostics base is
`fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`. Fetch external source into a separate
worktree, preserving an unmodified original:

```bash
git fetch https://github.com/Kashu7100/Genesis.git fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab
git worktree add --detach ../kashu-shell-pr22-reproduce fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab
git worktree add --detach ../kashu-shell-pr22-diagnostics-reproduce fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab
git -C ../kashu-shell-pr22-diagnostics-reproduce apply \
  "$PWD/research/rigid_stress/shell_comparison/pr22-diagnostics.patch"
```

Compare `source.json`/`extra-source0.json` engine and asset byte inventories with
the actual checkout; a revision string alone is insufficient for the diagnostic
variant. PR/API responses are retained in the archive's `meta/`. The adapter's
local imports avoid its initial circular-import failure. No historical repaired
shell solver or IPC replacement is required.

## Environment and primary campaigns

Primary paired timing uses
`/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python` for **both** engines:
Python3.10.20, Torch2.11.0+cu129, Quadrants1.3.3, NumPy2.2.6, SciPy1.15.3,
Trimesh5.1.1, Pydantic2.14. Independent FEM/collection/rendering use the established
`rigid-stress-physical` environment with scikit-fem and Matplotlib, Torch2.8.
The shell environment lacks scikit-fem: the first collector failed before any
analysis, and `collect-v2/v3` use the reference environment. Do not silently
change the primary timing environment to resolve an offline-analysis dependency.
Exact package/import/GPU/CPU/Slurm identities are in the result receipts.

Canonical task recipes and the actual launchers are in
`reproduction.tar.gz:campaigns/<campaign>/`. To rerun, extract that campaign's
`tasks.json`, replace **output/input checkout paths only as appropriate**, and
submit with a fresh output. Keep model/numerical parameters intact unless
declaring a new variant. For example, on this installation:

```bash
python3 research/rigid_stress/shell_comparison/submit.py \
  --source "$PWD" --output "$task_data/runs/reproduce-validated-native-v2" \
  --tasks "$task_run/validated-native-v2/tasks.json" \
  --python /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python \
  --minutes 60 --memory-gb 96
```

This native recipe retains its original alternate `source` path to this native
worktree; when relocating, update it too. External `paired-normal-v1` uses the
original PR checkout as primary source and native as alternate source; diagnostics
use the separately patched checkout. Never apply the hook patch to the original
normal baseline. The exact archived recipes carry all flags and absolute input
identities, rather than relying on evolving CLI defaults.

| Executed campaign | Purpose / job |
| --- | --- |
| environment-v1; controls-v1 | Original import, uniform/bending fields and sphere offsets;725859/725861 |
| prepare-inputs-v2; prepare-loads-v2; static-inputs-v1 | Matched meshes, finite cloud/volume/shell loads; exact submissions in archive |
| native-elastic-v2; shell-prescribed-v2; analyse-prescribed-v1 | Analytic plate and same-load compatible fields/actual equilibrium |
| prepare-sphere-v1; analytic-sphere-v1 | Curved Lamé/thin-sphere control;725956/725967 |
| diagnostic-controls-v2; press-controls-v3; contact-ledger-v1; contact-audit-v1 | Contact candidates, impulses, cap/dt and geometry/energy; last audit725975 |
| full-shell-history-v1; full-native-history-v1 | Eight-condition900step application diagnostics;725893/725896 |
| peak-traces-v2; fresh-restarts-v1 | Dense raw peak locations/fields and repeated identical frozen processes;725965/725951 |
| lifecycle-v1; lifecycle-one-step-v1; cache-ablation-v1/v2; close-sheets-v1 | Public restore/batch/partial reset, private warm-start isolation, unsupported shell contact |
| paired-normal-v1; validated-native-v2 | Four widths, full scene3repeats;725917/725947; **v2 native counters are primary** |
| paired-diagnostics-v1; profiles-v1 | Separate process memory and60moving-step trace/aggregation;725921/725954 |
| capacity-v1 | Explicit OOM versus Jacobian guard and short/full resource scopes;725944 |
| static-cost-v1; static-inverse-tuning-v1 | Prescribed full-field cost and inverse memory-cap ablation |
| collect-v3; figures-v2 | Final tables/derived plot inputs and inspected actual exports;725982/725983 |

Normal scene scope is900warm +3x900timed; no host trajectories, memory poll or
profiler in these rates. Stress validity runs persist original invalid counters
across resets and read counter deltas outside timers. Shell diagnostic residuals
are maxima over substeps, with a separate recursive squared-budget ratio.
Do not rename those ratios L2 residuals or interpret cap0 as proof of convergence.
Dense history and memory/trace rates are intrusive, excluded from headline rates.

## Rebuild tables and figures

To regenerate all tables from the retained **complete data-side** runs:

```bash
python3 research/rigid_stress/shell_comparison/submit.py \
  --source "$PWD" --output "$task_data/runs/reproduce-shell-collection" \
  --tasks "$task_run/collect-v3/tasks.json" \
  --python /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python --minutes 20
```

To replot **only from the compact Git packet**, use a Slurm task whose module is
`research.rigid_stress.shell_comparison.plot` and arguments are:

```text
--runs <packet>/derived/plot-inputs
--summary <packet>/derived/result.json
```

Pass that JSON task to `submit.py` with the reference Python and a new output.
The final executed `figures-v2/tasks.json` gives the full two-task recipe; replace
its `--runs/--summary` with the packet paths and its dependent `--figures` with
the new allocation's `group0/figures`. Its second task,
`figure_qa`, uses the isolated PDF library at
`<task_data>/assets/pymupdf-1.26.5`; it does not modify a shared environment.
The original renderer's exact snapshot is recoverable with `restore_snapshot.py`.
Run `python -m ...` inside the same container/Slurm wrapper, never directly on login.

Figures use editable SVG text, embedded TrueType PDF fonts and450dpi RGB PNGs.
The first layout's clipped right colourbar was corrected in `figures-v2`;
out-of-view Matplotlib tick objects that are never drawn are excluded from
bounding-box QA. Actual exported PDF rasters were also inspected. This is
repository/report artwork, not a claim of journal acceptance or damage validation.

## Verification and limits

Verify packet payload SHA256/size with `manifest.json`, source closure with its
per-campaign `analysis-index.json`, and raw/derived separation with
`data-artifacts.json`. Source archives include failed attempts and Slurm logs;
do not filter them when reconstructing the execution history. Pre-existing
full-RL/fracture campaigns are outside this task. Production engine changes are
not required to reproduce this comparison.
