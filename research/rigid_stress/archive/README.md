# Rigid stress archive

The native feature, current usage/results, useful independent references and
all regression tests remain active. Superseded research lives in the existing
Git history, rather than a second large directory in the checkout.
Historical instructions are records and do not override [AGENTS.md](../AGENTS.md).

## Find an original

The complete pre-cleanup tree is pinned at
[`1606ee3a9e9628a935ad5e446984f920af3adc66`](https://github.com/Roushelfy/Genesis/tree/1606ee3a9e9628a935ad5e446984f920af3adc66/research/rigid_stress).
The [catalogue](catalogue.tsv) records former paths/prefixes, counts, bytes,
reasons and that revision. Wildcard groups list only archived members; use
Git below for exact filenames and blob identities. No shared history was
rewritten. These checkout removals do not reduce Git history size.

| Former area | Disposition and reason |
| --- | --- |
| Standalone CPU/Torch/cuSPARSE/cuDSS solvers and their diagnostics | Pinned historical source; production uses the native Quadrants feature. Test-required CPU/FEM oracles remain importable at their original paths. |
| Physical/contact convergence campaigns, experimental drivers and patches | Pinned historical source/evidence; original unresolved results remain available without reopening their old goals. Current R/L/M reproduction dependencies remain active. |
| One-off shell launchers, publishers, configs and scheduler dumps | Pinned history; retain current benchmark/snapshot, metadata audit, application replay and focused-test commands. The packet's executed renderer/source replaces separate plotting copies. |
| Dated progress, intermediate audits and superseded plans | Pinned reports; current API/model, usage, final application and throughput results remain in `docs/`. |
| Earlier evidence packets and baseline-scaling-v1 | Pinned original bytes/manifests; retain the final baseline-v2, selected history4/history0, comparison and source-recipe packets. |
| `docs/figures/practical-application/` | Exact duplicates of blobs in the retained `practical-application-v2` packet. The application report now links those canonical blobs. Other dated generated figures are pinned history. |
| Original imported `reference/` experiments | Pinned tree and its `evidence/import_manifest.json`; active regression oracles remain separate. |

List exact archived filenames, or inspect a file without restoring it:

```bash
task_archive_revision=1606ee3a9e9628a935ad5e446984f920af3adc66
git diff --diff-filter=D --name-only "$task_archive_revision" HEAD -- research/rigid_stress
git ls-tree -rl "$task_archive_revision" -- research/rigid_stress/reference
git show "$task_archive_revision:research/rigid_stress/cpu.py"
```

The three entrypoints preceding the goal-setting commit remain at
[`47499f6ee71e940ec35281449242bfcc8d9b13b3`](https://github.com/Roushelfy/Genesis/tree/47499f6ee71e940ec35281449242bfcc8d9b13b3/research/rigid_stress).
For example, `git show 47499f6ee71e940ec35281449242bfcc8d9b13b3:research/rigid_stress/GOAL.md`
recovers the full former goal chronology.

## Retrieve evidence without changing the active checkout

From this repository, select a former path from the catalogue or pinned tree.
Use a fresh directory on the data mount; request only the packet you need:

```bash
task_archive_revision=1606ee3a9e9628a935ad5e446984f920af3adc66
task_old_path=research/rigid_stress/evidence/20261017-throughput/baseline-scaling-v1
task_restore="$RIGID_STRESS_DATA_ROOT/runs/archive-baseline-v1"
mkdir "$task_restore"
git archive "$task_archive_revision" -- "$task_old_path" | tar -x -C "$task_restore"
git ls-tree -rl "$task_archive_revision" -- "$task_old_path"
```

An exact file identity can be checked independently:

```bash
task_old_file="$task_old_path/manifest.json"
test "$(git hash-object "$task_restore/$task_old_file")" = \
  "$(git rev-parse "$task_archive_revision:$task_old_file")"
```

Then use the packet's original SHA256 manifest and decompression/reconstruction
instructions. Original absolute runtime/source paths are provenance, not new
local destinations. Resolve a missing historical relative path in the same
pinned tree. Active packet files, manifests and their bytes are unchanged.

## Run historical code in isolation

Use a fresh sparse worktree under data so archived imports cannot shadow the
current package. This restores code/tests/assets without checking out the
entire historical evidence collection:

```bash
task_archive_revision=1606ee3a9e9628a935ad5e446984f920af3adc66
task_historical="$RIGID_STRESS_DATA_ROOT/runs/archive-source"
git worktree add --detach --no-checkout "$task_historical" "$task_archive_revision"
git -C "$task_historical" sparse-checkout set --no-cone \
  '/genesis/' '/examples/' '/tests/' '/pyproject.toml' \
  '/research/__init__.py' '/research/rigid_stress/*.py' \
  '/research/rigid_stress/*.patch' '/research/rigid_stress/requirements-*.txt' \
  '/research/rigid_stress/scripts/' '/research/rigid_stress/docs/'
git -C "$task_historical" checkout --detach "$task_archive_revision"
```

Historical tools may require optional dependencies and their recorded inputs.
Retrieve required evidence explicitly as above. Run numerical work through
Slurm/gs-srun with fresh data-side output/cache paths. Do not execute archived
scripts in the active checkout or restore old agent instructions over it.

## Current retained evidence and protected source

Seven canonical packets remain: the five final throughput/source packets under
`evidence/20261017-throughput/`, and `practical-application-v2` plus
`practical-reproduction-v1` under `evidence/20261014-physical/`.
Their manifests preserve original source/configuration identities and hashes.
The exact baseline/selected source reconstruction is documented in
[the throughput guide](../docs/RUNNING_PRACTICAL_THROUGHPUT.md).

At cleanup, Slurm job716318 was pending with `DependencyNeverSatisfied`.
Its original `scripts/audit_nonlinear_u_tail.sh` and complete nine-file source
list were retained unchanged, including their import dependencies. No job,
runtime directory or other worktree was cancelled, migrated or deleted.
These protected historical helpers are not a new active research goal.
See [the cleanup result](../docs/BRANCH_CLEANUP_RESULTS.md) for verification and
before/after counts.
