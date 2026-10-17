# Rigid stress archive index

Historical records here are references, not active agent instructions.
The active task is [branch cleanup](../GOAL.md).

## Initial entrypoint archival, 2026-10-17

This goal-setting commit simplifies the three active entrypoints.
Their complete previous versions remain in pinned commit
[`47499f6ee71e`](https://github.com/Roushelfy/Genesis/commit/47499f6ee71e940ec35281449242bfcc8d9b13b3);
they are not copied into the checkout.

| Former entrypoint | Why archived | Exact historical version |
|---|---|---|
| research/rigid_stress/GOAL.md | Near-thousand-line accumulation of superseded stages and checkpoints | [Original GOAL.md](https://github.com/Roushelfy/Genesis/blob/47499f6ee71e940ec35281449242bfcc8d9b13b3/research/rigid_stress/GOAL.md) |
| research/rigid_stress/AGENTS.md | Repeated reading orders and obsolete stage priorities | [Original AGENTS.md](https://github.com/Roushelfy/Genesis/blob/47499f6ee71e940ec35281449242bfcc8d9b13b3/research/rigid_stress/AGENTS.md) |
| research/rigid_stress/README.md | Mixed current usage, older workload rates and standalone prototype instructions | [Original README.md](https://github.com/Roushelfy/Genesis/blob/47499f6ee71e940ec35281449242bfcc8d9b13b3/research/rigid_stress/README.md) |

Inspect any original without changing the checkout, for example:

```bash
git show 47499f6ee71e940ec35281449242bfcc8d9b13b3:research/rigid_stress/GOAL.md
```

Use the same pinned revision with the other paths in the table. Historical
relative links should be opened in that pinned GitHub tree or an isolated
checkout of the revision. Do not restore old instructions over the active goal.

## Broader cleanup still to implement

This is the starting archive index, not a claim that research/code/evidence
cleanup is complete. The implementing agent should extend it with actual
classifications, destinations or pinned revisions and restoration commands.
[Cleanup requirements](../docs/BRANCH_CLEANUP_GOAL.md) cover dependencies,
canonical evidence, active workflows and verification. Keep this index useful
rather than building another per-run audit chronology.
