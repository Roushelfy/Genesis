# Prepared validation after storage recovery

**Not submitted or executed.** The shared data destination still fails an
exclusive create/write/fsync probe with errno122. An additional probe
at12:13UTC confirms the same storage failure. Observer/evidence controls below
are prepared for subsequent execution. Static source checks are not Slurm
validation.

[The prepared task file](QIPC_PENDING_TASKS_20261018.json) has eight independent
groups and nine tasks. It reuses existing inputs and completed trajectories.
It does not resume incomplete QIPC histories or complete the reference goal.

| Group | Actual task | Evidence required before using its output |
| --- | --- | --- |
|0 | Synthetic interrupted-journal admission controls | Missing/corrupt result, truncated/middle-corrupt journal, failed/missing terminal receipts and absent flags cannot establish completion; damaged bytes stay unchanged; configured caps are used. |
|1 | Loaded same-state full/max Shell control, two environments | Actual420-tick history precedes20 test ticks. At every physical substep, independent Torch VM from full Cauchy must agree with the max-only interval reduction. A single-environment NaN probe must yield positive infinity in only that environment. No physical step follows the deliberate invalid probe. |
|2 | Normal max-only Shell application wiring |420 warmup ticks and40 measured ticks, one repeat; callback count must match all physical substeps. This is a wiring control, not final secondary timing. |
|3 | Signed residual/energy/penetration/friction/cap audit | Include completed sources and interrupted prefixes. Missing flags/cap observations stay unknown; missing terminal receipts do not become successful finite histories. All field/reference qualification gates remain separate. |
|4 | Complete per-finger force/moment time norms | Verify actual physical rigid points and nonrigid force closure for every candidate. Keep opposing finger loads separate. Include surface3/wall2 and halved friction smoothing. |
|5 | Refreshed full/max/memory/profile/cold cost audit | Preserve each original scope and worker receipt. Normalize a legacy max-only scope label only from its explicit output-mode arguments, retaining the original label. No primary rate comes from memory polling/profiling. |
|6 | New metric lookup and three-way comparison for completed surface2/wall4 | Rebuild reference element/barycentric lookup for this mesh; do not reuse surface1 indices. The same1280 material columns remain. The comparison follows metric generation sequentially in this group. |
|7 | Surface2 versus surface3 at fixed wall2 | Actual complete histories on the same geometry and common observation domain; no qualified-accuracy claim follows merely from a smaller contrast. |

Outputs from the scientific audits stay provisional until the relevant controls
and terminal receipts pass. A result JSON without successful process evidence
does not satisfy the interrupted-history admission control.

After the established data root is writable, use this existing submitter from
the task worktree:

```bash
task_work=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
task_sources=/mnt/home/zhaofeng/workspace/Genesis/.worktrees
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference
python3 research/rigid_stress/shell_comparison/submit.py \
  --source "$task_sources/kashu-shell-pr22-substep-peaks" \
  --output "$task_run/validation-after-storage-v1" \
  --tasks "$task_work/research/rigid_stress/docs/QIPC_PENDING_TASKS_20261018.json" \
  --python /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python \
  --minutes 60 --concurrency 3 --keep-going
```

The submitter still uses `rtx-mid`, `GENESIS_IMAGE_VER=1_26 gs-srun`, isolated
data-side caches and durable logs/results. Do not run the workers directly on
the login node. The prepared config contains group6's expected future generated
metric paths; changing the output root/name requires changing those two paths
in a fresh copied task config. Never reuse or overwrite an attempted campaign.
An alternate persistent output root may read the original inputs in place.

The Shell engine source is the existing pinned PR22 worktree plus the separate
[read-only substep patch](../shell_comparison/pr22-substep-peaks.patch).
There is no active task GPU writer importing it. Full-field and max-only output
have different temporal payloads: full exports the last-substep face/fiber field,
while max covers every physical substep of the current control interval. New
receipts state this explicitly and count observed substeps. Any final paired
timing must retain its actual source/callback overhead; a lower output-memory
footprint does not establish that max-only is faster.

The full primary benchmarks and compact checkpoint remain immutable. These
prepared observer/audit changes do not retroactively validate a reference,
overwrite previous measurements, or provide a capped QIPC resume mechanism.
