# GPU throughput evidence packet

Read [the report](../../docs/GPU_THROUGHPUT_20261025.md) and
[the predeclared budgets/execution record](../../docs/GPU_THROUGHPUT_PROGRESS_20261025.md).
Raw runtime files, full tensor/load arrays, logs, traces and captured analysis
sources remain under:

`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261025-gpu-throughput/`

- `conditions/` contains the unchanged tiled four-case bank and the 64 distinct
  pose/friction/phase bank. Repetition to B1024/B8192 does not create more unique
  physical conditions.
- `receipts/` contains submitted task lists and Slurm receipts, including failed
  attempts. Each task source is pinned; otherwise the submission's source
  inventory supplies the common source.
- `metadata/` contains preservation, source/artifact hashes, numerical
  requirements and compact benchmark distributions/profiles.
- `results/` contains compact candidate/validation results. Large payloads are
  referenced by their external path, size and SHA256 in the artifact manifest.

Reported rejected candidates remain rejected: q8 evolving fit, q12 field/peak
budget, four-column history stress budget, five-mesh spatial budgets and the
heterogeneous volume fit. Their failures are not omitted from the completion
accounting. The first radius audit, first mother-grid construction and first
capacity-test invocation also remain as failed research attempts.

Use the existing stdlib submission helper with a new explicit data output
location. Do not overwrite a retained campaign or edit its captured sources.
Production numerical code is entirely Quadrants. CPU reference fixtures and
publication aggregation are clearly separate from native/GPU speed claims.
