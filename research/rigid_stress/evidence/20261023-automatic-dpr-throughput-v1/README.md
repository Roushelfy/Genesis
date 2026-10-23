# Automatic DPR throughput evidence

See [the report](../../docs/DPR_THROUGHPUT_20261023.md) for the numerical method,
measured tradeoffs, recommended configuration and remaining physical limits.
This packet complements the [original implementation evidence](../20261023-automatic-dpr-v1/);
it does not replace its failed bitwise comparison or coarse physical errors.

The final production candidate is `db84114d1`. Later commits update research
audits and documentation. Runtime arrays, logs and frozen engines remain at
the original data/workspace paths recorded in the receipts.

- `example-tasks.json` reproduces a B64 full-output, 130-frame Hertz run from
  this branch. `conditions.npz` is the unchanged four-condition fixture;
  `diverse-conditions.npz` adds eight independently seeded conditions.
- `timing.json` keeps every repeat, invalid-step counts, memory, settings,
  source hashes and GPU identities. Per-environment warmup telemetry remains
  in the hashed raw results under data. Its medians retain the original results'
  hashes. Matched configurations share each campaign group allocation.
- `profiles.json` separates intrusive stage wall times from recorded kernels.
  CUDA graph capture omits kernels from Quadrants' profiler. Explicit secondary
  graph-disabled profiles reveal individual kernels under different scheduling;
  their times are not additive production costs or pure launch overhead.
- `results/` keeps numerical controls, lifecycle and same-input field audits,
  as well as failed cheaper quadratures. The full source inventories and large
  field arrays are identified by `campaigns.json` rather than copied into Git.
- `campaigns/` keeps original submitted arguments and job receipts, including
  compiler/setup failures and the pending-only cancelled timing arrangement.
  These historical tasks retain absolute paths; use the portable example for
  a new checkout.

From this branch's root, use a fresh data output directory:

```bash
python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$PWD" \
  --tasks research/rigid_stress/evidence/20261023-automatic-dpr-throughput-v1/example-tasks.json \
  --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/automatic-dpr-throughput-example \
  --python /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python \
  --minutes 20 --memory-gb 12 --cpus 1 --node-local-cache
```

This uses Slurm `rtx-mid`, image `1_26`, the existing environment and private
node caches. Full stress fields remain on the GPU during primary timing.
Sharp collision boxes are retained. `box_edge_policy="planar"` explicitly
approximates their edge curvature for a finite footprint; numerical acceptance
does not qualify a physical sharp-edge peak or a successful robotic grasp.
