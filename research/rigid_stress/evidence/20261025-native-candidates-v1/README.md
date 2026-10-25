# Native candidate evidence packet

Read the [result report](../../docs/NATIVE_CANDIDATE_RESULTS_20261025.md),
[integration goal](../../docs/NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md)
and [execution ledger](../../docs/NATIVE_CANDIDATE_PROGRESS_20261025.md).
This packet is being assembled while the final comparisons finish; an absent
task receipt is not a pass or a completed retention decision.

Raw arrays, full fields, traces, job logs and captured analysis sources stay in:

`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261025-native-candidate-integration/`

The original heterogeneous volume frame-221 failure remains in the preceding
`20261025-gpu-throughput/heterogeneous-volume-audit-v1/` campaign. Q32/q64 load
budget failures, the automatic q256 reference-path switch, the first strict
bitwise-radius audit, v8 dtype failures and the v10 heterogeneous history
displacement failures are retained. A successful runtime fit does not imply
passing the separate complete-field accuracy budget.

`receipts/` preserves task lists and Slurm submissions. `metadata/` contains
source pins and external artifact hashes, numerical summaries, repeated
latencies, preparation and memory inventories, profiles and preservation
checks. Large payloads are referenced rather than copied into Git. The small
[four-condition bank](../20261025-gpu-throughput-v1/conditions/tiled-four.npz)
and [64-condition bank](../20261025-gpu-throughput-v1/conditions/heterogeneous-64.npz)
are reused from the preceding packet. Tiling these banks to larger batches
does not add distinct physical conditions.

## Reproduce a bounded native control

Create a separate checkout at the experiment pin. Never edit a checkout while
its captured jobs are running. V13 `d5641351a127f674b4a19bf3d921cc3be0200789`
contains all native candidates with input-major H; V11
`7bbfa4faa8295f21ea9ce17c8c3ef7391042aab5` contains direct atlas accumulation,
guarded history and output-major H. V12
`b8b3ac7ed64b629f34cd41d904cedc92f3122619` fixes explicit angular integration's
fan selection and permits q128. Each older campaign's own captured
`analysis-source/` and hashes are authoritative for its historical workers.
Latest research workers must not be assumed compatible with every older pin.

From the task checkout, use the stdlib submission helper; scientific Python
runs inside the allocated GPU job, not on the login node:

```bash
ENGINE=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-candidates-v13
RUN=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/native-candidate-reproduction
PY=/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python
python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$ENGINE" --output "$RUN/kernels" \
  --tasks research/rigid_stress/evidence/20261025-native-candidates-v1/receipts/native-kernels-v13/tasks.json \
  --python "$PY" --minutes 45 --memory-gb 24 --cpus 1 \
  --keep-going --node-local-cache
```

Use a fresh output directory. Complete trajectory and benchmark receipts are
replayed with the same helper and their recorded engine pin. Replace absolute
condition paths in a copied task JSON with the two tracked banks above when
relocating. Keep all other parameters, original acceptance, full 600-step
moving trajectories, reset policy and output mode. Q128/angular is required
for the qualified heterogeneous volume comparisons. Q32/q64 repair-cost rows
are explicitly unqualified controls.

The synchronized headline times exclude policy inference and research host
exports. Intrusive traces and memory-sampling runs are separate. Shared
inventory, scratch/output/state bytes and sampled GPU peaks have distinct
meanings; the 200 ms peak samples are lower bounds on instantaneous peaks.

## Rebuild the compact packet

`collect_gpu_throughput` and `package_native_candidates` use only the standard
library and may run on the login node. Store their runtime summary under data:

```bash
python3 -m research.rigid_stress.qipc_comparison.collect_gpu_throughput \
  --run "$RUN" --output "$RUN/meta/summary.json"
python3 -m research.rigid_stress.qipc_comparison.package_native_candidates \
  --run "$RUN" --summary "$RUN/meta/summary.json" \
  --output research/rigid_stress/evidence/20261025-native-candidates-v1
```

The final requirement audit is separate from collection: it links all fourteen
candidate/backend decisions to native numerical, tuning, complete-scene and
memory evidence and to the resulting maintained production code.
