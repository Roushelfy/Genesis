# Reproduce the practical throughput measurements

Use the environment/container in [RUNNING.md](RUNNING.md) and the data root in
[DATA_LOCATION.md](../../../DATA_LOCATION.md). Scientific execution uses Slurm
and `GENESIS_IMAGE_VER=1_26 gs-srun`; the snapshot and JSON aggregation utilities
use Python stdlib only. Use fresh output directories and explicit data-side
temporary/cache/log paths. This protocol measures inference rollout, not RL
training. Accuracy limits remain in the
[application report](PRACTICAL_APPLICATION_RESULTS_20261016.md).

## Freeze the source and condition bank

From the task checkout, reconstruct the exact selected benchmark source before
scheduling. This includes the executed measurement files as well as the engine:

```bash
set -euo pipefail
task_repo="$PWD"
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/throughput-reproduce-$(date -u +%Y%m%dT%H%M%SZ)"
task_native="$task_root/native-source"
task_recipe_packet="$task_repo/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1"
task_bank="$task_recipe_packet/inputs/conditions.npz"
task_revision="exact-selected-b0f2f9cdaac8"
mkdir -p "$task_root/cache/cuda"
/usr/bin/python3 "$task_recipe_packet/rebuild_snapshot.py" \
  --repository "$task_repo" --recipe "$task_recipe_packet/selected" \
  --output "$task_native" --report "$task_root/rebuild-selected.json"
sha256sum "$task_bank" "$task_native/snapshot-files.sha256" > "$task_root/input-identities.sha256"
```

The bank SHA256 must be
`1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa`.
The selected manifest SHA256 must be
`b0f2f9cdaac8b90cfffe6a691c1f2ef1cdb0c6eeb4d356e2632e72f7a072ec10`.
For the original baseline, choose the `baseline` recipe and history4 instead;
the recipe packet preserves its exact source identity. To measure new code,
use `scripts/snapshot_practical_native.py --output "$task_native"` in place of
the rebuild command with a fresh output, and record the actual Git revision.
A new snapshot is a new source identity. Do not label it as the original
`2da41428...` baseline or `4bc5a887...` candidate without an exact match.
The original snapshots and submission commands remain under the established
`runs/20261017-practical-throughput/` data directory. The earlier
[application reproduction guide](RUNNING_PRACTICAL_APPLICATION.md) identifies
the archived baseline's exact Git/source correspondence.

## Four scopes, fixed simulated duration

The example below reproduces the fixed-history4 B8192 workload. Set the three
variables before creating the script; use `task_history=0` for the independently
qualified fast configuration, and `task_envs=16384` for the recommended balance.
All three history0 widths B8192/B16384/B32768 are closed; B16384 reaches about98%
of the widest point's throughput with about half its memory/latency. Change only the batch
variable for a sweep, retaining eight-condition repetitions and all other
arguments. Wider cases need appropriate CPU memory and wall time; inspect the
measured capacity record instead of assuming an allocation will fit. Initial
workspace2 is separately labeled memory tuning with full contact growth.

```bash
task_envs=8192
task_history=4
task_contact_block_size=16
cat > "$task_root/workload.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
sha256sum "$task_bank" > "$task_root/conditions-before.sha256"
task_finish() {
  task_status=\$?
  trap - EXIT
  set +e
  sha256sum "$task_bank" > "$task_root/conditions-after.sha256"
  cmp "$task_root/conditions-before.sha256" "$task_root/conditions-after.sha256"
  task_bank_status=\$?
  if test "\$task_status" = 0; then
    if test "\$task_bank_status" != 0; then task_status=\$task_bank_status; fi
  fi
  printf '%s\n' "\$task_status" > "$task_root/returncode.txt"
  exit "\$task_status"
}
trap task_finish EXIT
for task_scope in rigid live policy-rigid policy; do
  bash "$task_native/research/rigid_stress/scripts/run_practical_native_benchmark.sh" \
    "$task_root/\$task_scope" --scope "\$task_scope" \
    --envs "$task_envs" --level 1 --warmup 900 --steps 2400 --repetitions 3 \
    --minimum-seconds 0 --varied --seed 623016 \
    --contact-model intrinsic --intrinsic-support contact \
    --intrinsic-load-partition body_source_prior --contact-block-size "$task_contact_block_size" \
    --history "$task_history" --output-mode max --egg-geometry mesh64 \
    --collision-method mpr --contact-patch off \
    --conditions "$task_bank" --source-revision "$task_revision" --compact-log
done
EOF
cat > "$task_root/job.sh" <<EOF
#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 /usr/bin/env \
  CUDA_CACHE_PATH="$task_root/cache/cuda" bash "$task_root/workload.sh"
EOF
sbatch --parsable --array=0-0 --partition=rtx-mid --gres=gpu:1 \
  --cpus-per-task=8 --mem=128G --time=03:00:00 \
  --output="$task_root/job-%A_%a.log" "$task_root/job.sh"
```

The wrapper places each scope's temporary files, caches, logs, actual import
origins, hardware, source closures and benchmark JSON in that scope's directory.
The outer launch script omits nounset because the shared `gs-srun` helper probes
optional unset environment variables; the workload and benchmark wrappers keep
their strict exit/source/input checks.
Check terminal scheduler state, both actual exits, every snapshot hash and the
post-run inventory before using a rate. The benchmark JSON must contain three
2400-step rows,900 warmup steps and zero invalid steps/scatter overflows. Keep
the exact material, numerical options, controller and policy metadata with it.

For a same-GPU implementation comparison, run separate warmed processes in
baseline/candidate/candidate/baseline order inside one allocation. The existing
`meta/*-paired-live-b2048-v1.sh` records the precise measured ABBA layout.
Smaller initial `--contact-block-size` is a separately declared allocation
tuning; the actual body-union workspace must grow to contain every valid source.

## Observe memory and work separately

Use a fresh diagnostic run with `--repetitions 1`, retaining900 warmup and2400
timed steps. Inside the same GPU allocation/container as the benchmark, start
the sampler before the process and stop it after exit:

```bash
task_memory_csv="$task_root/process-memory-live.csv"
task_memory_stderr="$task_root/process-memory-live.stderr"
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory \
  --format=csv --loop-ms=100 > "$task_memory_csv" 2> "$task_memory_stderr" &
task_sampler_pid=$!
trap 'kill "$task_sampler_pid" 2>/dev/null || true; wait "$task_sampler_pid" || true' EXIT
bash "$task_native/research/rigid_stress/scripts/run_practical_native_benchmark.sh" \
  "$task_root/memory-live" --scope live \
  --envs 8192 --level 1 --warmup 900 --steps 2400 --repetitions 1 \
  --minimum-seconds 0 --varied --seed 623016 \
  --contact-model intrinsic --intrinsic-support contact \
  --intrinsic-load-partition body_source_prior --contact-block-size 16 \
  --history 4 --output-mode max --egg-geometry mesh64 \
  --collision-method mpr --contact-patch off \
  --conditions "$task_bank" --source-revision "$task_revision" --compact-log
kill -0 "$task_sampler_pid"
kill "$task_sampler_pid"
wait "$task_sampler_pid" || true
trap - EXIT
```

Set both sampler paths under the fresh data-side run. Match samples to the
benchmark's actual GPU UUID and sum compute processes per timestamp. This is a
sampled lower-bound process peak including setup/growth, not native-buffer bytes,
Torch allocator statistics, end-of-run free memory or a whole-device peak.
The retained `meta/*memory*four-scopes*.sh` scripts include exit traps, condition
hashes and sampler-liveness checks. Keep these observations out of headline
timings. Add `--trajectory-metrics` for a separate device work-counter pass;
history prediction-hit counts require the updated observer source.

For a moving profile, use another fresh live run with `--trace --trace-steps 600`.
Keep profiler timings separate and distinguish them from the frozen `profile`
scope. One600-step window covers the scripted period; post-trace validity and
reset counts must close. CPU annotations and GPU durations overlap.

`scripts/summarize_trace.py --trace "$task_trace" --output "$task_overlap"`
also reports the timestamp intersection of CPU API spans and GPU kernels.
It uses stdlib metadata only and may run on the host. API duration is not pure
CPU overhead or synchronization time; do not add it to GPU durations.

After freezing a scheduler-backed summary, independently reread its original
records with the stdlib-only metadata auditor from the checkout root:

```bash
/usr/bin/python3 research/rigid_stress/scripts/audit_practical_throughput_metadata.py \
  --summary "$task_summary" --contract "$task_contract" --output "$task_audit"
```

Use the frozen `meta/baseline-contract-v1.json` and a fresh data-side output.
Both the summarizer and auditor accept `--history 4` (default) or `--history 0`.
Pass the same explicit setting to each tool when analyzing the separate history0
tuning. The output records its reference history; all other frozen physical,
numerical and policy settings remain matched. Real NVML sampling files mark a
run intrusive even if a submission journal omits that annotation, and the
independent auditor rejects trace/observer/sampling artifacts in headline rows.
This verifies all eligible raw repeat rates, exact physical/numerical settings,
controller/policy/precision, per-environment failure counts and the unchanged
policy budget. Source/scheduler closure and independent physical field/lifecycle
checks remain separate evidence. The plotter retains independently complete
four-scope rate and memory widths, leaving missing CSV values blank.

## Bounded numerical and lifecycle checks

The candidate's full-field and complete-grasp comparison is reproducible with
`scripts/compare_practical_optimization.py`, using the frozen original data root,
the four-state field run, both complete-grasp runs and
`PRACTICAL_REWARD_PHASE_AUDIT_20261016.json`. Run it inside a CPU Slurm allocation
through `gs-srun`, with `--output` under data. Its exact executed CLI and input
hashes are retained in the comparison closure and `meta/*comparison-v5*` records.
It matches traction samples bijectively before comparison and retains the
original field/load/residual/wrench gates. Nominal-only reward rows do not
replace the retained cross-model/radius uncertainty assessment.

`research/rigid_stress/check_practical_large_batch.py` accepts `--conditions`,
`--envs`, `--contact-block-size` and a fresh `--output`. Run it inside a GPU Slurm
allocation with the selected immutable engine on `PYTHONPATH`. It checks full
fields, checkpoints, partial resets and the subsequent untouched environments,
then exercises complete zero-load release/recontact on a frozen mapper input.
This changes observation/reset work and is never throughput evidence.

Both the large-batch checker and `research/rigid_stress/practical_reward_proxy.py`
accept `--history 0` or `--history 4` (default4) and record the selected size.
The checker verifies the corresponding storage state. Keep this configuration
tuning separate from the fixed-history4 before/after code comparison.
`scripts/compare_practical_large_batch.py --reference-run "$task_reference"`
`--candidate-run "$task_candidate" --output "$task_comparison"` checks every
saved tensor/von-Mises/peak component and selected phase/environment at original
comparison bounds. Execute this numerical comparison through CPU Slurm/gs-srun
using terminal qualified producer runs and a fresh data-side output.

For figures, use the exact renderer, input, SVG/PDF/PNG and QA in the
[baseline figure packet](../evidence/20261017-throughput/baseline-scaling-v2/README.md).
Final selected [history4](../evidence/20261017-throughput/selected-history4-scaling-v1/README.md),
[history0](../evidence/20261017-throughput/selected-history0-scaling-v1/README.md)
and [direct comparison](../evidence/20261017-throughput/baseline-selected-comparison-v1/README.md)
packets provide the exact executed sources, compressed inputs, independent raw
audits and accepted exports. Decompress under data and use the archived renderer
and style together; the runtime QA file predates the separate visual review.
The final report and final metadata closure identify the closed headline and
intrusive results; historical checkpoints preserve their dated pending states.
No speedup or capacity claim follows merely from a successful submission.

## Reconstruct exact sources and inputs

The [source recipe packet](../evidence/20261017-throughput/source-recipes-v1/README.md)
rebuilds both the original baseline and selected899-file trees from committed
Git objects and small exact overrides. Both recipes have been executed and
pass every file size/SHA256 and inventory check. It includes the exact condition
bank, original contract and executed environment metadata. The stdlib/Git rebuild
may run on the host, with its fresh225MB output and report explicitly under data.
The normal scientific benchmark wrapper then runs through Slurm/gs-srun.
