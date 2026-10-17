# Reproduce the practical throughput measurements

Use the environment/container in [RUNNING.md](RUNNING.md) and the data root in
[DATA_LOCATION.md](../../../DATA_LOCATION.md). Scientific execution uses Slurm
and `GENESIS_IMAGE_VER=1_26 gs-srun`; the snapshot and JSON aggregation utilities
use Python stdlib only. Use fresh output directories and explicit data-side
temporary/cache/log paths. This protocol measures inference rollout, not RL
training. Accuracy limits remain in the
[application report](PRACTICAL_APPLICATION_RESULTS_20261016.md).

## Freeze the source and condition bank

From the selected task checkout, capture the actual source before scheduling:

```bash
set -euo pipefail
task_repo="$PWD"
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/throughput-reproduce-$(date -u +%Y%m%dT%H%M%SZ)"
task_native="$task_root/native-source"
task_bank="$task_repo/research/rigid_stress/evidence/20261014-physical/practical-application-v2/blobs/1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa.npz"
task_revision=$(git rev-parse HEAD)
mkdir -p "$task_root/cache/cuda"
python3 research/rigid_stress/scripts/snapshot_practical_native.py --output "$task_native"
sha256sum "$task_bank" "$task_native/snapshot-files.sha256" > "$task_root/input-identities.sha256"
```

The bank SHA256 must be
`1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa`.
A new snapshot is a new source identity. Do not label it as the original
`2da41428...` baseline or `4bc5a887...` candidate without an exact match.
The original snapshots and submission commands remain under the established
`runs/20261017-practical-throughput/` data directory. The earlier
[application reproduction guide](RUNNING_PRACTICAL_APPLICATION.md) identifies
the archived baseline's exact Git/source correspondence.

## Four scopes, fixed simulated duration

The example below reproduces the published B8192 workload. Change only `--envs`
for a batch sweep, retaining the eight-condition repetitions and all other
arguments. Wider cases need appropriate CPU memory and wall time; inspect the
measured capacity record instead of assuming an allocation will fit.

```bash
cat > "$task_root/workload.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
for task_scope in rigid live policy-rigid policy; do
  bash "$task_native/research/rigid_stress/scripts/run_practical_native_benchmark.sh" \
    "$task_root/\$task_scope" --scope "\$task_scope" \
    --envs 8192 --level 1 --warmup 900 --steps 2400 --repetitions 3 \
    --minimum-seconds 0 --varied --seed 623016 \
    --contact-model intrinsic --intrinsic-support contact \
    --intrinsic-load-partition body_source_prior --contact-block-size 16 \
    --history 4 --output-mode max --egg-geometry mesh64 \
    --collision-method mpr --contact-patch off \
    --conditions "$task_bank" --source-revision "$task_revision" --compact-log
done
EOF
cat > "$task_root/job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
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
task_memory_csv="$task_root/memory-live.csv"
task_memory_stderr="$task_root/memory-live.stderr"
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
The main experiment report and machine-readable checkpoint retain which results
are closed, intrusive, failed or still pending. No final speedup or capacity
claim follows merely from a successful submission.

## Reconstruct exact sources and inputs

The [source recipe packet](../evidence/20261017-throughput/source-recipes-v1/README.md)
rebuilds both the original baseline and selected899-file trees from committed
Git objects and small exact overrides. Both recipes have been executed and
pass every file size/SHA256 and inventory check. It includes the exact condition
bank, original contract and executed environment metadata. The stdlib/Git rebuild
may run on the host, with its fresh225MB output and report explicitly under data.
The normal scientific benchmark wrapper then runs through Slurm/gs-srun.
