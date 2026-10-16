# Reproduce the practical application settings

Use the existing reference environment and cluster container described in
[RUNNING.md](RUNNING.md) and [DATA_LOCATION.md](../../../DATA_LOCATION.md).
The wrappers accept `PHYSICAL_VALIDATION_ENV`; they set separate data-side
runtime/cache paths and verify actual source imports. Numerical execution must
use Slurm and `gs-srun`. Snapshot/archive operations below use Python stdlib only.
Choose fresh output directories. No package installation or new large environment
is required on the current machine.

## Exact frozen conditions

The8-condition bank is byte-preserved in the [compact archive](../evidence/20261014-physical/practical-application-v2/manifest.json).
SHA256: `1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa`.
It includes poses, friction, nominal radii, delays and actual controller targets.
Repeating it across32 environments preserves the intended comparison.

From the task checkout root:

```bash
set -euo pipefail
task_repo="$PWD"
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/practical-reproduce-$(date -u +%Y%m%dT%H%M%SZ)"
task_snapshot="$task_root/native-source"
task_bank="$task_repo/research/rigid_stress/evidence/20261014-physical/practical-application-v2/blobs/1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa.npz"
task_revision=$(git rev-parse HEAD)
python3 research/rigid_stress/scripts/snapshot_practical_native.py --output "$task_snapshot"
mkdir -p "$task_root/cache/cuda"
```

The original899-file snapshot is exactly reconstructible from tracked checkpoint
`cd42aed09a36c1bb71eb00bc702e76da41967ac0`; the production run recorded c8333ea38.
The publisher independently matches every tracked source/asset Git blob to the
actual immutable snapshot. A newer task revision may be used only with its own
source manifest; do not relabel changed code as the old measurement.

## Four matched cost scopes on one allocated GPU

Create the batch script in the data directory. This runs four separate Python
processes on the same allocated GPU. Each scope has900 warmup steps and three
fixed2400-step repeats. No intrusive trajectory observer is enabled.

```bash
cat > "$task_root/job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /mnt/data/shared/config/env.sh
for task_scope in rigid live policy-rigid policy; do
    GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 /usr/bin/env \
        CUDA_CACHE_PATH="$task_root/cache/cuda" \
        bash "$task_snapshot/research/rigid_stress/scripts/run_practical_native_benchmark.sh" \
        "$task_root/\$task_scope" --scope "\$task_scope" \
        --envs 32 --level 1 --warmup 900 --steps 2400 --repetitions 3 \
        --minimum-seconds 0 --varied --seed 623016 \
        --contact-model intrinsic --intrinsic-support contact \
        --intrinsic-load-partition body_source_prior \
        --history 4 --output-mode max --egg-geometry mesh64 \
        --collision-method mpr --contact-patch off \
        --conditions "$task_bank" --source-revision "$task_revision" --compact-log
done
EOF
sbatch --parsable --array=0-0 --partition=rtx-mid --gres=gpu:1 \
    --cpus-per-task=8 --mem=48G --time=01:00:00 \
    --output="$task_root/job-%A_%a.log" "$task_root/job.sh"
```

For the fast simplified law, replace `--contact-model intrinsic` with
`--contact-model pressure` in a fresh experiment. Preserve all other physical
settings and rerun its paired baselines. Its diagnosed local-cone limitation
remains; a successful numerical run does not qualify its local pressure law.
Check scheduler completion, actual return codes, complete benchmark JSON,
source-before/after verification, zero invalid-step count and actual GPU UUIDs
before using any rates. The archived original commands and orchestration scripts
provide the exact measured process/container layout.

## Controlled raw-stress reward traces

Inside a GPU Slurm allocation using the same `gs-srun` prefix, run:

```bash
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 \
    bash "$task_repo/research/rigid_stress/scripts/run_practical_reward_proxy.sh" \
    "$task_root/reward-source-prior" "$task_snapshot" \
    --conditions "$task_bank" --model body_source_prior --low-limit 0.2 --radius-scale 1
```

Use fresh outputs for `--model pressure`, `--radius-scale 0.5`, and the original
`--low-limit 1` comparison. All arm targets, friction and weak/release timing stay
matched. Each run duplicates the8 conditions at low/4N strong-force limits and
records three1200-step restarts after900 warmup. Observation is intrusive and
its elapsed time is not throughput evidence. Raw NPZ traces retain maximum stress,
height, phase, actual force limits and exposure. Jointly assess lift and stress;
do not infer reliable decisions within the measured model/radius band.

Audit the already-computed whole-episode peak scalars without another simulation:

```bash
python3 research/rigid_stress/scripts/summarize_practical_reward_logs.py \
    --manifest research/rigid_stress/evidence/20261014-physical/practical-application-v2/manifest.json \
    --output "$task_root/whole-episode-peak-audit.json"
```

This stdlib log aggregation keeps whole-episode and lift-window conclusions
separate. Both raw time scopes can favor different force choices; a lift-window
reward comparison does not establish a whole-episode safety threshold.

## Reconstruct the published R/L/M inputs

Restore the compact contact packets into the data directory. This verifies all
stored/original file hashes and the restored source closure. It needs no fresh
contact solve and no full original2640-state audit.

```bash
python3 research/rigid_stress/scripts/restore_practical_run.py \
    --manifest research/rigid_stress/evidence/20261014-physical/practical-application-v2/manifest.json \
    --run practical-franka-four-contact-packets-v3 \
    --output "$task_root/contact-packets"
```

Inside a GPU Slurm allocation with `gs-srun`, run:

```bash
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 \
    bash "$task_repo/research/rigid_stress/scripts/reconstruct_franka_packets.sh" \
    "$task_root/reconstructed" "$task_snapshot" "$task_root/contact-packets" \
    --radii 0.003 0.006 --models pressure componentwise pair_union body_source_prior
```

For a bounded handoff check, use `--radii 0.006 --models body_source_prior`.
The actual published53-file packet restoration verifies1994054B exactly.
The fresh four-state handoff720758 completes at unchanged original numerical
gates; its largest raw-peak relative difference from the original32-field
comparison is1.010e-11. The [89-entry reproduction packet](../evidence/20261014-physical/practical-reproduction-v1/manifest.json)
retains its actual closure. Source dictionaries and old absolute provenance paths
are retained as evidence; consumers use the explicit restored packet CLI path.

The default supports complete max output. Switch the native benchmark to
`--output-mode full` when a full field is needed; its extra cost is a separate
scope. Existing [API documentation](NATIVE_API.md) and qualified max/full/reset
regressions remain applicable. Model, radius, material and uncertainty limits
are in [the application report](PRACTICAL_APPLICATION_RESULTS_20261016.md).
