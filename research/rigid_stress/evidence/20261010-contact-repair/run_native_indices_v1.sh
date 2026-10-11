#!/usr/bin/env bash
set -euo pipefail
task_batch=${1:?batch required}
task_seed=${2:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-lifecycle-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=59b16811-native-scheduling-v6-controller-v3-link-invalidation-v4-indices-v1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_indices_v1/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_indices_genesis_v1/$SLURM_JOB_ID"
task_variants=(original ranges resets combined)
if [[ "$task_batch" == 32768 ]]; then task_variants=(original combined); fi
for task_variant in "${task_variants[@]}"; do
    task_flags=()
    case "$task_variant" in
        ranges) task_flags=(--control-ranges) ;;
        resets) task_flags=(--resident-resets) ;;
        combined) task_flags=(--control-ranges --resident-resets) ;;
    esac
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_native_indices.py "${task_flags[@]}" --scope policy --envs "$task_batch" --seed "$task_seed" --varied --warmup 900 --steps 2400 --repetitions 3 --minimum-seconds 0 --compact-log --output "$task_run/native-indices-$task_variant-policy-b$task_batch-seed$task_seed-v1.json" > "$task_run/native-indices-$task_variant-policy-b$task_batch-seed$task_seed-v1.log" 2>&1
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_benchmarks.py --output "$task_run/native-indices-policy-b$task_batch-seed$task_seed-v1-summary.json" "$task_run"/native-indices-*-policy-b"$task_batch"-seed"$task_seed"-v1.json > "$task_run/native-indices-policy-b$task_batch-seed$task_seed-v1-summary.log" 2>&1
