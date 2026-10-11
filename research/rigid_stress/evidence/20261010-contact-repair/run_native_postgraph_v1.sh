#!/usr/bin/env bash
set -euo pipefail
task_batch=${1:?batch required}
task_seed=${2:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=460e87f5-native-production-indices-v8-postgraph-probe-v1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_v1/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_genesis_v1/$SLURM_JOB_ID"
if [[ "$task_batch" == 1024 ]]; then
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_native_postgraph.py --fused-post --oracle --envs 16 --seed "$task_seed" --steps 2400 --output-mode full --output "$task_run/postgraph-full-oracle-seed$task_seed-v1.json" > "$task_run/postgraph-full-oracle-seed$task_seed-v1.log" 2>&1
fi
for task_variant in original fused; do
    task_flags=()
    if [[ "$task_variant" == fused ]]; then task_flags=(--fused-post); fi
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_native_postgraph.py "${task_flags[@]}" --scope policy --envs "$task_batch" --seed "$task_seed" --varied --warmup 900 --steps 2400 --repetitions 3 --minimum-seconds 0 --compact-log --output "$task_run/postgraph-$task_variant-policy-b$task_batch-seed$task_seed-v1.json" > "$task_run/postgraph-$task_variant-policy-b$task_batch-seed$task_seed-v1.log" 2>&1
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_benchmarks.py --output "$task_run/postgraph-policy-b$task_batch-seed$task_seed-v1-summary.json" "$task_run/postgraph-original-policy-b$task_batch-seed$task_seed-v1.json" "$task_run/postgraph-fused-policy-b$task_batch-seed$task_seed-v1.json" > "$task_run/postgraph-policy-b$task_batch-seed$task_seed-v1-summary.log" 2>&1
