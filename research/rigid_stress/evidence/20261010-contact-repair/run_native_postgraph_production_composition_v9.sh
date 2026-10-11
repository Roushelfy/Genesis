#!/usr/bin/env bash
set -euo pipefail
task_seed=${1:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-postgraph-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=460e87f5-native-postgraph-production-v9
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_production_composition_v9/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_production_genesis_composition_v9/$SLURM_JOB_ID"
task_inputs=()
for task_batch in 1024 2048; do
    for task_scope in live policy; do
        task_routes=(off on)
        if [[ "$task_scope" == policy ]]; then task_routes=(on off); fi
        for task_route in "${task_routes[@]}"; do
            task_name="native-postgraph-production-$task_route-$task_scope-b$task_batch-seed$task_seed-v9"
            "$RIGID_STRESS_ENV/bin/python" examples/speed_benchmark/rigid_stress.py --rigid-post-fusion "$task_route" --scope "$task_scope" --envs "$task_batch" --seed "$task_seed" --varied --warmup 900 --steps 2400 --repetitions 3 --minimum-seconds 10 --compact-log --output "$task_run/$task_name.json" > "$task_run/$task_name.log" 2>&1
            task_inputs+=("$task_run/$task_name.json")
        done
    done
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_benchmarks.py --output "$task_run/native-postgraph-production-composition-seed$task_seed-v9-summary.json" "${task_inputs[@]}" > "$task_run/native-postgraph-production-composition-seed$task_seed-v9-summary.log" 2>&1
