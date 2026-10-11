#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-postgraph-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=460e87f5-native-postgraph-production-v9
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_production_oracles_v9/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_production_genesis_oracles_v9/$SLURM_JOB_ID"
task_inputs=()
for task_seed in 510000 623001; do
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/verify_native_postgraph.py --seed "$task_seed" --output "$task_run/native-postgraph-production-equivalence-seed$task_seed-v9.json" > "$task_run/native-postgraph-production-equivalence-seed$task_seed-v9.log" 2>&1
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/verify_native_index_resets.py --seed "$task_seed" --output "$task_run/native-postgraph-production-index-reset-seed$task_seed-v9.json" > "$task_run/native-postgraph-production-index-reset-seed$task_seed-v9.log" 2>&1
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/native_trajectory_oracle.py --envs 16 --steps 1800 --seed "$task_seed" --output-mode full --contact-wrench-reuse on --packed-block-size 512 --coalesced-residual on --contact-moment-reuse on --output "$task_run/native-postgraph-production-full-oracle-seed$task_seed-v9.json" > "$task_run/native-postgraph-production-full-oracle-seed$task_seed-v9.log" 2>&1
    task_inputs+=("$task_run/native-postgraph-production-full-oracle-seed$task_seed-v9.json")
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/native_trajectory_oracle.py --envs 8 --steps 1800 --seed 510000 --substeps 4 --scatter-tasks-per-env 1 --output-mode full --contact-wrench-reuse on --packed-block-size 512 --coalesced-residual on --contact-moment-reuse on --output "$task_run/native-postgraph-production-overflow-substeps4-v9.json" > "$task_run/native-postgraph-production-overflow-substeps4-v9.log" 2>&1
task_inputs+=("$task_run/native-postgraph-production-overflow-substeps4-v9.json")
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_native_audits.py --output "$task_run/native-postgraph-production-full-oracle-v9-summary.json" "${task_inputs[@]}" > "$task_run/native-postgraph-production-full-oracle-v9-summary.log" 2>&1
