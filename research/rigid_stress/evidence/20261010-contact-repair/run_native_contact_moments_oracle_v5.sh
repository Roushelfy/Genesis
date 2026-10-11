#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-final-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=59b16811-native-wrench-block-residual-moments-v5
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_contact_moments_oracle_v5/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_contact_moments_oracle_genesis_v5/$SLURM_JOB_ID"
for task_seed in 510000 623001; do
    "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/native_trajectory_oracle.py --envs 16 --steps 1800 --seed "$task_seed" --output-mode full --contact-wrench-reuse on --packed-block-size 512 --coalesced-residual on --contact-moment-reuse on --output "$task_run/native-contact-moments-oracle-seed$task_seed-v5.json" > "$task_run/native-contact-moments-oracle-seed$task_seed-v5.log" 2>&1
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/native_trajectory_oracle.py --envs 8 --steps 1800 --seed 510000 --substeps 4 --scatter-tasks-per-env 1 --output-mode full --contact-wrench-reuse on --packed-block-size 512 --coalesced-residual on --contact-moment-reuse on --output "$task_run/native-contact-moments-overflow-substeps4-v5.json" > "$task_run/native-contact-moments-overflow-substeps4-v5.log" 2>&1
