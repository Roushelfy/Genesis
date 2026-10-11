#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-final-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=59b16811-native-wrench-block-residual-moments-v4
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_contact_moments_tests_v4/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_contact_moments_tests_genesis_v4/$SLURM_JOB_ID"
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress.py --backend=gpu -n 4 --timeout=1200 > "$task_run/native-contact-moments-gpu-v4.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress.py --backend=cpu -n 4 --timeout=1200 > "$task_run/native-contact-moments-cpu-v4.log" 2>&1
