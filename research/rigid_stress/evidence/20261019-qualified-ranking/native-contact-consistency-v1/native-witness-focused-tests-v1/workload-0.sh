#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0
test ! -e "$task_run"
mkdir -p "$task_run"
task_scratch=''
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; if test -n "$task_scratch"; then rm -rf -- "$task_scratch"; fi; exit "$task_status"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
task_scratch=$(mktemp -d /tmp/gs-shell-compare.XXXXXX)
task_cache="$task_scratch/cache"
task_tmp="$task_scratch/tmp"
printf "%s\n" "$task_scratch" > "$task_run/node-local-scratch.txt"
df -Pk "$task_scratch" > "$task_run/node-local-filesystem.txt"
mkdir -p "$task_tmp" "$task_cache"
export TMPDIR="$task_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GS_CACHE_FILE_PATH="$task_cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_cache/quadrants"
export CUDA_CACHE_PATH="$task_cache/cuda" NUMBA_CACHE_DIR="$task_cache/numba"
export MPLCONFIGDIR="$task_cache/matplotlib"
export QIPC_RTC_CACHE="$task_cache/qipc_rtc"
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
campaign_status=0
if ( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.run_focused_tests --tests /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency/tests/rigid/test_stress_witness.py --backend gpu --select 'native_collision_witnesses_and_lifecycle and False-False' --engine-root /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/before ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/before.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/before.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/before.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.run_focused_tests --tests /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency/tests/rigid/test_stress_witness.py --backend gpu --select native_collision_witnesses_and_lifecycle --engine-root /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/after > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/after.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/after.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-witness-focused-tests-v1/group0/after.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
