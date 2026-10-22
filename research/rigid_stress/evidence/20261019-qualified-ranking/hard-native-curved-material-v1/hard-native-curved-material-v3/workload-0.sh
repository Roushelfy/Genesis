#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_curved_material
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_curved_material
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_native_curved_material --backend cpu --geometry /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-quadratic-rest-surface-v1/group0/geometry/geometry.npz --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/curved-material > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/curved-material.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/curved-material.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/curved-material.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_native_hertz_shell --backend cpu --quadrature 16 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/shell-q16 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/shell-q16.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/shell-q16.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/shell-q16.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_native_hertz_recovery --backend cpu --reference /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-hertz-residual-v1/group0/study/coarse.npz --far-field --rotated --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/normal-component > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/normal-component.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/normal-component.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group0/normal-component.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
