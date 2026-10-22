#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_quadric_curvature
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_quadric_curvature
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_quadric_curvature --backend cpu --fit quadric --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/quadric > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/quadric.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/quadric.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/quadric.returncode.txt
  campaign_status=1
fi
if ( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_parallel_boundary && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_parallel_boundary /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_quadric_curvature --backend cpu --fit baseline --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/baseline ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/baseline.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/baseline.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/baseline.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_precomputed_surface --backend cpu --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/surface > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/surface.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/surface.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/surface.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_hertz_regions --backend cpu --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/regions > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/regions.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/regions.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-quadric-curvature-v1/group0/regions.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
