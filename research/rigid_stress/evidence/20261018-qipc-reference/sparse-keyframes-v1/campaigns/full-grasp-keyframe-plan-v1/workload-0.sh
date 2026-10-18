#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/group0
test ! -e "$task_run"
mkdir -p "$task_run"
task_scratch=''
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; if test -n "$task_scratch"; then rm -rf -- "$task_scratch"; fi; exit "$task_status"' EXIT
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.keyframe_plan --references /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-fresh-coarse-v1/group0/case0 /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-fresh-coarse-v1/group1/case2 /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-roundoff-reference-v1/group2/case7 --motion /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-input-v1/group0/motion/motion.npz --phases 0.22 0.46 0.67 0.73 0.92 --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/group0/phase-keys > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/group0/phase-keys.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/group0/phase-keys.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/full-grasp-keyframe-plan-v1/group0/phase-keys.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
