#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-capped-roundoff
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-capped-roundoff
campaign_status=0
if /mnt/data/zhaofeng/venvs/qipc-owned-coarse-tree-check01-20261015/bin/python3.12 -m research.rigid_stress.qipc_comparison.contact_window --baseline /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-fresh-coarse-v1/group0/case0 --protocol capped --duration 1.05 --window-start 0.85 --keyframes 0.89 0.95 1.05 --timing-mode whole-window --prewarm-elastic-observer --defer-field-io --linear-solver partition_pcg --linear-tolerance 1e-2 --velocity-tolerance 1e-7 --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/qipc-owned-coarse-tree-check01-20261015/bin/python3.12 -m research.rigid_stress.qipc_comparison.contact_window --baseline /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-fresh-coarse-v1/group0/case0 --protocol capped --duration 1.05 --window-start 0.85 --keyframes 0.89 0.95 1.05 --timing-mode whole-window --prewarm-elastic-observer --defer-field-io --linear-solver partition_pcg --linear-tolerance 1e-2 --velocity-tolerance 1e-8 --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose-newton-mid > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose-newton-mid.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose-newton-mid.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-partition-loose-v1/group0/cap-partition-loose-newton-mid.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
