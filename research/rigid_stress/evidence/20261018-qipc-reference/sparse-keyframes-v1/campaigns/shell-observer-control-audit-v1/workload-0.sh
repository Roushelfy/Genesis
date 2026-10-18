#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.compare_cadence --macro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group0/stock-sparse-r0 --micro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group0/stock-dense-r0 --kind shell --comparison-scope stress-sampling --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-dense-r0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-dense-r0.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-dense-r0.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-dense-r0.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.compare_cadence --macro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group0/stock-sparse-r0 --micro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group0/stock-sparse-r1 --kind shell --comparison-scope stress-sampling --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-sparse-r1 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-sparse-r1.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-sparse-r1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/stock-sparse-r1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.compare_cadence --macro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group1/volume-surface2-sparse-r0 --micro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group1/volume-surface2-dense-r0 --kind shell --comparison-scope stress-sampling --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-dense-r0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-dense-r0.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-dense-r0.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-dense-r0.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.compare_cadence --macro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group1/volume-surface2-sparse-r0 --micro /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/group1/volume-surface2-sparse-r1 --kind shell --comparison-scope stress-sampling --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-sparse-r1 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-sparse-r1.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-sparse-r1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-audit-v1/group0/volume-surface2-sparse-r1.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
