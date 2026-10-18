#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/group1
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.compare_fields --reference /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/roundoff-prescribed-v1/group0/snh --dpr /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/native-prescribed-sparse-v1/group0/dpr --scr /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/native-prescribed-sparse-v1/group1/scr --columns /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/surface-metric-v1/group0/metric/columns.npz --surfaces /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/surface-metric-v1/group0/metric/surfaces.npz --snapshot-times 1.33 2.77 4.0200000000000005 4.38 5.5200000000000005 --shell /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1/volume-surface2 --skip-snapshot-payloads --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/group1/volume-surface2 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/group1/volume-surface2.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/group1/volume-surface2.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-sparse-fields-nopayload-v1/group1/volume-surface2.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
