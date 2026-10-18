#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/group0
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-roundoff
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-roundoff
campaign_status=0
if /mnt/data/zhaofeng/venvs/qipc-owned-coarse-tree-check01-20261015/bin/python3.12 -m research.rigid_stress.qipc_comparison.prescribed_reference --motion /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-input-v1/group0/motion/motion.npz --conditions /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/qualify-tight-v1/group0/evaluation/case0.npz --mjcf /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-qipc-validation/genesis/assets/xml/franka_emika_panda/panda.xml --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/controls-v1/group0/inputs/base1-surface1-wall4.npz --duration 6.0 --energy snh --dbc-stiffness 1e7 --linear-tolerance 1e-2 --velocity-tolerance 1e-9 --keyframes 1.33 2.77 4.0200000000000005 4.38 5.5200000000000005 --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/group0/snh > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/group0/snh.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/group0/snh.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reference-prescribed-sparse-v1/group0/snh.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
