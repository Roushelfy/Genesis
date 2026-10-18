#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/kashu-shell-pr22
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/kashu-shell-pr22
campaign_status=0
if ( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/kashu-shell-pr22-diagnostics && env PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/kashu-shell-pr22-diagnostics /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.shell_grasp --model shell --envs 1 --precision 64 --conditions /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/qualify-tight-v1/group0/evaluation/case0.npz --inputs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/qualify-tight-v1/group0/evaluation/surface2 --wall-volume-input /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/wall-and-complete-fields-v1/group0/wall/surface2.npz --motion /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/prescribed-input-v1/group0/motion/motion.npz --sim-dt 0.001 --substeps 1 --steps 6000 --record-every 10 --coupling-friction 0.7191016138165071 --instrumented --iterations 300 --tolerance 1e-8 --coarse-interval 1 --keyframes 1.33 2.77 4.0200000000000005 4.38 5.5200000000000005 --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1/volume-surface2 ) > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1/volume-surface2.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1/volume-surface2.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-prescribed-sparse-v1/group1/volume-surface2.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
