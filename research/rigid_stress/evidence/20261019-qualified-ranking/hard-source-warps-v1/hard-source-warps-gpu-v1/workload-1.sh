#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_source_warps
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_source_warps
campaign_status=0
export GS_CACHE_FILE_PATH="$task_cache/3c73aa81c1d5ceb9/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_cache/3c73aa81c1d5ceb9/quadrants"
if ( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_edge_projection && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_edge_projection /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_material_sources --backend gpu --precision 32 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group1/curved-material/estimated.npz --queries /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-edge-projection-gpu-v2/group1/baseline/fields.npz --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/baseline ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/baseline.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/baseline.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/baseline.returncode.txt
  campaign_status=1
fi
export GS_CACHE_FILE_PATH="$task_cache/e8f8f3fb31e9b4d9/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_cache/e8f8f3fb31e9b4d9/quadrants"
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.check_material_sources --backend gpu --precision 32 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-native-curved-material-v3/group1/curved-material/estimated.npz --queries /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-edge-projection-gpu-v2/group1/baseline/fields.npz --baseline /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/baseline/fields.npz --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/candidate > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/candidate.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/candidate.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-source-warps-gpu-v1/group1/candidate.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
