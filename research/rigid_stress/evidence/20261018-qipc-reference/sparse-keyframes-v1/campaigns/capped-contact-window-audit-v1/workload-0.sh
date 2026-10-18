#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0
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
export PYTHONPATH=/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-capped-roundoff
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-capped-roundoff
campaign_status=0
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.reference_uncertainty --reference /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-control --candidates /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-partition-original /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-loose /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-loose-newton-mid --surface-mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/controls-v1/group0/inputs/base1-surface2-wall4.npz --no-column-export --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-fields > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-fields.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-fields.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-fields.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.reaction_history --reference /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-control --candidates /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-partition-original /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-loose /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1/group0/cap-linear-loose-newton-mid --layout /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/reaction-layout-v1/group0/layout/layout.npz --finger-names left_finger right_finger --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-reactions > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-reactions.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-reactions.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-reactions.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.window_audit --profiles /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-v1 --fields /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-fields/result.json --reactions /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/keyframe-reactions --plan /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/meta/capped-contact-window-v1-plan.json --reference-name cap-linear-control --no-observer-control --output /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/window-audit > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/window-audit.log 2>&1; then
  printf '%s\n' 0 > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/window-audit.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/capped-contact-window-audit-v1/group0/window-audit.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
