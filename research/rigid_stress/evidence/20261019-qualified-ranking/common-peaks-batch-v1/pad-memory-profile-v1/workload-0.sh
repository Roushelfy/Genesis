#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.pad_memory_profile --task /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/memory-b1-full-dpr-v1.json --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/dpr > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/dpr.log 2>&1
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.audit_pad_batch --runs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/dpr/run --data-root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/dpr-audit > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/dpr-audit.log 2>&1
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.pad_memory_profile --task /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/memory-b1-full-scr-v1.json --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/scr > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/scr.log 2>&1
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.audit_pad_batch --runs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/scr/run --data-root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/scr-audit > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/scr-audit.log 2>&1
( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.pad_memory_profile --task /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/memory-b1-full-shell-v1.json --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/shell ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/shell.log 2>&1
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.audit_pad_batch --runs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/shell/run --data-root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/shell-audit > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/pad-memory-profile-v1/group0/shell-audit.log 2>&1
