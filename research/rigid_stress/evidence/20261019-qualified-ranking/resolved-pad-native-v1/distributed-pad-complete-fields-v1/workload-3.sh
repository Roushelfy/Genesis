#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-complete-fields-v1/group3
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-complete-fields-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m research.rigid_stress.qipc_comparison.contact_fem_exact_columns --meshes /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/finite-area-controls-v1/group0/native-meshes/land-contact-0.npz /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/finite-area-controls-v1/group0/native-meshes/land-contact-0.npz --runs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-transfer-cpu-v1/group0/comparison/frames/distributed_outer-level1.npz /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-every-step-v1/group0/native/frames/stress-level1.npz --loads 0.1 --surface-quadrature 25 --depth-quadrature 4 --physical-volume --common-peaks --export-surface-fields --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-complete-fields-v1/group3/comparison > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/distributed-pad-complete-fields-v1/group3/comparison.log 2>&1
