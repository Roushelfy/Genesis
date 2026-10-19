#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.finite_area_shell --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/finite-area-isolated-mesh-v1/group0/meshes/land-contact-1.npz --field /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-fixed-point-refinement-v1/group1/reference/order-0/load-0.1.npz --dt 0.001 --outer-iterations 80 --tolerance 1e-4 --velocity-tolerance 1e-10 --iterations 8000 --coarse-patches 12 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/group0/replay ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/group0/replay.log 2>&1
( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-physical-pad /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.finite_area_shell --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/finite-area-isolated-mesh-v1/group0/meshes/land-contact-1.npz --field /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-fixed-point-refinement-v1/group1/reference/order-0/load-0.1.npz --dt 0.001 --outer-iterations 80 --tolerance 1e-4 --velocity-tolerance 1e-10 --iterations 8000 --coarse-patches 12 --solve-pad-contact --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/group0/solved-pad ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/physical-pad-shell250-static-controls-v1/group0/solved-pad.log 2>&1
