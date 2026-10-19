#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking
( cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-pad-friction && env PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/shell-pad-friction /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.plate_shell_coupled --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/finite-area-isolated-mesh-v1/group0/meshes/land-contact-1.npz --pad-geometry /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-shell-spatial-v2/group1/geometry/pad-geometry.npz --loads 0.1 0.1 --dt 0.001 --ramp 0.05 --hold 0.15 --record-stride 1 --tolerance 1e-4 --velocity-tolerance 1e-10 --iterations 8000 --coarse-patches 12 --floating-sums --friction 0.3 --gravity-x 1 --gravity-start 0.2 --gravity-ramp 0.05 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0/fine ) > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0/fine.log 2>&1
/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.audit_coupled_shell --runs /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0/fine --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0/audit > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/compliant-pad-friction250-history-v1/group0/audit.log 2>&1
