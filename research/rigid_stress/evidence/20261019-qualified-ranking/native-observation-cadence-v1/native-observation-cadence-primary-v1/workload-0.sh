#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
campaign_status=0
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method rigid --output-mode max --stress-stride 1 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/rigid-max-s1 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/rigid-max-s1.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/rigid-max-s1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/rigid-max-s1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode max --stress-stride 1 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s1 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s1.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode full --stress-stride 1 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s1 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s1.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode max --stress-stride 10 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s10 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s10.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s10.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s10.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode full --stress-stride 10 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s10 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s10.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s10.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s10.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode max --stress-stride 50 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s50 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s50.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s50.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-max-s50.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method dpr --output-mode full --stress-stride 50 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s50 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s50.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s50.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/dpr-full-s50.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode max --stress-stride 1 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s1 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s1.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode full --stress-stride 1 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s1 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s1.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s1.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s1.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode max --stress-stride 10 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s10 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s10.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s10.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s10.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode full --stress-stride 10 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s10 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s10.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s10.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s10.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode max --stress-stride 50 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s50 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s50.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s50.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-max-s50.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m research.rigid_stress.qipc_comparison.native_sparse_production --method scr --output-mode full --stress-stride 50 --envs 8 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --warmup 900 --steps 1200 --repetitions 3 --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s50 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s50.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s50.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-observation-cadence-primary-v1/group0/scr-full-s50.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
