#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0
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
export PYTHONPATH=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/analysis-source:/mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/native-witness-consistency
campaign_status=0
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m examples.speed_benchmark.rigid_stress --level 1 --warmup 900 --steps 1200 --repetitions 3 --minimum-seconds 0 --varied --seed 623016 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --egg-geometry mesh64 --collision-method mpr --contact-patch off --history 0 --compact-log --source-revision 1ff39b0775200120737225fdfb8631c9b1fc5fa7 --envs 1 --scope rigid --contact-model pressure --intrinsic-load-partition body_source_prior --output-mode max --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/rigid.json > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/rigid.json.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/rigid.json.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/rigid.json.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m examples.speed_benchmark.rigid_stress --level 1 --warmup 900 --steps 1200 --repetitions 3 --minimum-seconds 0 --varied --seed 623016 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --egg-geometry mesh64 --collision-method mpr --contact-patch off --history 0 --compact-log --source-revision 1ff39b0775200120737225fdfb8631c9b1fc5fa7 --envs 1 --scope live --contact-model pressure --intrinsic-load-partition body_source_prior --output-mode max --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-max.json > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-max.json.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-max.json.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-max.json.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m examples.speed_benchmark.rigid_stress --level 1 --warmup 900 --steps 1200 --repetitions 3 --minimum-seconds 0 --varied --seed 623016 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --egg-geometry mesh64 --collision-method mpr --contact-patch off --history 0 --compact-log --source-revision 1ff39b0775200120737225fdfb8631c9b1fc5fa7 --envs 1 --scope live --contact-model pressure --intrinsic-load-partition body_source_prior --output-mode full --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-full.json > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-full.json.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-full.json.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/dpr-full.json.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m examples.speed_benchmark.rigid_stress --level 1 --warmup 900 --steps 1200 --repetitions 3 --minimum-seconds 0 --varied --seed 623016 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --egg-geometry mesh64 --collision-method mpr --contact-patch off --history 0 --compact-log --source-revision 1ff39b0775200120737225fdfb8631c9b1fc5fa7 --envs 1 --scope live --contact-model intrinsic --intrinsic-load-partition body_source_prior --output-mode max --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-max.json > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-max.json.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-max.json.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-max.json.returncode.txt
  campaign_status=1
fi
if /mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python -m examples.speed_benchmark.rigid_stress --level 1 --warmup 900 --steps 1200 --repetitions 3 --minimum-seconds 0 --varied --seed 623016 --conditions /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-production-throughput/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz --egg-geometry mesh64 --collision-method mpr --contact-patch off --history 0 --compact-log --source-revision 1ff39b0775200120737225fdfb8631c9b1fc5fa7 --envs 1 --scope live --contact-model intrinsic --intrinsic-load-partition body_source_prior --output-mode full --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-full.json > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-full.json.log 2>&1; then
  printf '%s\n' 0 > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-full.json.returncode.txt
else
  task_exit=$?
  printf '%s\n' "$task_exit" > /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/native-production-canonical-pilot-v1/group0/scr-full.json.returncode.txt
  campaign_status=1
fi
exit "$campaign_status"
