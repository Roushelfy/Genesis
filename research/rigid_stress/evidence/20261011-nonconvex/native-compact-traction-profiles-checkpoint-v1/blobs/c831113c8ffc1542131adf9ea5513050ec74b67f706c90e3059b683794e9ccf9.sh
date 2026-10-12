#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
task_batch=${1:?Batch required}
task_seed=${2:?Seed required}
bash "$task_root/run_compact_traction_profiles_v4.sh" off live "$task_batch" "$task_seed"
bash "$task_root/run_compact_traction_profiles_v4.sh" on live "$task_batch" "$task_seed"
