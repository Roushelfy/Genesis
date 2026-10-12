#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_segment_bounds_v1.sh" off profile
bash "$task_root/run_segment_bounds_v1.sh" on profile
