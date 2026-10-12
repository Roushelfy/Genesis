#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_sample_lane_retry_v1.sh" 0 live
bash "$task_root/run_sample_lane_retry_v1.sh" 8 live
