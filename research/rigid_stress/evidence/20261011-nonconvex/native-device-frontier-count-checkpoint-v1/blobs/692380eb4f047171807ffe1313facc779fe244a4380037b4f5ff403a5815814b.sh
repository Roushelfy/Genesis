#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_native_device_frontier_count_regression_v1.sh" cuda
bash "$task_root/run_device_frontier_count_v1.sh" off live
bash "$task_root/run_device_frontier_count_v1.sh" on live
