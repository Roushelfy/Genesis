#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_port_bounds_v2.sh" off profile
bash "$task_root/run_port_bounds_v2.sh" on profile
bash "$task_root/run_native_port_bounds_regression_v2.sh" cuda
