#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_port_bounds_v2.sh" off live
bash "$task_root/run_port_bounds_v2.sh" on live
