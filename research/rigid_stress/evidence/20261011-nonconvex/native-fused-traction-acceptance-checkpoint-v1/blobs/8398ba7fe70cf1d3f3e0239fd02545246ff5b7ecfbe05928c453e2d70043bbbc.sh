#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_fused_traction_acceptance_v1.sh" off live
bash "$task_root/run_fused_traction_acceptance_v1.sh" on live
