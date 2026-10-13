#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_sampler_warp_live_v1.sh" off 16384 734001
bash "$task_root/run_sampler_warp_live_v1.sh" on 16384 734001
