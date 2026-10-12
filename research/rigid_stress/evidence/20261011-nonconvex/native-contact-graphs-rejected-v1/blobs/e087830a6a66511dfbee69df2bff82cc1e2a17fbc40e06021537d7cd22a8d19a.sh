#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_native_contact_graph_v3.sh" off profile
bash "$task_root/run_native_contact_graph_v3.sh" on profile
