#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_contact_tail_storage_v1.py" > "$task_root/native-contact-tail-storage-publish-v1.log" 2>&1
