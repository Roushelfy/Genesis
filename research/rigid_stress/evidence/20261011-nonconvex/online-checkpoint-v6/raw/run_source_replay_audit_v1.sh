#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/source-replay-audit-v1"
mkdir -p "$task_root/source-replay-audit-launch-v1/tmp"
export TMPDIR="$task_root/source-replay-audit-launch-v1/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_root/audit_source_replay_v1.py" > "$task_root/source-replay-audit-launch-v1/audit.log" 2>&1
