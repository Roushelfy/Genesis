#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/exact-602-socp-v1"
mkdir -p "$task_run/tmp" "$task_run/tools"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
hostname > "$task_run/node.txt"
"$RIGID_STRESS_ENV/bin/python" -m pip install --target "$task_run/tools" --no-deps clarabel > "$task_run/install.log" 2>&1
sha256sum "$task_root/probe_exact_602_socp_v1.py" > "$task_run/source.sha256"
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_exact_602_socp_v1.py" > "$task_run/probe.log" 2>&1
sha256sum "$task_root/probe_exact_602_socp_v1.py" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
