#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/exact-602-search-continuous-v1"
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_exact_602_search_continuous_v1.py" > "$task_root/exact-602-search-continuous-v1.log" 2>&1
sha256sum "$task_root/probe_exact_602_search_continuous_v1.py" "$task_run/summary.json" > "$task_run/source.sha256"
