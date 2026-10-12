#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
"$RIGID_STRESS_ENV/bin/python" "$task_root/extract_602_fixture_v2.py" > "$task_root/extract-602-fixture-v2.log" 2>&1
