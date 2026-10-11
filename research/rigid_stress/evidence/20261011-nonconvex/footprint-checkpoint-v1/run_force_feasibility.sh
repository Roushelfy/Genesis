#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/fit-diagnosis-v1"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$run/inspect_feasibility.py" > "$run/force-feasibility.log" 2>&1
