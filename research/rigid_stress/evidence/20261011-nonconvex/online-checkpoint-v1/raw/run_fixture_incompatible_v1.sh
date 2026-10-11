#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/export_fixture_incompatible_v1.py" > "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/export_fixture_incompatible_v1.log" 2>&1
