#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/extract_pair_support_fixtures_v1.py"
