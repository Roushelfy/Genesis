#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
failed=0
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/footprint-regression-v5-cpu" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu || failed=1
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/footprint-regression-v5-cuda" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda || failed=1
exit "$failed"
