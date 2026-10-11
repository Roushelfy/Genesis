#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/profile-surface-fan-v2"
mkdir -p "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/audit_continuum_cones_v7.py" > "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/continuum-cone-v7.log" 2>&1
bash "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/run_profile_surface_fan_v2.sh"
