#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/footprint-publication-v2"
mkdir -p "$run/tmp"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/archive_footprint_checkpoint_v2.py" --source-revision 273e67ba6e0ac600de1d4d27b9df0f8154610be7 --repo "$PWD" --raw "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex" --destination "$PWD/research/rigid_stress/evidence/20261011-nonconvex/footprint-checkpoint-v1" > "$run/archive.log" 2>&1
