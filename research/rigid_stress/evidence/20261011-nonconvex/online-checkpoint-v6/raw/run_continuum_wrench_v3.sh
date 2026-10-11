#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/continuum-wrench-v3"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
cp "$task_root/audit_continuum_wrench_v2.py" "$task_run/source/"
sha256sum "$task_root/audit_continuum_wrench_v2.py" > "$task_run/source.sha256"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$RIGID_STRESS_ENV/bin/python" "$task_root/audit_continuum_wrench_v2.py" --snapshots "$task_root/online-square-fan-v1/results/failure/snapshot-710.npz" --output "$task_run/audit.json" > "$task_run/audit.log" 2>&1
sha256sum "$task_root/audit_continuum_wrench_v2.py" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
