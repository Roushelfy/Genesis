#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/validate-native-body-baseline-v1"
test ! -e "$task_run"
mkdir -p "$task_run"
sha256sum "$task_root/validate_native_body_baseline_v1.py" "$task_root/run_validate_native_body_baseline_v1.sh" > "$task_run/source.sha256"
hostname > "$task_run/node.txt"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/validate_native_body_baseline_v1.py" > "$task_run/validation.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "$task_root/validate_native_body_baseline_v1.py" "$task_root/run_validate_native_body_baseline_v1.sh" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
