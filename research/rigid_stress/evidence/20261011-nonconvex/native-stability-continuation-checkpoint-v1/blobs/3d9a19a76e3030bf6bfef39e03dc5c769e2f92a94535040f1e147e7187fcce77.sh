#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-group-failure-audit-v2"
test ! -e "$task_run"
mkdir -p "$task_run/source"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/audit_native_group_failure_v2.py" "$task_root/run_native_group_failure_audit_v2.sh" "$task_root/native-certified-failure-capture-v1/b4096-623103/failure/before-rigid-integration.npz" research/rigid_stress/{wrench,mechanics,field_cpu}.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/audit_native_group_failure_v2.py" > "$task_run/audit.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
