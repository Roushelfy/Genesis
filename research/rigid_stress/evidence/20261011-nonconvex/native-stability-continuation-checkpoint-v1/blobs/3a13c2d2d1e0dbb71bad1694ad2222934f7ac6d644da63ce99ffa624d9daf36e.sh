#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/publish-shared-edge-fixture-v1"
test ! -e "$task_run"
mkdir -p "$task_run"
task_sources=("$task_root/publish_shared_edge_fixtures_v1.py" "$task_root/run_publish_shared_edge_fixtures_v1.sh" "$task_root/native-snapshot-repeat-capture-v1/b2048-623102/failure/before-after-recovery.npz")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/publish_shared_edge_fixtures_v1.py" > "$task_run/publication.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
