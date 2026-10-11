#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/fit-diagnosis-fan-v1"
mkdir -p "$run/tmp" "$run/source"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$run/cache/quadrants"
export GS_CACHE_FILE_PATH="$run/cache/genesis"
sources=(genesis/engine/solvers/rigid/stress/*.py research/rigid_stress/diagnose_native_traction_snapshot.py)
sha256sum "${sources[@]}" > "$run/source.sha256"
cp --parents "${sources[@]}" "$run/source"
hostname > "$run/node.txt"
failed=0
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.diagnose_native_traction_snapshot --run-dir "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/live-mapping-v3" --anchor-fan --output "$run/results.json" > "$run/diagnosis.log" 2>&1 || failed=1
sha256sum "${sources[@]}" > "$run/source-after.sha256"
cmp "$run/source.sha256" "$run/source-after.sha256"
exit "$failed"
