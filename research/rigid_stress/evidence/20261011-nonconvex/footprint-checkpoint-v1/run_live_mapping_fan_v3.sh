#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/live-mapping-fan-v3"
mkdir -p "$run/tmp" "$run/source"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$run/cache/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$run/cache/quadrants"
sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/rigid_solver.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py research/rigid_stress/probe_nonconvex_live_mapping.py)
sha256sum "${sources[@]}" > "$run/source.sha256"
cp --parents "${sources[@]}" "$run/source"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.csv"
hostname > "$run/node.txt"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)' > "$run/import.log" 2>&1
failed=0
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.probe_nonconvex_live_mapping --envs 4 --steps 640 --seed 623001 --anchor-fan --output "$run/results.json" > "$run/probe.log" 2>&1 || failed=1
sha256sum "${sources[@]}" > "$run/source-after.sha256"
cmp "$run/source.sha256" "$run/source-after.sha256"
exit "$failed"
