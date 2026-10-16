#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/source-snapshots/native-intrinsic-heap-edge-mutation-v1
source research/rigid_stress/scripts/cluster_env.sh
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/intrinsic-path-heap-mutated-edge-regression-v1
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" QD_OFFLINE_CACHE=0 PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
mapfile -t task_sources < <(python3 -c 'from pathlib import Path; print("\n".join(str(p) for root in ["genesis","tests"] for p in sorted(Path(root).rglob("*.py"))))')
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(python3 -c 'from pathlib import Path; print("\n".join(str(p) for p in sorted(Path("research/rigid_stress").glob("*.py"))))')
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
cp intentional-mutation.patch "$task_run/"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -m pytest -n 0 --backend cpu -m required tests/rigid/test_stress_footprint.py -k u-False-False-False --maxfail=1 --basetemp "$task_run/pytest-tmp" -o "cache_dir=$task_run/pytest-cache" --junitxml "$task_run/junit.xml" > "$task_run/pytest.log" 2>&1
