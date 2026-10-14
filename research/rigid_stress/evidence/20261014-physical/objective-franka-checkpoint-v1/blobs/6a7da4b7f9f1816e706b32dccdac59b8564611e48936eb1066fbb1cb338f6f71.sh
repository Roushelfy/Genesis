#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
task_sources=(research/rigid_stress/capture_franka_reference.py research/rigid_stress/scripts/{capture_franka_reference,cluster_env}.sh
    examples/rigid/{franka_egg_stress,egg_stress_controller}.py genesis/options/rigid_stress.py
    genesis/assets/xml/franka_emika_panda/panda.xml)
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(rg --files genesis/engine/solvers/rigid -g '*.py')
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(rg --files examples/rigid/assets/hollow_egg/level1)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata
import json
import platform
import sys
print(json.dumps({"python": sys.version, "platform": platform.platform(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-fem", "ipctk", "quadrants", "torch")}}, indent=2))
PY
"$task_env/bin/python" -m research.rigid_stress.capture_franka_reference --output "$task_run/results" "$@" \
    > "$task_run/capture.log" 2>&1
