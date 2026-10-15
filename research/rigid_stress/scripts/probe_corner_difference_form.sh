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
task_sources=(research/rigid_stress/{probe_corner_difference_form,diagnose_native_corner_arithmetic,summarize_fine_references}.py
    research/rigid_stress/scripts/{probe_corner_difference_form,cluster_env}.sh)
"$task_env/bin/python" - <<'PY' > "$task_run/additional-sources.txt"
from pathlib import Path
paths=sorted(Path("genesis/engine/solvers/rigid/stress").rglob("*.py"))
assert len(paths)>10
print("\n".join(str(path) for path in [*paths,Path("genesis/options/rigid_stress.py")]))
PY
while IFS= read -r task_source; do task_sources+=("$task_source"); done < "$task_run/additional-sources.txt"
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/hardware.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/import-origin.json"
import json,genesis
from pathlib import Path
import research.rigid_stress.probe_corner_difference_form as task_module
assert Path(genesis.__file__).resolve().is_relative_to(Path.cwd())
assert Path(task_module.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({"genesis":genesis.__file__,"diagnostic":task_module.__file__},indent=2))
PY
"$task_env/bin/python" -m research.rigid_stress.probe_corner_difference_form --output "$task_run/results" "$@" \
    > "$task_run/replay.log" 2>&1
