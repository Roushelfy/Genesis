#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_module=${2:?Native validation module required}
shift 2
test "$task_module" = load_model_validation || test "$task_module" = qualify_intrinsic_path_heap || test "$task_module" = audit_native_finite_models
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 QD_OFFLINE_CACHE=0
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
sha256sum -c snapshot-files.sha256 > "$task_run/snapshot-before-verification.txt"
cp snapshot.json snapshot-files.sha256 "$task_run/"
task_sources=()
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(
    python3 - <<'PY'
from pathlib import Path
files = list(Path('genesis').rglob('*.py')) + list(Path('research/rigid_stress').glob('*.py'))
files += [Path('research/rigid_stress/scripts/run_isolated_native_validation.sh'),
          Path('research/rigid_stress/scripts/cluster_env.sh')]
for path in sorted(set(files)):
    print(path)
PY
)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
    task_hash_status=$?
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
    task_compare_status=$?
    sha256sum -c snapshot-files.sha256 > "$task_run/snapshot-after-verification.txt"
    task_snapshot_status=$?
    if test "$task_status" = 0; then
        if test "$task_hash_status" != 0; then task_status=$task_hash_status
        elif test "$task_compare_status" != 0; then task_status=$task_compare_status
        elif test "$task_snapshot_status" != 0; then task_status=$task_snapshot_status
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$task_module" "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import genesis
assert Path(genesis.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({'genesis':genesis.__file__,'python':sys.version,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in
                             ('numpy','scipy','scikit-fem','pyamg','quadrants','torch')}},indent=2))
PY
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
fi
"$task_env/bin/python" -m "research.rigid_stress.$task_module" \
    --output "$task_run/results" "$@" > "$task_run/validation.log" 2>&1
