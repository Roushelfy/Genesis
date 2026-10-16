#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh application benchmark directory required}
shift
test -f snapshot-files.sha256
test ! -e "$task_run"
mkdir -p "$task_run/results" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export QD_OFFLINE_CACHE=0 NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
sha256sum -c snapshot-files.sha256 > "$task_run/snapshot-before-verification.txt"
cp snapshot.json snapshot-files.sha256 "$task_run/"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    sha256sum -c snapshot-files.sha256 > "$task_run/snapshot-after-verification.txt"
    task_hash_status=$?
    "$task_env/bin/python" - "$PWD" <<'PY' > "$task_run/snapshot-inventory-after.json"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);expected={row['path'] for row in json.loads((root/'snapshot.json').read_text())['files']}
actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
extra=actual-expected-{'snapshot.json','snapshot-files.sha256'};missing=expected-actual
print(json.dumps({'extra_files':sorted(extra),'missing_files':sorted(missing),'passed':not extra and not missing},indent=2))
assert not extra and not missing
PY
    task_inventory_status=$?
    if test "$task_status" = 0; then
        if test "$task_hash_status" != 0; then task_status=$task_hash_status
        elif test "$task_inventory_status" != 0; then task_status=$task_inventory_status
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata,json,platform,sys
from pathlib import Path
import genesis
from examples.rigid import franka_egg_stress
assert Path(genesis.__file__).resolve().is_relative_to(Path.cwd())
assert Path(franka_egg_stress.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({'python':sys.version,'platform':platform.platform(),'genesis':genesis.__file__,
                 'application':franka_egg_stress.__file__,
                 'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','quadrants','torch','trimesh','mujoco']}},indent=2))
PY
"$task_env/bin/python" -m examples.speed_benchmark.rigid_stress \
    "$@" --output "$task_run/results/benchmark.json" > "$task_run/benchmark.log" 2>&1
