#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_native=${2:?Immutable native snapshot required}
task_parent=${3:?Closed packet extraction run required}
shift 3
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export QD_OFFLINE_CACHE=0 NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
export CUDA_CACHE_PATH="$task_run/cache/cuda"
task_sources=(research/rigid_stress/{reconstruct_franka_packets,native_physical_sequence,finite_traction_reference,load_model_validation,profile_reference_sensitivity,native_load_samples,qualify_reference_scaling,spherical_shell_validation,validation_meshes}.py
    research/rigid_stress/scripts/{reconstruct_franka_packets,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
"$task_env/bin/python" - "$task_parent" "$task_run/source" <<'PY'
import hashlib,shutil,sys
from pathlib import Path
parent,destination=map(Path,sys.argv[1:])
assert (parent/'returncode.txt').read_text().strip()=='0'
assert (parent/'source.sha256').read_bytes()==(parent/'source-after.sha256').read_bytes()
for line in (parent/'source.sha256').read_text().splitlines():
    expected,relative=line.split('  ',1)
    source=parent/'source'/relative
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    target=destination/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
PY
cp --parents "${task_sources[@]}" "$task_run/source"
"$task_env/bin/python" - "$task_run/source" <<'PY' > "$task_run/source.sha256"
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
for path in sorted(root.rglob('*')):
    if path.is_file():print(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root)))
PY
(cd "$task_native"; sha256sum -c snapshot-files.sha256) > "$task_run/native-snapshot-before.txt"
cp "$task_native/snapshot.json" "$task_native/snapshot-files.sha256" "$task_run/"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    (cd "$task_run/source"; sha256sum -c "$task_run/source.sha256") > "$task_run/source-verification-after.txt"
    task_check=$?
    (cd "$task_run/source"; cut -c 67- "$task_run/source.sha256" | xargs sha256sum) > "$task_run/source-after.sha256"
    sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"
    cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"
    task_input_check=$?
    (cd "$task_native"; sha256sum -c snapshot-files.sha256) > "$task_run/native-snapshot-after.txt"
    task_native_check=$?
    if test "$task_status" = 0; then
        if test "$task_check" != 0; then task_status=$task_check
        elif test "$task_input_check" != 0; then task_status=$task_input_check
        elif test "$task_native_check" != 0; then task_status=$task_native_check
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
(
    cd "$task_run/source"
    export PYTHONPATH="$PWD:$task_native"
    "$task_env/bin/python" - "$task_native" <<'PY' > "$task_run/import-origins.txt"
import sys
from pathlib import Path
import genesis
from research.rigid_stress import reconstruct_franka_packets
assert Path(genesis.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert Path(reconstruct_franka_packets.__file__).resolve().is_relative_to(Path.cwd())
print(genesis.__file__)
print(reconstruct_franka_packets.__file__)
PY
    "$task_env/bin/python" -m research.rigid_stress.reconstruct_franka_packets \
        --packets "$task_parent" --output "$task_run/results" "$@" > "$task_run/reconstruction.log" 2>&1
)
