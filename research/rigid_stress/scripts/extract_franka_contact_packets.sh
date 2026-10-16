#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_parent=${2:?Closed reference source run required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
task_sources=(research/rigid_stress/{extract_franka_contact_packets,compare_original_contact_feedback}.py
    research/rigid_stress/scripts/{extract_franka_contact_packets,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
"$task_env/bin/python" - "$task_parent" "$task_run/source" <<'PY'
import hashlib,shutil,sys
from pathlib import Path
parent,destination=map(Path,sys.argv[1:])
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
    if test "$task_status" = 0; then
        if test "$task_check" != 0; then task_status=$task_check
        elif test "$task_input_check" != 0; then task_status=$task_input_check
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
cp "$task_parent/source.sha256" "$task_run/parent-source.sha256"
(
    cd "$task_run/source"
    export PYTHONPATH="$PWD"
    "$task_env/bin/python" -m research.rigid_stress.extract_franka_contact_packets \
        --output "$task_run/results" "$@" > "$task_run/extraction.log" 2>&1
)
