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
task_sources=(research/rigid_stress/{force_control_blocks,contact_reference,skfem_reference}.py
    research/rigid_stress/scripts/run_force_control_blocks.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
printf '%s\n' "$*" > "$task_run/arguments.txt"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.force_control_blocks --output "$task_run/solution" "$@" \
    > "$task_run/contact.log" 2>&1 || task_status=$?
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
