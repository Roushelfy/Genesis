#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{skfem_reference,contact_reference}.py
    research/rigid_stress/scripts/run_contact_convergence.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
task_status=0
for task_resolution in 3 4 5; do
    for task_dhat in 1e-5 5e-6 2.5e-6 1.25e-6; do
        task_case="$task_run/n${task_resolution}-d${task_dhat}"
        task_case_status=0
        "$task_env/bin/python" -m research.rigid_stress.contact_reference \
            --resolution "$task_resolution" --dhat "$task_dhat" --friction 0 \
            --output "$task_case" > "$task_case.log" 2>&1 || task_case_status=$?
        printf '%s\n' "$task_case_status" > "$task_case.returncode"
        if [[ "$task_case_status" != 0 ]]; then task_status=1; fi
    done
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
