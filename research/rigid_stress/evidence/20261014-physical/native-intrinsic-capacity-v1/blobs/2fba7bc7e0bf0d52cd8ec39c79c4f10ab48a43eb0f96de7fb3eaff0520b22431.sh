#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{contact_reference,skfem_reference,bonded_block_reference}.py
    research/rigid_stress/scripts/{run_contact_friction_controls,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
task_status=0
for task_resolution in 3 4 5; do
    for task_shear in 0 5e-6; do
        task_name="bonded-n$task_resolution-s$task_shear"
        "$task_env/bin/python" -m research.rigid_stress.bonded_block_reference \
            --resolution "$task_resolution" --shear "$task_shear" --output "$task_run/$task_name" \
            > "$task_run/$task_name.log" 2>&1 || task_status=$?
    done
    for task_dhat in 2.5e-6 1.25e-6; do
        for task_fraction in 0.25 0.125; do
            task_increment=$("$task_env/bin/python" -c "print($task_dhat * $task_fraction)")
            task_name="ipc-n$task_resolution-d$task_dhat-i$task_fraction"
            task_code=0
            "$task_env/bin/python" -m research.rigid_stress.contact_reference \
                --resolution "$task_resolution" --dhat "$task_dhat" --friction 0.5 \
                --shear-after-close 5e-6 --contact-load-increment "$task_increment" \
                --output "$task_run/$task_name" > "$task_run/$task_name.log" 2>&1 || task_code=$?
            printf '%s\n' "$task_code" > "$task_run/$task_name.returncode"
            if test "$task_code" -ne 0; then task_status=$task_code; fi
        done
    done
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
