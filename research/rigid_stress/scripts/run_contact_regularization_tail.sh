#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Tail index required}}
test "$task_index" -ge 0 && test "$task_index" -lt 6
task_resolution=$((3 + task_index / 2))
task_dhats=(6.25e-7 3.125e-7)
task_increments=(1.5625e-7 7.8125e-8)
task_level=$((task_index % 2))
bash research/rigid_stress/scripts/run_contact_reference.sh \
    "$task_root/contact-friction-tail-n$task_resolution-l$task_level-v1" \
    --resolution "$task_resolution" --dhat "${task_dhats[task_level]}" --friction 0.5 \
    --shear-after-close 5e-6 --contact-load-increment "${task_increments[task_level]}"
