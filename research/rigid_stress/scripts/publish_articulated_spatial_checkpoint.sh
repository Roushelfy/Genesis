#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_meta="$task_base/articulated-spatial-publication-v1"
task_destination=${1:-research/rigid_stress/evidence/20261014-physical/articulated-spatial-checkpoint-v1}
mkdir -p "$task_meta"
task_runs=(franka-spatial-comparison-control-v1 franka-spatial-comparison-control-v2
    franka-articulated-input-0-v1 franka-articulated-input-1-v1 franka-articulated-input-2-v1
    franka-articulated-audit-v1 franka-articulated-audit-v2
    franka-articulated-physical-mass-input-v2 franka-frozen-contact-control-v1)
task_args=()
for task_name in "${task_runs[@]}"; do
    test -f "$task_base/$task_name/source-after.sha256"
    cmp "$task_base/$task_name/source.sha256" "$task_base/$task_name/source-after.sha256"
    if [[ "$task_name" == franka-spatial-comparison-control-v1 ]]; then
        test "$(cat "$task_base/$task_name/returncode.txt")" = 1
    else
        test "$(cat "$task_base/$task_name/returncode.txt")" = 0
    fi
    task_args+=("$task_base/$task_name")
done
sacct -j 714509,714510,714514,714519,714523,714524,714531 -X \
    --format=JobID,State,ExitCode,Elapsed -n -P > "$task_meta/jobs.txt"
python3 - "$task_meta" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
summary = {
    "scope": "Actual articulated contact-law tails, corrected physical mass, spatial-probe and frozen-contact controls",
    "contact_force_adjacent_l2_changes": [0.008239255812573338, 0.0022287144191546473],
    "spatial_probe_control": "2400 material points,10 common times; all equal-constructor changes zero",
    "working_mass_limitation": "Original audit714519 near-static energy checks did not establish physical mass; audit714523 removes the source-defined implicit damping and checks all576 motion frames per setting.",
    "physical_mass_observer_equivalence": "576 rows and12672 original npy payloads byte-identical; new working/physical mass, corrections and generalized efforts are additional readouts",
    "frozen_contact_control": "Every original npy payload in83 full protocol NPZ archives is byte-identical",
    "failed_comparison": "714509 failed wrong probe-array orientation; corrected714510 passes",
    "limitations": "No full deformable arm, spatial/contact/stress/history convergence, feedback/model or physical accuracy-cost acceptance is inferred. Live experiments are excluded.",
    "goal_completion": False,
}
(p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
task_logs=(franka-spatial-comparison-control-v1-slurm-714509_0.log
    franka-spatial-comparison-control-v2-slurm-714510_0.log
    franka-articulated-input-v1-slurm-714514_0.log franka-articulated-input-v1-slurm-714514_1.log
    franka-articulated-input-v1-slurm-714514_2.log franka-articulated-audit-v1-slurm-714519_0.log
    franka-articulated-audit-v2-slurm-714523_0.log franka-articulated-physical-mass-input-v2-slurm-714524_0.log
    franka-frozen-contact-control-v1-slurm-714531_0.log)
task_extra=("$task_meta/jobs.txt" "$task_base/articulated-spatial-submissions-v1.txt")
for task_log in "${task_logs[@]}"; do task_extra+=("$task_base/$task_log"); done
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_destination" \
    --staging-root "$task_base/publication-staging" --run "${task_args[@]}" \
    --extra "${task_extra[@]}" --summary "$task_meta/summary.json" \
    --limitations 'Complete original arrays/config/source/hardware, successful and failed controls are retained. Working versus physical mass is explicit. Scheduled/running reference studies are excluded; the full physical acceptance scope remains unfinished.'
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_destination/manifest.json" --originals --output "$task_meta/verification.json"
cp "$task_meta/verification.json" "$task_destination/verification.json"
