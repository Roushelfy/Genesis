#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/axis-publication-tools-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{refine_fixed_geometry,summarize_fixed_geometry,summarize_axis_scan,force_control_blocks,grasp_contact_reference,contact_reference,diagnose_contact_reference}.py
    research/rigid_stress/scripts/publish_axis_checkpoint.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
git diff HEAD > "$task_tools/source.patch"
task_names=(axis-summary-v1 fixed-poly-summary-v1 fixed-poly-u-v1 fixed-poly-ellipsoid-v1
    force-control-blocks-n3-v1 force-control-blocks-n3-v2 force-control-blocks-n4-v2 force-control-blocks-n5-v2
    contact-affine-regression-v1 force-control-roundoff-regression-v3
    grasp-u-coupled-v1 grasp-u-coupled-v2 grasp-u-coupled-v3 grasp-u-diagnostic-v1)
for task_geometry in u capsule; do
    for task_setting in q6-p8 q14-p8 q10-p4 q10-p16; do
        task_names+=("axis-$task_geometry-$task_setting-v1" "axis-$task_geometry-$task_setting-v1-probes")
    done
    for task_level in 2 3 4; do task_names+=("ref-quad-$task_geometry-l$task_level-v1"); done
    for task_radius in 0.001 0.002 0.004; do
        task_names+=("radius-$task_geometry-r$task_radius-v1" "radius-$task_geometry-r$task_radius-v1-launcher")
    done
done
for task_geometry in ellipsoid u; do
    for task_level in 0 1 2; do
        task_names+=("fixed-poly-ref-$task_geometry-l$task_level-v1" "fixed-poly-ref-$task_geometry-l$task_level-v1-probes" "fixed-poly-ref-$task_geometry-l$task_level-v1-launcher")
    done
done
for task_geometry in ellipsoid capsule u thin_neck cavity narrow_gap egg; do
    task_names+=("fine-matrix-mesh-$task_geometry-v1")
done
for task_name in "${task_names[@]}"; do test -f "$task_epoch/$task_name/source-after.sha256"; done
TASK_AXIS_EPOCH="$task_epoch" TASK_AXIS_NAMES="${task_names[*]}" "$task_env/bin/python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["TASK_AXIS_EPOCH"])
names = os.environ["TASK_AXIS_NAMES"].split()
for name in names:
    run = root / name
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), name
    for line in (run / "source.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (name, relative)
axes = json.loads((root / "axis-summary-v1/results/summary.json").read_text())
fixed = json.loads((root / "fixed-poly-summary-v1/results/summary.json").read_text())
print(json.dumps({"scope": "Physical error separation, radius sensitivity and contact force-control progress",
                  "verified_source_closures": names, "axis_scan": axes, "fixed_polyhedron": fixed,
                  "goal_completion": False}, indent=2))
PY
task_jobs=(712677 712678 712679 712702 712715 712719 712720 712721 712726 712727
    712742 712744 712765 712767 712775 712784 712792 712793 712795 712803 712813)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=()
for task_job in "${task_jobs[@]}"; do
    if test "$task_job" = 712795 || test "$task_job" = 712813; then continue; fi
    for task_path in "$task_epoch"/*"$task_job"_*.log "$task_epoch"/*"$task_job"_*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/physical-axes-checkpoint-$task_version"
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Progress toward the complete physical goal. Fixed boundary polyhedra separate volume changes; reference quadrature and native quadrature/ports are measured separately with byte-identical frozen model inputs. Radius changes remain declared different physical problems. U raw peaks grow on fixed reentrant polygonal edges. Fine smooth-CAD reference sweeps remain outside this checkpoint while running. Both force-control normal and prior displacement-control regressions pass unchanged residual criteria. The coupled full-U pilot retains its invalid-initial-geometry and initial-support failures; no failed reference is counted as physically validated. Later coupled attempts, complete contact/refinement/profile/count sensitivity, full egg/Franka and warmed cost acceptance remain required."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
