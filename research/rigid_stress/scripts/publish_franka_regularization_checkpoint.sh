#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/franka-regularization-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{capture_franka_fixture,audit_franka_contact_regularization,dynamic_franka_reference,qualify_dynamic_franka,replay_dynamic_franka,audit_dynamic_franka_axes,plot_franka_regularization}.py
    research/rigid_stress/scripts/{capture_franka_fixture.sh,audit_franka_contact_regularization.sh,dynamic_franka_reference.sh,replay_dynamic_franka.sh,audit_dynamic_franka_axes.sh,plot_franka_regularization.sh,publish_franka_regularization_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(franka-contact-reg-0-v1 franka-contact-reg-1-v1 franka-contact-reg-2-v1
    franka-contact-reg-audit-v1 franka-inertial-qualification-v1 franka-inertial-release-v1
    franka-inertial-dt-0-v1 franka-inertial-dt-1-v1 franka-inertial-axis-audit-v1
    franka-inertial-axis-audit-v2 franka-inertial-replay-v1 franka-inertial-replay-v2
    franka-contact-reg-plots-v1 franka-contact-reg-plots-v2)
TASK_REG_EPOCH="$task_epoch" TASK_REG_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root=Path(os.environ['TASK_REG_EPOCH'])
closures=[]
for name in os.environ['TASK_REG_NAMES'].split():
    run=root/name
    before=(run/'source.sha256').read_bytes()
    assert before==(run/'source-after.sha256').read_bytes(),name
    for line in before.decode().splitlines():
        expected,relative=line.split('  ',1)
        assert hashlib.sha256((run/'source'/relative).read_bytes()).hexdigest()==expected,(name,relative)
    closures.append({'run':name,'returncode':int((run/'returncode.txt').read_text())})
reg=json.loads((root/'franka-contact-reg-audit-v1/results/summary.json').read_text())
assert len(reg['cases'])==3 and reg['fixed_integration_interval_s']==0.00125
dynamic=[]
for name in ('franka-inertial-dt-0-v1','franka-inertial-release-v1','franka-inertial-dt-1-v1'):
    r=json.loads((root/name/'results/summary.json').read_text())
    assert max(x['complete_dynamic_residual_N'] for x in r['steps'])<=1e-9
    assert max(x['complete_body_dynamic_residual_N'] for x in r['steps'])<=1e-9
    dynamic.append({'run':name,'steps':len(r['steps']),'max_complete_residual_N':max(x['complete_dynamic_residual_N'] for x in r['steps'])})
assert sum(x['steps'] for x in dynamic)==140
replay=json.loads((root/'franka-inertial-replay-v2/results/summary.json').read_text())
assert replay['actual_native_vs_independent_linear_tensor_relative_difference']<1e-8
qa=json.loads((root/'franka-contact-reg-plots-v2/results/rendered_qa.json').read_text())
assert qa['passed'] and qa['manual_rendered_preview_inspected'] and qa['svg_editable_text_elements']>10
report={'scope':'Raw Panda contact-only regularization, solved inertial reference/time axes and actual native replay',
        'source_closures':closures,'contact_only_regularization':reg,'dynamic_solve_gates':dynamic,
        'temporal_axis':json.loads((root/'franka-inertial-axis-audit-v2/results/summary.json').read_text()),
        'native_replay':replay,'figure_qa':qa,
        'earlier_attempts':'Original replay v1 omits required surface_triangles and fails before native recovery. Plot v1 omits an imported inactive mesh-generator definition from its source list; selected v2 captures it. Earlier energy-free temporal audit is retained, selected v2 adds body elastic and full kinetic energy.',
        'live_jobs_excluded':['714003_0','714003_3'],'goal_completion':False}
print(json.dumps(report,indent=2))
PY
task_jobs=(714299 714330 714334 714335 714337 714341 714352 714356 714359 714365 714367)
task_ids=$(IFS=,;printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}";do task_runs+=("$task_epoch/$task_name");done
task_extra=("$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt" "$task_tools/summary.json"
    "$task_epoch/grasp-u-plots-v3/style/nature_mpl_style.py")
while IFS= read -r -d '' task_source;do task_extra+=("$task_source");done < <(find "$task_tools/source" -type f -print0)
for task_job in "${task_jobs[@]}";do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err;do
        if test -f "$task_path";then task_extra+=("$task_path");fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/franka-regularization-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_tools/summary.json" \
    --limitations "Rigid contact-only time constants change at fixed1.25ms dt and actual weld parameters, with raw penetrations and finite-face geometry checks. Default overlap approaches wall thickness; the3.010% force tail remains unqualified. Independent inertial actual-finger references solve140 steps with unchanged dynamic force gates and three temporal controls at one coarse spatial/contact setting. One actual native nodal replay is qualified; it is not finite-footprint/model/spatial accuracy. Original asset/source-coverage failures remain exact. Different initializers, friction/contact regularization, longer histories and omitted full-arm/finger-equality behavior require matching. Live U/full egg are excluded. The complete original physical/model/feedback/cost objective remains active."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
