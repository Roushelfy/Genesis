#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/feedback-time-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib, json, pathlib, subprocess, sys
p, meta = map(pathlib.Path, sys.argv[1:])
runs = {
    "franka-articulated-temporal-coarse-v1": "715012_0",
    "franka-articulated-temporal-middle-v1": "715012_1",
    "franka-articulated-temporal-fine-v1": "715012_2",
    "franka-articulated-temporal-audit-v1": "715017_0",
    "franka-rigid-egg-coupled-pilot-v1": "715028_0",
    "franka-rigid-egg-coupled-pilot-v2": "715032_0",
    "franka-rigid-egg-coupled-pilot-v3": "715035_0",
    "franka-rigid-egg-hold-v1": "715038_0",
    "franka-rigid-feedback-audit-pilot-v1": "715036_0",
    "franka-rigid-feedback-audit-hold-v1": "715039_0",
    "franka-rigid-feedback-audit-hold-v2": "715051_0",
    "franka-feedback-native-pilot-v1": "715040_0",
    "franka-feedback-native-hold-v1": "715045_0",
    "franka-temporal-feedback-figure-v1": "715050_0"
}
ids = ','.join(sorted({j.split('_')[0] for j in runs.values()}))
jobs = subprocess.check_output(['sacct', '-X', '-j', ids, '-o', 'JobID%40,State,ExitCode,Elapsed,End,NodeList', '-n', '-P'], text=True)
(meta/'jobs.txt').write_text(jobs)
commands = subprocess.check_output(['sacct', '-X', '-j', ids, '-o', 'JobID%40,SubmitLine%12000', '-n', '-P'], text=True)
(meta/'job-commands.txt').write_text(commands)
states = {r.split('|')[0].strip(): r.split('|')[1].strip() for r in jobs.splitlines() if r.strip()}
closures, extras = [], set()
for name, job in runs.items():
    run = p/name
    code = int((run/'returncode.txt').read_text())
    assert states[job] == ('COMPLETED' if code == 0 else 'FAILED'), (name, states[job], code)
    for prefix in ['source', 'input-source']:
        before = run/(prefix+'.sha256')
        if not before.exists(): continue
        assert before.read_bytes() == (run/(prefix+'-after.sha256')).read_bytes()
        for line in before.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            source = run/prefix/relative
            if pathlib.Path(relative).is_absolute() and prefix == 'input-source':
                source = run/prefix/'nature_mpl_style.py'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            closures.append({'run': name, 'identity': prefix, 'file': relative, 'sha256': expected})
    for suffix in ['log', 'err']:
        extras.update(p.glob('*'+job+'.'+suffix))
dependencies = []
for name in ['articulated-u-checkpoint-v1']:
    manifest = pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
    dependencies.append({'manifest': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
def report(name):
    return json.loads((p/name/'results/summary.json').read_text())
summary = {
    'scope': 'Matched full-arm rigid/deformable contact feedback, native nodal replay and fixed-law temporal diagnostics',
    'runs': [{'run': n, 'job': j, 'state': states[j], 'returncode': int((p/n/'returncode.txt').read_text())} for n,j in runs.items()],
    'source_closures': closures, 'dependencies': dependencies,
    'temporal_audit': report('franka-articulated-temporal-audit-v1'),
    'rigid_pilot': report('franka-rigid-egg-coupled-pilot-v3'),
    'feedback_audit': report('franka-rigid-feedback-audit-hold-v2'),
    'native_replay': report('franka-feedback-native-hold-v1'),
    'figure_qa': json.loads((p/'franka-temporal-feedback-figure-v1/figure/rendered_qa.json').read_text()),
    'failures': ['Rigid pilot v1/v2 fail the unchanged1e-3 directional-tangent gate at perturbations1e-9/1e-10. The complete accepted64-target held state and source/logs are retained. The extended perturbation sequence reaches2.625e-4/4.284e-4 at1e-11 without changing the gate or physical law.'],
    'excluded_fine_jobs': ['714462_1','714503_0','714840_0','714874_3','714874_7','714883_1','715021_0'],
    'goal_completion': False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/n)+'\n' for n in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'extras.txt').write_text(''.join(str(f)+'\n' for f in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_identities':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/feedback-time-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Temporal and resolved-contact feedback are qualified only for the declared short hold at one coarse spatial/contact setting. Original approach/lift/angular history, spatial/contact/regularization tails, Genesis contact-law equivalence, finite-footprint model mapping and physically verified warmed costs remain incomplete. Exact successes and failed diagnostics are retained. Earlier deformable/capture inputs remain in the named immutable dependency.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_dest/verification.json"
python3 - "$task_dest/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text())
print(json.dumps({'files':len(d['files']),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
PY
