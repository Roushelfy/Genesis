#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/articulated-u-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib, json, pathlib, subprocess, sys
p, meta = map(pathlib.Path, sys.argv[1:])
runs = {
    'franka-articulated-coupled-pilot-v1': '714971_0',
    'franka-articulated-hold-v2': '714984_0',
    'franka-fixed-arm-hold-v2': '714984_1',
    'franka-articulated-hold-v3': '714999_0',
    'franka-fixed-arm-hold-v3': '714999_1',
    'franka-articulated-pilot-audit-v1': '715004_0',
    'franka-articulated-support-audit-v3': '715005_0',
    'franka-articulated-support-audit-v4': '715019_0',
    'grasp-u-spatial-y-l1-v1': '713653_0',
    'grasp-u-spatial-y-l2-v1': '713653_1',
    'grasp-u-spatial-y-l3-v1': '714003_0',
    'u-three-level-audit-v1': '715011_0',
    'u-three-level-audit-v2': '715016_0',
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
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            closures.append({'run': name, 'identity': prefix, 'file': relative, 'sha256': expected})
    for suffix in ['log', 'err']:
        extras.update(p.glob('*'+job+'.'+suffix))
for name in ['grasp-u-spatial-y-l1-v1','grasp-u-spatial-y-l2-v1','grasp-u-spatial-y-l3-v1']:
    extras.add(pathlib.Path(json.loads((p/name/'solution/configuration.json').read_text())['mesh']))
capture = p/'franka-articulated-physical-mass-input-v2'
captured = json.loads((capture/'results/summary.json').read_text())
frame = [r for r in captured['captures'] if r['mode']=='stationary_lift_hold'][-1]
extras.update([capture/'results/summary.json', capture/'results'/frame['geometry_snapshot'],
               capture/'source/examples/rigid/assets/hollow_egg/level1/elastic.npz'])
extras.update(capture/'results'/g['mesh'] for g in captured['geometry'] if g.get('mesh'))
dependencies = []
for name in ['articulated-spatial-checkpoint-v1', 'local-tail-articulated-checkpoint-v1']:
    manifest = pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
    dependencies.append({'manifest': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
def report(name):
    return json.loads((p/name/'results/summary.json').read_text())
summary = {
    'scope': 'Solved independent full-arm/FEM contact hold, explicit support control and third U spatial level',
    'runs': [{'run': n, 'job': j, 'state': states[j], 'returncode': int((p/n/'returncode.txt').read_text())} for n,j in runs.items()],
    'source_closures': closures, 'dependencies': dependencies,
    'coupled_pilot': report('franka-articulated-coupled-pilot-v1'),
    'independent_support_audit': report('franka-articulated-support-audit-v4'),
    'u_spatial_tail': report('u-three-level-audit-v2'),
    'failures': ['Full-arm v2 Newton stalls at step31 above original budgets; its exact failed state is retained.',
                 'Fixed-arm v2 fresh-step check fails at step69. Only states through68 were saved by that driver.',
                 'U audit v1 compared raw-coordinate and gauge-projected stiffness arithmetic at1e-12. Audit v2 retains this comparison in consistent gauge and separately verifies raw-coordinate balance at original1e-9.'],
    'live_jobs_excluded': ['714462_1','714503_0','714840_0','714874_3','714874_7','714883_1','715012_0','715012_1','715012_2','715017_0','715021_0'],
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
task_dest="research/rigid_stress/evidence/20261014-physical/articulated-u-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The solved short hold uses a held-body initializer and zero initial velocities at one coarse setting. Original approach/lift/angular history, temporal/spatial/contact/regularization tails, feedback versus load mapping, applicable model comparisons and warmed physical accuracy/cost remain incomplete. U third-level tensors still change12.2%. Successes and failures retain exact original files; live jobs are excluded. The earlier articulated-spatial manifest retains complete capture provenance without duplicating every motion frame here.' \
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
