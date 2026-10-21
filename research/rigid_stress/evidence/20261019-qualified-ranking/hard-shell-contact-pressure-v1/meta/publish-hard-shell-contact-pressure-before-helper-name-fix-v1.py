"""Freeze independent shell pressure results and the original failed controls."""
from pathlib import Path
import json
import subprocess
import ast
import difflib

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-shell-contact-pressure-v1', 'hard-shell-contact-pressure-v2',
    'hard-shell-contact-pressure-v3', 'hard-shell-contact-pressure-inputs-v1',
    'hard-shell-contact-pressure-inputs-v2', 'hard-shell-contact-pressure-v4',
    'hard-shell-contact-pressure-refinement-v1', 'hard-shell-contact-pressure-grid97-v1',
    'hard-shell-contact-pressure-grid97-refinement-v1', 'hard-shell-contact-pressure-inspection-v1',
    'hard-shell-contact-kernel-v1']
jobs = [str(read(N/name/'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for index, (campaign, job) in enumerate(zip(campaigns,jobs)):
    expected = ['FAILED','1:0'] if index < 3 else ['COMPLETED','0:0']
    for group,tasks in enumerate(read(N/campaign/'tasks.json')):
        assert observed[f'{job}_{group}'] == expected, (campaign,group,observed)
        assert (N/campaign/f'group{group}/returncode.txt').read_text().strip() == ('1' if index < 3 else '0')
        for task in tasks:
            path = N/campaign/f'group{group}'/task['name']/'result.json'
            if index < 3:
                assert not path.exists()
            else:
                result = read(path)
                assert result.get('execution_completed', result.get('passed', False)) or 'boundary' in result
inspection_path = N/'hard-shell-contact-pressure-inspection-v1/group0/inspection/result.json'
inspection = read(inspection_path)
assert inspection['passed'] and len(inspection['rows']) == 3
assert not inspection['independent_complete_stress_qualified'] and not inspection['whole_goal_complete']
for binding in inspection['bindings']:
    for key in ['parent_result', 'refinement_result', 'parent_pressures', 'refined_pressures','mesh_file']:
        assert identity(Path(binding[key]['path'])) == binding[key]
source_campaigns = {'hertz_reference': 'hard-shell-contact-pressure-inspection-v1',
    'study_shell_contact_pressure': 'hard-shell-contact-pressure-inspection-v1',
    'refine_shell_contact_pressure': 'hard-shell-contact-pressure-grid97-refinement-v1',
    'inspect_shell_contact_pressure': 'hard-shell-contact-pressure-inspection-v1',
    'check_rectangular_compliance': 'hard-shell-contact-kernel-v1'}
bindings=[]
metadata_changes=[]
for module,campaign in source_campaigns.items():
    rel=Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured=N/campaign/'analysis-source'/rel
    current = WORK/rel
    exact = captured.read_bytes() == current.read_bytes()
    if not exact:
        assert module == 'refine_shell_contact_pressure'
        def without_start_scope(path):
            tree=ast.parse(path.read_text())
            count=0
            for node in ast.walk(tree):
                if isinstance(node,ast.Dict):
                    for index,(key,value) in enumerate(zip(node.keys,node.values)):
                        if isinstance(key,ast.Constant) and key.value == 'scope' and isinstance(value,ast.Constant) and str(value.value).startswith('One added 97x97 scalar pressure grid.'):
                            node.values[index]=ast.Constant(value='SAME_START_SCOPE_DESCRIPTION')
                            count+=1
            assert count == 1
            return ast.dump(tree,include_attributes=False)
        assert without_start_scope(captured) == without_start_scope(current)
        metadata_changes.append({'captured':identity(captured),'current':identity(current),
            'only_start_scope_description_differs':True,
            'patch':''.join(difflib.unified_diff(captured.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='captured',tofile='current'))})
    bindings.append({'captured':identity(captured),'current':identity(current),'exact_match':exact,'scientific_AST_unchanged':True})
C=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_reproducible')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=C,text=True).strip()
for campaign in campaigns:
    for row in read(N/campaign/'source.json')['files']:
        actual=identity(C/row['path'])
        assert all(row[key] == actual[key] for key in ['bytes','sha256'])
originals=[read(N/campaign/group/'pressure/result.json') for campaign,group in [
    ('hard-shell-contact-pressure-v4','group0'),('hard-shell-contact-pressure-refinement-v1','group0'),
    ('hard-shell-contact-pressure-refinement-v1','group1')]]
assert all(not result['discrete_pressure_refinement_targets_met'] for result in originals)
receipt={'passed':True, 'scheduler':accounting, 'original_failed_jobs':jobs[:3],
    'original_grid33_grid65_target_failures_preserved':True, 'current_source_bindings':bindings, 'metadata_only_source_corrections':metadata_changes,
    'pressure_inspection':identity(inspection_path), 'inspection':inspection,
    'public_engine_alias':'d4503b9e42a7f8f8555d128b437bb45b3193cf75',
    'scope':'Independent scalar normal-contact pressure/support on one fixed hollow polar-land egg. Hertz sets a bounded search window, not the pressure or support. Three preserved initial coordinate-transform failures are diagnosed as DiscreteField subclass writes and repaired by explicit NumPy conversion. Three body meshes, pressure grids through97 and3/5 normal-source tables retain original failed33/65 refinement gates. The final read-only inspection reports separate measured grid, source and body-mesh variations, pressure/moment-axis errors and peak pressure. Offline source and pressure solves are not native throughput. No independent complete-stress, friction-history, arbitrary-geometry or public estimator qualification follows.',
    'whole_goal_complete':False}
terminal_path=N/'meta/hard-shell-contact-pressure-terminal-v1.json'
with terminal_path.open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
kernel = read(N/'hard-shell-contact-kernel-v1/group0/kernel/result.json')
assert kernel['passed'] and identity(Path(kernel['baseline']['path'])) == kernel['baseline']
files=[SCRIPT,HELPER,HELPER_SOURCE,terminal_path,Path(kernel['baseline']['path'])]
external=set()
for campaign in campaigns:
    files.extend(sources(campaign)+terminal(campaign))
    external.update((N/campaign).glob('group*/**/*.npz'))
for binding in inspection['bindings']:
    external.add(Path(binding['mesh_file']['path']))
large={path for path in files if path.stat().st_size > 2*1024*1024}
files=[path for path in files if path not in large]
external.update(large)
packet=freeze('hard-shell-contact-pressure-v1',files,sorted(external),receipt['scope'])
(DEST/packet['packet']/'README.md').write_text('# Independent hollow-shell normal contact pressure\n\n'+receipt['scope']+'\n\nReplay the captured scalar studies and97-grid refinements with their captured workers and exact completed parent arrays. Use public engine alias `d4503b9e4`, the pinned scientific Python, `rtx-mid`, image `1_26`, explicit data outputs and private caches. Preserve original failures and pressure-grid gates. The final inspector independently reduces complete saved nodal-pressure/force/gap arrays and binds all input hashes. These are conditional pressure controls; whole-scene speeds and complete/peak stress errors require their own matching evidence.\n')
with (N/'meta/hard-shell-contact-pressure-publication-v1.json').open('x') as f: json.dump(packet,f,indent=2);f.write('\n')
print(json.dumps(packet))
