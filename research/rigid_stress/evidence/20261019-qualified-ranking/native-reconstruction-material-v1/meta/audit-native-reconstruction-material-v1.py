"""Verify terminal reconstruction, normal material and prescribed-load metadata."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
receipts = {}
def receipt(path):
    path = Path(path)
    if str(path) not in receipts:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):digest.update(block)
        receipts[str(path)] = {'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    return receipts[str(path)]
def read(path):
    path = Path(path);receipt(path);return json.loads(path.read_text())
def verify(entry):
    assert receipt(entry['path']) == entry, entry['path']

scheduler = subprocess.check_output(['sacct','-j','728702,728738,728746,728750,728755,728761,728762,728773,728615','-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
expected=['728702_0','728746_0','728750_0','728755_0','728761_0','728773_0','728615_0']+[f'728738_{i}' for i in range(5)]+[f'728762_{i}' for i in range(4)]
assert all(f'{job}|COMPLETED|0:0' in scheduler.splitlines() for job in expected)
core=read(ROOT/'native-pad-reconstruction-full-audit-v1/group0/audit/result.json')
assert core['passed'] and len(core['cases']) == 10
verify(core['mesh'])
for case in core['cases']:
    assert case['passed']
    for key in ('capture','original','quadrature','replay','field'):verify(case[key])
    capture=read(case['capture']['path']);assert capture['passed']
    match=next(c for c in capture['cases'] if c['original']==case['original'])
    assert match['quadrature']==case['quadrature'] and match['replay']==case['replay']
    assert match['capture_checks']['passed'] and max(match['equivalence'].values()) <= 1e-8
    verify(match['physical_state'])
    start=read(Path(case['capture']['path']).parent/'start.json')
    for key in ('original_start','original_result','original_journal'):verify(start[key])
core_metrics=('independent_surface_integration_relative_error','complete_equation_relative_error','rigid_projection_correction_relative','complete_corner_stress_relative_error','maximum_stress_component_relative_error')
core_maxima={key:max(c[key] for c in core['cases']) for key in core_metrics}
assert core_maxima[core_metrics[0]] <= 1e-10 and max(core_maxima.values()) <= 1e-8

base=ROOT/'solid-pad-stvk-normal-load-controls-v1/group0'
physical=read(base/'reference/result.json');assert physical['completed'] and physical['steps']==400 and physical['mechanical_history_passed'];assert all(physical['history_checks'].values()) and all(w['mechanical_window_passed'] for w in physical['windows'])
assert read(base/'analytic/result.json')['passed']
moment=read(base/'moment/result.json');assert moment['passed'] and moment['history_admitted'];verify(moment['run'])
fields={f['field']['path']:f['field'] for f in moment['frames']}
for f in moment['frames']:assert f['physical_momentum_passed'];verify(f['field'])
static=read(ROOT/'compliant-pad-fixed-point-refinement-v1/group0/reference/result.json')
assert all(c['pad_equations_admitted'] for c in static['cases'])
static_fields={c['field']['path']:c['field'] for c in static['cases']}
metrics=('tangent_relative_l2','membrane_relative_l2','bending_relative_l2','outer_tangent_relative_l2','inner_tangent_relative_l2')
material=[]
for load in (.3,1.):
    cases={}
    for q in (9,25):
        data=read(base/f'load{load:g}-q{q}/result.json');assert data['physical_volume_weighting'] and data['common_peaks_evaluated'] and data['surface_quadrature_points_per_face']==q
        case=data['cases'][0];assert case['load_n']==load
        for i,raw in enumerate(case['raw']):
            verify(raw['field']);verify(raw['mesh'])
            assert raw['field']==(static_fields if i==0 else fields)[raw['field']['path']]
        cases[q]=case['adjacent_comparisons'][0]
    c=cases[25]
    peaks={p['metric']:p['relative_difference'] for p in c['common_spatial_measures']['tangent_peaks']}
    row={'load_n':load,'field_changes':{k:c[k] for k in metrics},'raw_mises_relative_change':c['raw_mises_relative_change'],'raw_tension_relative_change':c['raw_tension_relative_change'],'common_peaks':peaks,'q9_q25_metric_absolute_difference':max(abs(c[k]-cases[9][k]) for k in metrics)}
    row['reporting_bands_passed']=max(row['field_changes'].values())<=.005 and max(c['raw_mises_relative_change'],c['raw_tension_relative_change'],*peaks.values())<=.01
    assert row['reporting_bands_passed']
    material.append(row)

prescribed=[]
for group in range(5):
    folder=ROOT/'finite-area-prescribed-common-controls-v1'/f'group{group}'
    assert (folder/'returncode.txt').read_text().strip()=='0'
    for p in sorted(folder.glob('*/result.json')):
        data=read(p);assert data['physical_volume_weighting'] and data['common_peaks_evaluated']
        case=data['cases'][0]
        for raw in case['raw']:verify(raw['field']);verify(raw['mesh'])
        c=case['adjacent_comparisons'][0]
        prescribed.append({'comparison':p.parent.name,'field_changes':{k:c[k] for k in metrics},'raw_mises_relative_change':c['raw_mises_relative_change'],'raw_tension_relative_change':c['raw_tension_relative_change'],'common_peaks':{v['metric']:v['relative_difference'] for v in c['common_spatial_measures']['tangent_peaks']}})
assert len(prescribed)==10
result={'metadata_and_hash_audit_passed':True,'scheduler':scheduler,'core_cases':10,'core_maxima':core_maxima,'normal_material_changes':material,'prescribed_comparisons':prescribed,'receipts':list(receipts.values()),'whole_goal_complete':False,'scope':'Terminal exact native quadrature, independent P2 assembly/same-RHS core and same500um normal material-model controls, plus prescribed-C2 q9/q25 comparisons. Core numerical agreement does not qualify contact/reconstruction physics. Normal material changes are not finest-mesh bounds. Prescribed surface2.083% still fails2%;125um/three-wall prescribed and fine friction references remain separate pending work.'}
with (ROOT/'meta/native-reconstruction-material-v1-audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps({'verified_receipts':len(receipts),'core_maxima':core_maxima,'material':material},indent=2))
