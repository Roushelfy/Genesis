"""Freeze completed independent stress and native cone-memory evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import struct
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C = WORK.parent / 'hard_contact_reproducible'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C, text=True).strip()

physical = ['hard-bilinear-stress-kernel-v1', 'hard-shell-pressure-stress-v1',
    'hard-shell-pressure-stress-inspection-v1', 'hard-shell-pressure-stress-inspection-v2']
physical += [f'hard-shell-pressure-stress-figure-v{i}' for i in range(1, 5)]
native = [f'hard-cone-history-v{i}' for i in range(1, 6)]
native += [f'hard-native-cone-history-v{i}' for i in range(1, 4)]
native += ['hard-native-cone-vectors-v1', 'hard-cone-penalty-paired-v1']
native += [f'hard-cone-warp-paired-v{i}' for i in range(1, 5)]
native += ['hard-cone-warp-inspection-v1']
native += ['hard-cone-baseline-final-v1']
campaigns = physical + native
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(campaigns, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        code = (root / 'returncode.txt').read_text().strip()
        expected = ['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0']
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code, observed)
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1]})
        if code == '0':
            for task in tasks:
                result = read(root / task['name'] / 'result.json')
                assert result.get('passed', result.get('execution_completed', False))
    for row in read(N / name / 'source.json')['files']:
        actual = identity(C / row['path'])
        assert all(actual[key] == row[key] for key in ['bytes', 'sha256'])

inspection = read(N / 'hard-shell-pressure-stress-inspection-v2/group0/inspection/result.json')
assert inspection['passed'] and inspection['conditional_single_case_reference_targets_met']
assert all(inspection['conditional_candidate_envelopes_within_targets'][mesh][candidate]
    for mesh in ['coarse', 'medium', 'fine'] for candidate in ['hertz-point', 'hertz-distributed'])
assert all(not inspection['conditional_candidate_envelopes_within_targets'][mesh]['hertz-only']
    for mesh in ['coarse', 'medium', 'fine'])
assert not inspection['native_scene_accuracy_qualified'] and not inspection['whole_goal_complete']
layout = read(N / 'hard-cone-warp-inspection-v1/group0/inspection/result.json')
assert layout['passed'] and all(row['three_repeat_ranges_separate'] for row in layout['timings'])
assert not layout['independent_general_vector_friction_physics_qualified']
cpu = read(N / 'hard-cone-history-v5/group0/study/result.json')
assert cpu['force_and_continuous_cone_gates_met'] and cpu['constant_load_sample_stress_targets_met']
assert not cpu['independent_variable_load_vector_physics_qualified']

bindings = []
owned = {
    'bilinear_normal_stress': 'hard-shell-pressure-stress-v1',
    'check_bilinear_normal_stress': 'hard-bilinear-stress-kernel-v1',
    'study_shell_pressure_stress': 'hard-shell-pressure-stress-v1',
    'inspect_shell_pressure_stress': 'hard-shell-pressure-stress-inspection-v2',
    'plot_shell_pressure_stress': 'hard-shell-pressure-stress-figure-v4',
    'hertz_history_cone': 'hard-cone-history-v5',
    'study_hertz_cone_history': 'hard-cone-history-v5',
    'native_hertz_cone_history': 'hard-cone-warp-paired-v4',
    'native_hertz_cone_cooperative': 'hard-cone-warp-paired-v4',
    'check_native_hertz_cone': 'hard-cone-warp-inspection-v1',
    'check_native_cone_vectors': 'hard-cone-warp-paired-v4',
    'check_native_cone_baseline': 'hard-cone-baseline-final-v1',
    'check_native_cone_serial': 'hard-cone-warp-paired-v4',
    'inspect_native_cone_layout': 'hard-cone-warp-inspection-v1',
}
for module, name in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / name / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})

figure = N / 'hard-shell-pressure-stress-figure-v4/group0/figure'
assert read(figure / 'result.json')['passed']
for name in ['shell-pressure-stress_preview.png', 'shell-pressure-stress_grayscale.png']:
    assert (figure / name).read_bytes() == (N / 'hard-shell-pressure-stress-figure-v3/group0/figure' / name).read_bytes()
png = (figure / 'shell-pressure-stress.png').read_bytes()
assert struct.unpack('>IIBB', png[16:26]) == (4322, 3614, 8, 2)
pdf = (figure / 'shell-pressure-stress.pdf').read_bytes()
fonts = sorted(set(re.findall(rb'/BaseFont\s*/([^\s/>]+)', pdf)))
assert len(fonts) == 3 and pdf.count(b'/FontFile2') == 3
visual = {'passed': True, 'reviewed_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewed_color_and_grayscale_previews': [identity(figure / name) for name in ['shell-pressure-stress_preview.png', 'shell-pressure-stress_grayscale.png']],
    'final_previews_byte_equal_to_visually_inspected_v3': True, 'text_layout_qa_issues': [],
    'inspection': 'Actual final-size color/grayscale previews inspected. No clipped labels or legend/data overlap. Model lines have distinct styles; heatmaps are un-smoothed retained values. Signed principal-stress colorbar and MPa/um labels are explicit. Gray band is observed deterministic variation, not confidence. Peak markers refer to full sampled bank; scope is one normal polar case.',
    'rgb_png': {'width': 4322, 'height': 3614, 'bits': 8, 'color_type': 2, 'dpi': 600},
    'pdf_embedded_truetype_fonts': [font.decode() for font in fonts],
    'pdf_font_check': 'pdffonts unavailable; direct PDF FontFile2/BaseFont inspection confirms all three embedded TrueType faces.',
    'whole_goal_complete': False}
visual_path = N / 'meta/shell-pressure-stress-visual-review-v1.json'
if visual_path.exists():
    old = read(visual_path)
    visual['reviewed_UTC'] = old['reviewed_UTC']
    assert visual == old
else:
    with visual_path.open('x') as stream: json.dump(visual, stream, indent=2); stream.write('\n')

def add_input_ids(value, external):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values(): add_input_ids(item, external)
    elif isinstance(value, list):
        for item in value: add_input_ids(item, external)

packets = []
for kind, names, packet_name, scope in [
    ('physical', physical, 'hard-shell-pressure-stress-v1', 'Independent-pressure full stress comparison on one fixed normal polar-land hollow shell. The fine97 pressure is held fixed across three body meshes; independent bilinear Boussinesq tensors combine with retained25 scalar normal regular responses. Separate body, pressure, source, integration and edge-sampling changes pass the original conditional gates. Coarse central-source Hertz correction has2.188% tensor and5.023%/0.771% tensile/VM peak errors; observed envelopes3.479% and8.007%/0.819% pass5%/10% budgets. Hertz-only78.24% tensor failure, raw peak-coordinate switches, signed cross terms and both failed figure-QA attempts remain explicit. Complete arrays remain data-side. No level1 native, general friction/contact or scene-cost qualification follows.'),
    ('native', native, 'hard-cone-history-native-v1', 'Separate cone-preserving approximate radial-memory model with continuous sufficient Coulomb slope disks and exact force. Old line-clipping source/failures are untouched. CPU scalar physical samples and variable-load/vector feasibility controls pass, preserving actual failed solver flags and independent KKT certificates. Quadrants CPU/GPU scalar/lifecycle and97-step vector/frame checks pass. Same-allocation cooperative versus serial rho0.1 history cost is0.808/4.411ms for16 intervals B1024 and1.625/18.774ms for32; every audited benchmark frame passes, with an independent saved-array layout comparison. Initial optimizer, iteration-cap, compiler and scalar-test-order failures remain retained. These are history component costs, not general variable-vector physical or full native stress-scene qualification.')]:
    files = [SCRIPT, HELPER, HELPER_SOURCE]
    files.append(N / 'meta/publish-shell-stress-and-cone-history-before-baseline-binding-v1.py')
    external = set()
    for name in names:
        files.extend(sources(name) + terminal(name))
        external.update((N / name).glob('group*/**/*.npz'))
        for start_path in (N / name).glob('group*/**/start.json'):
            add_input_ids(read(start_path), external)
    if kind == 'physical':
        files += [visual_path, N / 'meta/nature_mpl_style-shell-stress-v1.py']
        files += [path for path in figure.iterdir() if path.suffix in ['.svg', '.pdf', '.png', '.csv']]
        for parent in read(N / 'hard-shell-pressure-stress-v1/tasks.json')[0][0]['args'][1:4]:
            external.add(Path(parent) / 'normal-source-table.npz')
        for level in ['coarse', 'medium', 'fine']:
            for suffix in ['-mesh.npz', '-fields.npz']:
                external.add(N / 'hard-shell-residual-v1/group0/study' / (level + suffix))
    receipt = {'passed': True, 'actual_scheduler': accounting,
        'campaign_states': {name: states[name] for name in names},
        'current_source_bindings': [row for row in bindings if ('pressure_stress' in row['current']['path'] or 'bilinear' in row['current']['path']) == (kind == 'physical')],
        'public_engine_alias': 'd4503b9e42a7f8f8555d128b437bb45b3193cf75',
        'preserved_failures_and_captured_workers': True, 'scope': scope, 'whole_goal_complete': False}
    receipt_path = N / f'meta/{packet_name}-terminal-v1.json'
    if receipt_path.exists(): assert read(receipt_path) == receipt
    else:
        with receipt_path.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    files.append(receipt_path)
    large = {path for path in files if path.stat().st_size > 2 * 1024**2 and path.suffix not in ['.svg', '.pdf', '.png']}
    files = [path for path in files if path not in large]
    external.update(large)
    packet = freeze(packet_name, files, sorted(external - set(files)), scope)
    (DEST / packet_name / 'README.md').write_text('# ' + packet_name + '\n\n' + scope + '\n\nReplay captured module/task lists with exact hashed completed inputs, public engine alias`d4503b9e4`, the pinned scientific Python, Slurm`rtx-mid`, image`1_26`, explicit new data outputs and private node-local caches. Archives retain actual failed and successful source variants. Decode gzip payloads into a separate replay directory; leave original evidence immutable. Full fields/history arrays stay at externally hashed data paths. The active estimator integration and matching physical-accuracy/scene-throughput goal remains open.\n')
    packets.append(packet)
with (N / 'meta/shell-stress-and-cone-history-publication-v1.json').open('x') as stream:
    json.dump(packets, stream, indent=2); stream.write('\n')
print(json.dumps(packets))
