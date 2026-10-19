"""Freeze independent process-memory controls and exact source-bound fresh profiles."""
import gzip
import hashlib
import io
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
WORK = pathlib.Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
DEST = WORK / 'research/rigid_stress/evidence/20261019-qualified-ranking'
META = RUN / 'meta'
SCRIPT = pathlib.Path(__file__)


def identity(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def freeze(name, files, external, scope):
    target = DEST / name
    target.mkdir(exist_ok=False)
    rows = []
    for source in sorted(set(files)):
        original = source.read_bytes()
        relative = source.relative_to(RUN)
        encoded = original
        compressed = bool(original) and (source.suffix == '.log' or (source.suffix == '.json' and len(original) > 8192) or source.suffix == '.patch')
        if compressed:
            memory = io.BytesIO()
            with gzip.GzipFile(filename='', mode='wb', fileobj=memory, mtime=0) as stream:
                stream.write(original)
            encoded = memory.getvalue()
            relative = pathlib.Path(str(relative) + '.gz')
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(encoded)
        rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(), 'encoding': 'gzip' if compressed else 'identity', 'original_bytes': len(original), 'original_sha256': hashlib.sha256(original).hexdigest()})
    manifest = {'files': rows, 'external_files': [identity(path) for path in external], 'scope': scope, 'payloads': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows)}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nManifest records encoded and original SHA-256/size. Gzip uses mtime0 and an empty filename. Decode to a separate working directory when replaying; immutable archive bytes stay unchanged. Large trajectory/field arrays remain at the explicit data-side locations in the receipts. The original1ff39b077 engine can be reconstructed from the public-base overlay in [the source packet](../native-public-source-overlay-v1/README.md). Captured worker modules retain publication-time variants. Live fine histories and pending method/ranking/figure outputs are excluded.\n')
    return {'packet': name, 'payloads': len(rows), 'bytes': manifest['payload_bytes'], 'external_arrays': len(external)}


def captured(campaign, modules):
    root = RUN / campaign
    files = [root / n for n in ('submission.json', 'source.json', 'source.patch', 'analysis-files.json', 'tasks.json', 'job.sh')]
    files += list(root.glob('workload-*.sh'))
    for module in modules:
        files.append(root / 'analysis-source/research/rigid_stress' / module)
    return files


def terminal(campaign):
    root = RUN / campaign
    return [p for p in root.glob('group*/**/*') if p.is_file() and p.suffix in ('.json', '.txt', '.log') and 'cache' not in p.parts and 'tmp' not in p.parts] + list(root.glob('logs/*.log'))

expected = {job: ('COMPLETED', '0:0') for job in ['730520_0','730526_0','730531_0','730531_1','730533_0','730533_1','730536_0','730538_0','730539_0','730541_0']}
accounting = subprocess.check_output(['sacct','-j','730520,730526,730531,730533,730536,730538,730539,730541','-P','--noheader','--format=JobID,State,ExitCode,Elapsed'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == state for job,state in expected.items()), observed
control = json.loads((RUN / 'process-memory-controls-v1/group0/controls/result.json').read_text())
assert control['passed'] and len(control['checks']) == 14 and all(control['checks'].values())
assert control['legacy_hold_gap_seconds'] >= 2.5 and control['independent_hold_sample_count'] >= 10 and control['independent_hold_maximum_sample_gap_seconds'] < 1
contract = json.loads((META / 'process-memory-profile-contract-v1.json').read_text())
control_contract = json.loads((META / 'process-memory-contract-v1.json').read_text())
assert contract['workers']['shell_comparison/memory.py'] == control_contract['workers']['shell_comparison/memory.py']
module = json.loads((RUN / 'native-module-memory-process-audit-v1/group0/audit/result.json').read_text())
small = json.loads((RUN / 'native-production-memory128-512-process-audit-v1/group0/audit/result.json').read_text())
large = json.loads((RUN / 'native-production-memory2048-8192-process-audit-v1/group0/audit/result.json').read_text())
for audit,count in [(module,4),(small,10),(large,10)]:
    assert audit['passed'] and audit['independent_sampler_required'] and len(audit['rows']) == count
coverage = [row['independent_polling'] for audit in [module,small,large] for row in audit['rows']]
assert len(coverage) == 24 and all(row['method'] == 'independent_process_nvidia_smi' and row['collector_returncode'] == 0 for row in coverage)
summary = json.loads((RUN / 'native-scene-process-memory-summary-v1/group0/summary/result.json').read_text())
original = json.loads((RUN / 'native-scene-summary-v2/group0/summary/result.json').read_text())
assert summary['passed'] and len(summary['rows']) == 61 and all(row['passed'] for row in summary['checks'])
key = lambda row: (row['cohort'],row['batch'],row['mode'],row['stress_stride'])
old_rows = {key(row): row for row in original['rows']}
for row in summary['rows']:
    old = old_rows[key(row)]
    assert {k:v for k,v in row.items() if k != 'sampled_process_memory'} == {k:v for k,v in old.items() if k != 'sampled_process_memory'}
assert sum(row['sampled_process_memory'] is not None for row in summary['rows']) == 20
campaigns = ['process-memory-controls-v1','native-module-memory-process-v1','native-production-memory128-512-process-v1','native-production-memory2048-8192-process-v1','native-module-memory-process-audit-v1','native-production-memory128-512-process-audit-v1','native-production-memory2048-8192-process-audit-v1','native-scene-process-memory-summary-v1']
for campaign in campaigns:
    for name,sha in contract['workers'].items():
        if campaign == 'process-memory-controls-v1' and name not in control_contract['workers']:
            continue
        path = RUN / campaign / 'analysis-source/research/rigid_stress' / name
        if path.is_file():
            expected_sha = control_contract['workers'].get(name,sha) if campaign == 'process-memory-controls-v1' else sha
            assert identity(path)['sha256'] == expected_sha, (campaign,name)
profile_wrappers = json.loads((META / 'process-memory-wrapper-equivalence-v1.json').read_text())
assert len(profile_wrappers['proofs']) == 2 and all(row['entire_module_AST_identical_after_only_removing_new_PID_import_and_output_key'] for row in profile_wrappers['proofs'])
terminal_receipt = {'accounting':accounting,'control_checks':14,'profiles':24,'module_binding_checks':len(module['checks']),'scene_binding_checks':len(summary['checks']),'scene_receipts':len(summary['receipts']),'query_gap_target_seconds':1.0,'all_observed_query_gaps_within_target':all(row['observed_query_gap_within_target'] for row in coverage),'maximum_observed_query_gap_seconds':max(row['maximum_observed_query_gap_seconds'] for row in coverage),'primary61_rows_unchanged_except_separate_memory':True,'whole_goal_complete':False}
(META / 'independent-process-memory-terminal-v1.json').write_text(json.dumps(terminal_receipt,indent=2)+'\n')
files = [SCRIPT] + [META / name for name in ['process-memory-controls-v1-tasks.json','process-memory-contract-v1.json','process-memory-profile-contract-v1.json','process-memory-wrapper-equivalence-v1.json','process-memory-original-profile-walltimes-v1.json','native-module-memory-process-v1-tasks.json','native-production-memory128-512-process-v1-tasks.json','native-production-memory2048-8192-process-v1-tasks.json','native-module-memory-process-audit-v1-tasks.json','native-production-memory128-512-process-audit-v1-tasks.json','native-production-memory2048-8192-process-audit-v1-tasks.json','native-scene-process-memory-summary-v1-tasks.json','independent-process-memory-terminal-v1.json']]
modules = ['shell_comparison/memory.py'] + ['qipc_comparison/'+name for name in ['artifacts.py','check_process_memory.py','process_memory_receipts.py','native_module_memory.py','native_production_memory.py','native_production_controls.py','native_observation_cadence.py','native_module_replay_audit.py','audit_native_module_memory.py','audit_native_production_memory.py','native_scene_summary.py']]
for campaign in campaigns:
    present = [name for name in modules if (RUN / campaign / 'analysis-source/research/rigid_stress' / name).is_file()]
    files += captured(campaign,present) + terminal(campaign)
    files += list((RUN / campaign).glob('group*/**/*.csv'))
files.append(RUN / 'process-memory-controls-v1/group0/controls/failing-bin/nvidia-smi')
external = list((RUN / 'native-module-memory-process-v1').glob('group*/**/*.npz'))
raw_benchmarks = [path for path in files if path.name == 'benchmark.json']
files = [path for path in files if path.name != 'benchmark.json']
external += raw_benchmarks
files.append(META / 'independent-process-memory-draft-migration-v1.json')
row = freeze('independent-process-memory-v2',files,external,'Independent stdlib-only child-process NVIDIA sampling replaces the workload-GIL-bound thread for separate profiles. Actual3s GIL-hold, query-error/child cleanup and disabled-mode integration controls pass14 checks; wrapper AST proof permits only the new profile-PID import/record. All20 exact native-scene configurations and4 complete-history module profiles retain original numerical drivers/arguments/acceptance and pass source/configuration/collector/complete-poll audits. Actual query/sample timing and the predeclared1s query-gap target remain explicit; no continuous memory-peak or hardware-capacity bound is claimed. The61 primary performance rows and all physical-error cells are unchanged except their separately bound memory observations. Twenty full raw benchmark JSON files and eight complete replay arrays remain on data with exact hashes; the oversized untracked first draft is preserved on data with a verified migration journal. Original thread profiles, live fine writers and queued final friction products remain intact.')
(META / 'independent-process-memory-publication-v2.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row))
