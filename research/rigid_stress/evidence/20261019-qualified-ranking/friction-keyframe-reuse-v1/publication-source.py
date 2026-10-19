"""Publish terminal keyframe costs, failed observations and response reuse controls."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/friction-keyframe-reuse-v1')
CAMPAIGNS = {
    'solid-pad-stvk-response-reuse-control-v1': [0],
    'pad-friction-keyframe-cost-v1': [0, 0, 0],
    'pad-friction-scr-rejection-capture-v1': [1],
    'pad-friction-scr-input-inspection-v1': [0],
    'pad-friction-source-cones-v1': [0, 0],
    'pad-friction-scr-final-inspection-v1': [0],
}
SUPPORT = ('artifacts', 'p2_stvk', 'p2_elastic', 'plate_solid_pad', 'solid_pad_step', 'modal_pad_dynamics', 'matrix_free_pad_dynamics', 'condensed_pad_dynamics', 'cvxopt_factor', 'compliant_plate_reference', 'common_column_peaks', 'contact_fem_exact_columns', 'check_matrix_free_pad', 'check_p2_stvk', 'audit_solid_pad', 'pad_primary_cost', 'pad_cost', 'plate_native_pad', 'plate_shell_coupled', 'native_pad_recovery', 'native_reference_replay', 'audit_native_pad', 'audit_coupled_shell', 'audit_pad_primary_cost', 'inspect_pad_rejection', 'capture_native_pad_input')

def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    for campaign, codes in CAMPAIGNS.items():
        for group, code in enumerate(codes):
            assert (ROOT/campaign/f'group{group}/returncode.txt').read_text().strip() == str(code)
    assert json.loads((ROOT/'meta/solid-pad-stvk-response-reuse-admission-v1.json').read_text())['passed']
    for group in range(3):
        assert json.loads((ROOT/f'pad-friction-keyframe-cost-v1/group{group}/primary-audit/result.json').read_text())['passed']
    parent = ROOT/'pad-primary-friction-cost-v1'
    assert (parent/'group1/returncode.txt').read_text().strip() == '1'
    DESTINATION.mkdir(exist_ok=False)
    files, omitted, copied = [], [], set()
    def copy(path, relative):
        if str(relative) in copied: return
        copied.add(str(relative))
        receipt = {'source': str(path), 'copy': str(relative), **identity(path)}
        if receipt['bytes'] > 1_000_000:
            omitted.append(receipt); return
        target = DESTINATION/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert identity(path) == identity(target)
        files.append(receipt)
    def sources(folder):
        paths = set()
        for pattern in ('source*.json', 'source*.patch', 'extra-source*', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        tasks = json.loads((folder/'tasks.json').read_text())
        modules = {t['module'] for g in tasks for t in g}
        modules.update('research.rigid_stress.qipc_comparison.'+name for name in SUPPORT)
        for module in modules:
            path = folder/'analysis-source'/(module.replace('.', '/')+'.py')
            if path.exists(): paths.add(path)
        for group in tasks:
            for task in group:
                args = task['args']
                if '--task' in args:
                    copy(Path(args[args.index('--task')+1]), Path(args[args.index('--task')+1]).relative_to(ROOT))
        return paths
    for campaign in CAMPAIGNS:
        folder = ROOT/campaign
        paths = sources(folder)
        for pattern in ('group*/returncode.txt', 'group*/*.log', 'logs/*.log', 'group*/**/result.json', 'group*/**/start.json', 'group*/**/cost.json', 'group*/**/steps.jsonl', 'group*/**/input.npz', 'group*/**/input-ready.json', 'group*/**/rejected-stress.npz'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        for path in sorted(paths): copy(path, path.relative_to(ROOT))
    paths = sources(parent)
    for pattern in ('returncode.txt', '*.log', '**/start.json', '**/result.json', '**/cost.json', '**/steps.jsonl'):
        paths.update(p for p in (parent/'group1').glob(pattern) if p.is_file())
    for path in sorted(paths): copy(path, path.relative_to(ROOT))
    for name in ('solid-pad-stvk-response-reuse-control-v1-targets.json', 'solid-pad-stvk-response-reuse-admission-v1.json', 'audit-response-reuse-v1.py'):
        copy(ROOT/'meta'/name, Path('meta')/name)
    copy(Path(__file__), Path('publication-source.py'))
    manifest = {
        'scope': 'Admitted B1 full-field friction keyframe costs at stride200, retained failed every-step SCR histories and exact pre-call/rejected inputs, independent source-cone certificates, and validated optional full-compliance response reuse. No native every-step friction admission or spatial reference qualification is inferred. All selected execution groups are terminal; active fine-reference writers are excluded.',
        'files': files, 'large_files_retained_on_data': omitted,
        'verified_file_count': len(files), 'verified_bytes': sum(r['bytes'] for r in files),
        'whole_goal_complete': False,
    }
    (DESTINATION/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for r in files:
        assert identity(DESTINATION/r['copy']) == {'bytes': r['bytes'], 'sha256': r['sha256']}
    print(json.dumps({k: manifest[k] for k in ('verified_file_count', 'verified_bytes')}))

if __name__ == '__main__': main()
