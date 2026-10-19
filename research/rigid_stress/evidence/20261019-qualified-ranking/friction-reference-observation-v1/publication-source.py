"""Publish terminal friction reference and observation evidence, preserving failures."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/friction-reference-observation-v1')
CAMPAIGNS = {'solid-pad-stvk-spatial500-v2': [0], 'stvk-spatial500-fields-v2': [0, 0, 0], 'pad-friction-dpr-rejection-capture-v1': [1], 'pad-friction-rejection-inspection-v1': [0]}
SUPPORT = ('artifacts', 'p2_stvk', 'p2_elastic', 'plate_solid_pad', 'solid_pad_step', 'modal_pad_dynamics', 'matrix_free_pad_dynamics', 'condensed_pad_dynamics', 'cvxopt_factor', 'compliant_plate_reference', 'common_column_peaks', 'contact_fem_exact_columns', 'pad_primary_cost', 'pad_cost', 'plate_native_pad', 'plate_shell_coupled', 'native_pad_recovery', 'native_reference_replay', 'audit_native_pad', 'audit_coupled_shell', 'audit_pad_primary_cost', 'inspect_pad_rejection')

def identity(path):
    return {'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    for campaign, codes in CAMPAIGNS.items():
        for group, code in enumerate(codes):
            assert (ROOT/campaign/f'group{group}/returncode.txt').read_text().strip() == str(code)
    parent=ROOT/'pad-primary-friction-cost-v1'
    for group,code in ((0,1),(2,0)):
        assert (parent/f'group{group}/returncode.txt').read_text().strip()==str(code)
    assert json.loads((parent/'group2/primary-audit/result.json').read_text())['passed']
    DESTINATION.mkdir(exist_ok=False)
    files,omitted,copied=[],[],set()
    def copy(path,relative):
        if str(relative) in copied:return
        copied.add(str(relative))
        receipt={'source':str(path),'copy':str(relative),**identity(path)}
        if receipt['bytes']>1_000_000:
            omitted.append(receipt);return
        target=DESTINATION/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);assert identity(path)==identity(target);files.append(receipt)
    def campaign_sources(folder):
        paths=set()
        for pattern in ('source*.json','source*.patch','extra-source*','submission.json','tasks.json','analysis-files.json','job.sh','workload-*.sh'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        tasks=json.loads((folder/'tasks.json').read_text())
        modules={t['module'] for g in tasks for t in g}
        modules.update('research.rigid_stress.qipc_comparison.'+name for name in SUPPORT)
        for module in modules:
            path=folder/'analysis-source'/(module.replace('.','/')+'.py')
            if path.exists():paths.add(path)
        return paths
    for campaign in CAMPAIGNS:
        folder=ROOT/campaign;paths=campaign_sources(folder)
        for pattern in ('group*/returncode.txt','group*/*.log','logs/*.log','group*/*/result.json','group*/*/start.json','group*/*/cost.json','group*/*/frames/rejected-stress.npz'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        for path in sorted(paths):copy(path,path.relative_to(ROOT))
    paths=campaign_sources(parent)
    for group in (0,2):
        for pattern in ('returncode.txt','*.log','*/start.json','*/result.json','*/cost.json'):
            paths.update(p for p in (parent/f'group{group}').glob(pattern) if p.is_file())
    paths.add(parent/'group1/normal-compatibility/result.json')
    for path in sorted(paths):copy(path,path.relative_to(ROOT))
    copy(Path(__file__),Path('publication-source.py'))
    manifest={'scope':'Completed500um/two-wall nonlinear history and physical-momentum audit, complete q9/q25 refinement and linear/StVK normal contrasts, original and reproduced DPR every-step friction-input rejection, read-only actual-source inspection, and independently admitted Shell1000-step friction primary cost. Only terminal groups0/2 and the completed normal-compatibility leaf are selected from the active primary-friction array; SCR runtime is excluded. Fine250um/wall controls and optional response-reuse control remain in progress. No friction continuum qualification or DPR every-step timing rank is claimed.','files':files,'large_files_retained_on_data':omitted,'verified_file_count':len(files),'verified_bytes':sum(r['bytes'] for r in files),'whole_goal_complete':False}
    (DESTINATION/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for r in files:assert identity(DESTINATION/r['copy'])=={'bytes':r['bytes'],'sha256':r['sha256']}
    print(json.dumps({k:manifest[k] for k in ('verified_file_count','verified_bytes')}))

if __name__=='__main__':main()
