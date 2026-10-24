"""Focused implementation audit; no mesh sweep or performance rerun."""
import json,hashlib
from pathlib import Path
import numpy as np
from plate_atlas import grid,ShellModel,probe_operator,rel,H,E,NU,HERE
from shell_cpu.contacts import Contact,ShellDPRTransfer
from shell_cpu.review_checks import independent_rectangle_weights

def main():
    x=np.array([-.02,-.003,.007,.02]);y=np.array([-.02,-.006,.005,.02]);v,q=grid(x,y)
    m=ShellModel(v,q,h=H,E=E,nu=NU)
    p=[]
    for ids in q:
        for xi,eta in m.gauss_points:
            N=.25*(1+np.array([-1,1,1,-1])*xi)*(1+np.array([-1,-1,1,1])*eta)
            p.append(N@v[ids,:2])
    S=probe_operator(m,x,y,np.array(p));rng=np.random.default_rng(89);u=rng.normal(size=m.ndof)*1e-7
    s=m.stress(u);expected=np.stack([s['local_top'],s['local_bottom']]).reshape(2,-1,3)
    operator_error=rel((S@u).reshape(2,-1,3),expected)
    transfer=ShellDPRTransfer(m);controls=[]
    for center,radius in [((.0007,.0003),.0024),((-.003,-.006),.0024),((.0052,-.004),.0036)]:
        w=independent_rectangle_weights(v,q,np.r_[center,H/2],radius)
        force=np.array([.013,-.005,-.05]);f=np.zeros((len(v),6));f[:,:3]=w[:,None]*force
        f[:,3:]=w[:,None]*np.cross([0,0,H/2],force)
        actual,info=transfer.transfer(Contact(np.r_[center,H/2],force,np.array([0.,0.,1.]),(50.,70.),radius=radius,friction=.6),order=32)
        controls.append(dict(center=center,radius=radius,cartesian_vs_existing_transfer_relative=rel(actual,f.ravel()),info=info))
    d=json.loads((HERE/'results/results.json').read_text());e=json.loads((HERE/'egg_results/results.json').read_text())
    gate_values=[g for row in d['gates'] for gg in row['models'].values() for g in gg]
    tests=dict(flat_common_probe_operator_matches_core=operator_error<1e-12,
       cartesian_hertz_weights_match_existing_transfer=all(c['cartesian_vs_existing_transfer_relative']<1e-8 for c in controls),
       all_plate_source_wrenches_pass=max(g.get('force_relative',0) for g in gate_values)<1e-9 and max(g.get('moment_normalized',0) for g in gate_values)<1e-8,
       all_plate_linear_solves_pass=max(g.get('linear_residual',0) for g in gate_values)<1e-8,
       coarse_plus_delta_identity=all(r['coarse_delta_identity']<1e-12 for r in d['rows']),
       unload_exact=next(r for r in d['rows'] if r['case']=='release')['coarse_delta_identity']==0,
       all_egg_sources_numerically_admitted=not e['failures'],
       balanced_egg_no_inertia_relief_needed=max(i['raw_wrench_norm'] for r in e['rows'] for gg in r['checks'].values() for i in gg)<1e-10,
       egg_source_is_normal_only=max(k['fixture_relative_tangential_roundoff'] for r in e['rows'] for gg in r['checks'].values() for i in gg for k in i['contact_checks'])<1e-14)
    out=dict(scope='Numerical implementation controls only. Physical/reference convergence is not a pass gate in this audit and remains explicitly unresolved.',operator_relative=operator_error,finite_pressure_controls=controls,checks=tests,all_checks_passed=all(tests.values()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    assert out['all_checks_passed']

if __name__=='__main__':main()
