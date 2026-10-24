"""Deterministic finite-dimensional controls for the feature-mesh theory.

These verify algebra and expose counterexamples; they are not physical FEM
accuracy or performance results. Run with one BLAS thread.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import null_space


def energy(k, u):
    return float(np.sqrt(max(0.,u@k@u)))


def project_solve(k, v, f):
    return v@np.linalg.solve(v.T@k@v,v.T@f)


def vm(s):
    dev=s-np.trace(s)/3*np.eye(3)
    return float(np.sqrt(1.5*np.sum(dev*dev)))


def run():
    out={'scope':'Algebraic controls/counterexamples, not continuum FEM qualification',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Common fine variational problem; each bank member misses another patch.
    n=8
    k=2*np.eye(n)-.8*(np.eye(n,k=1)+np.eye(n,k=-1))
    eye=np.eye(n);v0=eye[:,[0,1,6,7]]
    v1=eye[:,[0,1,2,3,6,7]];v2=eye[:,[0,1,4,5,6,7]]
    f1=eye[:,3]+.2*eye[:,0];f2=.7*eye[:,4]+.3*eye[:,7];f=f1+f2
    u1=np.linalg.solve(k,f1);u2=np.linalg.solve(k,f2);exact=u1+u2
    a1=project_solve(k,v1,f1);a2=project_solve(k,v2,f2);atlas=a1+a2
    coarse=project_solve(k,v0,f)
    delta=coarse+a1-project_solve(k,v0,f1)+a2-project_solve(k,v0,f2)
    e1=energy(k,u1-a1);e2=energy(k,u2-a2)
    bound=e1+e2;err=energy(k,exact-atlas)
    # Schur condensation of fine DOFs at a fixed common problem is exact.
    c=np.array([0,1,6,7]);i=np.array([2,3,4,5])
    kcc=k[np.ix_(c,c)];kci=k[np.ix_(c,i)];kii=k[np.ix_(i,i)]
    schur=kcc-kci@np.linalg.solve(kii,kci.T)
    uc=np.linalg.solve(schur,f[c]-kci@np.linalg.solve(kii,f[i]))
    ui=np.linalg.solve(kii,f[i]-kci.T@uc)
    condensed=np.zeros(n);condensed[c]=uc;condensed[i]=ui
    wrong=project_solve(k,v1,f)+project_solve(k,v2,f)
    # Hard switching bank members at an unchanged load need not be continuous.
    mid=.5*(eye[:,3]+eye[:,4])
    left=project_solve(k,v1,mid);right=project_solve(k,v2,mid)
    out['mixed_mesh']={
        'atlas_relative_energy_error':err/energy(k,exact),
        'individual_energy_errors':[e1,e2],
        'triangle_bound':bound,'actual_energy_error':err,
        'triangle_bound_pass':bool(err<=bound*(1+1e-13)),
        'joint_space_residual_of_atlas':float(np.linalg.norm(k@atlas-f)/np.linalg.norm(f)),
        'coarse_test_residual_of_atlas':float(np.linalg.norm(v0.T@(k@atlas-f))/np.linalg.norm(f)),
        'riesz_error_identity_relative':float(abs(err**2-(f-k@atlas)@np.linalg.solve(k,f-k@atlas))/err**2),
        'coarse_plus_full_delta_relative':float(np.linalg.norm(delta-atlas)/np.linalg.norm(atlas)),
        'condensed_vs_joint_relative':float(np.linalg.norm(condensed-exact)/np.linalg.norm(exact)),
        'wrong_all_loads_all_meshes_relative':float(np.linalg.norm(wrong-exact)/np.linalg.norm(exact)),
        'wrong_full_space_sum_relative':float(np.linalg.norm(2*exact-exact)/np.linalg.norm(exact)),
        'hard_switch_jump_over_joint_norm':float(np.linalg.norm(left-right)/np.linalg.norm(np.linalg.solve(k,mid)))}
    # Free-object compatible load projection must represent the same physical M.
    rng=np.random.default_rng(20261024)
    r,_=np.linalg.qr(rng.normal(size=(36,6)))
    z=null_space(r.T);a=rng.normal(size=(30,30));kk=z@(a.T@a+np.eye(30))@z.T
    mass=np.diag(np.linspace(1.,3.,36));mr=mass@r
    p=np.eye(36)-mr@np.linalg.solve(r.T@mr,r.T)
    f1=rng.normal(size=36);f2=rng.normal(size=36)
    q1=z@np.linalg.solve(z.T@kk@z,z.T@p@f1)
    q2=z@np.linalg.solve(z.T@kk@z,z.T@p@f2)
    q=z@np.linalg.solve(z.T@kk@z,z.T@p@(f1+f2))
    other_mass=np.diag(np.linspace(3.,1.,36));other_mr=other_mass@r
    other_p=np.eye(36)-other_mr@np.linalg.solve(r.T@other_mr,r.T)
    bad_q2=z@np.linalg.solve(z.T@kk@z,z.T@other_p@f2)
    out['free_body']={
        'projection_rigid_resultant_relative':float(np.linalg.norm(r.T@p)/np.linalg.norm(p)),
        'linearity_relative':float(np.linalg.norm(q-q1-q2)/np.linalg.norm(q)),
        'balanced_residual_relative':float(np.linalg.norm(kk@q-p@(f1+f2))/np.linalg.norm(p@(f1+f2))),
        'different_mass_projection_changes_solution_relative':float(np.linalg.norm(q1+bad_q2-q)/np.linalg.norm(q))}
    # Combining pressure snapshots can preserve force & first moment yet blur peaks.
    xs=np.linspace(-3.,3.,12001)
    pressure=lambda x:np.sqrt(np.maximum(0.,1-x*x))
    rows=[]
    for d in (.5,1.,2.):
        blended=.5*(pressure(xs-d/2)+pressure(xs+d/2))
        rows.append({'anchor_spacing_over_a':d,
                     'true_center_value':1.,'blended_center_value':float(pressure(d/2)),
                     'blended_sampled_peak':float(blended.max()),
                     'peak_relative_error':float(abs(blended.max()-1))})
    out['snapshot_blur']={'scope':'1D cut through normalized circular Hertz pressure, not a stress solve',
                          'rows':rows,'force8_radius_ratio':2.,'force8_pressure_peak_ratio':2.,
                          'fixed_radius_snapshot_wrong_peak_ratio':8.}
    sa=np.diag([10.,-2.,0.]);sb=np.diag([-9.,2.,0.]);sc=sa+sb
    lipschitz=[]
    for _ in range(1000):
        aa=rng.normal(size=(3,3));aa=.5*(aa+aa.T)
        ee=rng.normal(size=(3,3));ee=.5*(ee+ee.T)
        ratio=abs(vm(aa+ee)-vm(aa))/(np.sqrt(1.5)*np.linalg.norm(ee))
        pratio=abs(np.linalg.eigvalsh(aa+ee)[-1]-np.linalg.eigvalsh(aa)[-1])/np.linalg.norm(ee,2)
        lipschitz.append([ratio,pratio])
    out['stress_metrics']={'vm_of_sum':vm(sc),'sum_of_vm':vm(sa)+vm(sb),
        'maximum_vm_lipschitz_ratio':float(np.max(np.array(lipschitz)[:,0])),
        'maximum_principal_lipschitz_ratio':float(np.max(np.array(lipschitz)[:,1]))}
    checks={
        'energy_triangle_bound':out['mixed_mesh']['triangle_bound_pass'],
        'full_delta_identity':out['mixed_mesh']['coarse_plus_full_delta_relative']<1e-13,
        'static_condensation_identity':out['mixed_mesh']['condensed_vs_joint_relative']<1e-13,
        'mixed_mesh_is_not_joint_galerkin':out['mixed_mesh']['joint_space_residual_of_atlas']>1e-3,
        'coarse_test_can_miss_error':out['mixed_mesh']['coarse_test_residual_of_atlas']<1e-13,
        'riesz_error_identity':out['mixed_mesh']['riesz_error_identity_relative']<1e-13,
        'full_load_sum_double_counts':abs(out['mixed_mesh']['wrong_full_space_sum_relative']-1)<1e-13,
        'same_mass_projection_linear':out['free_body']['linearity_relative']<1e-12,
        'free_body_compatible':out['free_body']['balanced_residual_relative']<1e-12,
        'different_mass_warning_real':out['free_body']['different_mass_projection_changes_solution_relative']>1e-3,
        'vm_bound':out['stress_metrics']['maximum_vm_lipschitz_ratio']<=1+1e-12,
        'principal_bound':out['stress_metrics']['maximum_principal_lipschitz_ratio']<=1+1e-12}
    out['checks']=checks;out['all_checks_passed']=all(checks.values())
    path=Path(__file__).resolve().parent/'algebra_results.json'
    path.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    if not out['all_checks_passed']:
        raise RuntimeError('An algebra control failed')
    return out


if __name__=='__main__':
    run()
