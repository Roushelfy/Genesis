"""Independent serial shell reference and analytic plate controls.

The reference uses the installed pyfe3d Quad4 implementation, not the prototype
stiffness or strain operators.  Quad4 is a Hughes-style mixed-integration shell,
so curved/faceted-mesh differences from the prototype's covariant MITC4 are
discretization differences; exact equality is not an acceptance requirement.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import importlib.util

import numpy as np
import scipy.linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def availability():
    import pyfe3d
    return {
        "pyfe3d_version": pyfe3d.__version__,
        "code_aster_binary": shutil.which("run_aster"),
        "as_run_binary": shutil.which("as_run"),
        "code_aster_module": importlib.util.find_spec("code_aster") is not None,
        "opensees_note": "3.8.0.0 and 3.7.1.2 installed, but both imports aborted in MPI network-interface initialization; no OpenSees solve executed",
        "reference_scope": "independent prescribed-load linear shell FEM, not an independent contact solve",
    }


def rigid_modes(vertices):
    x = np.asarray(vertices)
    r = np.zeros((6 * len(x), 6))
    center = x.mean(axis=0)
    for i, point in enumerate(x - center):
        r[6*i:6*i+3, :3] = np.eye(3)
        for j in range(3):
            r[6*i:6*i+3, 3+j] = np.cross(np.eye(3)[j], point)
        r[6*i+3:6*i+6, 3:] = np.eye(3)
    return r


class Pyfe3dReference:
    """pyfe3d Quad4 assembly, constrained sparse solve and native strain probe."""
    def __init__(self, vertices, quads, E=1.e10, nu=.3, h=.0005,
                 drilling_model=0):
        from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE
        from pyfe3d.shellprop import ShellProp
        self.vertices = np.asarray(vertices, dtype=float)
        self.quads = np.asarray(quads, dtype=np.int64)
        self.E, self.nu, self.h = E, nu, h
        self.ndof = 6 * len(vertices)
        self.drilling_model = drilling_model
        p = ShellProp()
        p.h = h
        C = E / (1. - nu**2)
        p.A11 = p.A22 = C*h
        p.A12 = C*h*nu
        p.A66 = E*h/(2*(1+nu))
        p.D11 = p.D22 = C*h**3/12
        p.D12 = C*h**3*nu/12
        p.D66 = E*h**3/(24*(1+nu))
        p.A44 = p.A55 = (5./6.)*E*h/(2*(1+nu))
        self.prop = p
        self.probe = Quad4Probe()
        size = Quad4Data().KC0_SPARSE_SIZE
        row = np.zeros(size*len(quads), dtype=INT)
        col = row.copy()
        val = np.zeros(len(row), dtype=DOUBLE)
        self.elements = []
        xyz = np.ascontiguousarray(self.vertices.ravel())
        for e, nodes in enumerate(self.quads):
            q = Quad4(self.probe)
            q.eid = e+1
            for i, node in enumerate(nodes, 1):
                setattr(q, "n"+str(i), int(node)+1)
                setattr(q, "c"+str(i), int(node)*6)
            q.drilling_model = drilling_model
            q.K6ROT = 100.
            q.init_k_KC0 = e*size
            q.update_rotation_matrix(xyz)
            q.update_probe_xe(xyz)
            q.update_KC0(row, col, val, p)
            self.elements.append(q)
        self.K = coo_matrix((val,(row,col)),shape=(self.ndof,self.ndof)).tocsr()
        self.R = rigid_modes(self.vertices)
        self._factor = None

    def solve(self, force, fixed_dofs=None):
        f = np.asarray(force)
        if fixed_dofs is None:
            # Remove six physical null modes using a rank-revealing selection.
            # Loads must already be balanced; this does not silently alter them.
            _, _, piv = scipy.linalg.qr(self.R.T, pivoting=True, mode="economic")
            fixed_dofs = piv[:6]
        fixed = np.unique(np.asarray(fixed_dofs,dtype=int))
        mask = np.ones(self.ndof,dtype=bool); mask[fixed]=False
        key = tuple(fixed)
        if self._factor is None or self._factor[0] != key:
            self._factor=(key,mask,splu(self.K[mask][:,mask].tocsc()))
        q = np.zeros_like(f,dtype=float)
        q[mask]=self._factor[2].solve(np.ascontiguousarray(f[mask]))
        return q

    def stress(self, q):
        u = np.ascontiguousarray(np.asarray(q).ravel())
        x = np.ascontiguousarray(self.vertices.ravel())
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        N, M, S = [], [], []
        for element in self.elements:
            element.update_probe_xe(x)
            element.update_probe_ue(u)
            ue = np.asarray(self.probe.ue)
            R=np.array([[element.r11,element.r12,element.r13],
                        [element.r21,element.r22,element.r23],
                        [element.r31,element.r32,element.r33]])
            ns, ms, ss = [], [], []
            for xi, eta in [(gp[0],gp[0]),(gp[1],gp[0]),(gp[1],gp[1]),(gp[0],gp[1])]:
                self.probe.update_BL(xi,eta,self.drilling_model)
                eps=np.array([np.asarray(self.probe.BLexx)@ue,
                              np.asarray(self.probe.BLeyy)@ue,
                              np.asarray(self.probe.BLgxy)@ue])
                kap=np.array([np.asarray(self.probe.BLkxx)@ue,
                              np.asarray(self.probe.BLkyy)@ue,
                              np.asarray(self.probe.BLkxy)@ue])
                n=self.prop.A@eps; m=self.prop.D@kap
                surfaces=[]
                for zsign in [-1,1]:
                    s=n/self.h+zsign*6*m/self.h**2
                    sl=np.array([[s[0],s[2],0.],[s[2],s[1],0.],[0.,0.,0.]])
                    surfaces.append(R@sl@R.T)
                ns.append(n);ms.append(m);ss.append(surfaces)
            N.append(ns);M.append(ms);S.append(ss)
        S=np.asarray(S)
        return {"N":np.asarray(N),"M":np.asarray(M),
                "sigma_bottom":S[:,:,0],"sigma_top":S[:,:,1]}


def plate_mesh(n, side=.1):
    x=np.linspace(0,side,n+1)
    vertices=np.array([[xx,yy,0.] for yy in x for xx in x])
    quads=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i
            quads.append([a,a+1,a+n+2,a+n+1])
    return vertices,np.asarray(quads)


def plate_uniform_load(vertices,quads,pressure):
    f=np.zeros(6*len(vertices))
    for face in quads:
        xyz=vertices[face]
        area=np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[3]-xyz[0]))
        f[6*face+2]+=pressure*area/4
    return f


def plate_simply_supported_dofs(vertices,side=.1):
    fixed=[]
    for i,(x,y,z) in enumerate(vertices):
        fixed.extend([6*i,6*i+1,6*i+5])
        if np.isclose(x,0) or np.isclose(x,side) or np.isclose(y,0) or np.isclose(y,side):
            fixed.append(6*i+2)
    return np.array(fixed)


def navier_square_center(E=1.e10,nu=.3,h=.0005,side=.1,pressure=10.,terms=151):
    """Classical simply supported Kirchhoff plate under uniform pressure."""
    m=np.arange(1,terms+1,2,dtype=float);n=m.copy()
    mm,nn=np.meshgrid(m,n,indexing='ij')
    signs=np.sin(mm*np.pi/2)*np.sin(nn*np.pi/2)
    s=(mm/side)**2+(nn/side)**2
    D=E*h**3/(12*(1-nu**2))
    w=16*pressure/(D*np.pi**6)*np.sum(signs/(mm*nn*s**2))
    Mx=16*pressure/np.pi**4*np.sum(signs*((mm/side)**2+nu*(nn/side)**2)/(mm*nn*s**2))
    return {"center_w_m":float(w),"center_M_N":float(Mx),
            "center_surface_stress_pa":float(6*Mx/h**2),"terms":terms}


def navier_square_field(xy,E=1.e10,nu=.3,h=.0005,side=.1,pressure=10.,terms=151):
    """Independent Fourier-series membrane-free bending stresses at points."""
    pts=np.asarray(xy)
    m=np.arange(1,terms+1,2,dtype=float)
    mm,nn=np.meshgrid(m,m,indexing='ij')
    S=(mm/side)**2+(nn/side)**2
    D=E*h**3/(12*(1-nu**2))
    W=16*pressure/(D*np.pi**6*mm*nn*S**2)
    sx=np.sin(np.pi*pts[:,0,None]*m/side)
    sy=np.sin(np.pi*pts[:,1,None]*m/side)
    cx=np.cos(np.pi*pts[:,0,None]*m/side)
    cy=np.cos(np.pi*pts[:,1,None]*m/side)
    w=np.sum((sx@W)*sy,axis=1)
    wxx=-np.pi**2*np.sum((sx@(W*(mm/side)**2))*sy,axis=1)
    wyy=-np.pi**2*np.sum((sx@(W*(nn/side)**2))*sy,axis=1)
    wxy=np.pi**2*np.sum((cx@(W*mm*nn/side**2))*cy,axis=1)
    M=np.column_stack((-D*(wxx+nu*wyy),-D*(wyy+nu*wxx),-D*(1-nu)*wxy))
    st=6*M/h**2
    tensor=np.zeros((len(pts),3,3))
    tensor[:,0,0]=st[:,0];tensor[:,1,1]=st[:,1]
    tensor[:,0,1]=tensor[:,1,0]=st[:,2]
    return w,M,tensor


def run_plate_reference(ns=(4,8,16,32),out=None):
    E,nu,h,side,p=1.e10,.3,.0005,.1,10.
    exact=navier_square_center(E,nu,h,side,p)
    rows=[]
    for n in ns:
        v,e=plate_mesh(n,side)
        model=Pyfe3dReference(v,e,E,nu,h)
        f=plate_uniform_load(v,e,p)
        q=model.solve(f,plate_simply_supported_dofs(v,side))
        center=np.argmin(np.linalg.norm(v-np.array([side/2,side/2,0]),axis=1))
        w=q[6*center+2]
        rows.append({"n":n,"nodes":len(v),"dofs":len(q),"center_w_m":float(w),
                     "relative_w_error":float(abs(w-exact['center_w_m'])/exact['center_w_m'])})
    result={"availability":availability(),"analytic":exact,"rows":rows}
    if out:
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(out).write_text(json.dumps(result,indent=2))
    return result


def compare_with_core(vertices,quads,force,*,E=1.e10,nu=.3,h=.0005,
                      fixed_dofs=None,core_model=None,reference_model=None):
    """Compare independent formulations under exactly the same nodal loads.

    For a free shell, rigid-body inertia relief is applied explicitly once and
    the resulting vector is supplied identically to both solvers.  Returned
    relief is recorded so this cannot be mistaken for the unmodified load.
    """
    from .core import ShellModel
    own=core_model or ShellModel(vertices,quads,E=E,nu=nu,h=h)
    ref=reference_model or Pyfe3dReference(vertices,quads,E=E,nu=nu,h=h)
    f=np.asarray(force,dtype=float)
    relief=np.zeros(6)
    if fixed_dofs is None:
        f,relief=own.balance_load(f)
        fixed=np.asarray(own.gauge_dofs)
    else:
        fixed=np.asarray(fixed_dofs)
    a=own.solve(f,fixed_dofs=fixed,balance=False)
    b=ref.solve(f,fixed_dofs=fixed)
    sa,sb=own.stress(a),ref.stress(b)
    ta=np.stack([sa['sigma_bottom'],sa['sigma_top']])
    tb=np.stack([sb['sigma_bottom'],sb['sigma_top']])
    weights=own.gauss_weights
    err=np.sqrt(np.sum(weights[None,:,:,None,None]*(ta-tb)**2)/
                max(np.sum(weights[None,:,:,None,None]*tb**2),1e-300))
    qa,qb=a.copy(),b.copy()
    if fixed_dofs is None:
        R=own.rigid_modes
        qa-=R@np.linalg.lstsq(R,qa,rcond=None)[0]
        qb-=R@np.linalg.lstsq(R,qb,rcond=None)[0]
    transa,transb=qa.reshape((-1,6))[:,:3],qb.reshape((-1,6))[:,:3]
    def peaks(t):
        mean=np.trace(t,axis1=-2,axis2=-1)/3
        dev=t-mean[...,None,None]*np.eye(3)
        vm=np.sqrt(1.5*np.sum(dev*dev,axis=(-2,-1)))
        tensile=np.linalg.eigvalsh(t)[...,-1]
        return float(vm.max()),float(tensile.max())
    pa,pb=peaks(ta),peaks(tb)
    return {
        'displacement_relative':float(np.linalg.norm(transa-transb)/max(np.linalg.norm(transb),1e-300)),
        'surface_tensor_relative_L2':float(err),
        'core_max_surface_stress_pa':float(max(np.max(np.linalg.norm(sa['sigma_top'],axis=(-2,-1))),np.max(np.linalg.norm(sa['sigma_bottom'],axis=(-2,-1))))),
        'reference_max_surface_stress_pa':float(max(np.max(np.linalg.norm(sb['sigma_top'],axis=(-2,-1))),np.max(np.linalg.norm(sb['sigma_bottom'],axis=(-2,-1))))),
        'surface_stress_max_metric':'Frobenius tensor norm; additional explicit VM/tensile maxima below',
        'core_max_surface_vm_pa':pa[0], 'reference_max_surface_vm_pa':pb[0],
        'core_max_surface_tensile_pa':pa[1], 'reference_max_surface_tensile_pa':pb[1],
        'relative_compliance_difference':float(abs(f@a-f@b)/max(abs(f@b),1e-300)),
        'explicit_relief_coefficients':relief.tolist(),
        'same_applied_load':True,
        'reference':'pyfe3d0.10.0 Quad4 mixed integration; independent flat-faceted shell discretization',
        'scope':'prescribed-load shell-discretization comparison, not physical contact-reference validation',
    },a,b


def run_plate_comparison(ns=(4,8,16),out=None):
    from .core import ShellModel
    E,nu,h,side,p=1.e10,.3,.0005,.1,10.
    exact=navier_square_center(E,nu,h,side,p)
    rows=[]
    for n in ns:
        v,e=plate_mesh(n,side)
        a=ShellModel(v,e,E=E,nu=nu,h=h)
        b=Pyfe3dReference(v,e,E=E,nu=nu,h=h)
        f=plate_uniform_load(v,e,p);fixed=plate_simply_supported_dofs(v,side)
        row,qa,qb=compare_with_core(v,e,f,E=E,nu=nu,h=h,fixed_dofs=fixed,
                                  core_model=a,reference_model=b)
        i=np.argmin(np.linalg.norm(v-[side/2,side/2,0],axis=1))
        row.update(n=n,nodes=len(v),dofs=len(qa),
                   core_center_w_m=float(qa[6*i+2]),reference_center_w_m=float(qb[6*i+2]),
                   core_center_w_vs_Kirchhoff=float(abs(qa[6*i+2]/exact['center_w_m']-1)),
                   reference_center_w_vs_Kirchhoff=float(abs(qb[6*i+2]/exact['center_w_m']-1)))
        gp=a.gauss_points
        signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        N=.25*(1+gp[:,0,None]*signs[:,0])*(1+gp[:,1,None]*signs[:,1])
        positions=np.einsum('gn,enp->egp',N,v[e])
        _,_,analytic_tensor=navier_square_field(positions.reshape((-1,3))[:,:2],E,nu,h,side,p)
        analytic_tensor=analytic_tensor.reshape((-1,4,3,3))
        for name,model,q in [('core',a,qa),('reference',b,qb)]:
            tensor=model.stress(q)['sigma_top']
            row[name+'_surface_tensor_vs_Kirchhoff']=float(np.sqrt(
                np.sum(a.gauss_weights[:,:,None,None]*(tensor-analytic_tensor)**2)/
                np.sum(a.gauss_weights[:,:,None,None]*analytic_tensor**2)))
        rows.append(row)
    result={'availability':availability(),'analytic':exact,'rows':rows}
    if out:
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(out).write_text(json.dumps(result,indent=2))
    return result


def run_egg_comparison(ns=(4,8,12),out=None):
    from .geometry import egg_mesh,egg_geometry
    from .core import ShellModel
    from .contacts import Contact,ShellDPRTransfer
    E,nu,h,Fn=1.e10,.3,.0005,.2
    axes=(.018,.018,.026);beta=.15
    s=(-1+np.sqrt(1+8*beta**2))/(4*beta)
    xe=axes[0]*(1+beta*s)*np.sqrt(1-s*s)
    points=np.array([[xe,0.,axes[2]*s],[-xe,0.,axes[2]*s]])
    normals,curvature=egg_geometry(points,axes,beta)
    rows=[]
    for n in ns:
        v,e=egg_mesh(n,axes,beta)
        directors=egg_geometry(v,axes,beta)[0]
        own=ShellModel(v,e,E=E,nu=nu,h=h,directors=directors)
        ref=Pyfe3dReference(v,e,E=E,nu=nu,h=h)
        transfer=ShellDPRTransfer(own)
        for radius in (None,.002):
            for shear in (0.,.03):
                contacts=[]
                for side,(point,normal,k) in enumerate(zip(points,normals,curvature)):
                    force=-Fn*normal+np.array([0.,shear*(1 if side==0 else -1),0.])
                    contacts.append(Contact(point+.5*h*normal,force,normal,k,1,radius,.5))
                try:
                    f,records=transfer.transfer_many(contacts,order=24)
                    row,qa,qb=compare_with_core(v,e,f,E=E,nu=nu,h=h,core_model=own,reference_model=ref)
                    row.update(passed=True,n=n,nodes=len(v),quads=len(e),dofs=own.ndof,
                               radius_mode='automatic_hertz' if radius is None else 'manual_2mm',
                               shear_N=shear,contacts=records)
                except Exception as exc:
                    row=dict(passed=False,n=n,radius_mode='automatic_hertz' if radius is None else 'manual_2mm',
                             shear_N=shear,error=repr(exc))
                rows.append(row)
                print('egg reference',row,flush=True)
        if out:
            Path(out).parent.mkdir(parents=True,exist_ok=True)
            Path(out).write_text(json.dumps({'availability':availability(),'rows':rows},indent=2))
    return {'availability':availability(),'rows':rows,
            'scope':'independent shell FEM under identical DPR nodal loads; not independently solved contact or3Dstress'}


def uniform_surface_pressure(vertices,quads,pressure):
    """Consistent vector pressure on each bilinear reference midsurface."""
    f=np.zeros(6*len(vertices))
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    g=np.polynomial.legendre.leggauss(3)
    for face in quads:
        xyz=vertices[face]
        for xi,wx in zip(*g):
            for eta,wy in zip(*g):
                N=.25*(1+signs[:,0]*xi)*(1+signs[:,1]*eta)
                dxi=.25*signs[:,0]*(1+signs[:,1]*eta)
                deta=.25*signs[:,1]*(1+signs[:,0]*xi)
                df=pressure*np.cross(dxi@xyz,deta@xyz)*wx*wy
                for k,node in enumerate(face):
                    f[6*node:6*node+3]+=N[k]*df
    return f


def run_sphere_comparison(ns=(2,4,8),out=None):
    from .geometry import sphere_mesh
    from .core import ShellModel
    E,nu,h,R,p=1.e10,.3,.0005,.02,100.
    exact_u=p*R**2*(1-nu)/(2*E*h)
    exact_stress=p*R/(2*h)
    rows=[]
    for n in ns:
        v,e=sphere_mesh(n,R)
        a=ShellModel(v,e,E=E,nu=nu,h=h,directors=v/R)
        b=Pyfe3dReference(v,e,E=E,nu=nu,h=h)
        f=uniform_surface_pressure(v,e,p)
        row,qa,qb=compare_with_core(v,e,f,E=E,nu=nu,h=h,core_model=a,reference_model=b)
        sa,sb=a.stress(qa),b.stress(qb)
        normal=v/R
        for name,q,s in [('core',qa,sa),('reference',qb,sb)]:
            u=q.reshape((-1,6))[:,:3]
            radial=np.einsum('ij,ij->i',u,normal)
            sm=.5*(s['sigma_top']+s['sigma_bottom'])
            meanstress=np.trace(sm,axis1=-2,axis2=-1)/2
            avg=np.average(meanstress,weights=a.gauss_weights)
            row[name+'_mean_radial_u_m']=float(np.mean(radial))
            row[name+'_mean_radial_u_error']=float(abs(np.mean(radial)/exact_u-1))
            row[name+'_mean_membrane_stress_pa']=float(avg)
            row[name+'_mean_membrane_stress_error']=float(abs(avg/exact_stress-1))
        row.update(n=n,nodes=len(v),quads=len(e),dofs=a.ndof)
        rows.append(row)
        print('sphere reference',n,row,flush=True)
    result={'analytic_radial_u_m':exact_u,'analytic_membrane_stress_pa':exact_stress,
            'availability':availability(),'rows':rows,
            'scope':'uniform pressure control of shell discretization, not local contact model'}
    if out:
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(out).write_text(json.dumps(result,indent=2))
    return result


def plot_saved_results(directory='shell_cpu/reference_results'):
    """Reproduce the compact validation figure from the retained JSON data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(directory)
    plate=json.loads((p/'plate_analytic_field.json').read_text())['rows']
    sphere=json.loads((p/'sphere_independent.json').read_text())['rows']
    sphere+=json.loads((p/'sphere_independent_n16.json').read_text())['rows']
    egg=json.loads((p/'egg_independent_repaired.json').read_text())['rows']
    egg+=json.loads((p/'egg_independent_refined.json').read_text())['rows']
    egg=sorted([r for r in egg if r.get('passed') and r['radius_mode']=='automatic_hertz'
                and r['shear_N']==0],key=lambda r:r['n'])
    fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    cases=[(ax[0,0],plate,'core_center_w_vs_Kirchhoff','reference_center_w_vs_Kirchhoff',
            'Plate: center displacement vs Navier'),
           (ax[0,1],plate,'core_surface_tensor_vs_Kirchhoff','reference_surface_tensor_vs_Kirchhoff',
            'Plate: full surface-tensor L2 vs Navier'),
           (ax[1,0],sphere,'core_mean_membrane_stress_error','reference_mean_membrane_stress_error',
            'Sphere: mean membrane stress vs pR/(2h)')]
    for a,rows,key1,key2,title in cases:
        for key,label in [(key1,'Prototype MITC4'),(key2,'Independent pyfe3d')]:
            a.loglog([r['dofs'] for r in rows],[100*r[key] for r in rows],'o-',label=label)
        a.set_title(title);a.set_ylabel('relative error (%)')
    for key,label in [('core_max_surface_vm_pa','Prototype MITC4'),
                      ('reference_max_surface_vm_pa','Independent pyfe3d')]:
        ax[1,1].semilogx([r['dofs'] for r in egg],[r[key]/1e6 for r in egg],'o-',label=label)
    ax[1,1].set_title('Egg: tiny-footprint peak remains unresolved')
    ax[1,1].set_ylabel('surface von Mises peak (MPa)')
    for a in ax.flat:
        a.set_xlabel('scalar degrees of freedom');a.grid(True,which='both',alpha=.25)
    ax[0,0].legend();ax[1,1].legend()
    fig.suptitle('CPU shell validation: analytic controls and independent prescribed-load FEM',fontsize=12)
    fig.savefig(p/'independent_validation.png',dpi=180)
    plt.close(fig)
    return str(p/'independent_validation.png')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['plate-reference','plate','sphere','egg'],default='plate')
    parser.add_argument('--levels',help='comma-separated mesh subdivisions, e.g.4,8,16')
    parser.add_argument('--output',help='result JSON path')
    args=parser.parse_args()
    functions={'plate-reference':run_plate_reference,'plate':run_plate_comparison,
               'sphere':run_sphere_comparison,'egg':run_egg_comparison}
    kwargs={}
    if args.levels:
        kwargs['ns']=tuple(map(int,args.levels.split(',')))
    path=args.output or 'shell_cpu/reference_results/'+args.suite+'_comparison.json'
    print(json.dumps(functions[args.suite](out=path,**kwargs),indent=2))
