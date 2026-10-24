"""Bounded CPU thin-shell versus independent P2 solid sphere comparison.

Both use a shared reference midsurface and its normal-offset thickness.
The solid is a separate 3D P2 discretization from the existing DPR CPU oracle.
This prescribed-pressure check does not solve unilateral contact or validate
the Hertz footprint.  Its continuum control is the 3D Lame sphere solution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from .core import ShellModel, quad_shape
from .geometry import sphere_mesh
from .contacts import Contact, ShellDPRTransfer, hertz_radius

def _find_repository():
    for parent in Path(__file__).resolve().parents:
        if (parent / "research/rigid_stress/mechanics.py").is_file():
            return parent
    # Standalone scratch experiment layout used before copying into the branch.
    candidate = Path(__file__).resolve().parents[1] / "genesis-shell-study"
    if (candidate / "research/rigid_stress/mechanics.py").is_file():
        return candidate
    raise FileNotFoundError("Run inside a Genesis checkout containing the existing CPU P2 reference")


REPOSITORY = _find_repository()
sys.path.insert(0, str(REPOSITORY))
from research.rigid_stress.mechanics import P2Shell, shape_gradients


def extrude(vertices, quads, h, layers):
    normal = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    z = np.linspace(h / 2, -h / 2, layers + 1)
    xyz = np.concatenate([vertices + depth * normal for depth in z])
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    n = len(vertices)
    tets = []
    for layer in range(layers):
        for triangle in np.sort(triangles, axis=1):
            a, b, c = triangle + layer * n
            d, e, f = triangle + (layer + 1) * n
            tets.extend([[a, b, c, f], [a, b, e, f], [a, d, e, f]])
    return xyz, np.asarray(tets), triangles


def solid_pressure(fem, triangles, pressure, layer_offset):
    """Exact consistent P2 nodal loads for constant pressure on flat faces."""
    faces = triangles + layer_offset
    x = fem.base_xyz[faces]
    area_normal = .5 * np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    for i in range(len(faces)):
        if area_normal[i] @ x[i].mean(axis=0) < 0:
            area_normal[i] *= -1
    force = np.zeros((len(fem.xyz), 3))
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    keys = edges[:, :, 0] * fem.nbase + edges[:, :, 1]
    mids = fem.nbase + np.searchsorted(fem.edge_keys, keys)
    for i in range(3):
        np.add.at(force, mids[:, i], pressure * area_normal / 3)
    return force.ravel()


def shell_inner_pressure(model, pressure, order=4):
    """Independent virtual-work pressure integral on actual offset inner face."""
    force = np.zeros((len(model.vertices), 6))
    g, w = np.polynomial.legendre.leggauss(order)
    z = -model.h / 2
    for ids in model.quads:
        x, d = model.vertices[ids], model.directors[ids]
        offset_x = x + z * d
        for i, xi in enumerate(g):
            for j, eta in enumerate(g):
                n, dn = quad_shape(xi, eta)
                tangent = offset_x.T @ dn
                area_vector = np.cross(tangent[:, 0], tangent[:, 1])
                traction = pressure * area_vector * w[i] * w[j]
                np.add.at(force[:, :3], ids, n[:, None] * traction)
                np.add.at(force[:, 3:], ids, n[:, None] * z * np.cross(d, traction))
    return force.ravel()


def lame(radius, inner, outer, pressure, E, nu):
    a = pressure * inner**3 / (outer**3 - inner**3)
    b = pressure * inner**3 * outer**3 / (outer**3 - inner**3)
    ur = ((1 - 2 * nu) * a * radius + (1 + nu) * b / (2 * radius**2)) / E
    hoop = a + b / (2 * radius**3)
    radial = a - b / radius**3
    return ur, hoop, radial


def solid_layer_stress(fem, displacement):
    bary = np.full((1, 4), .25)
    pieces = []
    for lo, hi, strain in fem.strain(displacement, bary):
        stress = np.einsum("ij,eqj->eqi", fem.d, strain)[:, 0]
        points = fem.base_xyz[fem.base_tets[lo:hi]].mean(axis=1)
        norm = points / np.linalg.norm(points, axis=1)[:, None]
        tensor = np.zeros((len(stress), 3, 3))
        tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = stress[:, :3].T
        tensor[:, 0, 1] = tensor[:, 1, 0] = stress[:, 3]
        tensor[:, 1, 2] = tensor[:, 2, 1] = stress[:, 4]
        tensor[:, 0, 2] = tensor[:, 2, 0] = stress[:, 5]
        radial = np.einsum("ei,eij,ej->e", norm, tensor, norm)
        hoop = .5 * (np.trace(tensor, axis1=1, axis2=2) - radial)
        pieces.append((points, hoop, radial))
    return (np.concatenate([x[k] for x in pieces]) for k in range(3))


def solid_disk_load(fem, faces, point, normal, force, radius, radial_order=24, angular_order=64):
    """Independent pressure-disk quadrature projected along F onto P2 triangles.

    This uses the same declared directional disk load law as the shell, not
    its Q4 quadrature or interpolation implementation.  Surface geometry is
    separately triangulated; differences include this geometry approximation.
    """
    direction = force / np.linalg.norm(force)
    normal = normal / np.linalg.norm(normal)
    first = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    g, gw = np.polynomial.legendre.leggauss(radial_order)
    u, uw = .5*(g+1), .5*gw
    g, gw = np.polynomial.legendre.leggauss(angular_order)
    angle, aw = np.pi*(g+1), .5*gw
    radial = radius*np.sqrt(1-u*u)
    points = (point + radial[:,None,None]*(np.cos(angle)[None,:,None]*first +
                                          np.sin(angle)[None,:,None]*second)).reshape(-1,3)
    weights = (3*u[:,None]**2*uw[:,None]*aw[None,:]).ravel()
    xyz = fem.base_xyz[faces]
    edges = xyz[:,1:]-xyz[:,:1]
    norm = np.cross(edges[:,0],edges[:,1])
    norm /= np.linalg.norm(norm,axis=1)[:,None]
    den = norm @ direction
    # Only entry surfaces, so the opposite side cannot steal the loaded sheet.
    candidates = np.flatnonzero(den < -1e-10)
    xyz,edges,norm,den=xyz[candidates],edges[candidates],norm[candidates],den[candidates]
    gram = np.einsum('eai,ebi->eab',edges,edges)
    dual = np.linalg.solve(gram,edges)
    all_nodes=[]; all_forces=[]
    for lo in range(0,len(points),96):
        p=points[lo:lo+96]
        tau=np.einsum('eij,ej->ei',xyz[:,None,0,:]-p[None],norm)/den[:,None]
        hit=p[None]+tau[:,:,None]*direction
        uv=np.einsum('eai,eji->eja',dual,hit-xyz[:,None,0,:])
        ok=(uv[:,:,0]>=-1e-10)&(uv[:,:,1]>=-1e-10)&(uv.sum(axis=2)<=1+1e-10)
        cost=np.where(ok,np.abs(tau),np.inf)
        pick=np.argmin(cost,axis=0)
        if not np.isfinite(cost[pick,np.arange(len(p))]).all():
            raise RuntimeError('Pressure ray did not hit its material sheet')
        tri=faces[candidates[pick]]
        uv=uv[pick,np.arange(len(p))]
        bary=np.column_stack((1-uv.sum(axis=1),uv))
        shape=np.column_stack((bary*(2*bary-1),4*bary[:,0]*bary[:,1],
                               4*bary[:,1]*bary[:,2],4*bary[:,2]*bary[:,0]))
        ed=np.sort(tri[:,[[0,1],[1,2],[2,0]]],axis=2)
        key=ed[:,:,0]*fem.nbase+ed[:,:,1]
        nodes=np.column_stack((tri,fem.nbase+np.searchsorted(fem.edge_keys,key)))
        all_nodes.append(nodes)
        all_forces.append(shape[:,:,None]*weights[lo:lo+len(p),None,None]*force)
    out=np.zeros((len(fem.xyz),3))
    nodes=np.concatenate(all_nodes); values=np.concatenate(all_forces)
    np.add.at(out,nodes.ravel(),values.reshape(-1,3))
    fsum=out.sum(axis=0)
    moment=np.cross(fem.xyz-point,out).sum(axis=0)
    return out.ravel(),dict(force_relative=float(np.linalg.norm(fsum-force)/np.linalg.norm(force)),
                           moment_normalized=float(np.linalg.norm(moment)/(np.linalg.norm(force)*radius)))


def surface_sampling_gradients(solid, quads, layers, nvertices):
    """Solid owner elements and P2 gradients at corresponding Q4 face samples."""
    owners={}
    for e,tet in enumerate(solid.base_tets):
        for omit in range(4):
            face=tuple(sorted(np.delete(tet,omit)))
            owners[face]=(e,tet)
    elem=[]; barys=[]
    gp=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])/np.sqrt(3)
    for offset in (0,layers*nvertices):
        for quad in quads:
            for xi,eta in gp:
                u,v=.5*(xi+1),.5*(eta+1)
                if v<=u:
                    tri=quad[[0,1,2]]+offset; bb=[1-u,u-v,v]
                else:
                    tri=quad[[0,2,3]]+offset; bb=[1-v,u,v-u]
                e,tet=owners[tuple(sorted(tri))]
                bary=np.zeros(4)
                for vertex,value in zip(tri,bb):
                    bary[np.flatnonzero(tet==vertex)[0]]=value
                elem.append(e);barys.append(bary)
    elem=np.asarray(elem);bary=np.asarray(barys)
    gl=solid.glambda[elem]
    grad=(4*bary[:,:,None]-1)*gl
    edge_list=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edgegrad=4*(bary[:,edge_list[:,0],None]*gl[:,edge_list[:,1]]+
                bary[:,edge_list[:,1],None]*gl[:,edge_list[:,0]])
    grad=np.concatenate((grad,edgegrad),axis=1)
    return elem, grad


def surface_stress_at_shell_points(solid, displacement, quads, layers, nvertices):
    """Evaluate 3D solid stress on corresponding triangle positions at Q4 GPs."""
    elem, grad = surface_sampling_gradients(solid, quads, layers, nvertices)
    ue=displacement.reshape(-1,3)[solid.elements[elem]]
    deriv=np.einsum('eia,eib->eab',ue,grad)
    mu=solid.young/(2*(1+solid.poisson))
    lam=solid.young*solid.poisson/((1+solid.poisson)*(1-2*solid.poisson))
    sigma=mu*(deriv+deriv.transpose(0,2,1))+lam*np.trace(deriv,axis1=1,axis2=2)[:,None,None]*np.eye(3)
    return sigma.reshape(2,len(quads),4,3,3)


def run_disk_case(n=4,layers=1,Fn=.05,manual_radius=None):
    R,h,E,nu=.02,.0005,1e10,.3
    v,q=sphere_mesh(n,R)
    shell=ShellModel(v,q,E=E,nu=nu,h=h,directors=v/R)
    xyz,tets,tri=extrude(v,q,h,layers)
    solid=P2Shell(xyz,tets,tri,E,nu,1000.,layers)
    axis=np.array([.93,.31,.19]);axis/=np.linalg.norm(axis)
    radius=hertz_radius(Fn,E,nu,(1/(R+h/2),)*2) if manual_radius is None else manual_radius
    contacts=[Contact(sign*(R+h/2)*axis,-sign*Fn*axis,sign*axis,
                      (1/(R+h/2),)*2,radius=radius) for sign in (1,-1)]
    transfer=ShellDPRTransfer(shell)
    fs=np.zeros(shell.ndof);fp=np.zeros(solid.ndof);checks=[]
    for contact in contacts:
        ff,info=transfer.transfer(contact,order=24)
        fs+=ff
        fpp,check=solid_disk_load(solid,tri,contact.point,contact.normal,contact.force,radius)
        fp+=fpp;checks.append(check)
    us=shell.solve(fs);up,_,rp=solid.solve(fp)
    us=us-shell.rigid_modes@np.linalg.solve(shell.mass_gram,shell.mass_modes.T@us)
    up=solid.canonical(up)
    if layers%2==0:
        mids=layers//2*len(v)+np.arange(len(v))
    else:
        i=np.arange(len(v));a=(layers//2)*len(v)+i;b=(layers//2+1)*len(v)+i
        keys=np.minimum(a,b)*solid.nbase+np.maximum(a,b)
        mids=solid.nbase+np.searchsorted(solid.edge_keys,keys)
    pdisp=up.reshape(-1,3)[mids]
    sdisp=us.reshape(-1,6)[:,:3]
    pstress=surface_stress_at_shell_points(solid,up,q,layers,len(v))
    local=np.einsum('egai,segab,egbj->segij',shell.axes,pstress,shell.axes)
    pp=local[:,:,:,[0,1,0],[0,1,1]]
    ss=shell.stress(us)
    sp=np.stack((ss['local_top'],ss['local_bottom']))
    # This diagnostic uses equal weights for the three stored components
    # (xx, yy, xy), explicitly a Voigt-vector L2, not tensor Frobenius L2.
    weights=shell.gauss_weights[None,:,:,None]
    diff=np.sqrt(np.sum(weights*(sp-pp)**2)/np.sum(weights*pp**2))
    points=np.array([[quad_shape(xi,eta)[0]@v[face] for xi,eta in shell.gauss_points] for face in q])
    distance=np.minimum(np.linalg.norm(points-(R+h/2)*axis,axis=2),
                        np.linalg.norm(points+(R+h/2)*axis,axis=2))
    far=distance>max(2*np.sqrt(R*h),2*radius)
    fw=weights*far[None,:,:,None]
    far_difference=np.sqrt(np.sum(fw*(sp-pp)**2)/np.sum(fw*pp**2))
    vm=lambda x:np.sqrt(x[...,0]**2-x[...,0]*x[...,1]+x[...,1]**2+3*x[...,2]**2)
    # For the linear through-thickness fields, derive N and M from two faces.
    # These match the shell resultants' definitions, not 3D normal/shear stresses.
    pn=.5*h*(pp[0]+pp[1])
    pm=h*h/12*(pp[0]-pp[1])
    sn,sm=ss['N'],ss['M']
    nerr=np.sqrt(np.sum(weights[0]*(sn-pn)**2)/np.sum(weights[0]*pn**2))
    merr=np.sqrt(np.sum(weights[0]*(sm-pm)**2)/np.sum(weights[0]*pm**2))
    return dict(n=n,layers=layers,Fn=Fn,radius_m=radius,radius_over_h=radius/h,
                shell_dofs=shell.ndof,solid_dofs=solid.ndof,
                midpoint_displacement_relative_L2=float(np.linalg.norm(sdisp-pdisp)/np.linalg.norm(pdisp)),
                common_surface_inplane_voigt_relative_L2=float(diff),
                farfield_exclusion_distance_m=float(max(2*np.sqrt(R*h),2*radius)),
                farfield_inplane_voigt_relative_L2=float(far_difference),
                surface_linearized_membrane_relative_L2=float(nerr),
                surface_linearized_bending_relative_L2=float(merr),
                shell_to_solid_inplane_stress_norm_ratio=float(np.sqrt(np.sum(weights*sp**2)/np.sum(weights*pp**2))),
                shell_inplane_VM_peak_Pa=float(vm(sp).max()),solid_inplane_VM_peak_Pa=float(vm(pp).max()),
                shell_residual=float(shell.residual(us,fs)),solid_residual=float(rp),
                solid_transfer_checks=checks,
                scope='Projected finite Hertz-disk load on differently approximated Q4/tri surfaces. Same coarse vertex mesh, NOT a converged contact or continuum reference; shell in-plane stress only.')


def run_case(n=4, layers=1, R=.02, h=.0005, E=1e10, nu=.3, p=100.):
    vertices, quads = sphere_mesh(n, R)
    dirs = vertices / R
    t0 = time.perf_counter()
    shell = ShellModel(vertices, quads, E=E, nu=nu, h=h, directors=dirs)
    fs = shell_inner_pressure(shell, p)
    t1 = time.perf_counter()
    qs = shell.solve(fs)
    t2 = time.perf_counter()
    sigma = shell.stress(qs)
    mean_shell_u = np.average(np.sum(qs.reshape(-1, 6)[:, :3] * dirs, axis=1), weights=shell.nodal_mass)
    hoop_shell = .5 * (sigma["local_top"][:, :, 0] + sigma["local_top"][:, :, 1] +
                       sigma["local_bottom"][:, :, 0] + sigma["local_bottom"][:, :, 1]) / 2
    mean_shell_hoop = np.average(hoop_shell, weights=shell.gauss_weights)
    xyz, tets, triangles = extrude(vertices, quads, h, layers)
    t3 = time.perf_counter()
    solid = P2Shell(xyz, tets, triangles, E, nu, 1000., layers)
    fp = solid_pressure(solid, triangles, p, layers * len(vertices))
    t4 = time.perf_counter()
    qp, rhs, res = solid.solve(fp)
    t5 = time.perf_counter()
    # All base-layer displacements are compared against Lame at their actual radii.
    base_u = qp.reshape(-1, 3)[:solid.nbase]
    base_r = np.linalg.norm(xyz, axis=1)
    radial_u = np.sum(base_u * xyz / base_r[:, None], axis=1)
    exact_u, _, _ = lame(base_r, R-h/2, R+h/2, p, E, nu)
    exact_mid_u, exact_mid_hoop, exact_mid_radial = lame(R, R-h/2, R+h/2, p, E, nu)
    pos, hoop_p, radial_p = solid_layer_stress(solid, qp)
    _, exact_hoop_p, exact_radial_p = lame(np.linalg.norm(pos, axis=1), R-h/2, R+h/2, p, E, nu)
    volume = solid.volume
    solid_hoop_error = np.sqrt(np.sum(volume * (hoop_p-exact_hoop_p)**2) / np.sum(volume*exact_hoop_p**2))
    solid_radial_error = np.sqrt(np.sum(volume * (radial_p-exact_radial_p)**2) / np.sum(volume*exact_hoop_p**2))
    shell_wrench = shell.rigid_modes.T @ fs
    return dict(n=n, layers=layers, shell_vertices=len(vertices), shell_quads=len(quads),
                shell_dofs=shell.ndof, solid_tets=len(tets), solid_dofs=solid.ndof,
                parameters=dict(R=R,h=h,E=E,nu=nu,inner_pressure=p),
                shell_mid_radial_displacement_m=float(mean_shell_u), exact_mid_radial_displacement_m=float(exact_mid_u),
                shell_displacement_relative_error=float(abs(mean_shell_u/exact_mid_u-1)),
                solid_base_radial_displacement_relative_L2=float(np.linalg.norm(radial_u-exact_u)/np.linalg.norm(exact_u)),
                shell_mean_hoop_stress_Pa=float(mean_shell_hoop), exact_mid_hoop_stress_Pa=float(exact_mid_hoop),
                shell_mean_hoop_relative_error=float(abs(mean_shell_hoop/exact_mid_hoop-1)),
                solid_volume_hoop_relative_L2=float(solid_hoop_error),
                solid_radial_absolute_error_normalized_by_hoop=float(solid_radial_error),
                exact_mid_radial_stress_Pa=float(exact_mid_radial),
                shell_load_wrench=shell_wrench.tolist(),
                shell_residual=float(shell.residual(qs,fs)),solid_residual=float(res),
                seconds=dict(shell_assembly_load=t1-t0,shell_factor_solve=t2-t1,
                             solid_assembly_factor_load=t4-t3,solid_solve=t5-t4),
                scope="Prescribed inner pressure, same normal-offset reference geometry. 3D P2 solid and MITC4 shell; not a unilateral-contact or Code_Aster reference.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",nargs="+",type=int,default=[3,6,8])
    parser.add_argument("--layers",nargs="+",type=int,default=[1,2])
    parser.add_argument("--output",type=Path,default=Path("shell_cpu/solid_results/spheres.json"))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in args.n:
        for layers in args.layers:
            row=run_case(n,layers)
            rows.append(row)
            args.output.write_text(json.dumps(dict(rows=rows),indent=2)+"\n")
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
