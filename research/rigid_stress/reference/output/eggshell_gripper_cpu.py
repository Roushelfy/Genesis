#!/usr/bin/env python3
"""CPU stress recovery for an undeformed hollow egg and a force-controlled gripper.

Run beside rigid_stress_reference.py and rigid_stress_temporal_cpu_test.py:
  python eggshell_gripper_cpu.py --plot

Dependencies: NumPy, SciPy, threadpoolctl; --plot requires matplotlib and Pillow.
This is a prescribed finite-pad traction experiment, not a Genesis/contact
solver integration, a fracture test, or a calibrated model of a biological egg.
The material is an assumed homogeneous isotropic linear elastic solid. P1
volume tetrahedra are used through the shell thickness; bending accuracy and
peak stress must be checked separately before predicting physical failure.
"""
from pathlib import Path
from time import perf_counter
import argparse
import csv
import gc
import itertools
import json
import platform
import sys
import zipfile

import numpy as np
from scipy import linalg
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import rigid_stress_reference as ref
import rigid_stress_temporal_cpu_test as temporal


def sphere(level):
    a=(1+np.sqrt(5))/2
    vertices=np.array([[-1,a,0],[1,a,0],[-1,-a,0],[1,-a,0],
                       [0,-1,a],[0,1,a],[0,-1,-a],[0,1,-a],
                       [a,0,-1],[a,0,1],[-a,0,-1],[-a,0,1]],dtype=float)
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    faces=np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                    [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                    [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                    [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]],dtype=int)
    for _ in range(level):
        points=vertices.tolist();cache={};new=[]
        def midpoint(i,j):
            key=tuple(sorted((int(i),int(j))))
            if key not in cache:
                p=vertices[i]+vertices[j];p/=np.linalg.norm(p)
                cache[key]=len(points);points.append(p.tolist())
            return cache[key]
        for i,j,k in faces:
            ij,jk,ki=midpoint(i,j),midpoint(j,k),midpoint(k,i)
            new.extend([[i,ij,ki],[j,jk,ij],[k,ki,jk],[ij,jk,ki]])
        vertices=np.asarray(points);faces=np.asarray(new)
    return vertices,faces


def egg_mesh(level=4,layers=4,thickness=.0005):
    unit,tri=sphere(level)
    a,b,c=.022,.022,.030;taper=.18
    scale=1-taper*unit[:,2]
    outer=unit*np.column_stack([a*scale,b*scale,np.full(len(unit),c)])
    grad=np.column_stack([2*unit[:,0]/(a*scale),2*unit[:,1]/(b*scale),
                          2/c*(unit[:,2]+taper*(unit[:,0]**2+unit[:,1]**2)/scale)])
    normals=grad/np.linalg.norm(grad,axis=1)[:,None]
    xyz=np.concatenate([outer-depth*normals for depth in np.linspace(0,thickness,layers+1)])
    n=len(unit);tets=[]
    for layer in range(layers):
        for i,j,k in np.sort(tri,axis=1):
            A,B,C=np.array([i,j,k])+layer*n
            D,E,F=np.array([i,j,k])+(layer+1)*n
            tets.extend([[A,B,C,F],[A,B,E,F],[A,D,E,F]])
    tets=np.asarray(tets,dtype=int)
    # The spherical topology has two closed skins; radial layers fill only
    # the wall. Consistent global vertex ordering makes prism splits conform.
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    assert np.all(counts==2)
    tetxyz=xyz[tets]
    vols=np.abs(np.linalg.det(tetxyz[:,1:]-tetxyz[:,:1]))/6
    assert np.all(vols>0)
    metadata={'surface_refinement':level,'through_thickness_layers':layers,
              'thickness_m':thickness,'taper':taper,'outer_dimensions_m':np.ptp(outer,axis=0).tolist(),
              'outer_surface_vertices':n,'outer_surface_triangles':len(tri),
              'minimum_tet_volume_m3':float(vols.min())}
    return xyz,tets,tri,metadata


def pad_basis(fem,outer_faces,radius=.005,sigma=.002):
    points=fem.xyz[outer_faces];centers=points.mean(axis=1)
    areas=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/2
    # Three discrete gripper heights. Face quadrature gives a smooth pressure
    # taper to zero at the footprint edge; each vector column has net force 1 N.
    heights=[-.010,0.,.012]
    basis=np.zeros((fem.ndof,18));meta=[]
    for stage,z in enumerate(heights):
        for side,sign in enumerate([-1,1]):
            p=2*stage+side
            distance=np.linalg.norm(centers[:,[1,2]]-np.array([0,z]),axis=1)
            selected=np.flatnonzero((sign*centers[:,0]>0)&(distance<radius))
            assert len(selected)>2
            d=distance[selected]
            pressure=np.exp(-.5*(d/sigma)**2)*np.maximum(0.,1-(d/radius)**2)**2
            weight=areas[selected]*pressure
            nodal=np.zeros(len(fem.xyz))
            for a in range(3):np.add.at(nodal,outer_faces[selected,a],weight/3)
            nodal/=nodal.sum()
            for axis in range(3):basis[axis::3,3*p+axis]=nodal
            center=nodal@fem.xyz
            meta.append({'patch':p,'stage':stage,'side':sign,'height_m':z,
                         'pad_radius_m':radius,'pressure_sigma_m':sigma,
                         'triangles':len(selected),'geometric_area_m2':float(areas[selected].sum()),
                         'force_centroid_m':center.tolist(),'net_force_per_basis_N':1.})
    return basis,meta


class EggResponse:
    def __init__(self,level=4,layers=4,group_size=64):
        t=perf_counter()
        xyz,tets,outer,self.mesh_metadata=egg_mesh(level,layers)
        self.outer_faces=outer
        self.fem=ref.CachedElasticRecovery(xyz,tets,young=1e10,poisson=.3,density=2000.)
        f=self.fem
        self.offline_model_seconds=perf_counter()-t;t=perf_counter()
        self.contact_basis,self.patches=pad_basis(f,outer)
        self.weight_N=f.mass*9.81
        h=np.column_stack([self.contact_basis,f.consistent_body_force([0,0,-9.81])])
        self.rhs_basis=h-f.mr@linalg.cho_solve(f.g_factor,f.r.T@h)
        u=np.zeros_like(h);u[f.free]=f.factor.solve(np.asfortranarray(self.rhs_basis[f.free]))
        stress=np.einsum('eij,ejr->eir',f.stress_operator,u[f.element_dofs],optimize=False)
        A=np.empty_like(stress)
        A[:,0]=(stress[:,0]-stress[:,1])/np.sqrt(2)
        A[:,1]=(stress[:,1]-stress[:,2])/np.sqrt(2)
        A[:,2]=(stress[:,2]-stress[:,0])/np.sqrt(2)
        A[:,3:]=np.sqrt(3)*stress[:,3:]
        self.order=temporal.morton_order(xyz[tets].mean(axis=1))
        self.A=np.ascontiguousarray(A[self.order]);self.flat=self.A.reshape(-1,19)
        self.ne=len(tets);self.group_size=group_size;self.ng=(self.ne+group_size-1)//group_size
        self.starts=np.arange(self.ng)*group_size;self.counts=np.minimum(group_size,self.ne-self.starts)
        spectral=np.empty(self.ne)
        for i in range(0,self.ne,2048):spectral[i:i+2048]=np.linalg.svd(self.A[i:i+2048],compute_uv=False)[:,0]
        self.L=np.maximum.reduceat(spectral,self.starts)*(1+1e-12)
        self.Lcol=np.maximum.reduceat(np.linalg.norm(self.A,axis=1),self.starts,axis=0)*(1+1e-12)
        self.offline_response_seconds=perf_counter()-t
        self._bfree=np.ascontiguousarray(self.rhs_basis[f.free]);self._u=np.zeros(f.ndof)
        # Topology / rigid equilibrium validation, independent of temporal max.
        ncomp,_=connected_components(f.k,directed=False)
        assert ncomp==1
        assert len(f.surface_faces)==2*len(outer)
        basis_wrench=np.linalg.norm(f.r.T@self.rhs_basis)
        assert basis_wrench/max(np.linalg.norm(h),1.)<1e-10
        self.metadata={**self.mesh_metadata,'nodes':len(xyz),'tetrahedra':self.ne,'scalar_DOFs':f.ndof,
                       'young_Pa':f.young,'poisson':f.poisson,'density_kg_m3':f.density,
                       'mass_kg':f.mass,'weight_N':self.weight_N,'groups':self.ng,'load_basis_dimension':19,
                       'response_bytes_FP64':self.A.nbytes,'factorization_count':1,
                       'closed_boundary_components':2,'connected_body':True,'patches':self.patches,
                       'balanced_basis_wrench_norm':float(basis_wrench),
                       'offline_assembly_and_factor_s':self.offline_model_seconds,
                       'offline_response_and_bounds_s':self.offline_response_seconds}

    def values(self,q,indices=None):
        a=self.flat if indices is None else self.A[indices].reshape(-1,19)
        y=(a@q).reshape(-1,6)
        return np.einsum('ij,ij->i',y,y)

    def response_scan(self,q):
        v=self.values(q);i=int(v.argmax())
        return float(np.sqrt(v[i])),i

    def cached_lu(self,q):
        f=self.fem;self._u[f.free]=f.factor.solve(self._bfree@q)
        stress=np.einsum('eij,ej->ei',f.stress_operator,self._u[f.element_dofs],optimize=False)
        v=ref.von_mises(stress);i=int(v.argmax())
        return float(v[i]),i

    def physical_check(self,q):
        f=self.fem;raw=self.contact_basis@q[:18]+q[18]*f.consistent_body_force([0,0,-9.81])
        full=f.solve(raw,np.zeros(3));m=float(ref.von_mises(full.stress).max())
        canonical=f.remove_rigid_displacement(full.displacement).reshape(-1,3)
        return {'max_error':abs(self.response_scan(q)[0]-m)/max(m,1.),
                'full_relative_residual':float(full.residual_relative),
                'canonical_displacement_max_m':float(np.linalg.norm(canonical,axis=1).max()),
                'raw_net_wrench':(f.r.T@raw).tolist()}


class EggTemporal(temporal.TemporalMax):
    """Same bound algorithm; reference q dimensions follow the new basis."""
    def _full(self,q):
        m=self.model;v=m.values(q)
        padded=np.zeros(m.ng*m.group_size);padded[:m.ne]=v
        blocks=padded.reshape(m.ng,m.group_size);arg=blocks.argmax(axis=1)
        self.hot=m.starts+arg;self.group_refmax=np.sqrt(blocks[np.arange(m.ng),arg])
        self.qref=np.broadcast_to(q,(m.ng,len(q))).copy()
        i=int(v.argmax())
        return temporal.TrackResult(float(np.sqrt(v[i])),i,m.ne,m.ng,True,False)


def sequence(model,frames):
    t=np.linspace(0,1,frames);cases={}
    def clamp(F,stage,gravity=False):
        q=np.zeros(19);left,right=2*stage,2*stage+1
        q[3*left]=F;q[3*right]=-F
        if gravity:q[3*left+2]=q[3*right+2]=model.weight_N/2;q[18]=1
        return q
    # Exactly proportional loading is tested separately from the perturbed
    # lift and regrasp cases; this is where scalar maximum reuse is valid.
    ramp=np.interp(t,[0,.35,.65,1],[0,8,8,0])
    cases['fixed_pad_ramp']=np.array([clamp(F,1) for F in ramp])
    F=4+.5*np.sin(2*np.pi*t)
    q=np.array([clamp(v,1,True) for v in F])
    q[:,7]=.18*np.sin(4*np.pi*t);q[:,10]=-q[:,7]
    q[:,8]+=.10*np.sin(2*np.pi*t);q[:,11]-=.10*np.sin(2*np.pi*t)
    cases['grip_jitter']=q
    q=[]
    for i in range(frames):
        stage=min(2,3*i//frames)
        a=clamp(4+.4*np.sin(.06*i),stage,True)
        a[6*stage+1]=.08*np.sin(.04*i);a[6*stage+4]=-a[6*stage+1]
        q.append(a)
    cases['regrasp']=np.array(q)
    return cases


def measure(model,name,qs,repeats,rows):
    oracle=np.array([model.cached_lu(q)[0] for q in qs])
    scans=[model.response_scan(q) for q in qs]
    scanerr=max(abs(s[0]-v)/max(v,1.) for s,v in zip(scans,oracle))
    assert scanerr<1e-8,(name,scanerr)
    physical=[model.physical_check(qs[i]) for i in np.unique(np.linspace(0,len(qs)-1,5,dtype=int))]
    assert max(p['max_error'] for p in physical)<1e-8
    times={k:[] for k in ['cached_lu','response_scan','temporal']};saved=[]
    for q in qs[:3]:model.cached_lu(q);model.response_scan(q)
    for repeat in range(repeats):
        for method in times:
            tracker=EggTemporal(model);prev=None
            for i,q in enumerate(qs):
                start=perf_counter()
                if method=='temporal':out=tracker.query(q);v=out.maximum
                else:v=getattr(model,method)(q)[0]
                elapsed=perf_counter()-start;times[method].append(elapsed)
                assert abs(v-oracle[i])/max(oracle[i],1.)<1e-8
                if method=='temporal' and repeat==0:
                    old=oracle[i] if prev is None else float(np.sqrt(model.values(q,np.array([prev]))[0]))
                    row={'scenario':name,'frame':i,'grip_force_per_finger_N':float(np.max(np.abs(q[0:18:3]))),
                         'oracle_max_Pa':oracle[i],'temporal_max_Pa':v,
                         'relative_error':abs(v-oracle[i])/max(oracle[i],1.),
                         'temporal_first_repeat_ms':elapsed*1000,'evaluation_fraction':out.evaluated/model.ne,
                         'full_scan':int(out.full_scan),'identical_load_reuse':int(out.reused_identical),
                         'old_hotspot_max_Pa':old,'old_hotspot_underestimate':max(oracle[i]-old,0)/max(oracle[i],1.)}
                    rows.append(row);saved.append(out);prev=scans[i][1]
    out={'scenario':name,'frames':len(qs),'timings':{k:temporal.stats(v) for k,v in times.items()},
         'max_error':max(abs(s.maximum-v)/max(v,1.) for s,v in zip(saved,oracle)),
         'mean_evaluation_fraction':float(np.mean([s.evaluated/model.ne for s in saved])),
         'full_scan_frames':sum(s.full_scan for s in saved),'identical_load_reuse_frames':sum(s.reused_identical for s in saved),
         'peak_von_Mises_Pa':float(oracle.max()),'physical_checks':physical,
         'old_hotspot_worst_underestimate':max(r['old_hotspot_underestimate'] for r in rows if r['scenario']==name)}
    out['speedup_vs_response_scan']=out['timings']['response_scan']['p50_ms']/out['timings']['temporal']['p50_ms']
    if name=='fixed_pad_ramp':
        unit=np.zeros(19);unit[6]=1;unit[9]=-1;unit_max=model.response_scan(unit)[0]
        scalar_times=[];scalarerr=0.
        for _ in range(repeats):
            for q,v in zip(qs,oracle):
                start=perf_counter();value=abs(q[6])*unit_max;scalar_times.append(perf_counter()-start)
                scalarerr=max(scalarerr,abs(value-v)/max(v,1.))
        assert scalarerr<1e-8
        out['proportional_scalar_reuse']={'max_error':scalarerr,'timing':temporal.stats(scalar_times),
                                          'unit_force_max_Pa':unit_max}
    return out


def surface_parents(tets):
    faces=np.concatenate([tets[:,ids] for ids in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])])
    owner=np.tile(np.arange(len(tets)),4);keys=np.sort(faces,axis=1)
    _,ix,count=np.unique(keys,axis=0,return_index=True,return_counts=True)
    return faces[ix[count==1]],owner[ix[count==1]]


def plots(model,cases,summary,rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':'white',
                         'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    names={'fixed_pad_ramp':'Fixed pads: force ramp','grip_jitter':'Grip / shear variation','regrasp':'Regrasp at three heights'}
    fig,axs=plt.subplots(2,3,figsize=(13,6.5),layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
    for j,(name,label) in enumerate(names.items()):
        r=[r for r in rows if r['scenario']==name];frame=np.array([r['frame'] for r in r])
        axs[0,j].plot(frame,[r['oracle_max_Pa']/1e6 for r in r],color='#2474a6',lw=2,label='Full FEM')
        axs[0,j].plot(frame,[r['old_hotspot_max_Pa']/1e6 for r in r],color='#c54b43',ls='--',lw=1.1,label='Only old maximum element')
        mark=np.unique(np.r_[np.arange(0,len(r),8),(len(r)+2)//3,(2*len(r)+2)//3,len(r)-1])
        axs[0,j].plot(frame[mark],[r[i]['temporal_max_Pa']/1e6 for i in mark],'o',mfc='none',ms=4,color='#14806e',label='Temporal maximum')
        axs[0,j].set_title(label);axs[0,j].set_ylabel('Maximum von Mises (MPa)');axs[0,j].grid(alpha=.2)
        axs[0,j].set_xlabel('Frame')
        fraction=np.array([r['evaluation_fraction']*100 for r in r])
        axs[1,j].fill_between(frame,fraction,color='#2474a6',alpha=.15)
        axs[1,j].plot(frame,fraction,color='#2474a6',lw=1)
        axs[1,j].set_ylim(0,110);axs[1,j].set_ylabel('Response work (%)');axs[1,j].set_xlabel('Frame')
        axs[1,j].grid(alpha=.2)
        c=next(c for c in summary['cases'] if c['scenario']==name)
        axs[1,j].set_title(f"Median: full response {c['timings']['response_scan']['p50_ms']:.3f} ms / temporal {c['timings']['temporal']['p50_ms']:.3f} ms",fontsize=9)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f"Hollow egg / force-controlled gripper: {model.metadata['nodes']:,} nodes, {model.ne:,} tetrahedra\nCPU, one thread, FP64 | prescribed finite-area contact pads",fontsize=13)
    fig.savefig(output/'eggshell_gripper_cpu_results.png',dpi=170);plt.close(fig)

    f=model.fem;faces,owners=surface_parents(f.tetrahedra)
    n=model.mesh_metadata['outer_surface_vertices']
    # An actual cutaway: outer / inner skins and an interior section, never a
    # filled solid egg. Exterior rendering selects outer skin by vertex layer.
    outside=np.all(faces<n,axis=1);outer_faces=faces[outside];outer_owners=owners[outside]
    cent=f.xyz[faces].mean(axis=1);keep=cent[:,1]>=-.0001
    cutfaces=faces[keep];cutowners=owners[keep]
    from rigid_stress_visualize import slice_polygons
    section,section_owners=slice_polygons(f.xyz,f.tetrahedra,y=.00017)
    section3=[np.column_stack([p[:,0],np.full(len(p),.00017),p[:,1]]) for p in section]
    qs=cases['regrasp'];fields=[];outs=[];active=[];tracker=EggTemporal(model)
    inv=np.empty(model.ne,dtype=int);inv[model.order]=np.arange(model.ne)
    centers=f.xyz[f.tetrahedra].mean(axis=1)
    for q in qs:
        before=None if tracker.qref is None else tracker.qref.copy();out=tracker.query(q)
        changed=np.ones(model.ng,dtype=bool) if before is None else np.any(tracker.qref!=before,axis=1)
        v=np.sqrt(model.values(q));original=np.empty_like(v);original[model.order]=v
        fields.append(original/1e6);outs.append(out);active.append(changed)
    fields=np.asarray(fields);norm=Normalize(0,float(fields.max()));cmap=plt.get_cmap('inferno')
    def box(a,b):
        p=np.array(list(itertools.product(*zip(a,b))))
        return [p[ix] for ix in ([0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5])]
    def jaw_faces(stage):
        z=model.patches[2*stage]['height_m'];centers_patch=[np.array(model.patches[2*stage+s]['force_centroid_m']) for s in range(2)]
        boxes=[]
        for side,center in zip([-1,1],centers_patch):
            contact_x=center[0];near=contact_x+side*.0005;far=near+side*.008
            lo=[min(near,far),-.007,z-.006];hi=[max(near,far),.007,z+.006]
            boxes.extend(box(lo,hi))
        return boxes
    def setup(ax):
        ax.set_xlim(-.035,.035);ax.set_ylim(-.028,.028);ax.set_zlim(-.034,.034)
        ax.set_box_aspect((1.05,.85,1.15),zoom=.94);ax.view_init(elev=15,azim=-66);ax.set_axis_off()
    first=(len(qs)+2)//3;second=(2*len(qs)+2)//3
    selected=[first-1,first,second]
    fig=plt.figure(figsize=(12,5.7),layout='constrained');axes=[]
    for j,i in enumerate(selected):
        ax=fig.add_subplot(1,3,j+1,projection='3d',computed_zorder=False);axes.append(ax);setup(ax)
        color=cmap(norm(fields[i,cutowners]))
        ax.add_collection3d(Poly3DCollection(f.xyz[cutfaces],facecolors=color,edgecolors=color,linewidths=.03,antialiased=False,zorder=1))
        color=cmap(norm(fields[i,section_owners]))
        ax.add_collection3d(Poly3DCollection(section3,facecolors=color,edgecolors=color,linewidths=.04,antialiased=False,zorder=2))
        stage=min(2,3*i//len(qs))
        ax.add_collection3d(Poly3DCollection(jaw_faces(stage),facecolors='#b8c5d1',edgecolors='#566c7f',linewidths=.5,alpha=.55,zorder=3))
        hot=centers[int(fields[i].argmax())];ax.scatter(*hot,color='#26d6c4',edgecolor='black',s=100,marker='*',depthshade=False,zorder=5)
        ax.set_title(f"Frame {i} | grip height {model.patches[2*stage]['height_m']*1000:+.0f} mm\nmax = {fields[i].max():.2f} MPa",fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.7,pad=.025,label='Von Mises stress (MPa), common scale')
    fig.suptitle('Hollow eggshell with opposing rigid fingers: cutaway shows the empty cavity\nOriginal geometry | 0.5 mm wall | cyan star: projected global maximum element',fontsize=13)
    fig.savefig(output/'eggshell_gripper_stress_fields.png',dpi=170);plt.close(fig)

    fig=plt.figure(figsize=(10.7,6.3),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[2,1]);ax=fig.add_subplot(gs[0,0],projection='3d',computed_zorder=False);setup(ax)
    color=cmap(norm(fields[0,cutowners]));shell=Poly3DCollection(f.xyz[cutfaces],facecolors=color,edgecolors=color,linewidths=.03,antialiased=False,zorder=1);ax.add_collection3d(shell)
    color=cmap(norm(fields[0,section_owners]));cut=Poly3DCollection(section3,facecolors=color,edgecolors=color,linewidths=.03,antialiased=False,zorder=2);ax.add_collection3d(cut)
    jaws=Poly3DCollection(jaw_faces(0),facecolors='#b8c5d1',edgecolors='#566c7f',linewidths=.5,alpha=.55,zorder=3);ax.add_collection3d(jaws)
    ax.set_title('Actual stress on undeformed hollow shell',fontsize=10)
    refresh_ax=fig.add_subplot(gs[0,1]);mask=PolyCollection(section,array=np.zeros(len(section)),cmap=matplotlib.colors.ListedColormap(['#e4e8ed','#e49b2d']),norm=Normalize(0,1),edgecolors='none',antialiased=False)
    refresh_ax.add_collection(mask);refresh_ax.set_xlim(-.025,.025);refresh_ax.set_ylim(-.032,.032);refresh_ax.set_aspect('equal')
    refresh_ax.set_xlabel('x (m)');refresh_ax.set_ylabel('z (m)');refresh_ax.set_title('Section: orange refreshed / gray skipped groups',fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.65,pad=.01,label='Von Mises (MPa)')
    curve=fig.add_subplot(gs[1,:]);rr=[r for r in rows if r['scenario']=='regrasp'];frame=np.arange(len(rr))
    curve.plot(frame,[r['oracle_max_Pa']/1e6 for r in rr],color='#2474a6',label='Full FEM',lw=2)
    curve.plot(frame,[r['old_hotspot_max_Pa']/1e6 for r in rr],color='#c54b43',ls='--',label='Only previous maximum element',lw=1.2)
    cursor=curve.axvline(0,color='#444444',lw=1);marker,=curve.plot([],[],'o',mfc='none',color='#14806e',label='Temporal maximum')
    curve.set_xlabel('Frame');curve.set_ylabel('Max von Mises (MPa)');curve.grid(alpha=.2);curve.legend(loc='lower right',fontsize=8,frameon=False)
    title=fig.suptitle('',fontsize=12);writer=PillowWriter(fps=8)
    indices=sorted(set(range(0,len(qs),2))|{first-1,first,first+1,second-1,second,second+1,len(qs)-1})
    with writer.saving(fig,str(output/'eggshell_gripper_regrasp.gif'),dpi=100):
        for k,i in enumerate(indices):
            color=cmap(norm(fields[i,cutowners]));shell.set_facecolor(color);shell.set_edgecolor(color)
            color=cmap(norm(fields[i,section_owners]));cut.set_facecolor(color);cut.set_edgecolor(color)
            stage=min(2,3*i//len(qs));jaws.set_verts(jaw_faces(stage))
            mask.set_array(active[i][inv[section_owners]//model.group_size].astype(float))
            cursor.set_xdata([i,i]);marker.set_data([i],[outs[i].maximum/1e6])
            title.set_text(f"Rigid gripper / hollow egg | frame {i}/{len(qs)-1} | {np.max(np.abs(qs[i,:18:3])):.2f} N per finger\n"+
                           f"max = {outs[i].maximum/1e6:.2f} MPa | response work = {outs[i].evaluated/model.ne*100:.1f}%")
            writer.grab_frame()
            if k%20==0:print(json.dumps({'render_frame':i,'progress':f'{k+1}/{len(indices)}'}),flush=True)
    plt.close(fig)


def cpu_name():
    p=Path('/proc/cpuinfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('model name'):return line.split(':',1)[1].strip()
    return platform.processor() or 'unknown'


def bundle(output):
    files=[HERE/'eggshell_gripper_cpu.py',HERE/'rigid_stress_reference.py',HERE/'rigid_stress_temporal_cpu_test.py',HERE/'rigid_stress_visualize.py']
    files+=list(output.glob('eggshell_gripper_*'))
    zip_path=output/'eggshell_gripper_cpu_bundle.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        added=set()
        for p in files:
            if p.is_file() and p.name not in added and p!=zip_path:z.write(p,p.name);added.add(p.name)
    return zip_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--plot',action='store_true')
    parser.add_argument('--render-only',action='store_true',help='Use saved timings; rebuild only the FEM fields and figures.')
    parser.add_argument('--output-dir',type=Path,default=HERE);args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir
    with threadpool_limits(limits=1):
        if args.render_only:
            summary=json.loads((output/'eggshell_gripper_cpu_results.json').read_text())
            with (output/'eggshell_gripper_cpu_frames.csv').open() as f:
                rows=[{k:v if k=='scenario' else float(v) for k,v in r.items()} for r in csv.DictReader(f)]
            model=EggResponse(4,4);cases=sequence(model,summary['cases'][0]['frames'])
            summary['execution']['CPU']=cpu_name()
            (output/'eggshell_gripper_cpu_results.json').write_text(json.dumps(summary,indent=2))
            plots(model,cases,summary,rows,output)
            print(json.dumps({'render_only':True,'bundle':str(bundle(output))}),flush=True)
            return
        coarse=EggResponse(3,4)
        cq=np.zeros(19);cq[6]=4;cq[9]=-4
        coarse_max=coarse.response_scan(cq)[0];coarse_check=coarse.physical_check(cq);coarse_meta=coarse.metadata
        print(json.dumps({'coarse_nodes':coarse_meta['nodes'],'coarse_max_Pa':coarse_max}),flush=True)
        del coarse;gc.collect()
        model=EggResponse(4,4)
        print(json.dumps({'fine_nodes':model.metadata['nodes'],'fine_tets':model.ne,'offline_seconds':model.offline_model_seconds+model.offline_response_seconds}),flush=True)
        cases=sequence(model,args.frames);rows=[];measured=[]
        for name,qs in cases.items():
            result=measure(model,name,qs,args.repeats,rows);measured.append(result)
            print(json.dumps({'scenario':name,'timings':result['timings'],'fraction':result['mean_evaluation_fraction'],'error':result['max_error']}),flush=True)
        fine_max=model.response_scan(cq)[0]
        summary={'execution':{'CPU':cpu_name(),'BLAS_threads':1,'precision':'FP64','Genesis_tested':False,'GPU_tested':False},
                 'model':model.metadata,'cases':measured,'frames_verified':len(rows),'all_checks_passed':True,
                 'maximum_normalized_error':max(r['relative_error'] for r in rows),
                 'discretization_check':{'coarse':coarse_meta,'force_per_finger_N':4.,'coarse_max_Pa':coarse_max,'fine_max_Pa':fine_max,
                                         'relative_peak_change':abs(coarse_max-fine_max)/fine_max,'coarse_displacement_max_m':coarse_check['canonical_displacement_max_m']},
                 'limitations':['Prescribed pressure footprint / force-controlled jaws; no online collision or contact solve.',
                                'Synthetic homogeneous material; no fracture, buckling, geometric nonlinearities or biological calibration.',
                                'P1 solid tetrahedra through thickness; two resolutions do not establish stress convergence.',
                                'Discrete regrasp heights are inside a fixed 19-dimensional load basis.',
                                'Visualizations recover full fields outside maximum-query timings.']}
        np.savez_compressed(output/'eggshell_gripper_mesh.npz',xyz=model.fem.xyz,tetrahedra=model.fem.tetrahedra,outer_faces=model.outer_faces)
        (output/'eggshell_gripper_cpu_results.json').write_text(json.dumps(summary,indent=2))
        with (output/'eggshell_gripper_cpu_frames.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if args.plot:plots(model,cases,summary,rows,output)
        zip_path=bundle(output)
        print(json.dumps({'all_checks_passed':True,'frames':len(rows),'max_error':summary['maximum_normalized_error'],
                          'coarse_fine_peak_change':summary['discretization_check']['relative_peak_change'],
                          'bundle':str(zip_path)}),flush=True)


if __name__=='__main__':main()
