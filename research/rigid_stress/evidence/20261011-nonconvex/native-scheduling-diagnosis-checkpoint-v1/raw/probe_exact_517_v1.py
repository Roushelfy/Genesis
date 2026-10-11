"""Replay the first failed component, with immutable original source groups, rather than a later frame."""
import json
from pathlib import Path

import numpy as np
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_footprints, create_surface_fan_tractions, kernel_footprint_ports, kernel_surface_fan_profiles
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, create_traction_samples, kernel_traction_samples, kernel_fit_tractions
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.wrench import WrenchPatch
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from traction_warp_candidate_v2 import kernel_fit_tractions as kernel_fit_warp_v2
from traction_warp_candidate_v4 import kernel_fit_tractions as kernel_fit_warp_v4

root=Path(__file__).parent
output=root/'exact-517-replay-v1/results'
output.mkdir(parents=True,exist_ok=False)
gs.init(backend=gs.cuda,precision='64',logging_level='warning')
with np.load(root/'b1024-stage-failure-v1/results/first-error.npz') as saved:
    data={name:saved[name].copy() for name in saved.files}
active=data['chunk_contacts.valid'][0]
count=int(active.sum())
mesh=Path('examples/rigid/assets/hollow_egg/level1/elastic.npz')
options=gs.options.RigidStressOptions(mesh=mesh,contact_model='intrinsic',intrinsic_support='pair_union')
model=StressModel(options)
surface=StressSurface(10,model.info)
paths=StressSurfacePaths(model.info,8)
contacts,sources=create_contacts(count,1,0.006),create_sources(count,1)
info=create_traction_info(surface.info)
footprints=create_footprints(paths.info.positions.shape[0],count,1)
tractions=create_surface_fan_tractions(surface.info,count,1,shared_support=True)
for prefix,container in [('chunk_contacts',contacts),('chunk_sources',sources)]:
    for name,value in container.__dict__.items():
        key=prefix+'.'+name
        if key in data and len(value.shape)>0:
            array=data[key]
            if value.shape[0]==count:
                array=np.moveaxis(array[:,:count],0,1)
            value.from_numpy(np.ascontiguousarray(array))
kernel_footprint_ports(contacts,sources,footprints,surface.info,paths.info)
kernel_surface_fan_profiles(contacts,footprints,tractions,model.info,surface.info,paths.info,info)
kernel_bind_traction_support(contacts,sources,footprints,tractions)
epsilon=float(np.finfo(gs.np_float).eps)
records=[]
for label,function,limit in [('scalar128',kernel_fit_tractions,128),('warp_v2_128',kernel_fit_warp_v2,128),
                              ('warp_v4_128',kernel_fit_warp_v4,128),('scalar512',kernel_fit_tractions,512),
                              ('warp_v2_512',kernel_fit_warp_v2,512)]:
    function(contacts,sources,tractions,info,epsilon,9e-9,limit)
    fields={name:qd_to_numpy(tractions.__dict__[name],transpose=True,copy=True).tolist()
             for name in ('status','force_error','moment_error','evaluations','dual','scale','weight_sum','certificate','certificate_margin')}
    record={'variant':label,**fields}
    records.append(record)
    print(json.dumps(record),flush=True)
samples=create_traction_samples(tractions)
kernel_traction_samples(samples,tractions,surface.info,info)
positions=qd_to_numpy(samples.positions,transpose=True,copy=True)[0]
normals=qd_to_numpy(samples.normals,transpose=True,copy=True)[0]
weights=qd_to_numpy(tractions.profile,transpose=True,copy=True)[0]
starts=qd_to_numpy(tractions.support.start,transpose=True,copy=True)[0]
ends=qd_to_numpy(tractions.support.end,transpose=True,copy=True)[0]
friction=data['chunk_contacts.friction'][0,:count]
cpu=[]
try:
    for i_c in range(count):
        cloud_positions,cloud_normals,cloud_weights,cloud_mu=[],[],[],[]
        for i_s in range(starts[i_c],ends[i_c]):
            selected=weights[:,i_s]>0
            cloud_positions.append(positions[selected,i_s])
            cloud_normals.append(normals[selected,i_s])
            cloud_weights.append(weights[selected,i_s])
            cloud_mu.append(np.full(selected.sum(),friction[i_s]))
        xyz,n,w,mu=[np.concatenate(value) for value in (cloud_positions,cloud_normals,cloud_weights,cloud_mu)]
        patch=WrenchPatch(data['chunk_contacts.center'][0,i_c],data['chunk_contacts.force'][0,i_c],
                          data['chunk_contacts.radius'][0,i_c],mu,data['chunk_sources.anchor_moment'][0,i_c])
        force=cpu_resolved_wrench_fit(xyz,w,n,patch)
        point=data['chunk_contacts.position'][0,i_c]
        error_f=np.linalg.norm(force.sum(axis=0)-patch.force)/np.linalg.norm(patch.force)
        error_m=np.linalg.norm(np.cross(xyz-point,force).sum(axis=0)-data['chunk_sources.free_moment'][0,i_c])/(patch.radius*np.linalg.norm(patch.force))
        record={'component':i_c,'count':len(w),'normalized_force_error':float(error_f),'normalized_moment_error':float(error_m)}
        cpu.append(record)
        print(json.dumps({'cpu_oracle':record}),flush=True)
finally:
    np.savez_compressed(output/'fixture.npz',**{name:data['chunk_contacts.'+name][0,:count] for name in ('position','center','force','radius','friction','anchor_face')},
                         geom=data['chunk_sources.geom'][0,:count],pair=data['chunk_sources.pair'][0,:count],
                         free_moment=data['chunk_sources.free_moment'][0,:count],anchor_moment=data['chunk_sources.anchor_moment'][0,:count])
    (output/'summary.json').write_text(json.dumps({'records':records,'cpu_oracles':cpu,
        'geom':data['chunk_sources.geom'][0,:count].tolist(),'pair':data['chunk_sources.pair'][0,:count].tolist(),
        'radius':data['chunk_contacts.radius'][0,:count].tolist(),'friction':friction.tolist()},indent=2)+'\n')
