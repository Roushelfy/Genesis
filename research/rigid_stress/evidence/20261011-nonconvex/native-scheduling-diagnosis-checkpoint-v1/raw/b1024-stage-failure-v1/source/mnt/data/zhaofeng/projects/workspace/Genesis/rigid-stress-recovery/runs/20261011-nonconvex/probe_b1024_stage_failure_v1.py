"""Locate the first larger-batch candidate errno at an actual native pipeline boundary."""
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd
import genesis as gs
import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy
from traction_warp_candidate_v2 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

root=Path(__file__).parent
output=root/'b1024-stage-failure-v1/results'
output.mkdir(parents=True,exist_ok=False)
gs.init(backend=gs.cuda,precision='64',seed=623201,logging_level='warning')
workload=FrankaEgg(1024,seed=623201,varied=True,contact_model='intrinsic',intrinsic_support='pair_union',
                   contact_block_size=4,egg_geometry='mesh64',use_gjk_collision=True,enable_contact_patch=True,
                   conditions=root/'sample-warp-b1024-profile-v1/conditions.npz')
solver=workload.scene.rigid_solver
entry=solver.stress_recovery.links[0]
records=[]


def stage(name,function,*args,**kwargs):
    result=function(*args,**kwargs)
    if workload.tick>=510:
        errno=qd_to_numpy(solver._errno,transpose=True,copy=True)
        failed=np.flatnonzero(errno)
        if len(failed):
            arrays={'errno':errno}
            for prefix,container in [('contacts',entry.contacts),('sources',entry.intrinsic.raw_sources),
                                     ('chunk_contacts',entry.intrinsic.contacts),('chunk_sources',entry.intrinsic.sources),
                                     ('traction',entry.intrinsic.tractions),('packing',entry.intrinsic.packing)]:
                for item in intrinsic.iter_data(container,prefix):
                    if len(item.value.shape)>0 and all(item.value.shape) and 'profile' not in item.name:
                        value=qd_to_numpy(item.value,transpose=True,copy=True)
                        arrays[item.name]=value[failed] if value.shape[0]==1024 else value
            record={'tick':workload.tick,'first_error_stage':name,'errno_envs':failed.tolist(),
                    'chunk_active':arrays['chunk_contacts.valid'].tolist(),
                    'chunk_traction_status':arrays['traction.status'].tolist(),
                    'chunk_source_status':arrays['chunk_sources.status'].tolist(),
                    'raw_nonzero_status':np.argwhere(arrays['contacts.status']!=0).tolist()}
            records.append(record)
            np.savez_compressed(output/'first-error.npz',**arrays)
            (output/'first-error.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            raise RuntimeError('Captured first candidate load error at '+name)
    return result


functions={name:intrinsic.__dict__[name] for name in ('kernel_pack_intrinsic_contacts','kernel_pack_contact_groups',
    'kernel_load_intrinsic_contacts','kernel_bind_source_vertices','kernel_resolve_sources','kernel_footprint_ports',
    'kernel_surface_fan_profiles','kernel_bind_traction_support','kernel_certify_continuous_force')}
functions.update(kernel_fit_tractions=kernel_fit_tractions,kernel_commit_intrinsic_contacts=kernel_commit_intrinsic_contacts,
                 kernel_scatter_tractions=kernel_scatter_tractions)
patches=[patch.object(intrinsic,name,partial(stage,name,function)) for name,function in functions.items()]
complete=0
try:
    for context in patches:
        context.start()
    for tick in range(530):
        workload.step()
        complete+=1
        if tick%25==0:
            print(json.dumps({'tick':tick}),flush=True)
finally:
    for context in reversed(patches):
        context.stop()
    (output/'summary.json').write_text(json.dumps({'complete_steps':complete,'records':records},indent=2)+'\n')
