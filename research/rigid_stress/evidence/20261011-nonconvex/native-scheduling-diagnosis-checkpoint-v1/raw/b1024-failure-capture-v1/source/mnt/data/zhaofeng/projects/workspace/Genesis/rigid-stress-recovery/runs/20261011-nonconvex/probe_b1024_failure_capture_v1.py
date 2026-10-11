"""Capture actual larger-batch failure state and independently replay its complete current loads."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd
import genesis as gs
import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step, kernel_accept
from genesis.utils.misc import qd_to_numpy
from traction_warp_candidate_v2 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

root=Path(__file__).parent
output=root/'b1024-failure-capture-v1/results'
output.mkdir(parents=True,exist_ok=False)
gs.init(backend=gs.cuda,precision='64',seed=623201,logging_level='warning')
workload=FrankaEgg(1024,seed=623201,varied=True,contact_model='intrinsic',intrinsic_support='pair_union',
                   contact_block_size=4,egg_geometry='mesh64',use_gjk_collision=True,enable_contact_patch=True,
                   conditions=root/'sample-warp-b1024-profile-v1/conditions.npz')
solver=workload.scene.rigid_solver
entry=solver.stress_recovery.links[0]
completed=0
try:
    with patch.object(intrinsic,'kernel_fit_tractions',kernel_fit_tractions), \
         patch.object(intrinsic,'kernel_commit_intrinsic_contacts',kernel_commit_intrinsic_contacts), \
         patch.object(intrinsic,'kernel_scatter_tractions',kernel_scatter_tractions):
        for tick in range(600):
            workload.step()
            completed+=1
            if tick%25==0:
                print(json.dumps({'tick':tick}),flush=True)
finally:
    containers=[('contacts',entry.contacts),('sources',entry.intrinsic.raw_sources),('state',entry.state),
                ('chunk_contacts',entry.intrinsic.contacts),('chunk_sources',entry.intrinsic.sources),
                ('traction',entry.intrinsic.tractions),('packing',entry.intrinsic.packing)]
    arrays={}
    for prefix,container in containers:
        for item in intrinsic.iter_data(container,prefix):
            if len(item.value.shape)>0:
                arrays[item.name]=qd_to_numpy(item.value,transpose=True,copy=True)
    arrays['errno']=qd_to_numpy(solver._errno,transpose=True,copy=True)
    i_link=workload.link.idx
    for name in ('cfrc_applied_ang','cfrc_applied_vel','cfrc_coupling_ang','cfrc_coupling_vel'):
        value=solver.dyn_state.links.__dict__[name]
        arrays['rigid_'+name]=qd_to_numpy(value,transpose=True,copy=True)[:,i_link]
    np.savez_compressed(output/'failure-state.npz',**arrays)
    failed=np.flatnonzero(arrays['errno'])
    print(json.dumps({'completed_steps':completed,'errno_envs':failed.tolist(),
        'contact_status_nonzero':np.argwhere(arrays['contacts.status']!=0).tolist(),
        'applied_max':{name:float(np.abs(value).max()) for name,value in arrays.items() if name.startswith('rigid_')},
        'last_chunk_statuses':np.unique(arrays['traction.status']).tolist(),
        'invalid_steps':arrays['state.invalid_steps'][failed].tolist()}),flush=True)
    # Keep the captured component frame; do not use the already-integrated rigid pose.
    solver._errno.fill(0)
    kernel_begin_step(entry.state)
    entry.intrinsic.scatter(entry.contacts,entry.state,entry.model.info,entry.surface.info,entry.options,
                            solver.stress_recovery.source_epsilon,solver._errno)
    errno_scalar=qd_to_numpy(solver._errno,transpose=True,copy=True)
    status_scalar=qd_to_numpy(entry.contacts.status,transpose=True,copy=True)
    force_scalar=qd_to_numpy(entry.state.force,transpose=True,copy=True)
    np.savez_compressed(output/'scalar-replay.npz',errno=errno_scalar,status=status_scalar,nodal_force=force_scalar)
    summary={'completed_steps':completed,'failed_envs':failed.tolist(),
        'scalar_replay_errno_envs':np.flatnonzero(errno_scalar).tolist(),
        'scalar_replay_status_nonzero':np.argwhere(status_scalar!=0).tolist(),
        'wrench_max_normalized_force':float((qd_to_numpy(entry.contacts.force_error,transpose=True)/
            np.maximum(1e-300,np.linalg.norm(arrays['contacts.force'],axis=-1)))[arrays['contacts.valid']].max(initial=0))}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
