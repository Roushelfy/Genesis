"""Untimed captured-pose GJK/patch diagnosis with read-only producer tracing; no force modifications."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.collider import gjk
from genesis.utils import array_class
from genesis.utils.array_class import V_VEC, V, iter_data
from genesis.utils.misc import qd_to_numpy

directory = Path(__file__).parent
output = directory / "captured-gjk-patch-probe-v1"
with np.load(directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz") as asset:
    saved = {key:asset[key].copy() for key in asset.files}
gs.init(backend=gs.cuda,precision="64",logging_level="warning")
trace_vectors = V_VEC(3,dtype=gs.qd_float,shape=(5,2))
trace_called = V(dtype=gs.qd_int,shape=(2,))
original_multi = gjk.func_multi_contact

@qd.func
def func_trace_multi(i_ga:int,i_gb:int,i_b:int,i_f:int,pos_a:qd.types.vector(3),quat_a:qd.types.vector(4),pos_b:qd.types.vector(3),quat_b:qd.types.vector(4),gjk_state:array_class.GJKState,dyn_info:array_class.DynInfo,collider_info:array_class.ColliderInfo):
    is_selected = (i_ga == 14 or i_ga == 15) and i_gb == 24
    index = i_ga - 14
    if is_selected:
        trace_called[index] = 1
        trace_vectors[0,index] = gjk_state.polytope_faces.normal[i_b,i_f]
        trace_vectors[1,index] = gjk_state.witness.point_obj1[i_b,0]
        trace_vectors[2,index] = gjk_state.witness.point_obj2[i_b,0]
    original_multi(i_ga,i_gb,i_b,i_f,pos_a,quat_a,pos_b,quat_b,gjk_state,dyn_info,collider_info)
    if is_selected:
        trace_vectors[3,index] = gjk_state.witness.point_obj1[i_b,0]
        trace_vectors[4,index] = gjk_state.witness.point_obj2[i_b,0]

records = []
for is_patch in (False,True):
    workload = FrankaEgg(n_envs=1,stress=True,seed=623103,varied=True,contact_model="intrinsic",intrinsic_load_partition="body_source_prior",contact_block_size=4,egg_geometry="mesh64",use_gjk_collision=True,enable_contact_patch=is_patch)
    solver = workload.scene.rigid_solver
    assert solver.n_geoms == 25
    for name,state in (("rigid.links",solver.dyn_state.links),("rigid.geoms",solver.dyn_state.geoms)):
        for item in iter_data(state,name):
            if item.name not in saved:
                continue
            array = saved[item.name]
            if item.value.shape:
                array = np.moveaxis(array,0,len(item.value.shape)-1)
            item.value.from_numpy(np.ascontiguousarray(array) if array.ndim else array)
    trace_called.fill(0)
    trace_vectors.fill(0)
    with patch.object(gjk,"func_multi_contact",func_trace_multi):
        solver.collider.detection()
    qd.sync()
    solver.check_errno()
    state = solver.collider.collider_state
    count = int(qd_to_numpy(state.n_contacts)[0])
    order = qd_to_numpy(state.contact_sort_idx,transpose=True)[0,:count]
    fields = {name:qd_to_numpy(getattr(state.contact_data,name),transpose=True,copy=True)[0] for name in ("geom_a","geom_b","pos","normal","penetration","pair_idx")}
    selected = order[((fields["geom_a"][order] == 14) | (fields["geom_a"][order] == 15)) & (fields["geom_b"][order] == 24)]
    rows = [{name:values[slot].tolist() for name,values in fields.items()} for slot in selected]
    trace = qd_to_numpy(trace_vectors,transpose=True,copy=True)
    record = {"patch":is_patch,"selected_contacts":rows,"trace_called":qd_to_numpy(trace_called).tolist(),"trace_vectors":trace.tolist(),"trace_axis":"First dimension: original EPA face normal, original witness A/B, post-patch witness A/B; second dimension: geom14/geom15 against egg24."}
    records.append(record)
    print(json.dumps(record),flush=True)
(output / "summary.json").write_text(json.dumps({"scope":__doc__,"environment":3092,"records":records},indent=2)+"\n")
