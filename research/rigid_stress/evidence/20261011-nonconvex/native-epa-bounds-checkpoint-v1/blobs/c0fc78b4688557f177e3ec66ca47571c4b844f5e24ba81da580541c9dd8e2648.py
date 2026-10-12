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
output = directory / "captured-gjk-patch-probe-v8"
with np.load(directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz") as asset:
    saved = {key:asset[key].copy() for key in asset.files}
gs.init(backend=gs.cuda,precision="64",logging_level="warning")
trace_vectors = V_VEC(3,dtype=gs.qd_float,shape=(5,2))
trace_called = V(dtype=gs.qd_int,shape=(2,))
original_multi = gjk.func_multi_contact

@qd.func
def func_trace_multi(i_ga:int,i_gb:int,i_b:int,i_f:int,pos_a:qd.types.vector(3),quat_a:qd.types.vector(4),pos_b:qd.types.vector(3),quat_b:qd.types.vector(4),gjk_state:array_class.GJKState,dyn_info:array_class.DynInfo,collider_info:array_class.ColliderInfo):
    is_selected = (i_ga == 14 or i_ga == 15) and i_gb == 24
    normal = gjk_state.polytope_faces.normal[i_b,i_f]
    first = gjk_state.witness.point_obj1[i_b,0]
    second = gjk_state.witness.point_obj2[i_b,0]
    original_multi(i_ga,i_gb,i_b,i_f,pos_a,quat_a,pos_b,quat_b,gjk_state,dyn_info,collider_info)
    if is_selected:
        # Mesh-involved patches use at most4of8slots. These diagnostic writes use inactive slots only.
        assert gjk_state.n_witness[i_b] <= 4
        gjk_state.witness.point_obj1[i_b,6] = gjk_state.witness.point_obj1[i_b,0]
        gjk_state.witness.point_obj2[i_b,6] = gjk_state.witness.point_obj2[i_b,0]
        gjk_state.witness.point_obj1[i_b,4] = normal
        gjk_state.witness.point_obj2[i_b,4] = first
        gjk_state.witness.point_obj1[i_b,5] = second
        gjk_state.witness.point_obj2[i_b,5] = gs.qd_vec3(i_ga,i_gb,1435)

records = []

@qd.kernel
def kernel_read_face(gjk_state:array_class.GJKState, index:int, values:qd.Tensor):
    for once in range(1):
        face = gjk_state.nearest_face[index]
        values[0] = gs.qd_vec3(face,gjk_state.polytope.nverts[index],gjk_state.polytope.nfaces[index])
        values[1] = gjk_state.polytope_faces.normal[index,face]
        values[2] = gs.qd_vec3(gjk_state.polytope_faces.dist2[index,face],gjk_state.distance[index],gjk_state.n_witness[index])
        values[12] = gjk_state.polytope_faces.verts_idx[index,face]
        values[13] = gs.qd_vec3(gjk_state.polytope_faces.map_idx[index,face],gjk_state.polytope.nfaces_map[index],gjk_state.nsimplex[index])
        for corner in qd.static(range(3)):
            vertex = gjk_state.polytope_faces.verts_idx[index,face][corner]
            values[3+corner] = gjk_state.polytope_verts.mink[index,vertex]
            values[6+corner] = gjk_state.polytope_verts.obj1[index,vertex]
            values[9+corner] = gjk_state.polytope_verts.obj2[index,vertex]

@qd.kernel
def kernel_read_polytope(gjk_state:array_class.GJKState,index:int,vertices:qd.Tensor,faces:qd.Tensor,normals:qd.Tensor,map_values:qd.Tensor):
    for vertex in range(vertices.shape[0]):
        vertices[vertex] = gjk_state.polytope_verts.mink[index,vertex]
    for face in range(faces.shape[0]):
        faces[face] = gjk_state.polytope_faces.verts_idx[index,face]
        normals[face] = gjk_state.polytope_faces.normal[index,face]
        map_values[face] = gjk_state.polytope_faces.map_idx[index,face]

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
    producer_trace = {}
    if is_patch:
        for scratch_name in ("contact0_gjk_state","multicontact_gjk_state"):
            gjk_state = getattr(solver.collider.gjk,scratch_name)
            if gjk_state.witness.point_obj1.shape[1] > 6:
                arrays = [qd_to_numpy(getattr(gjk_state.witness,name),copy=True) for name in ("point_obj1","point_obj2")]
                ids = np.flatnonzero(np.isin(arrays[1][:,5,0],(14,15)) & (arrays[1][:,5,1] == 24) & (arrays[1][:,5,2] == 1435))
                producer_trace[scratch_name] = [{"scratch":int(index),"pair":arrays[1][index,5].tolist(),"EPA_face_normal":arrays[0][index,4].tolist(),"original_witness_A":arrays[1][index,4].tolist(),"original_witness_B":arrays[0][index,5].tolist(),"postpatch_witness_A":arrays[0][index,6].tolist(),"postpatch_witness_B":arrays[1][index,6].tolist()} for index in ids]
                for row in producer_trace[scratch_name]:
                    index = row["scratch"]
                    values = V_VEC(3,dtype=gs.qd_float,shape=(14,))
                    kernel_read_face(gjk_state,index,values)
                    face_values = qd_to_numpy(values,copy=True)
                    vertex_count,face_count = (int(x) for x in face_values[0,1:])
                    vertices = V_VEC(3,dtype=gs.qd_float,shape=(vertex_count,))
                    faces = V_VEC(3,dtype=gs.qd_int,shape=(face_count,))
                    normals = V_VEC(3,dtype=gs.qd_float,shape=(face_count,))
                    map_values = V(dtype=gs.qd_int,shape=(face_count,))
                    kernel_read_polytope(gjk_state,index,vertices,faces,normals,map_values)
                    row["nearest_face_values"] = face_values.tolist()
                    row["bound_exit_trace_A"] = arrays[0][index,7].tolist()
                    row["bound_exit_trace_B"] = arrays[1][index,7].tolist()
                    row["all_minkowski_vertices"] = qd_to_numpy(vertices,copy=True).tolist()
                    row["all_faces"] = qd_to_numpy(faces,copy=True).tolist()
                    row["all_face_normals"] = qd_to_numpy(normals,copy=True).tolist()
                    row["all_face_map"] = qd_to_numpy(map_values,copy=True).tolist()
    record = {"patch":is_patch,"selected_contacts":rows,"trace_called":qd_to_numpy(trace_called).tolist(),"trace_vectors":trace.tolist(),"trace_axis":"First dimension: original EPA face normal, original witness A/B, post-patch witness A/B; second dimension: geom14/geom15 against egg24."}
    record["producer_inactive_slot_trace"] = producer_trace
    records.append(record)
    print(json.dumps(record),flush=True)
(output / "summary.json").write_text(json.dumps({"scope":__doc__,"environment":3092,"records":records},indent=2)+"\n")
