"""Offline feasibility of an explicitly different aggregate-pair model; never alters original inputs or production."""

import json
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "exact-602-socp-v1/tools"))
import clarabel
import numpy as np
from scipy import sparse

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
with np.load(root / "policy-triangle-clip-v3/cpu/results/samples.npz") as saved:
    xyz = np.concatenate([saved["positions"][:saved["count"][i], i] for i in range(2)])
    normals = np.concatenate([saved["normals"][:saved["count"][i], i] for i in range(2)])
    mu = np.concatenate([np.full(saved["count"][i],saved["contacts.friction"][i]) for i in range(2)])
    inputs = {key:saved[key].copy() for key in saved.files if key.startswith(("contacts.","raw_sources."))}
records = []
for component in ("aggregate",):
    original_force = inputs["contacts.force"][:2]
    original_anchor = inputs["contacts.center"][:2]
    force, anchor, radius = original_force.sum(axis=0), original_anchor[0], inputs["contacts.radius"][0]
    moment = inputs["raw_sources.anchor_moment"][:2].sum(axis=0)
    moment += np.cross(original_anchor - anchor, original_force).sum(axis=0)
    scale = np.linalg.norm(force)
    relative = (xyz-anchor)/radius
    skew = np.zeros((len(xyz),3,3))
    skew[:,0,1],skew[:,0,2] = -relative[:,2],relative[:,1]
    skew[:,1,0],skew[:,1,2] = relative[:,2],-relative[:,0]
    skew[:,2,0],skew[:,2,1] = -relative[:,1],relative[:,0]
    operator = np.concatenate((np.broadcast_to(np.eye(3),skew.shape),skew),axis=1)
    target = np.concatenate((force,moment/radius))/scale
    cone_maps = np.concatenate((mu[:,None,None]*normals[:,None],np.eye(3)-normals[:,:,None]*normals[:,None,:]),axis=1)
    equality = sparse.csc_matrix(operator.transpose(1,0,2).reshape(6,-1))
    matrix = sparse.vstack((equality,sparse.block_diag(list(-cone_maps),format="csc")),format="csc")
    rhs = np.concatenate((target,np.zeros(4*len(xyz))))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    for field in ("tol_feas","tol_gap_abs","tol_gap_rel","tol_infeas_abs","tol_infeas_rel"):
        setattr(settings,field,1e-11)
    solver = clarabel.DefaultSolver(sparse.csc_matrix((3*len(xyz),3*len(xyz))),np.zeros(3*len(xyz)),matrix,rhs,
        [clarabel.ZeroConeT(6)]+[clarabel.SecondOrderConeT(4)]*len(xyz),settings)
    solution = solver.solve()
    f = np.asarray(solution.x).reshape(-1,3)
    compression = np.einsum("qi,qi->q",f,normals)
    excess = np.linalg.norm(f-compression[:,None]*normals,axis=1)-mu*compression
    ray = -np.asarray(solution.z)[:6]
    ray /= np.linalg.norm(ray)
    value = np.einsum("qij,i->qj",operator,ray)
    ray_normal = np.einsum("qi,qi->q",value,normals)
    polar = ray_normal+mu*np.linalg.norm(value-ray_normal[:,None]*normals,axis=1)
    mismatch = equality@solution.x-target
    record = {"component":component,"status":str(solution.status),"iterations":solution.iterations,"sample_count":len(xyz),
        "force_error":float(np.linalg.norm(mismatch[:3])),"moment_error":float(np.linalg.norm(mismatch[3:])),
        "minimum_compression":float(compression.min()),"maximum_cone_excess":float(excess.max()),
        "certificate_margin":float(target@ray),"certificate_polar_max":float(polar.max()),"ray":ray.tolist()}
    records.append(record)
    np.savez_compressed(output/f"component-{component}.npz",ray=ray,target=target,positions=xyz,normals=normals,friction=mu,forces=f)
    print(json.dumps(record),flush=True)
(output/"summary.json").write_text(json.dumps(records,indent=2)+"\n")
