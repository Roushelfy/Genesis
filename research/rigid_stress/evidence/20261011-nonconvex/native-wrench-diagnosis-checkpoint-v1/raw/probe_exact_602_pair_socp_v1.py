"""Independent circular-cone feasibility and separating certificate for the exact tick602 support."""
import importlib.metadata
import json
from pathlib import Path
import sys

root = Path(__file__).parent
output = root / 'exact-602-pair-socp-v1'
sys.path.insert(0, str(output / 'tools'))
import clarabel
import numpy as np
from scipy import sparse

with np.load(root / 'exact-602-replay-v1/results/cpu-component-0.npz') as saved:
    xyz, normals, weights, mu = (saved[key] for key in ('positions', 'normals', 'profiles', 'friction'))
with np.load(root / 'exact-602-replay-v1/results/fixture.npz') as saved:
    fixture = {key: saved[key].copy() for key in saved.files}
records = []
for component in ['pair']:
    force = fixture['force'].sum(axis=0)
    scale, radius = np.linalg.norm(force), fixture['radius'][1]
    anchor = fixture['center'][1]
    moment = np.cross(fixture['position']-anchor,fixture['force']).sum(axis=0)+fixture['free_moment'].sum(axis=0)
    relative = (xyz - anchor) / radius
    skew = np.zeros((len(xyz), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -relative[:, 2], relative[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = relative[:, 2], -relative[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -relative[:, 1], relative[:, 0]
    operator = np.concatenate((np.broadcast_to(np.eye(3), skew.shape), skew), axis=1)
    target = np.concatenate((force / scale, moment / (radius * scale)))
    cone_maps = np.concatenate((mu[:, None, None] * normals[:, None], np.eye(3) - normals[:, :, None] * normals[:, None, :]), axis=1)
    equality = sparse.csc_matrix(operator.transpose(1, 0, 2).reshape(6, -1))
    cones_matrix = sparse.block_diag(list(-cone_maps), format='csc')
    matrix = sparse.vstack((equality, cones_matrix), format='csc')
    rhs = np.concatenate((target, np.zeros(4 * len(xyz))))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_feas = 1e-11
    settings.tol_gap_abs = 1e-11
    settings.tol_gap_rel = 1e-11
    settings.tol_infeas_abs = 1e-11
    settings.tol_infeas_rel = 1e-11
    solver = clarabel.DefaultSolver(sparse.csc_matrix((3*len(xyz), 3*len(xyz))), np.zeros(3*len(xyz)),
                                   matrix, rhs, [clarabel.ZeroConeT(6)] + [clarabel.SecondOrderConeT(4)] * len(xyz), settings)
    solution = solver.solve()
    forces = np.asarray(solution.x).reshape(-1, 3)
    compression = np.einsum('qi,qi->q', forces, normals)
    excess = np.linalg.norm(forces - compression[:, None]*normals, axis=1) - mu*compression
    ray = -np.asarray(solution.z)[:6]
    ray /= np.linalg.norm(ray)
    value = np.einsum('qij,i->qj', operator, ray)
    ray_normal = np.einsum('qi,qi->q', value, normals)
    polar = ray_normal + mu*np.linalg.norm(value-ray_normal[:,None]*normals, axis=1)
    original_normal = force @ normals[0]
    record = {'component': component, 'status': str(solution.status), 'iterations': solution.iterations,
              'time_seconds': solution.solve_time, 'sample_count':len(xyz),
              'force_error': float(np.linalg.norm((equality@solution.x-target)[:3])),
              'moment_error': float(np.linalg.norm((equality@solution.x-target)[3:])),
              'minimum_compression_normalized': float(compression.min()),
              'maximum_cone_excess_normalized': float(excess.max()),
              'certificate_ray':ray.tolist(), 'certificate_polar_max':float(polar.max()),
              'certificate_margin':float(target@ray),
              'source_first_normal_compression_N':float(original_normal),
              'source_first_normal_tangential_ratio':float(np.linalg.norm(force-original_normal*normals[0])/original_normal)}
    records.append(record)
    np.savez_compressed(output / f'component-{component}.npz',force=forces,dual=np.asarray(solution.z),ray=ray,
                        positions=xyz,normals=normals,friction=mu,operator=operator,target=target)
    print(json.dumps(record), flush=True)
(output / 'summary.json').write_text(json.dumps({'clarabel_version':importlib.metadata.version('clarabel'), 'records':records},indent=2)+'\n')
