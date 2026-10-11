"""Offline integration diagnosis; no production solver or throughput claims."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
OUT = ROOT / 'nearby-fan-diagnosis-v1'
OUT.mkdir(exist_ok=True)
with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as data:
    vertices, faces, tets = (data[key] for key in ['vertices', 'surface_triangles', 'tetrahedra'])
triangles = vertices[faces]
normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
for i, face in enumerate(faces):
    owner = np.nonzero(np.sum(np.isin(tets, face), axis=1) == 3)[0]
    opposite = np.setdiff1d(tets[owner[0]], face)
    if normals[i] @ (vertices[opposite[0]] - triangles[i, 0]) < 0:
        normals[i] *= -1
with np.load(ROOT / 'online-mesh64-gjk-v1/results/failure/snapshot-147.npz') as data:
    anchor, position, force = (data[key][1, 2] for key in ['anchor', 'position', 'force'])
    radius, friction = (float(data[key][1, 2]) for key in ['radius', 'friction'])
    anchor_face = int(data['anchor_face'][1, 2])
moment = np.cross(position - anchor, force)
target = np.r_[force, moment / radius] / np.linalg.norm(force)


def nearest(triangle, point):
    origin = triangle[0]
    edges = (triangle[1:] - origin).T
    uv = np.linalg.solve(edges.T @ edges, edges.T @ (point - origin))
    choices = []
    if min(uv[0], uv[1], 1 - uv.sum()) >= 0:
        choices.append(origin + edges @ uv)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        edge = triangle[b] - triangle[a]
        t = np.clip(edge @ (point - triangle[a]) / (edge @ edge), 0, 1)
        choices.append(triangle[a] + t * edge)
    choices = np.array(choices)
    return choices[np.argmin(np.linalg.norm(choices - point, axis=1))]


def graph(s):
    used, vinverse = np.unique(faces, return_inverse=True)
    edges, einverse = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    inner = len(used) + einverse[:, None] * (s - 1) + np.arange(s - 1)
    ports = np.concatenate([vinverse.reshape(-1, 3), inner.reshape(len(faces), -1)], axis=1)
    positions = np.concatenate([vertices[used], ((1 - np.arange(1, s)[None, :, None] / s) * vertices[edges[:, 0], None] + np.arange(1, s)[None, :, None] / s * vertices[edges[:, 1], None]).reshape(-1, 3)])
    pairs = np.stack(np.broadcast_arrays(ports[:, :, None], ports[:, None, :]), axis=-1).reshape(-1, 2)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    length = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    start = len(positions)
    starts = ports[anchor_face]
    rows = np.r_[pairs[:, 0], np.full(len(starts), start)]
    cols = np.r_[pairs[:, 1], starts]
    weight = np.r_[length, np.linalg.norm(positions[starts] - anchor, axis=1)]
    matrix = coo_matrix((weight, (rows, cols)), shape=(start + 1, start + 1)).tocsr()
    distances = dijkstra(matrix, directed=True, indices=start)
    return ports, positions, distances[:-1]


def lp(points, ns, sides, outer):
    axis = np.eye(3)[np.argmin(np.abs(ns), axis=1)]
    first = np.cross(ns, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(ns, first)
    angle = np.arange(sides) * 2 * np.pi / sides
    mu = friction / np.cos(np.pi / sides) if outer else friction
    rays = ns[:, None] + mu * (first[:, None] * np.cos(angle)[None, :, None] + second[:, None] * np.sin(angle)[None, :, None])
    relative = (points - anchor) / radius
    columns = np.concatenate([rays, np.cross(relative[:, None], rays)], axis=2).reshape(-1, 6).T
    result = linprog(np.zeros(columns.shape[1]), A_eq=columns, b_eq=target, bounds=(0, None), options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    record = {'sides': sides, 'outer': outer, 'status': result.status, 'message': result.message}
    if result.success:
        f = np.sum(result.x.reshape(len(points), sides, 1) * rays, axis=1)
        compression = np.einsum('ij,ij->i', f, ns)
        record['wrench_error'] = float(np.linalg.norm(columns @ result.x - target))
        record['cone_violation'] = float(np.max(np.linalg.norm(f - compression[:, None] * ns, axis=1) - friction * compression))
    return record


centers = np.array([nearest(triangle, anchor) for triangle in triangles])
candidate_faces = np.nonzero(np.linalg.norm(centers - anchor, axis=1) < radius)[0]
q, w = np.polynomial.legendre.leggauss(10)
q, w = .5 * (q + 1), .5 * w
u, v = np.meshgrid(q, q, indexing='ij')
u, v = u.reshape(-1, 1), v.reshape(-1, 1)
records = []
for s in [2, 4, 8, 16]:
    ports, port_positions, port_distance = graph(s)
    points, indices, distances = [], [], []
    faces_summary = []
    for face in candidate_faces:
        center = centers[face]
        target_ports = ports[face]
        tri = triangles[face]
        face_points = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            if np.linalg.norm(np.cross(tri[a] - center, tri[b] - center)) > 1e-20:
                face_points.append((1 - u) * center + u * (1 - v) * tri[a] + u * v * tri[b])
        samples = np.concatenate(face_points)
        path_dist = np.min(port_distance[target_ports, None] + np.linalg.norm(port_positions[target_ports, None] - samples[None], axis=2), axis=0)
        if face == anchor_face:
            path_dist = np.linalg.norm(samples - anchor, axis=1)
        active = path_dist < radius
        faces_summary.append({'face': int(face), 'center': center.tolist(), 'euclidean_min': float(np.linalg.norm(center - anchor)), 'sample_min_distance': float(path_dist.min()), 'eligible_samples': int(active.sum())})
        points.append(samples[active])
        indices.extend([face] * active.sum())
        distances.extend(path_dist[active])
    points, indices = np.concatenate(points), np.array(indices)
    record = {'ports': s, 'faces': faces_summary, 'samples': len(points), 'lp': [lp(points, normals[indices], sides, outer) for sides, outer in [(128, False), (128, True), (512, True)]]}
    records.append(record)
    print(json.dumps(record), flush=True)
    np.savez_compressed(OUT / f'ports{s}.npz', positions=points, faces=indices, normals=normals[indices], distance=np.array(distances), target=target, anchor=anchor, position=position, force=force, radius=radius, friction=friction)
(OUT / 'audit.json').write_text(json.dumps({'scope': __doc__, 'anchor_face': anchor_face, 'candidate_faces': candidate_faces.tolist(), 'records': records}, indent=2) + '\n')
